
import sqlite3
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

DB_PATH = "ridehail_real.db"
SEED = 42
TEST_SIZE = 0.20
MAX_DURATION_MINUTES = 300

# --------------------------------------------------
# 1. LOAD ORIGINAL AND CLEANED HARMONIZED DATA
# --------------------------------------------------

with sqlite3.connect(DB_PATH) as conn:
    original = pd.read_sql_query(
        "SELECT * FROM harmonized_trip_events",
        conn,
    )

    cleaned = pd.read_sql_query(
        "SELECT * FROM harmonized_trip_events_clean",
        conn,
    )

original = original.reset_index(drop=True)
original["_record_id"] = np.arange(len(original))

required_columns = [
    "source",
    "trip_distance_miles",
    "trip_duration_minutes",
]

for name, df in [("Original", original), ("Cleaned", cleaned)]:
    missing = set(required_columns) - set(df.columns)
    if missing:
        raise ValueError(
            f"{name} dataset is missing required columns: {sorted(missing)}"
        )

    for col in ["trip_duration_minutes", "trip_distance_miles"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")


# --------------------------------------------------
# 2. DEFINE ELIGIBLE RECORDS AND COMMON TEST SET
# --------------------------------------------------

duration = original["trip_duration_minutes"].to_numpy(dtype=float)
distance = original["trip_distance_miles"].to_numpy(dtype=float)

eligible_mask = (
    np.isfinite(duration)
    & np.isfinite(distance)
    & (duration > 0)
    & (distance > 0)
    & (duration < MAX_DURATION_MINUTES)
)

eligible_original = original.loc[eligible_mask].copy()

# Confirm the cleaned table contains the same eligible records.
identity_columns = [
    "source",
    "pickup_location_id",
    "dropoff_location_id",
    "pickup_datetime",
    "dropoff_datetime",
    "trip_duration_minutes",
    "trip_distance_miles",
]

for col in identity_columns:
    if col not in cleaned.columns or col not in eligible_original.columns:
        raise ValueError(
            f"Cannot verify cleaned record identity: missing '{col}'."
        )

original_identity = (
    eligible_original[identity_columns]
    .sort_values(identity_columns, na_position="first")
    .reset_index(drop=True)
)

cleaned_identity = (
    cleaned[identity_columns]
    .sort_values(identity_columns, na_position="first")
    .reset_index(drop=True)
)

if not original_identity.equals(cleaned_identity):
    raise ValueError(
        "The cleaned dataset does not match the eligible original records. "
        "Check the cleaning and harmonization steps before proceeding."
    )

# Split eligible records once so both experiments use identical test records.
train_eligible, test = train_test_split(
    eligible_original,
    test_size=TEST_SIZE,
    random_state=SEED,
    stratify=eligible_original["source"],
)

test_ids = set(test["_record_id"])

# Original training data includes ineligible records, except held-out test rows.
train_original = original.loc[
    ~original["_record_id"].isin(test_ids)
].copy()

# Cleaned training data contains only eligible records outside the test set.
train_cleaned = eligible_original.loc[
    ~eligible_original["_record_id"].isin(test_ids)
].copy()


# --------------------------------------------------
# 3. DEFINE MODELS
# --------------------------------------------------

FEATURES = ["trip_distance_miles", "source"]
TARGET = "trip_duration_minutes"


def make_model(model_type):
    preprocessing = ColumnTransformer(
        transformers=[
            (
                "source",
                OneHotEncoder(handle_unknown="ignore"),
                ["source"],
            ),
            ("distance", "passthrough", ["trip_distance_miles"]),
        ]
    )

    if model_type == "Linear Regression":
        estimator = LinearRegression()

    elif model_type == "Random Forest":
        estimator = RandomForestRegressor(
            n_estimators=300,
            random_state=SEED,
            n_jobs=-1,
            min_samples_leaf=2,
        )

    else:
        raise ValueError(f"Unsupported model: {model_type}")

    return Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            ("model", estimator),
        ]
    )


# --------------------------------------------------
# 4. TRAIN AND EVALUATE COMBINED-DATASET MODELS
# --------------------------------------------------

X_test = test[FEATURES]
y_test = test[TARGET]

results = []

print("=" * 65)
print("COMBINED-DATASET MODEL PERFORMANCE COMPARISON")
print("=" * 65)

print(f"\nOriginal harmonized records: {len(original):,}")
print(f"Eligible records:            {len(eligible_original):,}")
print(f"Cleaned harmonized records:   {len(cleaned):,}")
print(f"Original training records:    {len(train_original):,}")
print(f"Cleaned training records:     {len(train_cleaned):,}")
print(f"Common held-out test records: {len(test):,}")

print("\nTest-set source distribution:")
print(test["source"].value_counts().to_string())

for model_type in ["Linear Regression", "Random Forest"]:
    for data_type, train_data in [
        ("Original", train_original),
        ("Cleaned", train_cleaned),
    ]:
        # Exclude unusable feature or target values from model fitting.
        valid_train = (
            np.isfinite(
                train_data["trip_duration_minutes"].to_numpy(dtype=float)
            )
            & np.isfinite(
                train_data["trip_distance_miles"].to_numpy(dtype=float)
            )
            & train_data["source"].notna().to_numpy()
        )

        model_train = train_data.loc[valid_train].copy()

        model = make_model(model_type)
        model.fit(
            model_train[FEATURES],
            model_train[TARGET],
        )

        predictions = model.predict(X_test)

        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        r2 = r2_score(y_test, predictions)

        results.append(
            {
                "model": model_type,
                "data": data_type,
                "training_records": len(model_train),
                "test_records": len(test),
                "MAE": mae,
                "RMSE": rmse,
                "R2": r2,
            }
        )

        print(f"\n{model_type} — {data_type} data")
        print(f"  Training records: {len(model_train):,}")
        print(f"  MAE:  {mae:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  R²:   {r2:.4f}")


# --------------------------------------------------
# 5. SAVE COMBINED-DATASET RESULTS
# --------------------------------------------------

results_df = pd.DataFrame(results)
results_df.to_csv("model_performance_results.csv", index=False)

print("\n" + "=" * 65)
print("FINAL RESULTS")
print("=" * 65)
print(results_df.round(4).to_string(index=False))

print("\nResults saved to: model_performance_results.csv")
