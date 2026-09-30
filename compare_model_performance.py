"""
The purpose of this experiment is to compare Linear Regression models trained on
undrifted harmonized data and harmonized data containing the
controlled Yellow Taxi trip-distance unit drift.
"""

import sqlite3

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split


DB_PATH = "ridehail_real.db"

CLEAN_TABLE = "harmonized_trips"
DRIFTED_TABLE = "harmonized_trips_drifted"

FEATURE = "trip_distance_miles"
TARGET = "trip_duration_minutes"

TEST_SIZE = 0.20
RANDOM_STATE = 42


with sqlite3.connect(DB_PATH) as connection:
    clean_data = pd.read_sql(
        f"SELECT * FROM {CLEAN_TABLE}",
        connection
    )

    drifted_data = pd.read_sql(
        f"SELECT * FROM {DRIFTED_TABLE}",
        connection
    )


print("=" * 60)
print("DOWNSTREAM CONSUMER EXPERIMENT")
print("=" * 60)

print()
print("Consumer task:")
print("Predict trip duration from trip distance.")

print()
print("Model:")
print("Linear Regression")

print()
print("Experimental conditions:")
print("1. Undrifted (harmonization-layer clean)")
print("2. Drifted (harmonization-layer unit drift)")

print()
print("Evaluation:")
print("Both models use the same held-out test set.")
print("Metrics: MAE and RMSE")


print("\nDATA OVERVIEW")
print("-" * 60)

print(f"Undrifted harmonized data: {len(clean_data):,} rows")
print(f"Drifted harmonized data:   {len(drifted_data):,} rows")

print()
print("Source distribution:")
print(clean_data["source"].value_counts())


print("\nDATA INSPECTION")
print("-" * 60)
print(clean_data[[FEATURE, TARGET]].describe())


print("\nMISSING VALUES")
print("-" * 60)
print(clean_data[[FEATURE, TARGET]].isna().sum())


if len(clean_data) != len(drifted_data):
    raise ValueError(
        "The undrifted and drifted datasets contain "
        "different numbers of rows."
    )


comparison_columns = [
    column
    for column in clean_data.columns
    if column != FEATURE
]

if not clean_data[comparison_columns].equals(
    drifted_data[comparison_columns]
):
    raise ValueError(
        "The drifted dataset does not preserve the same records "
        "and non-distance values as the undrifted dataset."
    )


sns.set_style("ticks")

plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=clean_data,
    x=FEATURE,
    y=TARGET,
    hue="source",
    alpha=0.5
)

plt.title("Trip Distance and Trip Duration")
plt.xlabel("Trip Distance (miles)")
plt.ylabel("Trip Duration (minutes)")
plt.tight_layout()
plt.show()


clean_train, clean_test = train_test_split(
    clean_data,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE
)

clean_train = clean_train.copy()
clean_test = clean_test.copy()

drifted_train = drifted_data.loc[clean_train.index].copy()


print("\nEXPERIMENTAL SPLIT")
print("-" * 60)
print(f"Training records: {len(clean_train):,}")
print(f"Test records:     {len(clean_test):,}")
print()
print("The same test records are used to evaluate both models.")


results = {}
models = {}


for condition, training_data in [
    ("Undrifted (harmonization-layer clean)", clean_train),
    ("Drifted (harmonization-layer unit drift)", drifted_train),
]:
    model = LinearRegression()

    model.fit(
        training_data[[FEATURE]],
        training_data[TARGET]
    )

    predictions = model.predict(clean_test[[FEATURE]])

    mae = mean_absolute_error(
        clean_test[TARGET],
        predictions
    )

    rmse = mean_squared_error(
        clean_test[TARGET],
        predictions
    ) ** 0.5

    results[condition] = {
        "MAE": mae,
        "RMSE": rmse,
        "Slope": model.coef_[0],
        "Intercept": model.intercept_,
    }

    models[condition] = model


undrifted_mae = results[
    "Undrifted (harmonization-layer clean)"
]["MAE"]

drifted_mae = results[
    "Drifted (harmonization-layer unit drift)"
]["MAE"]

undrifted_rmse = results[
    "Undrifted (harmonization-layer clean)"
]["RMSE"]

drifted_rmse = results[
    "Drifted (harmonization-layer unit drift)"
]["RMSE"]


mae_difference = drifted_mae - undrifted_mae
rmse_difference = drifted_rmse - undrifted_rmse

mae_percentage_change = (
    mae_difference / undrifted_mae
) * 100

rmse_percentage_change = (
    rmse_difference / undrifted_rmse
) * 100


print("\nMODEL EVALUATION")
print("-" * 60)
print(
    f"{'Condition':<48}"
    f"{'MAE (min)':>12}"
    f"{'RMSE (min)':>14}"
)
print("-" * 74)

for condition, metrics in results.items():
    print(
        f"{condition:<48}"
        f"{metrics['MAE']:>12.3f}"
        f"{metrics['RMSE']:>14.3f}"
    )


print("\nLEARNED REGRESSION RELATIONSHIPS")
print("-" * 60)

for condition, metrics in results.items():
    print(f"\n{condition}")
    print(
        f"Duration = {metrics['Slope']:.4f} × distance "
        f"+ {metrics['Intercept']:.4f}"
    )


result_plot = pd.DataFrame({
    "Training Condition": ["Undrifted", "Drifted"],
    "MAE": [undrifted_mae, drifted_mae],
    "RMSE": [undrifted_rmse, drifted_rmse],
})

error_plot = result_plot.melt(
    id_vars="Training Condition",
    value_vars=["MAE", "RMSE"],
    var_name="Metric",
    value_name="Error"
)


plt.figure(figsize=(9, 6))

ax = sns.barplot(
    data=error_plot,
    x="Training Condition",
    y="Error",
    hue="Metric"
)

plt.title("Prediction Error Under Different Data Conditions")
plt.xlabel("Training Condition")
plt.ylabel("Prediction Error (minutes)")

for container in ax.containers:
    ax.bar_label(container, fmt="%.3f", padding=3)

plt.tight_layout()
plt.show()


x_min = clean_test[FEATURE].min()
x_max = clean_test[FEATURE].max()

x_values = pd.DataFrame({
    FEATURE: [x_min, x_max]
})


plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=clean_test,
    x=FEATURE,
    y=TARGET,
    hue="source",
    alpha=0.35
)

for condition, model in models.items():
    predictions = model.predict(x_values)

    plt.plot(
        x_values[FEATURE],
        predictions,
        label=condition,
        linewidth=2
    )

plt.title("Learned Relationship Between Trip Distance and Duration")
plt.xlabel("Trip Distance (miles)")
plt.ylabel("Trip Duration (minutes)")
plt.legend()
plt.tight_layout()
plt.show()


print("\nDATA-QUALITY EFFECT")
print("-" * 60)
print(f"MAE change: {mae_difference:+.3f} minutes")
print(f"MAE relative change: {mae_percentage_change:+.2f}%")
print(f"Equivalent MAE change: {mae_difference * 60:+.1f} seconds")
print(f"\nRMSE change: {rmse_difference:+.3f} minutes")
print(f"RMSE relative change: {rmse_percentage_change:+.2f}%")


print("\nEXPERIMENT INTERPRETATION")
print("-" * 60)

if mae_difference > 0:
    print(
        "The harmonization-layer unit drift produced a higher MAE "
        "on the shared test set."
    )
elif mae_difference < 0:
    print(
        "The harmonization-layer unit drift produced a lower MAE "
        "on the shared test set."
    )
else:
    print("The two conditions produced the same MAE.")


if rmse_difference > 0:
    print(
        "RMSE was also higher under the harmonization-layer "
        "unit-drift condition."
    )
elif rmse_difference < 0:
    print(
        "RMSE was slightly lower under the harmonization-layer "
        "unit-drift condition."
    )
else:
    print("RMSE was unchanged between the two conditions.")


print(
    "\nThe experiment measures the downstream effect of this "
    "specific controlled harmonization-layer unit drift; it does "
    "not establish a universal effect of data contracts."
)
