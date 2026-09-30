"""
The purpose of this script is to compute the observed request-to-pickup responsiveness baseline
for the January 2024 HVFHV sample.
"""

import sqlite3

import pandas as pd


DB_PATH = "ridehail_real.db"
TABLE_NAME = "hvfhv_trip_events"


with sqlite3.connect(DB_PATH) as connection:
    df = pd.read_sql(
        f"SELECT * FROM {TABLE_NAME}",
        connection
    )


df["request_datetime"] = pd.to_datetime(df["request_datetime"])
df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])


missing_request = df["request_datetime"].isna().sum()
missing_pickup = df["pickup_datetime"].isna().sum()

print(f"Rows loaded: {len(df):,}")
print(f"Rows missing request_datetime: {missing_request}")
print(f"Rows missing pickup_datetime: {missing_pickup}")


valid = df.dropna(
    subset=["request_datetime", "pickup_datetime"]
).copy()

valid["wait_minutes"] = (
    (valid["pickup_datetime"] - valid["request_datetime"])
    .dt.total_seconds()
    / 60
)


negative_waits = (valid["wait_minutes"] < 0).sum()

print(f"Rows with negative wait time: {negative_waits}")


valid = valid[valid["wait_minutes"] >= 0]


print()
print("=" * 60)
print("REQUEST-TO-PICKUP WAIT TIME DISTRIBUTION")
print("=" * 60)
print(valid["wait_minutes"].describe())


print()
print("PERCENTAGE PICKED UP WITHIN SELECTED THRESHOLDS")

for threshold in [5, 10, 15, 20, 30]:
    percentage = (
        (valid["wait_minutes"] <= threshold).mean() * 100
    )

    print(
        f"Within {threshold:>2} minutes: "
        f"{percentage:.2f}%"
    )