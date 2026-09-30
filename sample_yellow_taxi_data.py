"""
The script Creates a reproducible sample distributed across all 31 days of
January 2024 and stores the result in the project database.
"""

import sqlite3

import pandas as pd


SOURCE_URL = (
    "https://d37ci6vzurychx.cloudfront.net/"
    "trip-data/yellow_tripdata_2024-01.parquet"
)

DB_PATH = "ridehail_real.db"
TABLE_NAME = "yellow_trip_events"

TARGET_ROWS = 2500
SEED = 42

KEEP_COLUMNS = [
    "VendorID",
    "tpep_pickup_datetime",
    "tpep_dropoff_datetime",
    "passenger_count",
    "trip_distance",
    "RatecodeID",
    "PULocationID",
    "DOLocationID",
    "payment_type",
    "fare_amount",
    "extra",
    "mta_tax",
    "tip_amount",
    "tolls_amount",
    "improvement_surcharge",
    "total_amount",
    "congestion_surcharge",
]


print("Loading Yellow Taxi source...")

df = pd.read_parquet(
    SOURCE_URL,
    columns=KEEP_COLUMNS
)

df = df[
    (df["tpep_pickup_datetime"] >= "2024-01-01")
    & (df["tpep_pickup_datetime"] < "2024-02-01")
].copy()

df["pickup_day"] = df["tpep_pickup_datetime"].dt.day

rows_per_day = TARGET_ROWS // 31

sampled = (
    df.groupby("pickup_day", group_keys=False)
    .apply(
        lambda group: group.sample(
            n=min(len(group), rows_per_day),
            random_state=SEED
        )
    )
)

if len(sampled) < TARGET_ROWS:
    remaining = df.drop(sampled.index)

    top_up = remaining.sample(
        n=min(TARGET_ROWS - len(sampled), len(remaining)),
        random_state=SEED
    )

    sampled = pd.concat([sampled, top_up])

sampled = (
    sampled
    .drop(columns="pickup_day")
    .reset_index(drop=True)
)

print(f"Rows sampled: {len(sampled):,}")

with sqlite3.connect(DB_PATH) as connection:
    sampled.to_sql(
        TABLE_NAME,
        connection,
        if_exists="replace",
        index=False
    )

print(f"Saved to {DB_PATH} as table '{TABLE_NAME}'")