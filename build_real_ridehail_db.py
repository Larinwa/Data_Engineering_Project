"""
The purpose of this script is to build the initial SQLite database from the January 2024 HVFHV source
and NYC taxi-zone lookup data.
"""

import sqlite3
import pandas as pd


PARQUET_FILE = "fhvhv_tripdata_2024-01.parquet"
ZONE_FILE = "taxi_zone_lookup.csv"
OUTPUT_DB = "ridehail_real.db"

TARGET_SAMPLE_SIZE = 2517
RANDOM_SEED = 42

HVFHV_TABLE = "hvfhv_trip_events"
ZONE_TABLE = "zones"


print("=" * 60)
print("BUILDING RIDE-HAIL DATABASE")
print("=" * 60)

print(f"Source: {PARQUET_FILE}")
print(f"Target HVFHV sample: {TARGET_SAMPLE_SIZE:,} rows")
print(f"Random seed: {RANDOM_SEED}")


print("\nLoading HVFHV source...")

hvfhv = pd.read_parquet(PARQUET_FILE)

sample = hvfhv.sample(
    n=TARGET_SAMPLE_SIZE,
    random_state=RANDOM_SEED
).reset_index(drop=True)


print(f"HVFHV rows sampled: {len(sample):,}")


print("\nChecking date coverage...")

pickup_dates = pd.to_datetime(
    sample["pickup_datetime"]
).dt.date

print(f"Earliest pickup date: {pickup_dates.min()}")
print(f"Latest pickup date:   {pickup_dates.max()}")
print(
    f"Distinct pickup days: "
    f"{pickup_dates.nunique()}"
)


print("\nLoading zone lookup...")

zones = pd.read_csv(ZONE_FILE)

print(f"Zone rows loaded: {len(zones):,}")


print("\nWriting database...")

with sqlite3.connect(OUTPUT_DB) as connection:

    zones.to_sql(
        ZONE_TABLE,
        connection,
        if_exists="replace",
        index=False
    )

    sample.to_sql(
        HVFHV_TABLE,
        connection,
        if_exists="replace",
        index=False
    )


print()
print("=" * 60)
print("DATABASE CREATED")
print("=" * 60)

print(f"Database: {OUTPUT_DB}")
print(f"Table: {HVFHV_TABLE} ({len(sample):,} rows)")
print(f"Table: {ZONE_TABLE} ({len(zones):,} rows)")