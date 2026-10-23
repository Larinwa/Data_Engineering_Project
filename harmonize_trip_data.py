
"""
Harmonize HVFHV and Yellow Taxi trip records into a shared canonical schema.
"""

import sqlite3

import pandas as pd


DB_PATH = "ridehail_real.db"
HVFHV_TABLE = "hvfhv_trip_events"
YELLOW_TABLE = "yellow_trip_events"
OUTPUT_TABLE = "harmonized_trip_events"

with sqlite3.connect(DB_PATH) as connection:
    hvfhv = pd.read_sql(
        f"SELECT * FROM {HVFHV_TABLE}",
        connection,
    )

    yellow = pd.read_sql(
        f"SELECT * FROM {YELLOW_TABLE}",
        connection,
    )


# Parse source timestamps before deriving canonical temporal fields.
for column in [
    "request_datetime",
    "pickup_datetime",
    "dropoff_datetime",
]:
    hvfhv[column] = pd.to_datetime(hvfhv[column], errors="coerce")

for column in [
    "tpep_pickup_datetime",
    "tpep_dropoff_datetime",
]:
    yellow[column] = pd.to_datetime(yellow[column], errors="coerce")


# Map HVFHV fields to the canonical schema.
hvfhv_canonical = pd.DataFrame({
    "source": "hvfhv",
    "pickup_location_id": hvfhv["PULocationID"],
    "dropoff_location_id": hvfhv["DOLocationID"],
    "pickup_datetime": hvfhv["pickup_datetime"],
    "dropoff_datetime": hvfhv["dropoff_datetime"],
    "trip_duration_minutes": (
        (
            hvfhv["dropoff_datetime"]
            - hvfhv["pickup_datetime"]
        ).dt.total_seconds() / 60
    ),
    "trip_distance_miles": hvfhv["trip_miles"],
    "base_passenger_fare": hvfhv["base_passenger_fare"],
    "yellow_taxi_selected_fare_components": pd.NA,
    "request_to_pickup_minutes": (
        (
            hvfhv["pickup_datetime"]
            - hvfhv["request_datetime"]
        ).dt.total_seconds() / 60
    ),
})


# Define the fare components currently used for Yellow Taxi.
yellow_fare_components = [
    "fare_amount",
    "extra",
    "mta_tax",
    "tolls_amount",
    "improvement_surcharge",
    "congestion_surcharge",
]


# Map Yellow Taxi fields to the same canonical structure.
yellow_canonical = pd.DataFrame({
    "source": "yellow_taxi",
    "pickup_location_id": yellow["PULocationID"],
    "dropoff_location_id": yellow["DOLocationID"],
    "pickup_datetime": yellow["tpep_pickup_datetime"],
    "dropoff_datetime": yellow["tpep_dropoff_datetime"],
    "trip_duration_minutes": (
        (
            yellow["tpep_dropoff_datetime"]
            - yellow["tpep_pickup_datetime"]
        ).dt.total_seconds() / 60
    ),
    "trip_distance_miles": yellow["trip_distance"],
    "base_passenger_fare": pd.NA,
    "yellow_taxi_selected_fare_components": (
        yellow[yellow_fare_components].sum(axis=1, min_count=1)
    ),
    "request_to_pickup_minutes": pd.NA,
})


# Combine records using the shared canonical field structure.
harmonized = pd.concat(
    [hvfhv_canonical, yellow_canonical],
    ignore_index=True,
)

with sqlite3.connect(DB_PATH) as connection:
    harmonized.to_sql(
        OUTPUT_TABLE,
        connection,
        if_exists="replace",
        index=False,
    )


print(f"HVFHV source: {HVFHV_TABLE}")
print(f"Yellow Taxi source: {YELLOW_TABLE}")
print(f"HVFHV rows harmonized: {len(hvfhv_canonical):,}")
print(f"Yellow Taxi rows harmonized: {len(yellow_canonical):,}")
print(f"Total harmonized rows: {len(harmonized):,}")
print(f"Saved to {DB_PATH} as table '{OUTPUT_TABLE}'")
