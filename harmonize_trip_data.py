"""
The purpose of this script is to harmonize two trip-data sources into a shared canonical schema.
"""

import sqlite3
import sys

import pandas as pd


DB_PATH = "ridehail_real.db"


if len(sys.argv) != 4:
    print(
        "Usage: python harmonize_trip_data.py "
        "<hvfhv_table> <yellow_table> <output_table>"
    )
    sys.exit(1)


HVFHV_TABLE = sys.argv[1]
YELLOW_TABLE = sys.argv[2]
OUTPUT_TABLE = sys.argv[3]


with sqlite3.connect(DB_PATH) as connection:
    hvfhv = pd.read_sql(
        f"SELECT * FROM {HVFHV_TABLE}",
        connection
    )

    yellow = pd.read_sql(
        f"SELECT * FROM {YELLOW_TABLE}",
        connection
    )


for column in [
    "request_datetime",
    "pickup_datetime",
    "dropoff_datetime",
]:
    hvfhv[column] = pd.to_datetime(hvfhv[column])

for column in [
    "tpep_pickup_datetime",
    "tpep_dropoff_datetime",
]:
    yellow[column] = pd.to_datetime(yellow[column])


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
    "total_fare": hvfhv["base_passenger_fare"],
    "request_to_pickup_minutes": (
        (
            hvfhv["pickup_datetime"]
            - hvfhv["request_datetime"]
        ).dt.total_seconds() / 60
    ),
})


yellow_fare_components = [
    "fare_amount",
    "extra",
    "mta_tax",
    "tolls_amount",
    "improvement_surcharge",
    "congestion_surcharge",
]


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
    "total_fare": yellow[yellow_fare_components].sum(axis=1),
    "request_to_pickup_minutes": pd.NA,
})


harmonized = pd.concat(
    [hvfhv_canonical, yellow_canonical],
    ignore_index=True
)


with sqlite3.connect(DB_PATH) as connection:
    harmonized.to_sql(
        OUTPUT_TABLE,
        connection,
        if_exists="replace",
        index=False
    )


print(f"HVFHV source: {HVFHV_TABLE}")
print(f"Yellow Taxi source: {YELLOW_TABLE}")
print(f"HVFHV rows harmonized: {len(hvfhv_canonical):,}")
print(f"Yellow Taxi rows harmonized: {len(yellow_canonical):,}")
print(f"Total harmonized rows: {len(harmonized):,}")
print(
    f"Saved to {DB_PATH} as table '{OUTPUT_TABLE}'"
)