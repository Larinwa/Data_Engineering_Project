"""
The purpose of this script is to compute observed SLO baselines for the sampled January 2024
Yellow Taxi data. It measures pickup-zone validity, dropoff-zone validity, fare validity,
and trip-duration validity to ground the Yellow Taxi data contract
in observed data.
"""

import sqlite3

import pandas as pd


DB_PATH = "ridehail_real.db"

with sqlite3.connect(DB_PATH) as connection:
    yellow = pd.read_sql(
        "SELECT * FROM yellow_trip_events",
        connection
    )
    zones = pd.read_sql(
        "SELECT * FROM zones",
        connection
    )


total_rows = len(yellow)
known_zone_ids = set(zones["LocationID"]) - {264, 265}


pickup_valid = yellow["PULocationID"].isin(known_zone_ids).sum()
print(
    f"Pickup zone validity:  "
    f"{pickup_valid}/{total_rows} = {pickup_valid / total_rows:.4%}"
)


dropoff_valid = yellow["DOLocationID"].isin(known_zone_ids).sum()
print(
    f"Dropoff zone validity: "
    f"{dropoff_valid}/{total_rows} = {dropoff_valid / total_rows:.4%}"
)


fare_valid = (yellow["fare_amount"] >= 0).sum()
print(
    f"Fare validity:         "
    f"{fare_valid}/{total_rows} = {fare_valid / total_rows:.4%}"
)


pickup = pd.to_datetime(yellow["tpep_pickup_datetime"])
dropoff = pd.to_datetime(yellow["tpep_dropoff_datetime"])

duration_minutes = (
    (dropoff - pickup).dt.total_seconds() / 60
)

duration_valid = (
    (duration_minutes > 0)
    & (duration_minutes < 300)
).sum()

print(
    f"Trip-duration validity: "
    f"{duration_valid}/{total_rows} = "
    f"{duration_valid / total_rows:.4%}"
)