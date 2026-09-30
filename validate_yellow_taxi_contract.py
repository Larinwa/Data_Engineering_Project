"""
This script validates Yellow Taxi trip data against the consumer-defined contract.
"""

import sqlite3

import pandas as pd
import yaml


DB_PATH = "ridehail_real.db"
CONTRACT_PATH = "yellow_taxi_data_contract.yaml"
TABLE_NAME = "yellow_trip_events"


with open(CONTRACT_PATH, "r", encoding="utf-8") as file:
    contract = yaml.safe_load(file)


with sqlite3.connect(DB_PATH) as connection:
    yellow = pd.read_sql(
        f"SELECT * FROM {TABLE_NAME}",
        connection
    )
    zones = pd.read_sql(
        "SELECT * FROM zones",
        connection
    )


total_rows = len(yellow)
known_zone_ids = set(zones["LocationID"]) - {264, 265}


pickup_validity = yellow["PULocationID"].isin(known_zone_ids).mean()

dropoff_validity = yellow["DOLocationID"].isin(known_zone_ids).mean()

fare_validity = (yellow["fare_amount"] >= 0).mean()

pickup = pd.to_datetime(yellow["tpep_pickup_datetime"])
dropoff = pd.to_datetime(yellow["tpep_dropoff_datetime"])

duration_minutes = (
    (dropoff - pickup).dt.total_seconds() / 60
)

trip_duration_validity = (
    (duration_minutes > 0)
    & (duration_minutes < 300)
).mean()


observed = {
    "completeness_pickup": pickup_validity,
    "completeness_dropoff": dropoff_validity,
    "fare_validity": fare_validity,
    "trip_duration_validity": trip_duration_validity,
}


print(
    f"Contract: {contract['contract']['name']} "
    f"(v{contract['contract']['version']})"
)
print(f"Rows checked: {total_rows:,}\n")


all_passed = True

for slo in contract["slos"]:
    name = slo["name"]

    if slo.get("status") == "NOT APPLICABLE":
        print(f"[NOT APPLICABLE] {name}")
        continue

    threshold = float(
        slo["threshold"].replace(">=", "").strip().replace("%", "")
    ) / 100

    observed_value = observed[name]
    passed = observed_value >= threshold

    if not passed:
        all_passed = False

    status = "PASS" if passed else "FAIL"

    print(
        f"[{status}] {name}: "
        f"{observed_value:.4%} "
        f"(threshold: >= {threshold:.0%})"
    )


print(
    f"\n{'CONTRACT SATISFIED' if all_passed else 'CONTRACT VIOLATED'}"
)