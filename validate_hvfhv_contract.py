"""
The purpose of this script is to validate an HVFHV trip table against the consumer-defined contract.
"""

import sqlite3
import sys

import pandas as pd
import yaml


DB_PATH = "ridehail_real.db"
CONTRACT_PATH = "hvfhv_data_contract.yaml"


if len(sys.argv) != 2:
    print(
        "Usage: python validate_hvfhv_contract.py "
        "<table_name>"
    )
    sys.exit(1)


TABLE_NAME = sys.argv[1]


with open(CONTRACT_PATH, "r", encoding="utf-8") as file:
    contract = yaml.safe_load(file)


with sqlite3.connect(DB_PATH) as connection:
    df = pd.read_sql(
        f"SELECT * FROM {TABLE_NAME}",
        connection
    )

    zones = pd.read_sql(
        "SELECT * FROM zones",
        connection
    )


known_zone_ids = set(zones["LocationID"]) - {264, 265}


print("=" * 60)
print("HVFHV CONTRACT VALIDATION")
print("=" * 60)
print(f"Contract: {contract['fundamentals']['name']}")
print(f"Database: {DB_PATH}")
print(f"Table: {TABLE_NAME}")
print(f"Rows checked: {len(df):,}")
print()


expected_fields = {
    field["name"]
    for field in contract["schema"]["fields"]
}

actual_fields = set(df.columns)

missing_fields = expected_fields - actual_fields

schema_passed = not missing_fields


df["request_datetime"] = pd.to_datetime(
    df["request_datetime"],
    errors="coerce"
)

df["pickup_datetime"] = pd.to_datetime(
    df["pickup_datetime"],
    errors="coerce"
)


valid_timestamps = df.dropna(
    subset=["request_datetime", "pickup_datetime"]
).copy()


valid_timestamps["wait_minutes"] = (
    (
        valid_timestamps["pickup_datetime"]
        - valid_timestamps["request_datetime"]
    )
    .dt.total_seconds()
    / 60
)


non_negative_waits = valid_timestamps[
    valid_timestamps["wait_minutes"] >= 0
]


observed = {
    "completeness_dropoff_zone": (
        df["DOLocationID"].isin(known_zone_ids)
    ).mean(),

    "completeness_pickup_zone": (
        df["PULocationID"].isin(known_zone_ids)
    ).mean(),

    "responsiveness": (
        non_negative_waits["wait_minutes"] <= 10
    ).mean()
    if len(non_negative_waits) > 0
    else 0.0,

    "request_pickup_ordering": (
        valid_timestamps["wait_minutes"] >= 0
    ).mean()
    if len(valid_timestamps) > 0
    else 0.0,
}


all_passed = schema_passed


print("VALIDATION RESULTS")
print("-" * 60)


if schema_passed:
    print("[PASS] Schema conformance")
else:
    print(
        "[FAIL] Schema conformance: "
        f"Missing required fields: {missing_fields}"
    )


for slo in contract["service_level_objectives"]:
    name = slo["name"]

    if slo.get("testable_live") is False:
        print(f"[NOT TESTABLE] {name}")
        continue

    threshold = (
        float(
            slo["threshold"]
            .replace(">=", "")
            .replace("%", "")
            .strip()
        )
        / 100
    )

    observed_value = observed[name]
    passed = observed_value >= threshold

    if not passed:
        all_passed = False

    status = "PASS" if passed else "FAIL"

    print(f"[{status}] {name}")
    print(
        f"       Observed: {observed_value:.2%} "
        f"(threshold: >= {threshold:.0%})"
    )


print()
print("=" * 60)


if all_passed:
    print("OVERALL RESULT: CONTRACT SATISFIED")
else:
    print("OVERALL RESULT: CONTRACT VIOLATED")


print("=" * 60)