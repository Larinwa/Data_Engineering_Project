
"""Validate Yellow Taxi data against its consumer-defined YAML contract."""

import re
import sqlite3

import pandas as pd
import yaml

DB_PATH = "ridehail_real.db"
CONTRACT_PATH = "yellow_taxi_data_contract.yaml"
TABLE_NAME = "yellow_trip_events"


def threshold_value(text):
    """Convert a threshold such as '>= 99%' into 0.99."""
    match = re.fullmatch(r"\s*>=\s*(\d+(?:\.\d+)?)\s*%\s*", str(text))
    if not match:
        raise ValueError(f"Unsupported threshold format: {text!r}")
    return float(match.group(1)) / 100


def main():
    with open(CONTRACT_PATH, encoding="utf-8") as file:
        contract = yaml.safe_load(file)

    with sqlite3.connect(DB_PATH) as conn:
        yellow = pd.read_sql_query(
            f'SELECT * FROM "{TABLE_NAME}"', conn
        )
        zones = pd.read_sql_query("SELECT LocationID FROM zones", conn)

    if yellow.empty:
        raise ValueError(f"{TABLE_NAME} contains no records.")

    print("=" * 68)
    print("YELLOW TAXI CONTRACT VALIDATION")
    print("=" * 68)
    print(f"Contract: {contract['contract']['name']}")
    print(f"Version: {contract['contract']['version']}")
    print(f"Rows checked: {len(yellow):,}\n")

    # Check schema fields declared in the YAML.
    fields = contract["schema"]["fields"]
    missing_fields = [
        field["name"] for field in fields
        if field["name"] not in yellow.columns
    ]

    all_passed = not missing_fields

    if missing_fields:
        print(f"[FAIL] Missing schema fields: {missing_fields}")
    else:
        print("[PASS] All contract fields are present.")

    pickup = pd.to_datetime(
        yellow["tpep_pickup_datetime"], errors="coerce"
    )
    dropoff = pd.to_datetime(
        yellow["tpep_dropoff_datetime"], errors="coerce"
    )
    duration_minutes = (
        (dropoff - pickup).dt.total_seconds() / 60
    )

    zones["LocationID"] = pd.to_numeric(
        zones["LocationID"], errors="coerce"
    )
    zone_ids = set(zones["LocationID"].dropna().astype(int))
    ordinary_zone_ids = zone_ids - {264, 265}

    pu = pd.to_numeric(yellow["PULocationID"], errors="coerce")
    do = pd.to_numeric(yellow["DOLocationID"], errors="coerce")
    distance = pd.to_numeric(yellow["trip_distance"], errors="coerce")
    fare = pd.to_numeric(yellow["fare_amount"], errors="coerce")

    observed = {
        "completeness_pickup": pu.isin(ordinary_zone_ids).mean(),
        "completeness_dropoff": do.isin(ordinary_zone_ids).mean(),
        "trip_duration_validity": (
            duration_minutes.notna()
            & (duration_minutes > 0)
            & (duration_minutes < 300)
        ).mean(),
        "trip_distance_validity": (
            distance.notna() & (distance > 0)
        ).mean(),
        "fare_validity": (
            fare.notna() & (fare >= 0)
        ).mean(),
    }

    print("\nDATA-QUALITY RESULTS")
    print("-" * 68)

    for slo in contract["slos"]:
        name = slo["name"]

        if slo.get("status") == "NOT APPLICABLE":
            print(f"[NOT APPLICABLE] {name}")
            continue

        if name not in observed:
            print(f"[REVIEW] No implemented metric for {name}")
            continue

        value = float(observed[name])
        threshold = threshold_value(slo["threshold"])
        passed = value >= threshold

        if not passed:
            all_passed = False

        print(
            f"[{'PASS' if passed else 'FAIL'}] {name}: "
            f"{value:.2%} (required >= {threshold:.0%}; "
            f"n={len(yellow):,})"
        )

    print("\nREVIEW COUNTS")
    print("-" * 68)
    print(f"Pickup IDs 264/265: {int(pu.isin([264, 265]).sum()):,}")
    print(f"Drop-off IDs 264/265: {int(do.isin([264, 265]).sum()):,}")
    print(f"Missing/unparseable pickup timestamps: {int(pickup.isna().sum()):,}")
    print(f"Missing/unparseable drop-off timestamps: {int(dropoff.isna().sum()):,}")
    print(f"Non-positive durations: {int((duration_minutes <= 0).sum()):,}")
    print(f"Durations >= 300 minutes: {int((duration_minutes >= 300).sum()):,}")
    print(f"Zero distances: {int((distance == 0).sum()):,}")
    print(f"Negative fares: {int((fare < 0).sum()):,}")

    print("\n" + "=" * 68)
    print(
        "OVERALL RESULT: CONTRACT SATISFIED"
        if all_passed
        else "OVERALL RESULT: CONTRACT VIOLATED"
    )
    print("Review counts are diagnostic and are not automatically defects.")
    print("=" * 68)


if __name__ == "__main__":
    main()
