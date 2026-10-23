
"""Validate the HVFHV sample against its consumer-defined YAML contract."""

import re
import sqlite3

import pandas as pd
import yaml

DB_PATH = "ridehail_real.db"
CONTRACT_PATH = "hvfhv_data_contract.yaml"
TABLE_NAME = "hvfhv_trip_events"


def threshold_value(text):
    """Convert a threshold such as '>= 95%' into 0.95."""
    match = re.fullmatch(
        r"\s*>=\s*(\d+(?:\.\d+)?)\s*%\s*",
        str(text),
    )
    if not match:
        raise ValueError(f"Unsupported threshold format: {text!r}")
    return float(match.group(1)) / 100


def main():
    with open(CONTRACT_PATH, encoding="utf-8") as file:
        contract = yaml.safe_load(file)

    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query(
            f'SELECT * FROM "{TABLE_NAME}"', conn
        )
        zones = pd.read_sql_query(
            "SELECT LocationID FROM zones", conn
        )

    if df.empty:
        raise ValueError(
            f"The table '{TABLE_NAME}' contains no records."
        )

    print("=" * 68)
    print("HVFHV CONTRACT VALIDATION")
    print("=" * 68)
    print(f"Contract: {contract['fundamentals']['name']}")
    print(f"Version: {contract['fundamentals']['version']}")
    print(f"Database: {DB_PATH}")
    print(f"Table: {TABLE_NAME}")
    print(f"Rows checked: {len(df):,}\n")

    # Check schema field presence.
    fields = contract["schema"]["fields"]
    missing_fields = [
        field["name"]
        for field in fields
        if field["name"] not in df.columns
    ]

    schema_passed = not missing_fields

    print("SCHEMA CHECKS")
    print("-" * 68)

    if missing_fields:
        print(f"[FAIL] Missing fields: {missing_fields}")
    else:
        print("[PASS] All contract fields are present.")

    # Check required fields for null values.
    for field in fields:
        name = field["name"]

        if name in df.columns and field.get("required", False):
            null_count = int(df[name].isna().sum())
            passed = null_count == 0

            print(
                f"[{'PASS' if passed else 'FAIL'}] "
                f"Required field {name}: {null_count:,} nulls"
            )

            if not passed:
                schema_passed = False

    # Parse timestamps without silently discarding invalid records.
    for name in [
        "request_datetime",
        "pickup_datetime",
        "dropoff_datetime",
    ]:
        df[name] = pd.to_datetime(df[name], errors="coerce")

    wait_minutes = (
        df["pickup_datetime"] - df["request_datetime"]
    ).dt.total_seconds() / 60

    elapsed_minutes = (
        df["dropoff_datetime"] - df["pickup_datetime"]
    ).dt.total_seconds() / 60

    # Compare location IDs with the supplied zone lookup.
    zones["LocationID"] = pd.to_numeric(
        zones["LocationID"], errors="coerce"
    )
    zone_ids = set(zones["LocationID"].dropna().astype(int))
    ordinary_zone_ids = zone_ids - {264, 265}

    pu = pd.to_numeric(df["PULocationID"], errors="coerce")
    do = pd.to_numeric(df["DOLocationID"], errors="coerce")
    distance = pd.to_numeric(df["trip_miles"], errors="coerce")
    fare = pd.to_numeric(df["base_passenger_fare"], errors="coerce")

    observed = {
        "completeness_dropoff_zone": do.isin(ordinary_zone_ids).mean(),
        "completeness_pickup_zone": pu.isin(ordinary_zone_ids).mean(),
        "request_pickup_ordering": (
            wait_minutes.notna() & (wait_minutes >= 0)
        ).mean(),
        "responsiveness": (
            wait_minutes.notna()
            & (wait_minutes >= 0)
            & (wait_minutes <= 10)
        ).mean(),
        "trip_duration_ordering": (
            elapsed_minutes.notna() & (elapsed_minutes > 0)
        ).mean(),
        "trip_distance_validity": (
            distance.notna() & (distance > 0)
        ).mean(),
        "fare_review": (
            fare.notna() & (fare >= 0)
        ).mean(),
    }

    print("\nDATA-QUALITY RESULTS")
    print("-" * 68)

    all_passed = schema_passed

    for slo in contract["service_level_objectives"]:
        name = slo["name"]

        if slo.get("testable_live") is False:
            print(f"[NOT TESTABLE] {name}")
            continue

        # Diagnostic only: the 60-second tolerance is provisional.
        if name == "trip_duration_consistency":
            reported = pd.to_numeric(
                df["trip_time"], errors="coerce"
            )
            calculated = elapsed_minutes * 60
            comparable = reported.notna() & calculated.notna()
            within_60s = (
                (reported - calculated).abs() <= 60
            ) & comparable

            print(
                f"[REVIEW] {name}: "
                f"{int(within_60s.sum()):,}/"
                f"{int(comparable.sum()):,} comparable records "
                "within 60 seconds; diagnostic only."
            )
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
            f"{value:.2%} "
            f"(required >= {threshold:.0%}; n={len(df):,})"
        )

    # Report diagnostic counts separately from contract decisions.
    print("\nREVIEW COUNTS")
    print("-" * 68)
    print(f"Pickup IDs 264/265: {int(pu.isin([264, 265]).sum()):,}")
    print(f"Drop-off IDs 264/265: {int(do.isin([264, 265]).sum()):,}")
    print(
        "Negative request-to-pickup intervals: "
        f"{int((wait_minutes < 0).sum()):,}"
    )
    print(
        "Missing/unparseable request or pickup timestamps: "
        f"{int(wait_minutes.isna().sum()):,}"
    )
    print(
        f"Non-positive elapsed trip durations: "
        f"{int((elapsed_minutes <= 0).sum()):,}"
    )
    print(
        f"Non-positive trip distances: "
        f"{int((distance <= 0).sum()):,}"
    )
    print(f"Negative base fares: {int((fare < 0).sum()):,}")

    print("\n" + "=" * 68)
    print(
        "OVERALL RESULT: CONTRACT SATISFIED"
        if all_passed
        else "OVERALL RESULT: CONTRACT VIOLATED"
    )
    print("REVIEW diagnostics are not counted as pass/fail.")
    print("=" * 68)


if __name__ == "__main__":
    main()
