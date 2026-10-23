
"""
Validate the harmonized trip table against the consumer-defined contract.
"""

import sqlite3

import numpy as np
import pandas as pd
import yaml


DB_PATH = "ridehail_real.db"
CONTRACT_PATH = "harmonization_contract.yaml"

HVFHV_SOURCE_TABLE = "hvfhv_trip_events_clean"
YELLOW_SOURCE_TABLE = "yellow_trip_events_clean"
TABLE_NAME = "harmonized_trip_events_drift"

SOURCES = ["hvfhv", "yellow_taxi"]


with open(CONTRACT_PATH, "r", encoding="utf-8") as file:
    contract = yaml.safe_load(file)

with sqlite3.connect(DB_PATH) as connection:
    harmonized = pd.read_sql(
        f"SELECT * FROM {TABLE_NAME}", connection
    )
    hvfhv_source_count = pd.read_sql(
        f"SELECT COUNT(*) AS n FROM {HVFHV_SOURCE_TABLE}",
        connection,
    )["n"][0]
    yellow_source_count = pd.read_sql(
        f"SELECT COUNT(*) AS n FROM {YELLOW_SOURCE_TABLE}",
        connection,
    )["n"][0]

print("=" * 68)
print("HARMONIZATION CONTRACT VALIDATION")
print("=" * 68)
print(f"Contract: {contract['contract']['name']}")
print(f"Version: {contract['contract']['version']}")
print(f"Database: {DB_PATH}")
print(f"Table: {TABLE_NAME}")
print(f"Rows checked: {len(harmonized):,}")
print()

all_passed = True


def report(name, passed, detail):
    global all_passed
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] {name}")
    print(f"       {detail}")
    if not passed:
        all_passed = False


# 1. Required canonical fields
required_fields = contract["canonical_schema"]["required_fields"]
missing_fields = [
    field for field in required_fields
    if field not in harmonized.columns
]

report(
    "canonical_schema_fields",
    not missing_fields,
    (
        "All required fields are present."
        if not missing_fields
        else f"Missing fields: {missing_fields}"
    ),
)

if missing_fields:
    print("\nValidation cannot continue because required fields are missing.")
    print("=" * 68)
    raise SystemExit(1)


# 2. Mapping completeness by source
source_counts = harmonized["source"].value_counts(dropna=False).to_dict()
expected_counts = {
    "hvfhv": int(hvfhv_source_count),
    "yellow_taxi": int(yellow_source_count),
}

print()
for source in SOURCES:
    observed = int(source_counts.get(source, 0))
    expected = expected_counts[source]
    passed = observed == expected

    report(
        f"mapping_completeness ({source})",
        passed,
        f"{observed:,}/{expected:,} rows preserved by source count.",
    )

# Check that only expected source labels occur.
unexpected_sources = [
    source for source in source_counts
    if source not in SOURCES
]
report(
    "source_identity",
    not unexpected_sources and harmonized["source"].notna().all(),
    (
        "Only expected source labels are present."
        if not unexpected_sources and harmonized["source"].notna().all()
        else f"Unexpected or null source labels: {unexpected_sources}"
    ),
)


# 3. Required-field completeness, reported separately by source
print()
for source in SOURCES:
    data = harmonized[harmonized["source"] == source]
    for field in required_fields:
        if len(data) == 0:
            completeness = 0.0
        else:
            completeness = data[field].notna().mean()

        report(
            f"canonical_completeness ({source}.{field})",
            completeness == 1.0,
            f"{completeness:.2%} non-null ({data[field].notna().sum():,}/{len(data):,}).",
        )


# 4. Source-specific field availability and fare semantics
print()
source_specific_rules = {
    "hvfhv": {
        "must_be_populated": ["base_passenger_fare"],
        "must_be_null": ["yellow_taxi_selected_fare_components"],
    },
    "yellow_taxi": {
        "must_be_populated": ["yellow_taxi_selected_fare_components"],
        "must_be_null": ["base_passenger_fare"],
    },
}

for source, rules in source_specific_rules.items():
    data = harmonized[harmonized["source"] == source]

    for field in rules["must_be_populated"]:
        populated = data[field].notna().mean() if len(data) else 0.0
        report(
            f"source_field_availability ({source}.{field})",
            populated == 1.0,
            f"{populated:.2%} populated.",
        )

    for field in rules["must_be_null"]:
        null_rate = data[field].isna().mean() if len(data) else 0.0
        report(
            f"source_field_nullability ({source}.{field})",
            null_rate == 1.0,
            f"{null_rate:.2%} null, as required for this source.",
        )

# Yellow Taxi request-to-pickup time is unavailable in this mapping.
yellow = harmonized[harmonized["source"] == "yellow_taxi"]
yellow_request_null_rate = (
    yellow["request_to_pickup_minutes"].isna().mean()
    if len(yellow) else 0.0
)
report(
    "source_field_nullability (yellow_taxi.request_to_pickup_minutes)",
    yellow_request_null_rate == 1.0,
    f"{yellow_request_null_rate:.2%} null, as specified in the binding.",
)


# 5. Timestamp and duration consistency
print()
for source in SOURCES:
    data = harmonized[harmonized["source"] == source].copy()

    pickup = pd.to_datetime(data["pickup_datetime"], errors="coerce")
    dropoff = pd.to_datetime(data["dropoff_datetime"], errors="coerce")
    duration = pd.to_numeric(
        data["trip_duration_minutes"], errors="coerce"
    )

    derived_duration = (dropoff - pickup).dt.total_seconds() / 60
    comparable = (
        derived_duration.notna()
        & duration.notna()
        & np.isfinite(derived_duration)
        & np.isfinite(duration)
    )

    matches = (
        np.isclose(
            derived_duration[comparable],
            duration[comparable],
            atol=1e-6,
            rtol=0,
        )
        if comparable.any()
        else np.array([], dtype=bool)
    )

    passed = (
        comparable.sum() == len(data)
        and matches.all()
    )

    report(
        f"duration_consistency ({source})",
        passed,
        (
            f"{int(matches.sum()):,}/{len(data):,} rows match the "
            "pickup/drop-off timestamp derivation."
        ),
    )


# 6. Speed plausibility: every source row remains in the denominator.
speed_check = next(
    check for check in contract["checks"]
    if check["name"] == "speed_plausibility"
)

speed_limit = float(speed_check["speed_limit_mph"])
speed_threshold = float(speed_check["minimum_plausible_pct"]) / 100

print()
for source in SOURCES:
    data = harmonized[harmonized["source"] == source]

    distance = pd.to_numeric(
        data["trip_distance_miles"], errors="coerce"
    )
    duration = pd.to_numeric(
        data["trip_duration_minutes"], errors="coerce"
    )

    valid_inputs = (
        distance.notna()
        & duration.notna()
        & np.isfinite(distance)
        & np.isfinite(duration)
        & (distance >= 0)
        & (duration > 0)
    )

    speed = pd.Series(np.nan, index=data.index, dtype=float)
    speed.loc[valid_inputs] = (
        distance.loc[valid_inputs] / (duration.loc[valid_inputs] / 60)
    )

    plausible = valid_inputs & speed.between(0, speed_limit)
    total = len(data)
    plausible_count = int(plausible.sum())
    observed = plausible_count / total if total else 0.0

    report(
        f"speed_plausibility ({source})",
        total > 0 and observed >= speed_threshold,
        (
            f"{plausible_count:,}/{total:,} plausible "
            f"({observed:.2%}); required >= {speed_threshold:.0%}; "
            f"range 0-{speed_limit:.0f} mph."
        ),
    )


# 7. Unit consistency is specified in the bindings and contract.
# Numeric values alone cannot independently prove their physical units.
print()
print(
    "[INFO] unit_consistency: canonical distance is declared in miles "
    "and durations in minutes. Confirming the declared units requires "
    "review of the source bindings and transformation logic."
)

print()
print("=" * 68)
print(
    "OVERALL RESULT: CONTRACT SATISFIED"
    if all_passed
    else "OVERALL RESULT: CONTRACT VIOLATED"
)
print("=" * 68)
