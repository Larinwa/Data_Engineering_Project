"""
This script validates a harmonized trip table against the consumer-defined
harmonization contract.
"""

import sqlite3
import sys
import pandas as pd
import yaml


DB_PATH = "ridehail_real.db"
CONTRACT_PATH = "harmonization_contract.yaml"


if len(sys.argv) != 4:
    print(
        "Usage: python validate_harmonization_contract.py "
        "<hvfhv_table> <yellow_table> <harmonized_table>"
    )
    sys.exit(1)


HVFHV_SOURCE_TABLE = sys.argv[1]
YELLOW_SOURCE_TABLE = sys.argv[2]
TABLE_NAME = sys.argv[3]


with open(CONTRACT_PATH, "r", encoding="utf-8") as file:
    contract = yaml.safe_load(file)


with sqlite3.connect(DB_PATH) as connection:
    harmonized = pd.read_sql(
        f"SELECT * FROM {TABLE_NAME}",
        connection
    )

    source_counts = pd.read_sql(
        f"""
        SELECT source, COUNT(*) AS row_count
        FROM {TABLE_NAME}
        GROUP BY source
        """,
        connection
    )

    hvfhv_source_count = pd.read_sql(
        f"SELECT COUNT(*) AS n FROM {HVFHV_SOURCE_TABLE}",
        connection
    )["n"][0]

    yellow_source_count = pd.read_sql(
        f"SELECT COUNT(*) AS n FROM {YELLOW_SOURCE_TABLE}",
        connection
    )["n"][0]


print("=" * 60)
print("HARMONIZATION CONTRACT VALIDATION")
print("=" * 60)
print(f"Contract: {contract['contract']['name']}")
print(f"Database: {DB_PATH}")
print(f"Table: {TABLE_NAME}")
print(f"Rows checked: {len(harmonized):,}")
print()

source_counts_dict = dict(
    zip(
        source_counts["source"],
        source_counts["row_count"]
    )
)

hvfhv_harmonized_count = source_counts_dict.get("hvfhv", 0)
yellow_harmonized_count = source_counts_dict.get("yellow_taxi", 0)

hvfhv_ratio = hvfhv_harmonized_count / hvfhv_source_count
yellow_ratio = yellow_harmonized_count / yellow_source_count

mapping_passed = (hvfhv_ratio == 1.0) and (yellow_ratio == 1.0)

print("VALIDATION RESULTS")
print("-" * 60)

status = "PASS" if mapping_passed else "FAIL"
print(f"[{status}] mapping_completeness")
print(
    f"       HVFHV:  {hvfhv_harmonized_count:,}/{hvfhv_source_count:,} "
    f"= {hvfhv_ratio:.2%}"
)
print(
    f"       Yellow: {yellow_harmonized_count:,}/{yellow_source_count:,} "
    f"= {yellow_ratio:.2%}"
)

all_passed = mapping_passed

speed_check = next(
    check
    for check in contract["checks"]
    if check["name"] == "speed_plausibility"
)

speed_threshold = float(
    speed_check["threshold"]
    .split(">=")[1]
    .split("%")[0]
    .strip()
) / 100

speed_limit = float(
    speed_check["metric"]
    .split("<=")[-1]
    .split("mph")[0]
    .strip()
)

print()

for source in ["hvfhv", "yellow_taxi"]:
    source_data = harmonized[
        harmonized["source"] == source
    ].copy()

    valid_duration = (
        source_data["trip_duration_minutes"] > 0
    )

    source_data = source_data[valid_duration].copy()

    implied_speed = (
        source_data["trip_distance_miles"]
        / (source_data["trip_duration_minutes"] / 60)
    )

    plausible = (
        (implied_speed >= 0)
        & (implied_speed <= speed_limit)
    )

    observed = plausible.mean()
    passed = observed >= speed_threshold

    if not passed:
        all_passed = False

    status = "PASS" if passed else "FAIL"

    print(
        f"[{status}] speed_plausibility ({source})"
    )
    print(
        f"       Observed: {observed:.2%} "
        f"(threshold: >= {speed_threshold:.0%})"
    )
    print(
        f"       Speed range: 0-{speed_limit:.0f} mph"
    )

print()
print("=" * 60)

if all_passed:
    print("OVERALL RESULT: CONTRACT SATISFIED")
else:
    print("OVERALL RESULT: CONTRACT VIOLATED")

print("=" * 60)
