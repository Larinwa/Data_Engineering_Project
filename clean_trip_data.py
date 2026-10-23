
"""This script applies documented modelling-eligibility rules to trip data."""

import sqlite3

import numpy as np
import pandas as pd

DB_PATH = "ridehail_real.db"

SOURCE_TABLES = {
    "hvfhv": "hvfhv_trip_events",
    "yellow_taxi": "yellow_trip_events",
}


def clean_source(df, source):
    df = df.copy()

    if source == "hvfhv":
        pickup_col = "pickup_datetime"
        dropoff_col = "dropoff_datetime"
        distance_col = "trip_miles"
    else:
        pickup_col = "tpep_pickup_datetime"
        dropoff_col = "tpep_dropoff_datetime"
        distance_col = "trip_distance"

    pickup = pd.to_datetime(df[pickup_col], errors="coerce")
    dropoff = pd.to_datetime(df[dropoff_col], errors="coerce")
    distance = pd.to_numeric(df[distance_col], errors="coerce")

    duration = (dropoff - pickup).dt.total_seconds() / 60

    valid_timestamps = pickup.notna() & dropoff.notna()
    valid_duration = duration.notna() & np.isfinite(duration) & (duration > 0)
    valid_distance = (
        distance.notna()
        & np.isfinite(distance)
        & (distance > 0)
    )

    # Keep the reasons separate for transparent reporting.
    excluded_timestamp = ~valid_timestamps
    excluded_duration = ~valid_duration
    excluded_distance = ~valid_distance
    excluded_long_duration = duration >= 300

    eligible = (
        valid_timestamps
        & valid_duration
        & valid_distance
        & (duration < 300)
    )

    # Add canonical duration for downstream modelling.
    df["trip_duration_minutes"] = duration

    clean = df.loc[eligible].copy()

    # Diagnostic flags do not automatically exclude these records.
    if source == "yellow_taxi":
        fare = pd.to_numeric(df["fare_amount"], errors="coerce")
        df["review_negative_fare"] = fare < 0
        df["review_long_duration"] = duration >= 300
        df["review_special_pickup_zone"] = df["PULocationID"].isin([264, 265])
        df["review_special_dropoff_zone"] = df["DOLocationID"].isin([264, 265])
    else:
        fare = pd.to_numeric(df["base_passenger_fare"], errors="coerce")
        df["review_negative_fare"] = fare < 0
        df["review_long_duration"] = duration >= 300

    print(f"\n{source.upper()}")
    print(f"Original records: {len(df):,}")
    print(f"Invalid/missing timestamps: {excluded_timestamp.sum():,}")
    print(f"Invalid/non-positive durations: {excluded_duration.sum():,}")
    print(f"Invalid/non-positive distances: {excluded_distance.sum():,}")
    print(f"Eligible records retained: {eligible.sum():,}")
    print(f"Records excluded: {(~eligible).sum():,}")
    print(f"Negative fares flagged for review: {(fare < 0).sum():,}")
    print(
        f"Durations >= 300 minutes excluded by rule: "
        f"{excluded_long_duration.sum():,}"
    )

    # Save all eligible records. Original source tables remain unchanged.
    return clean


def main():
    with sqlite3.connect(DB_PATH) as conn:
        for source, table in SOURCE_TABLES.items():
            original = pd.read_sql_query(
                f'SELECT * FROM "{table}"', conn
            )

            if original.empty:
                raise ValueError(f"Source table '{table}' is empty.")

            cleaned = clean_source(original, source)
            output_table = f"{table}_clean"

            cleaned.to_sql(
                output_table,
                conn,
                if_exists="replace",
                index=False,
            )

            print(f"Saved cleaned data to: {output_table}")

    print("\nCleaning complete. Original source tables were preserved.")


if __name__ == "__main__":
    main()
