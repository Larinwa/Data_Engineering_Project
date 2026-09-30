"""
This script creates a controlled corrupted version of the HVFHV trip data.
"""

import sqlite3
import numpy as np
import pandas as pd


DB_PATH = "ridehail_real.db"
SOURCE_TABLE = "hvfhv_trip_events"
CORRUPTED_TABLE = "hvfhv_trip_events_corrupted"

RANDOM_SEED = 42


rng = np.random.default_rng(RANDOM_SEED)


with sqlite3.connect(DB_PATH) as connection:
    trips = pd.read_sql(
        f"SELECT * FROM {SOURCE_TABLE}",
        connection
    )


print(f"Loaded {len(trips):,} rows from '{SOURCE_TABLE}'")


known_dropoff = ~trips["DOLocationID"].isin([264, 265])
known_indices = trips[known_dropoff].index

n_to_corrupt_zone = int(len(trips) * 0.10)

zone_indices = rng.choice(
    known_indices,
    size=n_to_corrupt_zone,
    replace=False
)

trips.loc[zone_indices, "DOLocationID"] = 265

print(
    f"Dropoff-zone violations introduced: "
    f"{n_to_corrupt_zone:,}"
)


n_to_delay = int(len(trips) * 0.10)

delay_indices = rng.choice(
    trips.index,
    size=n_to_delay,
    replace=False
)

trips["pickup_datetime"] = pd.to_datetime(
    trips["pickup_datetime"]
)

delay_minutes = rng.integers(
    15,
    26,
    size=n_to_delay
)

trips.loc[delay_indices, "pickup_datetime"] = (
    trips.loc[delay_indices, "pickup_datetime"]
    + pd.to_timedelta(delay_minutes, unit="m")
)


print(
    f"Responsiveness violations introduced: "
    f"{n_to_delay:,}"
)


with sqlite3.connect(DB_PATH) as connection:
    trips.to_sql(
        CORRUPTED_TABLE,
        connection,
        if_exists="replace",
        index=False
    )


print(
    f"Created '{CORRUPTED_TABLE}' in '{DB_PATH}'."
)
print(
    f"Original '{SOURCE_TABLE}' was preserved."
)