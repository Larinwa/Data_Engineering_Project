"""
This script creates a controlled semantic/unit-drift version of the Yellow Taxi data.
The experiment rescales trip_distance from miles to kilometres while
leaving the column name and database schema unchanged. 
"""

import sqlite3

import pandas as pd


DB_PATH = "ridehail_real.db"
SOURCE_TABLE = "yellow_trip_events"
DRIFTED_TABLE = "yellow_trip_events_drifted"

MILES_TO_KM = 1.60934


with sqlite3.connect(DB_PATH) as connection:
    yellow = pd.read_sql(
        f"SELECT * FROM {SOURCE_TABLE}",
        connection
    )

    drifted = yellow.copy()
    drifted["trip_distance"] = (
        drifted["trip_distance"] * MILES_TO_KM
    )

    drifted.to_sql(
        DRIFTED_TABLE,
        connection,
        if_exists="replace",
        index=False
    )


print(
    f"Created '{DRIFTED_TABLE}' from '{SOURCE_TABLE}'."
)
print(
    f"trip_distance values were rescaled by {MILES_TO_KM} "
    "to simulate an undocumented unit change from miles to kilometres."
)