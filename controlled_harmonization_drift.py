
import sqlite3

import pandas as pd

DB_PATH = "ridehail_real.db"

with sqlite3.connect(DB_PATH) as connection:
    data = pd.read_sql(
        "SELECT * FROM harmonized_trip_events",
        connection,
    )

    mask = data["source"] == "yellow_taxi"

    # Simulate miles converted to kilometres but still labelled as miles.
    data.loc[mask, "trip_distance_miles"] *= 1.60934

    data.to_sql(
        "harmonized_trip_events_drift",
        connection,
        if_exists="replace",
        index=False,
    )

print("Controlled-drift table created.")
print(f"HVFHV rows unchanged: {(~mask).sum():,}")
print(f"Yellow Taxi rows affected: {mask.sum():,}")
print("Original harmonized table preserved.")
