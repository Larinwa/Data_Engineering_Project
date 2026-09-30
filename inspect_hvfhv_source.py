"""
This script inspects the January 2024 HVFHV source file.
"""

import pyarrow.parquet as pq


FILE = "fhvhv_tripdata_2024-01.parquet"
SAMPLE_SIZE = 2000


parquet_file = pq.ParquetFile(FILE)

print("=" * 60)
print("HVFHV SOURCE INSPECTION")
print("=" * 60)
print(f"Source file: {FILE}")
print(f"Total rows: {parquet_file.metadata.num_rows:,}")
print(f"Total columns: {parquet_file.metadata.num_columns}")
print(f"Row groups: {parquet_file.num_row_groups}")

print()
print("SCHEMA")
print("-" * 60)

for field in parquet_file.schema_arrow:
    print(f"{field.name}: {field.type}")

print()
print("SAMPLE RECORDS")
print("-" * 60)

sample = next(
    parquet_file.iter_batches(batch_size=SAMPLE_SIZE)
).to_pandas()

print(f"Rows inspected: {len(sample):,}")
print()
print(sample.head())