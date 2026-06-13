import os
import pandas as pd

RADAR_ROOT = os.path.expanduser("~/hpc_boost_v2/data/radar")
CSV_FILE = os.path.join(RADAR_ROOT, "combined_hardware_trails.csv")

CHUNKSIZE = 100_000

metadata_cols = [
    "Filename", "full_label", "label", "binarylabel",
    "method", "goal", "family_gene",
]

header = pd.read_csv(CSV_FILE, nrows=0)
columns = list(header.columns)
hpc_cols = columns[2:57]

print("=" * 80)
print("RADAR FULL DATA SUMMARY")
print("=" * 80)
print(f"CSV: {CSV_FILE}")
print(f"Total columns: {len(columns)}")
print(f"HPC event columns: {len(hpc_cols)}")
print("\nHPC columns:")
for col in hpc_cols:
    print(f"  {col}")

total_rows = 0
sample_row_counts = {}
counts = {col: {} for col in metadata_cols if col != "Filename"}

for chunk in pd.read_csv(CSV_FILE, usecols=metadata_cols, chunksize=CHUNKSIZE, low_memory=False):
    total_rows += len(chunk)

    for sample_id, n in chunk["Filename"].value_counts().items():
        sample_row_counts[sample_id] = sample_row_counts.get(sample_id, 0) + int(n)

    for col in counts:
        for value, n in chunk[col].value_counts(dropna=False).items():
            key = str(value)
            counts[col][key] = counts[col].get(key, 0) + int(n)

print("\n" + "=" * 80)
print("COUNTS")
print("=" * 80)
print(f"Total rows: {total_rows:,}")
print(f"Unique samples: {len(sample_row_counts):,}")

row_counts = list(sample_row_counts.values())
print(f"Rows per sample min: {min(row_counts):,}")
print(f"Rows per sample median: {pd.Series(row_counts).median():,.0f}")
print(f"Rows per sample max: {max(row_counts):,}")

print("\nTop 20 samples by row count:")
for sample_id, n in sorted(sample_row_counts.items(), key=lambda x: x[1], reverse=True)[:20]:
    print(f"  {sample_id}: {n:,}")

for col, col_counts in counts.items():
    print("\n" + "-" * 80)
    print(col)
    for value, n in sorted(col_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {value}: {n:,}")
