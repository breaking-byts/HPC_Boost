"""
Step 3.1: Inspect the RaDaR dataset.
Goal: Understand folder structure, columns, sample counts, and data types.
"""
import os
import pandas as pd

RADAR_ROOT = os.path.expanduser("~/hpc_boost_v2/data/radar")

print("=" * 80)
print("RADAR DATASET INSPECTION")
print("=" * 80)

# 1. Find the combined_hardware_trails.csv file
csv_file = os.path.join(RADAR_ROOT, "combined_hardware_trails.csv")

if not os.path.exists(csv_file):
    print(f"Error: {csv_file} not found.")
    exit(1)

size_mb = os.path.getsize(csv_file) / (1024 * 1024)
print(f"\nFound dataset file: {csv_file} ({size_mb:.1f} MB)")

# 2. Inspect the file in detail
print("\n" + "=" * 80)
print("DETAILED INSPECTION")
print("=" * 80)

# Read just first 1000 rows to be fast
df = pd.read_csv(csv_file, nrows=1000)

print(f"\nShape (first 1000 rows): {df.shape}")
print(f"Total columns: {len(df.columns)}")

print(f"\n--- ALL COLUMN NAMES ({len(df.columns)}) ---")
for i, col in enumerate(df.columns):
    print(f"  [{i:3d}] {col:40s}  dtype={df[col].dtype}  nulls={df[col].isnull().sum()}")

print(f"\n--- FIRST 5 ROWS ---")
print(df.head())

print(f"\n--- DATA TYPES ---")
print(df.dtypes)

print(f"\n--- BASIC STATS (numeric columns only) ---")
print(df.describe())

# 3. Check for label/class columns
print(f"\n--- POTENTIAL LABEL COLUMNS ---")
for col in df.columns:
    if df[col].dtype == 'object' or df[col].nunique() < 50:
        print(f"  {col}: unique={df[col].nunique()}, values={df[col].unique()[:10]}")

# 4. We don't need to count rows per file since it's a single file.
# The total rows was already found by wc -l (3,473,471)
print(f"\n  {'TOTAL ROWS':20s}: 3,473,471 rows (approx)")


