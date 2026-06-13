import os
import glob

ZHOU_ROOT = os.path.expanduser("~/hpc_boost_v2/data/zhou_asiaccs")

print("=" * 80)
print("ZHOU DATASET INSPECTION")
print("=" * 80)

if not os.path.exists(ZHOU_ROOT):
    print(f"Error: {ZHOU_ROOT} not found.")
    exit(1)

print("\nTop-level contents:")
for item in sorted(os.listdir(ZHOU_ROOT)):
    path = os.path.join(ZHOU_ROOT, item)
    if os.path.isdir(path):
        count = sum(1 for _ in os.listdir(path))
        print(f"  [DIR]  {item:40s} ({count} items)")
    else:
        size_kb = os.path.getsize(path) / 1024
        print(f"  [FILE] {item:40s} ({size_kb:.1f} KB)")

print("\nAll files:")
for dirpath, _, filenames in os.walk(ZHOU_ROOT):
    for f in sorted(filenames):
        if f.startswith("."):
            continue
        path = os.path.join(dirpath, f)
        rel = os.path.relpath(path, ZHOU_ROOT)
        size_kb = os.path.getsize(path) / 1024
        print(f"  {rel:60s} ({size_kb:.1f} KB)")

for pattern in ["**/*.csv", "**/*.txt", "**/*.data", "**/*.npy"]:
    files = glob.glob(os.path.join(ZHOU_ROOT, pattern), recursive=True)
    if files:
        print(f"\nFound {len(files)} files matching {pattern}")
        first = files[0]
        print(f"  First file: {first}")
        with open(first, "r", errors="ignore") as fh:
            lines = fh.readlines()[:10]
        print("  First 10 lines:")
        for line in lines:
            print(f"    {line.rstrip()}")
