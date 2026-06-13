import ast
import os
import re

ROOT = os.path.expanduser(
    "~/hpc_boost_v2/data/zhou_asiaccs/results_for_ML/extracted/open_source_data"
)

print("=" * 80)
print("ZHOU COMPACT SUMMARY")
print("=" * 80)

for dirpath, _, filenames in os.walk(ROOT):
    for name in sorted(filenames):
        if not name.endswith(".txt"):
            continue

        path = os.path.join(dirpath, name)
        rel = os.path.relpath(path, ROOT)
        size_mb = os.path.getsize(path) / (1024 * 1024)

        print("\n" + "-" * 80)
        print(rel)
        print(f"size_mb: {size_mb:.2f}")

        with open(path, "r", errors="ignore") as f:
            text = f.read()

        print(f"chars: {len(text):,}")
        print(f"lines: {text.count(chr(10)) + 1:,}")

        hex_events = sorted(set(re.findall(r'"(0x[0-9A-Fa-f]+)"', text)))
        quoted_keys = sorted(set(re.findall(r'"([^"]+)"\s*:', text)))
        numeric_outer_keys = sorted(set(re.findall(r'(?<![\w"])(\d+)\s*:', text)))

        print(f"hex_event_keys: {len(hex_events)}")
        if hex_events:
            print(f"hex_event_key_sample: {hex_events[:20]}")

        print(f"quoted_keys_count: {len(quoted_keys)}")
        if quoted_keys:
            print(f"quoted_key_sample: {quoted_keys[:20]}")

        print(f"numeric_outer_keys_count: {len(numeric_outer_keys)}")
        if numeric_outer_keys:
            print(f"numeric_outer_key_sample: {numeric_outer_keys[:20]}")

        # Try parsing only smaller files safely.
        if size_mb <= 50:
            try:
                obj = ast.literal_eval(text)
                print(f"parsed_type: {type(obj).__name__}")
                if isinstance(obj, dict):
                    print(f"top_level_keys: {len(obj)}")
                    top_keys = list(obj.keys())[:10]
                    print(f"top_level_key_sample: {top_keys}")

                    first_key = top_keys[0] if top_keys else None
                    first_val = obj[first_key] if first_key is not None else None
                    print(f"first_value_type: {type(first_val).__name__}")

                    if isinstance(first_val, dict):
                        inner_keys = list(first_val.keys())[:10]
                        print(f"inner_key_sample: {inner_keys}")
                        if inner_keys:
                            arr = first_val[inner_keys[0]]
                            try:
                                print(f"first_series_len: {len(arr)}")
                            except TypeError:
                                pass
                    elif hasattr(first_val, "__len__"):
                        print(f"first_value_len: {len(first_val)}")
            except Exception as e:
                print(f"parse_error: {type(e).__name__}: {e}")
        else:
            print("parse_skipped: file over 50 MB")
