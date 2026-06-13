import os
import subprocess
import sys

def patch_file(filepath, replacements):
    with open(filepath, 'r') as f:
        content = f.read()
    
    for old, new in replacements:
        if old in content:
            content = content.replace(old, new)
        else:
            print(f"Warning: Could not find exactly this string in {filepath}:\n{old}")
            
    with open(filepath, 'w') as f:
        f.write(content)

def main():
    print("Patching codebase...")
    
    # 1. Patch run_comparison_final.py
    p1 = "hpc_boost_v2/experiments/exp2_baseline_comparison/run_comparison_final.py"
    if not os.path.exists(p1):
        print(f"ERROR: Cannot find {p1}. Please run this script from the parent directory of hpc_boost_v2.")
        sys.exit(1)
        
    r1 = [
        ('features[f"{col}_min"]  = float(np.min(vals))', 'features[f"{col}_min"]  = float(np.percentile(vals, 1))'),
        ('features[f"{col}_max"]  = float(np.max(vals))', 'features[f"{col}_max"]  = float(np.percentile(vals, 99))'),
        ('features[f"{col}_skew"] = float(skew(vals)) if n > 2 else 0.0', 'vals_64 = vals.astype(np.float64)\n        features[f"{col}_skew"] = float(skew(vals_64)) if n > 2 else 0.0'),
        ('features[f"{col}_kurt"] = float(kurtosis(vals)) if n > 2 else 0.0', 'features[f"{col}_kurt"] = float(kurtosis(vals_64)) if n > 2 else 0.0'),
        ('return (0, False)', 'return (1, False)')
    ]
    patch_file(p1, r1)
    
    # 2. Patch spread_score.py
    p2 = "hpc_boost_v2/src/ranking/spread_score.py"
    r2 = [
        ('return float(1.0 / iqr)', 'return float(iqr)')
    ]
    patch_file(p2, r2)
    
    print("Patching complete! Running the benchmark...")
    
    # Run the benchmark
    env = os.environ.copy()
    subprocess.run([sys.executable, p1], env=env, check=True)

if __name__ == "__main__":
    main()
