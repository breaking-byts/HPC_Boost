# HPC-Boost v2 — Detailed Server Execution Plan

> [!IMPORTANT]
> This plan is for **SSH-only access**. No BIOS, no malware execution, no `perf` collection.
> We work entirely with pre-collected datasets (RaDaR from Kaggle + Zhou from GitHub).

---

## Stage 1: Environment Setup (~30 min)

### Step 1.1 — Create project directory

```bash
mkdir -p ~/hpc_boost_v2
cd ~/hpc_boost_v2

# Create full directory tree
mkdir -p data/{radar,zhou_asiaccs,processed/{rankings,splits,results}}
mkdir -p src/{ranking,baselines,detection,features,recommender,utils}
mkdir -p experiments/{exp1_ranking_validation,exp2_baseline_comparison,exp3_category_analysis,exp4_sensitivity,exp5_recommender}
mkdir -p figures notebooks logs
```

**Verify**: `find ~/hpc_boost_v2 -type d | head -20` should show the tree.

---

### Step 1.2 — Set up Python environment

```bash
cd ~/hpc_boost_v2

# Option A: If conda is available
conda create -n hpcboost python=3.11 -y
conda activate hpcboost

# Option B: If only system Python is available
python3 -m venv venv
source venv/bin/activate
```

---

### Step 1.3 — Install dependencies

```bash
# Core scientific stack
pip install numpy scipy pandas scikit-learn statsmodels matplotlib seaborn

# HPC-Boost specific
pip install pymannkendall   # Mann-Kendall trend test (Score 2)
pip install arch            # Phillips-Perron unit root test (Score 4)

# ML models
pip install torch --index-url https://download.pytorch.org/whl/cpu  # CPU-only, saves space
pip install xgboost lightgbm

# Utilities
pip install tqdm joblib jupyter

# Save for reproducibility
pip freeze > requirements.txt
```

**Verify**: `python -c "import numpy, scipy, pandas, sklearn, statsmodels, pymannkendall, arch, torch, xgboost; print('All OK')"`.

---

### Step 1.4 — Create `__init__.py` files for the package

```bash
touch src/__init__.py
touch src/ranking/__init__.py
touch src/baselines/__init__.py
touch src/detection/__init__.py
touch src/features/__init__.py
touch src/recommender/__init__.py
touch src/utils/__init__.py
```

---

## Stage 2: Dataset Download (~15 min)

### Step 2.1 — Download Zhou et al. (AsiaCCS 2018) from GitHub

```bash
cd ~/hpc_boost_v2/data/zhou_asiaccs
git clone https://github.com/bu-icsg/Hardware_Performance_Counters_Can_Detect_Malware_Myth_or_Fact.git .
```

**Verify**: `ls -la` should show data files, code, README.

---

### Step 2.2 — Download RaDaR (JUGAAD Trails) from Kaggle

```bash
# Install Kaggle CLI
pip install kaggle

# Set up API token (you need to paste your kaggle.json)
mkdir -p ~/.kaggle
# Copy your API token: go to kaggle.com → Account → Create New Token → download kaggle.json
# Then: cp /path/to/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json

# Download the dataset
cd ~/hpc_boost_v2/data/radar
kaggle datasets download -d sareenakarapoola/open-malware-research-jugaad-trails

# Unzip
unzip open-malware-research-jugaad-trails.zip
rm open-malware-research-jugaad-trails.zip  # Clean up
```

> [!NOTE]
> If the Kaggle username in the URL is wrong, search on Kaggle for "Open Malware Research Jugaad Trails" and use the correct download command shown on the dataset page.

**Verify**: `find . -name "combined_hardware_trails.csv" | head -20` — should list multiple CSV files in different subfolders.

---

## Stage 3: Data Inspection & Understanding (~1-2 hours)

> [!IMPORTANT]
> This is the most important preliminary step. We need to understand the exact format of the data before writing any code. Every decision downstream depends on this.

### Step 3.1 — Inspect the RaDaR dataset structure

Create file `src/utils/inspect_radar.py`:

```python
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
```

**Run**: `cd ~/hpc_boost_v2 && python src/utils/inspect_radar.py 2>&1 | tee logs/radar_inspection.log`

**What to look for in the output**:
1. **Column names** — Which ones are HPC events? Which are metadata (sample ID, family, label)?
2. **How many rows per class** — This tells us dataset balance
3. **Data types** — Are HPC values integers or floats? Any strings mixed in?
4. **Null values** — Any missing data we need to handle?
5. **Unique values in label columns** — What are the exact malware family names?

> [!IMPORTANT]
> **Share this output with me.** The column names dictate how we write every subsequent script.

---

### Step 3.2 — Inspect the Zhou dataset

Create file `src/utils/inspect_zhou.py`:

```python
"""
Step 3.2: Inspect the Zhou et al. (AsiaCCS 2018) dataset.
"""
import os

ZHOU_ROOT = os.path.expanduser("~/hpc_boost_v2/data/zhou_asiaccs")

print("=" * 80)
print("ZHOU DATASET INSPECTION")
print("=" * 80)

# List top-level structure
print("\nTop-level contents:")
for item in sorted(os.listdir(ZHOU_ROOT)):
    path = os.path.join(ZHOU_ROOT, item)
    if os.path.isdir(path):
        count = sum(1 for _ in os.listdir(path))
        print(f"  [DIR]  {item:40s} ({count} items)")
    else:
        size_kb = os.path.getsize(path) / 1024
        print(f"  [FILE] {item:40s} ({size_kb:.1f} KB)")

# Recursively find all data files
print("\nAll files:")
for dirpath, dirnames, filenames in os.walk(ZHOU_ROOT):
    for f in sorted(filenames):
        if f.startswith('.'):
            continue
        path = os.path.join(dirpath, f)
        rel = os.path.relpath(path, ZHOU_ROOT)
        size_kb = os.path.getsize(path) / 1024
        print(f"  {rel:60s} ({size_kb:.1f} KB)")

# Try to read any CSV/data files
import glob
for pattern in ['**/*.csv', '**/*.txt', '**/*.data', '**/*.npy']:
    files = glob.glob(os.path.join(ZHOU_ROOT, pattern), recursive=True)
    if files:
        print(f"\nFound {len(files)} files matching {pattern}")
        # Inspect first one
        first = files[0]
        print(f"  First file: {first}")
        with open(first, 'r', errors='ignore') as fh:
            lines = fh.readlines()[:10]
        print(f"  First 10 lines:")
        for line in lines:
            print(f"    {line.rstrip()}")
```

**Run**: `cd ~/hpc_boost_v2 && python src/utils/inspect_zhou.py 2>&1 | tee logs/zhou_inspection.log`

---

### Step 3.3 — Create a unified data summary

After running Steps 3.1 and 3.2, create `logs/data_summary.md` documenting:

| Property | RaDaR | Zhou |
|---|---|---|
| Total samples | ? | 1,924 (962 + 962) |
| Malware classes | ? (expect 9) | Not labeled by class |
| HPC events | ? (expect ~54) | ? |
| Rows per sample | ? (time-series) | ? |
| Platform | Windows | Linux |
| Label column name | ? | ? |
| Sample ID column | ? | ? |

---

## Stage 4: Implement the Ranking Module (~3-5 days)

> [!IMPORTANT]
> This is the **intellectual core** of HPC-Boost. The ranking module assigns a quality score to each HPC event for each binary, determining which events are best for monitoring that specific binary.

### Step 4.1 — Data loader

Create `src/utils/data_loader.py`:

```python
"""
Unified data loader for RaDaR and Zhou datasets.
Adapts each dataset into a common format:
  - Dict[sample_id] → Dict[event_name] → np.ndarray (time-series)
  - Metadata: label (benign/malicious), family, class
"""
import os
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

class RadarDataLoader:
    """
    Loads the RaDaR dataset from Kaggle.
    
    Expected structure (update after Step 3.1 inspection):
    - Multiple folders, each containing combined_hardware_trails.csv
    - Each CSV has 71 columns: metadata + HPC event values
    - Each row is a time-series snapshot
    """
    
    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        self.csv_file = os.path.join(self.root_dir, "combined_hardware_trails.csv")
        
        # Extracted from our inspection:
        self.sample_id_column = 'Filename'
        self.label_column = 'full_label'  # Indicates normal vs malware name
        self.family_column = 'family_gene'
        self.goal_column = 'goal'
        
        # The 54 PMU events (columns 2 through 55 in the CSV)
        self.hpc_columns = [
            'Core_cyc', 'Ref_cyc', 'Instruct', 'Ins_Retd', 'ILenStal', 'DTLBLoadMissWD', 
            'DTLBStoreMissW', 'DTLBStrMiss_SH', 'DTLBStrMiss_WC', 'DTLBStrMiss_WD', 
            'FP_Assist_ANY', 'HW_Intrs_Rcvd', 'ICache_Misses', 'IDQ_All_DSB_C', 
            'IDQ_AllMite_UO', 'L1D_P_Miss_Oc', 'L3_LAT_C_Miss', 'M_Ld_LLCH.XS_M', 
            'M_Ld_LLCH.XS_N', 'MLdULLCM_LDRAM', 'M_Ld_Ret_L1Hit', 'M_Ld_Ret_L2Hit', 
            'M_Ld_Ret_L3Hit', 'Loop_uops', 'Dec_uops', 'Cach_uops', 'Uops', 'Macrofus', 
            'Uops_F.D.', 'res.stl.', 'uop_p0', 'uop_p1', 'uop_p2', 'uop_p3', 'uop_p4', 
            'uop_p5', 'uop_p05', 'BrMispred', 'Mov_elim', 'BrTaken', 'Mov_elim-', 
            'L1D_Miss', 'ITLBMissW', 'ITLBMissS', 'L1D_Rep', 'L2ReqAll', 'L2ReqPFms', 
            'Load_Hit_Pre', 'BrMispExec_Any', 'BrMispRetd_All', 'CPL_CYCLES(R0)', 
            'CPU_CLK_UNH_RF', 'DSB2MIT_SW_CNT', 'DTLBLoadMiss_W', 'DTLBLoadMissWC'
        ]
    
    def load_all(self) -> Tuple[pd.DataFrame, dict]:
        """Load the CSV and extract metadata."""
        if not os.path.exists(self.csv_file):
            raise FileNotFoundError(f"Dataset not found at {self.csv_file}")
            
        print(f"Loading {self.csv_file} (approx 1.2GB)...")
        df = pd.read_csv(self.csv_file)
        
        # Basic cleaning (handle the empty string vs 'normal' in labels if necessary)
        df['malware_class'] = df[self.family_column]
        
        metadata = {
            'total_rows': len(df),
            'classes': df['malware_class'].value_counts().to_dict(),
            'hpc_columns': self.hpc_columns,
            'num_hpc_events': len(self.hpc_columns),
            'unique_samples': df[self.sample_id_column].nunique()
        }
        
        return df, metadata
    
    def get_per_sample_timeseries(self, df: pd.DataFrame) -> Dict:
        """
        Group the DataFrame by sample ID to get per-sample time-series.
        Returns: {sample_id: {event_name: np.array, ...}, ...}
        
        NOTE: Update self.sample_id_column after Step 3.1 inspection.
        """
        samples = {}
        for sample_id, group in df.groupby(self.sample_id_column):
            samples[sample_id] = {
                'data': {col: group[col].values for col in self.hpc_columns},
                'label': group['malware_class'].iloc[0],
                'num_timesteps': len(group),
            }
        return samples


# Placeholder for Zhou loader — will be implemented after Step 3.2
class ZhouDataLoader:
    """Loader for Zhou et al. AsiaCCS 2018 dataset."""
    def __init__(self, root_dir: str):
        self.root_dir = root_dir
    
    def load_all(self):
        raise NotImplementedError("Implement after Step 3.2 inspection")
```

**Why**: We need a clean abstraction over the raw data so all downstream code (ranking, baselines, experiments) works with a consistent interface.

---

### Step 4.2 — Score 1: Spread Score

Create `src/ranking/spread_score.py`:

```python
"""
Score 1: Spread Score
Measures how tight/consistent an HPC event's value distribution is.
A tight distribution (low IQR) means deviations are easy to detect.

Mathematical definition:
  SpreadScore(e) = median_{r ∈ runs}( 1 / IQR_r(e) )

Why IQR not std: IQR is robust to outliers from context switches and interrupts.
Why median not mean: Robust to a single bad run.
"""
import numpy as np
from typing import List

def spread_score(time_series: np.ndarray) -> float:
    """
    Compute spread score for a single event's time-series data.
    
    Args:
        time_series: 1D array of HPC event values over time
        
    Returns:
        Spread score (higher = tighter distribution = better for detection)
    """
    if len(time_series) < 4:
        return 0.0
    
    q75, q25 = np.percentile(time_series, [75, 25])
    iqr = q75 - q25
    
    if iqr == 0:
        return float('inf')  # Perfectly consistent
    
    return 1.0 / iqr


def spread_score_multi_run(runs: List[np.ndarray]) -> float:
    """
    Compute spread score across multiple runs.
    Uses median across runs for robustness.
    
    Args:
        runs: List of 1D arrays, one per independent run
    """
    scores = [spread_score(run) for run in runs]
    # Filter out inf for median calculation
    finite_scores = [s for s in scores if np.isfinite(s)]
    
    if not finite_scores:
        return float('inf')
    
    return float(np.median(finite_scores))


def spread_score_single(time_series: np.ndarray) -> float:
    """
    Simplified version for pre-collected datasets where we only have
    one continuous trace (not multiple runs).
    
    Strategy: Split the trace into 5 equal segments and treat each
    as a separate "run" to simulate multi-run behavior.
    """
    n = len(time_series)
    if n < 20:
        return spread_score(time_series)
    
    segment_size = n // 5
    segments = [time_series[i*segment_size:(i+1)*segment_size] for i in range(5)]
    
    return spread_score_multi_run(segments)
```

---

### Step 4.3 — Score 2: Trend Score

Create `src/ranking/trend_score.py`:

```python
"""
Score 2: Trend Score
Measures consistency of time-series trend.
Good events show consistent behavior (similar trend) across runs.

Uses Mann-Kendall non-parametric trend test.
Mathematical definition:
  TrendScore(e) = |mean(S_r)| / std(S_r)   (signal-to-noise of trend)
"""
import numpy as np
import pymannkendall as mk
from typing import List

def trend_score(time_series: np.ndarray) -> float:
    """
    Compute Mann-Kendall S statistic for a single time-series.
    Returns the S statistic (positive = upward trend, negative = downward).
    """
    if len(time_series) < 10:
        return 0.0
    
    try:
        result = mk.original_test(time_series)
        return result.s
    except Exception:
        return 0.0


def trend_score_multi_run(runs: List[np.ndarray]) -> float:
    """
    Compute trend consistency across runs.
    High score = consistent trend direction and magnitude.
    """
    s_values = [trend_score(run) for run in runs]
    
    mean_s = np.mean(s_values)
    std_s = np.std(s_values)
    
    if std_s == 0:
        return float('inf')  # Perfect consistency
    
    return abs(mean_s) / std_s


def trend_score_single(time_series: np.ndarray) -> float:
    """
    For pre-collected datasets: split into 5 segments, compute consistency.
    """
    n = len(time_series)
    if n < 50:
        return abs(trend_score(time_series))
    
    segment_size = n // 5
    segments = [time_series[i*segment_size:(i+1)*segment_size] for i in range(5)]
    
    return trend_score_multi_run(segments)
```

---

### Step 4.4 — Score 3: Correlation Score

Create `src/ranking/correlation_score.py`:

```python
"""
Score 3: Correlation Score
Measures how independent this event is from all other events.
Independent events capture different system behavior = less redundancy.

Uses Spearman rank correlation (not Pearson):
  - Detects monotonic non-linear relationships
  - Non-parametric (no normality assumption)

Mathematical definition:
  CorrScore(e_i) = mean_{j≠i}(1 - |ρ_s(e_i, e_j)|)
"""
import numpy as np
from scipy.stats import spearmanr

def correlation_score(event_idx: int, all_events_matrix: np.ndarray) -> float:
    """
    Compute independence score for event at event_idx.
    
    Args:
        event_idx: Index of the target event
        all_events_matrix: 2D array of shape (num_timesteps, num_events)
        
    Returns:
        Independence score (higher = more independent = better)
    """
    target = all_events_matrix[:, event_idx]
    num_events = all_events_matrix.shape[1]
    
    correlations = []
    for j in range(num_events):
        if j == event_idx:
            continue
        other = all_events_matrix[:, j]
        
        # Skip constant columns
        if np.std(target) == 0 or np.std(other) == 0:
            correlations.append(0.0)
            continue
        
        rho, _ = spearmanr(target, other)
        if np.isnan(rho):
            rho = 0.0
        correlations.append(abs(rho))
    
    if not correlations:
        return 1.0
    
    # Average independence (1 - |correlation|)
    avg_independence = np.mean([1 - c for c in correlations])
    
    return float(avg_independence)
```

---

### Step 4.5 — Score 4: Stationarity Score

Create `src/ranking/stationarity_score.py`:

```python
"""
Score 4: Stationarity Score
Measures how stationary (stable statistical properties) the event is.
Anomaly detectors work best when the baseline is stationary.

Short series (< 100 points): PP + KPSS tests
Long series (≥ 100 points): ADF + PP tests
"""
import numpy as np
from statsmodels.tsa.stattools import adfuller, kpss

def stationarity_score(time_series: np.ndarray) -> float:
    """
    Compute stationarity score.
    Higher score = more stationary = better for anomaly detection.
    """
    T = len(time_series)
    
    if T < 20:
        return 0.0
    
    # Remove mean to avoid issues with constant offsets
    data = time_series - np.mean(time_series)
    
    # Skip constant series
    if np.std(data) < 1e-10:
        return float('inf')  # Perfectly stationary
    
    scores = []
    
    try:
        # ADF test (null: non-stationary)
        # More negative stat = more evidence of stationarity
        adf_stat, adf_pval, _, _, _, _ = adfuller(data, maxlag=min(T//4, 20))
        scores.append(-adf_stat)  # Negate so higher = more stationary
    except Exception:
        pass
    
    try:
        # KPSS test (null: stationary)
        # Lower stat = more evidence of stationarity
        kpss_stat, kpss_pval, _, _ = kpss(data, regression='c', nlags='auto')
        scores.append(-kpss_stat)  # Negate so higher = more stationary
    except Exception:
        pass
    
    if not scores:
        return 0.0
    
    return float(np.mean(scores))
```

---

### Step 4.6 — Combined Ranking Engine

Create `src/ranking/ranker.py`:

```python
"""
Combined Ranking Engine: The core of HPC-Boost.
Takes all 4 scores, normalizes them, and produces a final event ranking.
"""
import numpy as np
import pandas as pd
from typing import Dict, List
from .spread_score import spread_score_single
from .trend_score import trend_score_single
from .correlation_score import correlation_score
from .stationarity_score import stationarity_score

class HPCBoostRanker:
    """
    Ranks HPC events for a given binary/sample based on 4 statistical scores.
    """
    
    def __init__(self, weights: List[float] = None):
        """
        Args:
            weights: [w_spread, w_trend, w_correlation, w_stationarity]
                     Default: equal weights [0.25, 0.25, 0.25, 0.25]
        """
        self.weights = weights or [0.25, 0.25, 0.25, 0.25]
    
    def rank_events(self, event_matrix: np.ndarray, event_names: List[str]) -> pd.DataFrame:
        """
        Rank all events for a given sample.
        
        Args:
            event_matrix: shape (num_timesteps, num_events)
            event_names: list of event names, length = num_events
            
        Returns:
            DataFrame with columns: [event_name, spread, trend, correlation,
                                     stationarity, combined_score, rank]
        """
        num_events = event_matrix.shape[1]
        
        # Compute raw scores
        raw_spread = []
        raw_trend = []
        raw_corr = []
        raw_stat = []
        
        for i in range(num_events):
            ts = event_matrix[:, i]
            raw_spread.append(spread_score_single(ts))
            raw_trend.append(trend_score_single(ts))
            raw_corr.append(correlation_score(i, event_matrix))
            raw_stat.append(stationarity_score(ts))
        
        # Normalize each score to [0, 1] using min-max
        def normalize(scores):
            finite = [s for s in scores if np.isfinite(s)]
            if not finite or max(finite) == min(finite):
                return [0.5] * len(scores)
            lo, hi = min(finite), max(finite)
            return [min((s - lo) / (hi - lo), 1.0) if np.isfinite(s) else 1.0
                    for s in scores]
        
        norm_spread = normalize(raw_spread)
        norm_trend = normalize(raw_trend)
        norm_corr = normalize(raw_corr)
        norm_stat = normalize(raw_stat)
        
        # Combine with weights
        combined = []
        for i in range(num_events):
            score = (self.weights[0] * norm_spread[i] +
                     self.weights[1] * norm_trend[i] +
                     self.weights[2] * norm_corr[i] +
                     self.weights[3] * norm_stat[i])
            combined.append(score)
        
        # Build results DataFrame
        results = pd.DataFrame({
            'event_name': event_names,
            'spread_raw': raw_spread,
            'trend_raw': raw_trend,
            'correlation_raw': raw_corr,
            'stationarity_raw': raw_stat,
            'spread_norm': norm_spread,
            'trend_norm': norm_trend,
            'correlation_norm': norm_corr,
            'stationarity_norm': norm_stat,
            'combined_score': combined,
        })
        
        results = results.sort_values('combined_score', ascending=False)
        results['rank'] = range(1, len(results) + 1)
        
        return results
```

---

### Step 4.7 — Run ranking on the RaDaR dataset

Create `experiments/exp0_generate_rankings.py`:

```python
"""
Apply the HPC-Boost ranking module to the RaDaR dataset.
Generate per-sample event rankings and save them.

This validates the CORE CLAIM: different samples produce different rankings.
"""
import sys
sys.path.insert(0, '..')

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from src.ranking.ranker import HPCBoostRanker
from src.utils.data_loader import RadarDataLoader

# === CONFIGURATION (update after Stage 3 inspection) ===
RADAR_ROOT = os.path.expanduser("~/hpc_boost_v2/data/radar")
OUTPUT_DIR = os.path.expanduser("~/hpc_boost_v2/data/processed/rankings")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load data
loader = RadarDataLoader(RADAR_ROOT)
# TODO: Set loader.hpc_columns and loader.sample_id_column after Stage 3
df, metadata = loader.load_all()
print(f"Loaded {metadata['total_rows']} rows across {len(metadata['classes'])} classes")

# Get per-sample time-series
samples = loader.get_per_sample_timeseries(df)
print(f"Total samples: {len(samples)}")

# Initialize ranker
ranker = HPCBoostRanker()

# Rank events for each sample
all_rankings = {}
for sample_id in tqdm(samples, desc="Ranking events per sample"):
    sample = samples[sample_id]
    event_names = list(sample['data'].keys())
    
    # Build event matrix: (timesteps, events)
    arrays = [sample['data'][e] for e in event_names]
    min_len = min(len(a) for a in arrays)
    event_matrix = np.column_stack([a[:min_len] for a in arrays])
    
    # Rank
    ranking_df = ranker.rank_events(event_matrix, event_names)
    ranking_df['sample_id'] = sample_id
    ranking_df['malware_class'] = sample['label']
    
    all_rankings[sample_id] = ranking_df
    
    # Save individual ranking
    ranking_df.to_csv(
        os.path.join(OUTPUT_DIR, f"ranking_{sample_id}.csv"),
        index=False
    )

# Save combined rankings
combined = pd.concat(all_rankings.values(), ignore_index=True)
combined.to_csv(os.path.join(OUTPUT_DIR, "all_rankings.csv"), index=False)
print(f"\nSaved {len(all_rankings)} rankings to {OUTPUT_DIR}")

# === KEY VALIDATION: Do rankings differ across samples? ===
print("\n" + "=" * 80)
print("VALIDATION: Do top-4 events differ across samples?")
print("=" * 80)

top4_per_sample = {}
for sid, rdf in all_rankings.items():
    top4 = tuple(rdf.head(4)['event_name'].tolist())
    top4_per_sample[sid] = top4

unique_top4 = set(top4_per_sample.values())
print(f"  Total samples: {len(top4_per_sample)}")
print(f"  Unique top-4 event sets: {len(unique_top4)}")
print(f"  Ratio (unique/total): {len(unique_top4)/len(top4_per_sample):.2%}")

if len(unique_top4) > 1:
    print("\n  ✅ CORE CLAIM VALIDATED: Different samples have different optimal events!")
else:
    print("\n  ❌ WARNING: All samples have the same top-4. Check data quality.")
```

---

## Stage 5: Implement Baselines (~2 days)

### Step 5.1 — 2SMaRT Baseline (Pearson Correlation)

Create `src/baselines/two_smart.py`: Global Pearson correlation-based event selection (DATE 2019).

### Step 5.2 — PCA Baseline

Create `src/baselines/pca_selection.py`: PCA dimensionality reduction to select top-K principal components.

### Step 5.3 — Global-Fixed Baseline

Create `src/baselines/global_fixed.py`: Uses the 4 most cited events: `instructions`, `cycles`, `cache-misses`, `branch-misses`.

### Step 5.4 — Random Baseline

Create `src/baselines/random_selection.py`: Random 4-event subset, repeated 100 times and averaged.

---

## Stage 6: Implement Anomaly Detectors (~2 days)

### Step 6.1 — Detector Suite

Create `src/detection/detectors.py` with:
- **One-Class SVM** — trains only on benign data (unsupervised)
- **Isolation Forest** — detects anomalies by random partitioning
- **XGBoost** — supervised binary classifier (benign vs. malicious)
- **LSTM Autoencoder** — time-series anomaly detection (reconstruction error)

### Step 6.2 — Evaluation Metrics

Create `src/detection/metrics.py` with: Accuracy, F1, TPR, FPR, AUC-ROC, precision, recall.

---

## Stage 7: Run Experiments (~3-5 days)

### Experiment 1 — Ranking Validation
**Goal**: Show that top-ranked events give better detection than bottom-ranked events.
**Method**: Slide a window of 4 events across the ranking. At each position, train detectors and measure F1/AUC.
**Expected result**: Monotonically decreasing detection quality as rank decreases.

### Experiment 2 — Baseline Comparison
**Goal**: Show HPC-Boost per-binary ranking beats all baselines.
**Method**: Compare HPC-Boost vs. 2SMaRT vs. PCA vs. Global-Fixed vs. Random on same test set.
**Expected result**: HPC-Boost has highest average F1 and lowest FPR.

### Experiment 3 — Category Analysis (Addresses Reviewer R1)
**Goal**: Show that optimal events differ by malware category.
**Method**: For each malware class in RaDaR, compute the average top-4 events. Show overlap heatmap.
**Expected result**: Different classes favor different events (e.g., ransomware → crypto ops, trojans → branch events).

### Experiment 4 — Sensitivity Score (Supplementary)
**Goal**: Validate that statistically-ranked events also have the highest empirical sensitivity to malware.
**Method**: KS-test between benign and malicious distributions per event. Correlate with ranking.

---

## Stage 8: Draft Paper Sections (~ongoing)

### Step 8.1 — LaTeX Setup
```bash
mkdir -p ~/hpc_boost_v2/paper
cd ~/hpc_boost_v2/paper
# Download IEEE conference template or use overleaf
```

### Step 8.2 — Sections to Draft Now
1. **Related Work** — Updated with SUNDEW, RaDaR, Intel TDT, MTD (all new since 2021)
2. **Methodology** — Full mathematical formalization of all 4 scores
3. **Threat Model** — What attacks we defend against, what's in scope
4. **Problem Statement** — The 4-register constraint, why binary-specific selection matters

---

## Execution Timeline Summary

| Week | Stage | Deliverable |
|---|---|---|
| **Week 1, Day 1-2** | Stage 1-3 | Environment ready, datasets downloaded, data inspection complete |
| **Week 1, Day 3-5** | Stage 4 | Ranking module fully implemented, rankings generated on RaDaR |
| **Week 2, Day 1-2** | Stage 5 | All 4 baselines implemented |
| **Week 2, Day 3-4** | Stage 6 | Anomaly detector suite ready |
| **Week 2, Day 5 — Week 3** | Stage 7 | All 4 experiments run, figures generated |
| **Ongoing** | Stage 8 | Paper sections drafted |

> [!IMPORTANT]
> **Blocking dependency**: Everything after Stage 3 depends on knowing the exact column names and data format from the RaDaR CSV. Stage 3 output is the critical first deliverable.
