# HPC-Boost v2 — Task Tracker

## Current Constraint
- ✅ SSH access to dedicated server (i5-13450HX)
- ❌ No BIOS access yet (cannot disable HT, configure Turbo Boost)
- ❌ No full PC access for malware sandbox execution

**Strategy**: Start with pre-collected public datasets to develop and validate the ranking module, baseline comparisons, and ML pipeline. Switch to own data collection once full access is granted.

---

## Phase A: SSH-Only Work (Pre-collected Datasets)

### A1. Obtain Public Datasets
- [ ] Download **Zhou et al. (AsiaCCS 2018)** dataset from GitHub: `bu-icsg/Hardware_Performance_Counters_Can_Detect_Malware_Myth_or_Fact`
  - Contains: 962 malware + 962 benign HPC traces
  - Format: Need to inspect (likely CSV or similar)
  - Status: ✅ Publicly available, no registration needed
- [ ] Search Kaggle for **"Open Malware Research Jugaad Trails"** (RaDaR dataset)
  - Contains: 3471 malware samples, 54 HPC events, network + OS traces
  - Status: ✅ Downloaded and extracted
- [x] If RaDaR not on Kaggle, email **Sareena Karapoola / Chester Rebeiro** (IIT Madras RISE Lab) requesting access
  - Cite: CIKM 2022 paper (DOI: 10.1145/3511808.3557121)
- [ ] Explore **Open-Malware-Research-IIT-Madras** GitHub org (4 repos)

### A2. Environment Setup (SSH Server)
- [ ] Install Python 3.10+ environment (conda/venv)
- [ ] Install core packages: `numpy`, `scipy`, `pandas`, `scikit-learn`, `statsmodels`, `pymannkendall`, `matplotlib`
- [ ] Install ML packages: `torch`, `xgboost`, `lightgbm`
- [ ] Install binary analysis packages: `angr`, `capstone`, `networkx` (for later feature extraction)
- [ ] Create project directory structure:
  ```
  hpc_boost_v2/
  ├── data/
  │   ├── zhou_asiaccs/        # Public AsiaCCS 2018 dataset
  │   ├── radar/               # RaDaR dataset (when obtained)
  │   └── processed/
  ├── src/
  │   ├── ranking/             # Ranking module (4 scores)
  │   ├── features/            # Feature extractor
  │   ├── recommender/         # NN recommender
  │   ├── baselines/           # 2SMaRT, PCA, global-fixed
  │   ├── detection/           # Anomaly detection classifiers
  │   └── utils/               # Data loading, preprocessing
  ├── experiments/
  │   ├── exp1_ranking_validation/
  │   ├── exp2_baseline_comparison/
  │   ├── exp3_category_analysis/
  │   ├── exp4_overhead/
  │   └── exp5_adversarial_robustness/
  ├── figures/                 # Publication-quality plots
  └── notebooks/               # Exploration & analysis
  ```

### A3. Data Exploration & Understanding
- [ ] Inspect Zhou dataset: identify format, HPC events recorded, sample structure
- [x] Inspect RaDaR dataset: identify 54 HPC events, malware family labels, trace format
  - Confirmed 54 PMU events (e.g., `Core_cyc`, `L1D_Miss`, `BrMispred`)
  - 3471 unique samples across families like PUA, deceptor, trojan
- [ ] Document which HPC events overlap between Zhou, RaDaR, and our target i5-13450HX
- [ ] Generate descriptive statistics: sample counts per family, trace lengths, event distributions
- [ ] Create visualization: event value distributions for benign vs. malware (initial sanity check)

### A4. Implement Ranking Module (Core Contribution)
- [ ] Implement Score 1: **Spread Score** (IQR-based, median across runs)
- [ ] Implement Score 2: **Trend Score** (Mann-Kendall S statistic consistency)
- [ ] Implement Score 3: **Correlation Score** (Spearman inter-event independence)
- [ ] Implement Score 4: **Stationarity Score** (ADF/PP/KPSS with short/long distinction)
- [ ] Implement **combined ranking** (normalized score aggregation)
- [ ] Unit tests for each score function
- [ ] Apply ranking to Zhou dataset → generate per-binary event rankings
- [ ] Apply ranking to RaDaR dataset → generate per-binary event rankings
- [ ] Validate: do the top-ranked events differ across binaries? (core claim of HPC-Boost)

### A5. Implement Baseline Methods
- [ ] **2SMaRT baseline** (Sayadi, DATE 2019): Pearson correlation-based global event selection
- [ ] **PCA baseline** (Kadiyala, TECS 2020): PCA dimensionality reduction on HPC features
- [ ] **Global-fixed baseline**: Use the 4 most commonly cited events (instructions, cycles, cache-misses, branch-misses)
- [ ] **Random selection baseline**: Random event subset (controls for selection bias)
- [ ] **SUNDEW-style baseline** (Karapoola, TDSC 2024): Per-malware-class event selection

### A6. Implement Anomaly Detection Classifiers
- [ ] One-Class SVM (trains on benign only)
- [ ] Isolation Forest
- [ ] XGBoost (binary classification)
- [ ] LSTM Autoencoder (time-series anomaly detection, from previous team's work)
- [ ] Evaluation metrics: Accuracy, F1, TPR, FPR, AUC-ROC, detection latency

### A7. Run Experiments on Pre-collected Data
- [ ] **Experiment 1 — Ranking Validation**: Show that HPC-Boost ranked events yield better detection than randomly/uniformly selected events
- [ ] **Experiment 2 — Baseline Comparison**: HPC-Boost per-binary ranking vs. 2SMaRT vs. PCA vs. Global-Fixed
- [ ] **Experiment 3 — Category Analysis**: How does the optimal event set differ across malware categories (ransomware, trojan, backdoor, etc.)?
- [ ] **Experiment 4 — Sensitivity Score**: Compute KS-statistic between benign/malicious distributions per event; correlate with ranking
- [ ] Generate publication-quality figures for all experiments

### A8. Draft Paper Sections (Can Start Now)
- [ ] Related work section (with updated citations: SUNDEW, RaDaR, Intel TDT, MTD)
- [ ] Methodology section (ranking module formalization, all 4 scores with math)
- [ ] Threat model & problem formulation
- [ ] LaTeX template setup for target venue

---

## Phase B: Full Access Work (After BIOS + Bare-Metal Access)

### B1. Hardware Configuration
- [ ] Install Ubuntu bare-metal on the i5-13450HX laptop
- [ ] Configure BIOS: disable Hyper-Threading, disable Turbo Boost
- [ ] Set `perf_event_paranoid = -1`
- [ ] Set CPU governor to `performance`
- [ ] Identify P-core CPU IDs: `cat /sys/devices/cpu_core/cpus`
- [ ] Run `perf list` to catalog all available events on Raptor Lake
- [ ] Document PMU capabilities in `event_catalog.json`

### B2. Malware Sandbox Setup
- [ ] Set up KVM/QEMU VM with Ubuntu 22.04
- [ ] Configure network isolation
- [ ] Create clean snapshot + restoration script
- [ ] Test `perf` passthrough from host to VM

### B3. Our Own Data Collection
- [ ] Collect benign binary HPC traces (MiBench, SPEC, system utils)
- [ ] Collect malware HPC traces (VirusShare/MalwareBazaar Linux ELF samples)
- [ ] Apply ranking module to our own collected data
- [ ] Validate that findings from Phase A (pre-collected data) hold on our own data

### B4. Overhead Analysis (Reviewer R2/R4)
- [ ] Measure feature extraction time per binary
- [ ] Measure recommender inference latency
- [ ] Measure full pipeline overhead vs. baseline detection
- [ ] Compare overhead with prior work (2SMaRT, SUNDEW)

### B5. Final Experiments & Paper Completion
- [ ] Run all 5 experiments on our own collected data
- [ ] Cross-validate with RaDaR results
- [ ] Generate all final figures
- [ ] Complete paper writing
- [ ] Internal review and proofreading
