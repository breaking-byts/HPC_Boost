# Experiment Script: Deep Analysis & Corrected Design

## Scientific Issues Found

### Issue 1: Data Leakage in 2SMaRT ❌ CRITICAL
**Previous code**: `twosmart.fit(X_means, y, event_cols)` called ONCE on the full dataset BEFORE cross-validation.

**Why it matters**: 2SMaRT selects events using Pearson correlation with the malware label. If the test fold's labels influence event selection, 2SMaRT gets an unfair advantage — it has partial knowledge of the test data.

**Fix**: Move `twosmart.fit()` inside each fold, using only `X_train` and `y_train`.

---

### Issue 2: Data Leakage in Global-Fixed Fallback ❌ MODERATE
**Previous code**: When hardcoded events (`Instruct`, `Core_cyc`, `L1D_Miss`, `BrMispred`) are missing from the dataset, the script fills the gap using variance computed on the FULL dataset.

**Fix**: Compute fallback variance inside each fold using only `X_train`.

---

### Issue 3: HPC-Boost Rankings — Is This Leakage? ✅ NO
**Concern**: Rankings were pre-computed on the FULL dataset. Does this leak test info?

**Analysis**: No. Each sample's ranking uses ONLY its own time-series data. No labels, no other samples' data are used. This is analogous to per-sample feature engineering (like computing a sample's own mean/std). In deployment, this step happens naturally: you observe the binary's trace, compute its ranking, then classify.

**However**: This IS an advantage HPC-Boost has over baselines — it "sees" each test sample's features to decide which ones to use. This is the core methodological claim and must be transparently documented.

---

### Issue 4: Unfair Scoring — Sample Dropping ❌ MODERATE
**Previous code**: When HPC-Boost can't evaluate a sample (missing ranking), it silently skips it. F1 is then computed on the reduced subset. Baselines are scored on the full fold.

**Why it matters**: Comparing F1 scores computed on different-sized test sets is invalid.

**Fix**: Score ALL strategies on the identical test set. For HPC-Boost, if ranking is missing → predict 0 (benign). Track and report coverage.

**Justification for default=0**: With ~74% malware prevalence, defaulting to 0 actually PENALIZES HPC-Boost (most missing samples are malware → false negatives → lower recall → lower F1). This is the conservative choice.

---

### Issue 5: PCA Uses All 55 Events ⚠️ DOCUMENTED
**Issue**: PCA projects ALL 330 features (55 events × 6 stats) into 24 components. In real hardware, you can only monitor 4 PMU registers simultaneously. PCA is physically impossible to deploy.

**Fix**: Keep PCA but mark it clearly as `PCA*` (upper bound). Add methodology note.

---

### Issue 6: Detector Hyperparameters ⚠️ MINOR (affects absolute, not relative)
**OCSVM nu=0.3**: nu is the upper bound on margin errors. With clean benign training data, nu=0.3 means up to 30% of benign samples are allowed to be "outliers." This gives a loose boundary.

**IsolationForest contamination=0.3**: Expects 30% outliers in training data. But training data is ALL benign, so contamination should be much lower.

**Impact on comparison**: These parameters affect ALL strategies equally. The relative ranking of strategies is unaffected. Only absolute F1 values change.

**Fix**: Keep current parameters for fair comparison. Document that absolute values could improve with hyperparameter tuning.

---

### Issue 7: Random Baseline Sample Size ⚠️ MINOR
**Previous**: 3 random seeds. 

**Fix**: Use 5 seeds for more stable estimates.

---

### Issue 8: Class Imbalance ⚠️ DOCUMENTED
**Dataset**: 74% malware, 26% benign. Stratified CV ensures each fold preserves this ratio.

**For OCSVM/IF**: Train on benign only (~721 training samples per fold). Sufficient for stable models.

**For XGBoost**: Class imbalance could bias predictions. However, F1 metric (harmonic mean of precision and recall) is robust to imbalance.

---

### Issue 9: Feature Engineering Consistency ✅ OK
All strategies use the same 6 statistical features per event (mean, std, min, max, skew, kurtosis). The ONLY difference is WHICH events' features are used. This is correct — the experiment isolates the effect of event SELECTION.

---

### Issue 10: XGBoost Scaling ⚠️ HARMLESS
XGBoost is tree-based and invariant to monotonic feature transformations. Scaling doesn't help or hurt. Both baseline and worker apply scaling, so it's consistent. Keeping it for code consistency with OCSVM/IF.

---

## Technical Issues Found

### Issue T1: Memory — Raw Traces Not Freed ❌
**Previous**: After aggregation, the `traces` dict (~1-2 GB of DataFrames) remains in memory.

**Fix**: Explicit `del traces` after aggregation. Also delete the intermediate data structures.

---

### Issue T2: Nested Parallelism in XGBoost ❌
**Previous**: XGBClassifier defaults to n_jobs=1 in recent versions, but not guaranteed across all versions.

**Fix**: Explicit `n_jobs=1` in XGBClassifier within the worker. Also set `XGB_NTHREAD=1` environment variable.

---

### Issue T3: Pandas in Worker Processes ⚠️ PERFORMANCE
**Previous v2**: Workers received pandas DataFrames, causing expensive pickling.

**Fixed in last version**: Workers use numpy arrays + integer index lists. This is correct.

---

### Issue T4: Repeated Disk I/O for Rankings ⚠️ PERFORMANCE
**Previous v1/v2**: Each worker reads a CSV file from disk.

**Fixed in last version**: Rankings preloaded once, converted to index arrays.

---

### Issue T5: Silent Exception Swallowing ❌
**Previous**: Broad `except Exception: return 0` hides potential bugs.

**Fix**: Count and report exceptions. Print first N exception messages.

---

### Issue T6: Redundant Model Training in v2 ❌
**Previous v2**: Trained a model via `run_detector()`, computed metrics, then trained AGAIN for prediction.

**Fixed in last version**: Single training per sample.

---

## Corrected Design Summary

| Decision | Choice | Rationale |
|----------|--------|-----------|
| 2SMaRT fitting | Inside fold | Prevents label leakage |
| Global-Fixed fallback | Inside fold | Prevents variance leakage |
| HPC-Boost rankings | Pre-computed (OK) | No cross-sample leakage; documented as per-binary advantage |
| Missing ranking default | Predict 0 (benign) | Conservative; penalizes HPC-Boost; coverage tracked |
| Feature aggregation | 6 stats per event | Consistent across all strategies |
| Data type | float32 | Half the memory of float64 |
| Workers | numpy-only, rankings preloaded | No disk I/O, no pandas serialization |
| XGBoost in workers | n_jobs=1 explicit | Prevents nested parallelism |
| Memory | Explicit del of traces | Frees ~1-2 GB after aggregation |
| Exceptions | Counted and reported | Transparency |
| Random seeds | 5 | More stable averaging |
| PCA | Labeled as PCA* (upper bound) | Not deployable with 4 PMU registers |
