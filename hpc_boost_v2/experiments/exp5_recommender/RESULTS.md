# exp5 — Per-Binary Oracle Gate: Results & Verdict (2026-06-21)

Run: `python -m experiments.exp5_recommender.run_per_binary_oracle --dataset
labeled_dataset_track1.csv --n-estimators 100 --max-depth 4 --outer-folds 5
--inner-folds 3 --jobs 8 --target-tpr 0.95` (77 s). Raw artifacts in
`results/gate_summary.json`, `results/gate_metrics.json`, `results/gate_report.txt`.

Dataset: **Track-1 native x86-64 KVM real-PMU**, 45 binaries (22 malware =
18 Tsunami + 4 Kaiji; 23 diverse benign workloads), 5 reps × 4 event groups,
225 `(sample,rep)` rows, 13 events, exhaustive 715 four-event subsets.

## Metrics (pooled over 5 sample-grouped outer folds, tuned @ target-TPR 0.95)

| strategy | F1 | BalAcc | FPR | MCC | deployable? |
|---|---:|---:|---:|---:|---|
| majority | 0.6567 | 0.5000 | 1.0000 | 0.0000 | — |
| 2SMaRT (Pearson) | 0.8525 | 0.8423 | 0.2609 | 0.6973 | yes |
| best_global (training-only single subset) | 0.8918 | 0.8899 | 0.1565 | 0.7819 | yes |
| global (oracle single subset, post-hoc) | 0.9767 | 0.9773 | 0.0000 | 0.9564 | no |
| per_family (oracle) | 0.9815 | 0.9818 | 0.0000 | 0.9650 | no |
| **per_binary (oracle — the thesis)** | 0.9815 | 0.9818 | 0.0000 | 0.9650 | no (target) |
| per_run (oracle ceiling) | 0.9815 | 0.9818 | 0.0000 | 0.9650 | no |

## The decomposition (balanced accuracy) — the money result

Total oracle headroom over 2SMaRT = **+0.140**, split as:

| step | meaning | Δ balacc | share |
|---|---|---:|---:|
| 2SMaRT → best_global | better *single-subset search* (deployable) | +0.048 | 34% |
| best_global → global | *fixed-subset selection* ceiling (perfect single subset) | +0.087 | 62% |
| global → per_binary | **per-binary conditioning — the thesis** | **+0.0045** | **3%** |
| per_binary → per_run | per-execution ceiling | +0.000 | 0% |

**~96% of the achievable headroom is "choose one good fixed 4-event subset";
~3% is per-binary conditioning; ~0% is per-execution.** Per-binary routing used
11 distinct subsets over 45 binaries, but routing to them beats the single best
subset by only +0.0047 F1 (≈ within one run of 225) — the 11 subsets are
near-equivalent (echoing exp3 §5.5 candidate flatness).

## The three §9.3 gates

1. **STABILITY — STRONG PASS.** median rep-jaccard **0.979** (the set of subsets
   that correctly classify a binary is ~98% consistent across its 5 reps),
   best-subset-consistency **1.000** (each binary's best subset is correct on all
   5 reps), marginal feature-CV across reps **0.012**. Per-binary utility is
   highly stable, AND the separate-boot marginal-feature assembly is validated
   (~1% boot-to-boot drift).
2. **HEADROOM — FAIL.** per_binary − global = **+0.0045** balacc / +0.0047 F1,
   vs the +0.03–0.05 F1 bar. Conditioning adds essentially nothing over a single
   well-chosen subset.
3. **DIVERGENCE — FAIL (in value).** 11 distinct subsets are routed (count > 1),
   but they are near-equivalent, so divergence carries no performance.

## Verdict

**NO-GO on the per-binary _conditional_ recommender on this dataset.** This is the
*ceiling* (oracle) — a learned recommender cannot beat it — so there is nothing
for per-binary conditioning to learn beyond a single fixed subset. The gate did
its job: it stopped us before building a conditional model that cannot win here.

This **confirms and sharpens exp3 §5.1.1**, which predicted the realistically
attainable (group-level) headroom was small (~5.5 balacc points) and that the
dramatic gains lived only at the non-deployable per-sample granularity. On our
own per-binary data the conditioning headroom is even smaller (~0.5 balacc point).

### Dominant caveat (was pre-registered in PLAN.md §3)

The malware class is **bot-only / Tsunami-heavy (2 families)**. With ~2 malware
behaviors there is little binary-to-binary diversity for conditioning to exploit;
a single subset that separates "Tsunami/Kaiji vs benign" suffices (oracle global
FPR = 0). The benign set is diverse (23 workloads) but benign is the easy class
here. So the dataset structurally limits how much per-binary conditional headroom
*could* appear — this NO-GO is partly a property of the corpus, not only the thesis.

## Two positive, publishable findings

1. **Stability of per-binary HPC utility** (rep-jaccard 0.98) and **reproducibility
   of marginal counters across separate boots** (CV 1.2%) — validates the
   collection methodology and the per-binary target construction.
2. **Fixed-subset selection has real deployable headroom:** +0.048 balacc over
   2SMaRT from a better training-only single subset (best_global), and a +0.087
   ceiling if selection were perfect. A *static fixed-4-event recommender* (no
   per-binary conditioning) is the defensible, deployable win here — quantified next.

## Deployable fixed-subset recommender (ship A) — `recommend_fixed_subset.py`

Selects ONE 4-event subset from **training only** and deploys it on held-out
binaries (sample-grouped CV; all selectors at a train-tuned TPR-0.95 threshold for
a fair comparison). Run on track1 (60 s); artifacts in
`results/recommender_{summary,metrics}.json` + `recommender_report.txt`.

| selector | F1 | BalAcc | FPR | MCC | deployable? |
|---|---:|---:|---:|---:|---|
| 2SMaRT (Pearson) | 0.8490 | 0.8379 | 0.2696 | 0.6897 | yes |
| fixed @ inner-AUCPR | 0.8667 | 0.8597 | 0.2261 | 0.7281 | yes |
| **fixed @ inner-balanced-accuracy** | **0.8889** | **0.8858** | **0.1739** | **0.7754** | **yes (recommended)** |
| oracle single subset (ceiling) | — | 0.9773 | — | — | no (ref) |

- **A deployable fixed-4-event recommender beats 2SMaRT by +0.040 F1 / +0.048
  balanced accuracy** and cuts FPR 0.270 → 0.174. The real, honest win.
- **Select by inner balanced accuracy, not AUCPR** (0.886 vs 0.860 balacc): pick
  the subset by the metric you will report. It recovers +0.048 of the 0.139
  balacc gap to the oracle ceiling (~35%); the residual ~0.09 is irreducible
  single-subset-selection uncertainty.
- **CAVEAT — subset instability (flat top family).** The chosen 4-event subset is
  NOT consistent across folds: **5 distinct subsets over 5 folds** (echoes exp3
  §5.5 candidate flatness / §6.4 fold instability). Deployable *performance* is
  stable (~0.886) but the *identity* of the best subset is not — many
  near-equivalent subsets. The recommendation is one representative of a flat top
  family, not a uniquely optimal set. (This same flatness is why per-binary
  conditioning fails: many subsets work, so routing among them adds nothing.)
- All-data recommended representative (for deployment): **{cycles, branches,
  cache-misses, L1-dcache-stores}** (its inner balacc 0.982 is in-sample
  optimistic; the honest held-out number is 0.886).

## Decision options (for the next phase)

- **A. Pivot to fixed-subset selection** — write up exp5 as: (i) stability/repro
  validated; (ii) conditional routing not justified on per-binary data (confirms
  exp3 §5.1.1); (iii) a static best-single-subset recommender beats 2SMaRT by
  ~5–9 points. Deployable and honest. No more collection needed.
- **B. Broaden the malware corpus** — the conditional thesis was not given a fair
  test (2 families). Collect more autonomous x86-64 families (beyond
  Tsunami/Kaiji), rebuild track1, re-run this exact gate. Only worth it if a
  fair conditional test is the paper's goal.
- **C. Both** — ship A now; queue B as future work / a stronger second test.
