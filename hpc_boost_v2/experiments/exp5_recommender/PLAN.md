# exp5 — Per-Binary Recommender Pilot: Plan & Progress Log

Owner: HPC-Boost. Created 2026-06-21. This file is both the plan and a running
progress log (see **Status log** at the bottom). It is the durable record of the
recommender phase; update the log as work lands.

---

## 1. Objective

Turn the exp3 detection-aware oracle *headroom* (validated on RaDaR, see
`papers/hpc-boost-direction-validation.md`) into a **deployable per-binary PMU
event recommender**, using our own collected traces. Before building the
recommender, clear the paper's §9.3 go/no-go gate on our own data.

## 2. The conceptual pivot (why this phase exists)

- **exp3 / RaDaR:** per-*sample* routing was the **non-deployable ceiling** — a
  static recommender cannot choose a subset per execution. The headroom was real
  (~37 balanced-accuracy points over 2SMaRT) but ~85% of it lived at per-sample
  granularity (§5.1.1); only ~5.5 balanced-acc points were reachable at the
  group level a recommender could approximate.
- **exp5 / our traces:** our unit `sample` is a **binary**, traced over **5
  repeated runs**. A static binary recommender *can* choose one subset per
  binary. So **what was the non-deployable ceiling in exp3 (per-sample) becomes
  the deployable _target_ in exp5 (per-binary).** The repeated runs are what make
  that conversion measurable — and let us test stability.
- The non-deployable ceiling in exp5 is per-*run* routing (per `(binary, rep)`).

This reframing is the crux of the whole collection effort; state it explicitly
in any write-up so the per-sample/per-binary distinction is not conflated.

## 3. Data

- **Source:** `labeled_dataset.csv`, produced by
  `scratch/malware_collect/build_dataset.py` on the collection host.
- **Schema:** `label, family, sample, rep, group, ts_ms, <13 event columns>`
  (wide, zero-filled).
  - `sample` = binary id (sha256 dir) = **the recommendation unit**.
  - `rep` = repeated run (5 reps), revert-per-group.
  - `group` = event group `g000`–`g003`, ≤4 events each, **separate boots, NOT
    tick-aligned** (PMU-budget collection without multiplexing). `cycles` anchors
    every group.
- **13 events:** cycles, instructions, branches, branch-misses, cache-references,
  cache-misses, L1-dcache-loads, L1-dcache-load-misses, L1-dcache-stores,
  dTLB-loads, dTLB-load-misses, iTLB-loads, iTLB-load-misses.
- **Composition (per project state, verify on load):** malware = 22
  autonomous-runner samples (18 Tsunami + 4 Kaiji); benign = diversified
  workloads (heavy compute + light/idle), deliberately spread into the malware
  IPC cloud.
- **KNOWN LIMITATION — malware diversity:** bot-only / Tsunami-heavy (~2
  families). Mitigated by exp3's finding that the oracle headroom is **~entirely
  benign false-positive reduction** (§6.1: every malware sample had a correct
  candidate; the hard cases were all benign) plus a deliberately diverse benign
  set. The pilot can still test the operationally valuable question (does
  per-binary selection cut benign FPs?); the narrow malware mainly limits
  malware-side routing, which exp3 already found was not where the headroom was.
  **Document this, do not bury it.**

## 4. Methodology (consistent with exp3, adapted)

- **Features:** marginal per-event statistics (mean / std / min(p1) / max(p99) /
  skew / kurt) — exp3 `aggregate_trace`, reused verbatim. 13 events × 6 = **78
  features**; a 4-event subset = 24 features. No cross-event/joint features, so
  events from different collection groups can still be assembled into any 4-event
  subset at the feature level.
  - **Group handling:** aggregate each event ONLY from rows where it was actually
    measured (its home group / nonzero), per `(sample, rep)`. Marginal features
    are valid across separate-boot groups; the rep-to-rep variance tests
    boot-stability — which doubles as the **stability gate**.
- **Feature rows:** one per `(sample, rep)` (~190 rows). Enough to train XGBoost;
  routing/evaluation aggregated up to per-binary.
- **Candidate pool:** **exhaustive** C(13,4) = **715** four-event subsets (small
  event universe → no beam-search approximation needed; exp3 `run_beam_search`
  kept as a fallback for parity).
- **Detector / scaling:** XGBoost via exp3 `make_xgb` (100 trees, depth 4, hist)
  + per-subset `StandardScaler`. Reused verbatim.
- **CV:** **grouped by `sample`** (hold out whole binaries) — tests
  generalization to unseen binaries, the recommender's real job. Inner CV
  (sample-grouped) for candidate AUCPR scoring + train-only threshold tuning.
  Outer-test labels never touch candidate generation, scoring, or thresholds.
- **Baselines:** 2SMaRT (`select_twosmart`), best global subset, Majority. Reused.
- **Routing ladder (reframed for exp5):**
  `global → per_family → `**`per_binary (deployable target)`**` → per_run (ceiling)`.

## 5. The three go/no-go gates (paper §9.3)

1. **STABILITY** — across a binary's 5 reps, is per-subset utility / the best
   subset stable? Metrics: rank-correlation & top-k overlap of per-subset utility
   across reps; dispersion of the per-binary oracle's chosen subset. *No
   stability → no learnable target.*
2. **HEADROOM** — does the **per_binary** oracle beat the best **global** subset
   by **≥3–5 F1** (and on balanced accuracy / FPR)?
3. **DIVERGENCE** — do different binaries prefer different subsets? (routing
   diversity > 1 subset; `per_binary` > `per_family` > `global`).

**PASS** → build the recommender (static binary features → subset-utility target;
listwise/utility objective; eval top-1/top-k regret on held-out binaries, §10).
**FAIL** → fall back to FPR-safe subset-class prediction, or reframe scope.

## 6. Workstreams

- **A. Data acquisition** — *BLOCKED on host access.* Pull `labeled_dataset.csv`
  from the collection host (or a local copy) and sanity-check it: per-binary rep
  counts, class balance, nonzero-event coverage per group.
- **B. Oracle-gate pipeline** — build now, data-independent:
  `run_per_binary_oracle.py` (new loader + driver reusing exp3 primitives) +
  stability/divergence analysis + synthetic fixture + test.
- **C. Run gate** on real data; record results; make the PASS/FAIL call.
- **D. Recommender** (if PASS) — static binary feature extraction + utility-target
  construction + training + held-out-binary evaluation.
- **E. Diversity reassessment** — fold into gate interpretation; decide whether
  more malware families are needed before/after the gate.

## 7. Status log

- **2026-06-21** — Plan written; exp5 dir was empty. Confirmed the dataset schema
  from `build_dataset.py` and that all exp3 oracle primitives are reusable
  (`aggregate_trace`, `build_event_indices`, `score_subsets`,
  `fit_all_candidates`, `tune_candidate_thresholds`, `route_candidate_oracle`,
  `route_group_oracle`, `route_global_oracle`, `compute_group_oracle_metrics`,
  `compute_metrics`, `select_twosmart`). **Workstream A (data pull) is BLOCKED:**
  the safety policy requires explicit user authorization to SSH into the shared
  collection host (`iiitd@iiitd-ThinkPad-E470.local`); blanket autonomy does not
  cover it. Proceeding with Workstream B against the schema + a synthetic fixture
  so the gate runs the moment the dataset is local.
- **2026-06-21** — **Workstream B DONE.** Built `run_per_binary_oracle.py` (loader
  `load_per_binary` → 78 marginal features per `(sample,rep)`, nonzero-masked by
  home group; exhaustive 715-subset pool; sample-grouped outer/inner CV; the
  `global→per_family→per_binary→per_run` ladder; stability + divergence). Reuses
  exp3 primitives by **canonical import** (`from experiments.exp3_detection_aware_oracle
  import run_beam_oracle`) — required so joblib/loky workers can re-import the
  parallelized functions (importlib path-loading broke pickling). Added
  `tests/test_per_binary_oracle_gate.py`: synthetic fixture (2 malware families
  separable by different event groups, stable reps) + asserts the ladder is
  accuracy-monotonic (`global ≤ per_family ≤ per_binary ≤ per_run`, guaranteed by
  routing-freedom nesting) and outputs are well-formed. Verified: gate unit test
  passes; `python -m experiments.exp5_recommender.run_per_binary_oracle` CLI runs
  (report + `gate_summary.json`/`gate_metrics.json`/`gate_report.txt`); full suite
  47 passed / 4 pre-existing stale ranking fails. **Gate is ready to run on real
  data.** Still BLOCKED on Workstream A (dataset pull) — needs explicit host
  authorization or a local `labeled_dataset.csv`.
- **2026-06-21** — **Workstreams A + C DONE.** User authorized the host pull;
  reached the collection host at `10.110.14.249` (`.local` mDNS failed across
  subnets; resolved by IP, same /24 as this Mac). `pipeline_done=DONE`. Found the
  default `labeled_dataset.csv` is **STALE** (Jun 19, old invalid 28-sample
  malware set) and used `labeled_dataset_track1.csv` instead (Jun 21, the current
  native x86-64 KVM real-PMU track: 22 malware [18 Tsunami + 4 Kaiji] + 23 diverse
  benign). Pulled it (315 MB, byte-verified) to the gitignored
  `experiments/exp5_recommender/data/`; loader sanity-check clean (225×78, no
  NaN/zero-var). Ran the full gate (715 candidates, 100-tree XGBoost, 77 s).
  **VERDICT: STABILITY pass (rep-jaccard 0.98), HEADROOM fail, DIVERGENCE fail —
  NO-GO on the per-binary _conditional_ recommender** (~96% of the oracle headroom
  is fixed-single-subset selection, ~3% is conditioning). Confirms/sharpens exp3
  §5.1.1; dominant caveat = bot-only malware (2 families). Full analysis +
  decision options A/B/C in **`RESULTS.md`**. Workstream D (conditional
  recommender) is **not justified on this data**; awaiting user's A/B/C call.
- **2026-06-21** — User chose **C (ship A, queue B)**. **Ship A DONE.** Built
  `recommend_fixed_subset.py` (deployable fixed-subset selector: inner-AUCPR vs
  inner-balanced-accuracy criteria, sample-grouped CV, train-tuned thresholds;
  reuses gate/exp3 primitives; selection worker added to `run_per_binary_oracle.py`
  so loky can pickle it under `-m`). Added recommender test (3 exp5 tests now
  pass). Ran on track1 (60 s): **fixed @ balanced-accuracy beats 2SMaRT by +0.040
  F1 / +0.048 balacc, FPR 0.27→0.17** (balacc-selection > AUCPR-selection);
  recommended subset `{cycles, branches, cache-misses, L1-dcache-stores}`, but the
  subset is unstable across folds (5 distinct/5 — flat near-equivalent family,
  echoes exp3 §5.5/§6.4). Wrote the paper-ready memo
  `papers/hpc-boost-exp5-per-binary-gate.md` (full exp5 story). RESULTS.md updated
  with the recommender table. **Queue B** logged as task #6 (broaden malware →
  re-run gate). exp5 phase complete pending any committing.
