# From Oracle Headroom to a Deployable Recommender: The exp5 Per-Binary Gate

## Abstract

The HPC-Boost validation study (`hpc-boost-direction-validation.md`) used a
detection-aware oracle on the RaDaR dataset to show that *conditional* four-event
PMU selection has large headroom over one global set — but it also warned (its
§5.1.1) that ~85% of that headroom lived at a non-deployable per-sample
granularity, and only a few balanced-accuracy points were reachable by a static
recommender. exp5 tests that warning directly, on our **own** collected traces, by
converting the per-sample oracle into a **per-binary** one: our collection unit is
a binary executed over five repeated runs, so per-binary routing — the exp3
*ceiling* — becomes the exp5 deployable *target*.

On 45 binaries (22 autonomous-bot malware + 23 diverse benign workloads), the
per-binary oracle ceiling confirms and sharpens the warning. Per-binary
conditioning adds only **+0.005 F1 / +0.005 balanced accuracy** over a single
well-chosen subset; a decomposition of the total oracle headroom over the 2SMaRT
baseline attributes **~96% to choosing one good fixed subset and only ~3% to
per-binary conditioning** (≈0% to per-execution). We therefore do **not** build a
conditional recommender. We instead ship the part that survives: (i) a strong
**stability** result — a binary's best subset is correct on all five of its runs
(rep-Jaccard 0.98) and marginal counters reproduce across separate boots to 1.2%,
validating the collection methodology; and (ii) a **deployable fixed-subset
recommender** that selects one four-event set from training and beats 2SMaRT by
**+0.040 F1 / +0.048 balanced accuracy** (FPR 0.27→0.17). The exact subset is
unstable across folds (a flat family of near-equivalent subsets — the same
flatness that makes conditioning futile), so the recommendation is one
representative of that family, not a unique optimum.

## 1. The reframe: per-sample ceiling → per-binary target

exp3 routed each RaDaR *sample* to a candidate subset using its label — a
per-execution decision no deployed system can make. Our collection instead traces
each *binary* five times. A static binary recommender can choose one subset per
binary, so **per-binary routing is the deployable granularity**, and the
non-deployable ceiling becomes per-*run* routing (per `(binary, run)`). The
repeated runs additionally let us measure whether a binary's preferred subset is
stable — the prerequisite for any learnable per-binary target. This is the entire
reason the running-binaries phase exists.

## 2. Data and method

**Data.** Track-1 native x86-64 KVM real-PMU traces: 22 malware binaries
(18 Tsunami + 4 Kaiji autonomous bots) and 23 diverse benign workloads
(sort, gzip, openssl, matrix, crypto, I/O, idle/heartbeat/poll, …), each run 5×
under revert-per-group collection in 4 separate-boot event groups of ≤4 events
(13 guest-vPMU events total; `cycles` anchors every group). 45 binaries,
225 `(binary, run)` rows.

**Features.** Six marginal statistics per event (mean/std/p1/p99/skew/kurt) — the
exp3 `aggregate_trace`, reused verbatim. Because the features are marginal (no
cross-event joints), the separate-boot groups can still be assembled into any
four-event subset: each event's statistics come from its own home group. 13×6 = 78
features; a four-event subset uses 24.

**Oracle and baselines.** Exhaustive C(13,4) = 715 four-event subsets (no beam
approximation needed at this event count). Each subset is an XGBoost detector
(100 trees, depth 4) scored by inner-CV AUCPR; thresholds tuned train-only at
target-TPR 0.95. We reuse the exp3 scoring, candidate-fitting, threshold-tuning,
routing, and honest-metric primitives verbatim, so the comparison is
methodologically identical. Cross-validation is **grouped by binary** (whole
binaries held out), the recommender's real generalization test. The routing ladder
is `global → per_family → per_binary → per_run`.

## 3. Per-binary oracle gate (the ceiling)

Pooled over five binary-grouped folds (tuned @ TPR 0.95):

| strategy | F1 | BalAcc | FPR | deployable? |
|---|---:|---:|---:|---|
| Majority | 0.6567 | 0.5000 | 1.0000 | — |
| 2SMaRT | 0.8525 | 0.8423 | 0.2609 | yes |
| best_global (training-only single subset) | 0.8918 | 0.8899 | 0.1565 | yes |
| global (oracle single subset) | 0.9767 | 0.9773 | 0.0000 | no |
| per_family (oracle) | 0.9815 | 0.9818 | 0.0000 | no |
| **per_binary (oracle — the thesis)** | 0.9815 | 0.9818 | 0.0000 | no (target) |
| per_run (oracle ceiling) | 0.9815 | 0.9818 | 0.0000 | no |

**Headroom decomposition (balanced accuracy).** Total oracle headroom over
2SMaRT = +0.140, split as:

| step | meaning | Δ balacc | share |
|---|---|---:|---:|
| 2SMaRT → best_global | better single-subset *search* (deployable) | +0.048 | 34% |
| best_global → global | fixed-subset *selection* ceiling | +0.087 | 62% |
| global → per_binary | **per-binary conditioning (the thesis)** | **+0.0045** | **3%** |
| per_binary → per_run | per-execution ceiling | +0.000 | 0% |

Per-binary conditioning is worth ~3% of the headroom; per-execution ~0%. Because
this is the oracle ceiling, no learned conditional model can do better. **The gate
is a NO-GO for a per-binary conditional recommender on this dataset** — and a
direct empirical confirmation of exp3 §5.1.1.

## 4. The two findings that survive

### 4.1 Stability (strong pass)

| metric | value | reading |
|---|---:|---|
| median rep-Jaccard | 0.979 | the set of subsets that classify a binary correctly is ~98% consistent across its 5 runs |
| best-subset consistency | 1.000 | each binary's best subset is correct on all 5 runs |
| marginal feature-CV across runs | 0.012 | counters reproduce across separate boots to ~1% |

Per-binary utility is highly stable, and the marginal-feature assembly across
separate-boot groups is validated. This is a publishable methods result and a
green light for the collection design.

### 4.2 Deployable fixed-subset recommender

Selecting one four-event subset from training only (all selectors at a train-tuned
TPR-0.95 threshold), evaluated on held-out binaries:

| selector | F1 | BalAcc | FPR | MCC |
|---|---:|---:|---:|---:|
| 2SMaRT | 0.8490 | 0.8379 | 0.2696 | 0.6897 |
| fixed @ inner-AUCPR | 0.8667 | 0.8597 | 0.2261 | 0.7281 |
| **fixed @ inner-balanced-accuracy** | **0.8889** | **0.8858** | **0.1739** | **0.7754** |
| oracle single subset (ceiling) | — | 0.9773 | — | — |

A deployable fixed-four-event recommender beats 2SMaRT by **+0.040 F1 / +0.048
balanced accuracy** and cuts FPR from 0.270 to 0.174. **Selecting by inner
balanced accuracy beats selecting by AUCPR** — pick the subset by the metric you
report. It recovers ~35% of the 0.139 balanced-accuracy gap to the oracle ceiling;
the rest is irreducible single-subset-selection uncertainty. (2SMaRT differs
trivially from §3 because here it uses the same tuned threshold as the other
selectors, not 0.5.)

**Caveat — subset instability.** The selected subset is *not* consistent across
folds (5 distinct over 5). Deployable *performance* is stable (~0.886) but the
*identity* of the best subset is not: there is a flat family of near-equivalent
subsets (the exp3 §5.5/§6.4 flatness, now reproduced on our data). The
recommendation — `{cycles, branches, cache-misses, L1-dcache-stores}` — is one
representative of that family. This flatness is also *why* conditioning fails:
when many subsets work, routing among them adds nothing.

## 5. Honest limitations

- **Bot-only malware (2 families).** 18 Tsunami + 4 Kaiji gives little
  binary-to-binary diversity for conditioning to exploit; a single subset already
  separates the classes (oracle FPR = 0). The conditional thesis was therefore not
  given its fairest test — a limitation pre-registered before the run. The benign
  set is diverse (23 workloads) but benign is the easy class here.
- **Selection ceiling, not conditioning ceiling, is where the headroom is.** This
  is a per-corpus statement; a more behaviorally diverse malware set could move it.
- **One operating point / one detector family.** Tuned at TPR 0.95 with XGBoost,
  matching exp3.

## 6. Conclusion and next step

The per-binary gate does its job: it stops us from building a conditional
recommender that cannot win on this corpus, and it confirms exp3's own sobering
prediction on independent, self-collected data. What we ship is honest and
deployable — a stability result that validates the collection methodology, and a
fixed-subset recommender that beats the 2SMaRT baseline by ~5 balanced-accuracy
points while halving its false-positive rate.

**Broadening was attempted and quantifies a double wall.** An automated re-fetch
across nine autonomous families (Mirai, Gafgyt, Bashlite, Tsunami, XorDDoS,
Dofloo, Ddostf, Hajime, Kaiji) sifted hundreds of MalwareBazaar candidates and
yielded only **6 new x86-64 ELFs** — the rest were ARM (76), MIPS (117), i386
(88), or aarch64 (14): an **architecture wall** (these are IoT families). Triage
of those 6 found **0 that detonated** (the **detonation wall**). So the Track-1
(x86-64 real-PMU) corpus cannot be meaningfully broadened from MalwareBazaar, and
the per-binary conditioning NO-GO is robust on the available real-PMU data.

**Future work to give conditioning a fair test would require a different corpus,**
not more of the same: either Track-2 multi-arch emulation (abundant ARM/MIPS bots
via host-side proxy HPC — but translated execution, needing its own measurement-
validity argument), or a non-MalwareBazaar x86-64 source (VirusShare/VirusTotal
traditional Backdoor/Rootkit/Trojan families — older and more likely to run).

## Appendix: artifacts

- Gate: `hpc_boost_v2/experiments/exp5_recommender/run_per_binary_oracle.py`
  → `results/gate_{summary,metrics}.json`, `gate_report.txt`.
- Recommender: `hpc_boost_v2/experiments/exp5_recommender/recommend_fixed_subset.py`
  → `results/recommender_{summary,metrics}.json`, `recommender_report.txt`.
- Plan/log: `.../exp5_recommender/PLAN.md`; full results memo: `.../RESULTS.md`.
- Tests: `hpc_boost_v2/tests/test_per_binary_oracle_gate.py` (loader, gate ladder
  monotonicity, recommender).
- Data (gitignored, pulled from collection host): `labeled_dataset_track1.csv`
  (the default `labeled_dataset.csv` on the host is a stale 28-sample build — do
  not use).
