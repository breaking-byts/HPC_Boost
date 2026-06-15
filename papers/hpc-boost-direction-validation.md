# Detection-Aware Evidence for Binary-Specific PMU Event Selection in HPC-Boost

## Abstract

HPC-Boost is motivated by a hardware constraint: although modern processors
expose many hardware performance counter (HPC/PMU) events, runtime monitoring
can reliably observe only a small number at once. In our setting, the operating
constraint is four simultaneously monitored events. The central research
question is whether malware detection should use one globally selected four-event
set for all binaries, or whether the event set should vary by binary.

An earlier proof-of-concept used a hand-designed per-sample ranking heuristic and
found mixed evidence: the ranking beat a fixed global baseline but did not beat
the fold-local 2SMaRT Pearson-correlation baseline. That result was not a
decisive rejection of the thesis, because the ranking heuristic optimized
statistical interestingness rather than downstream detection utility.

We therefore ran a detection-aware candidate-pool oracle on the full RaDaR
dataset. For each outer training fold, the oracle generated a pool of promising
four-event subsets by beam search using only training data and inner-fold AUCPR.
It then compared the best global candidate against a non-deployable
Candidate-Oracle that uses the outer-test label only after prediction to route
each test sample to a retained candidate that predicts it correctly if one
exists.

The result is strong evidence that binary-conditional event choice is worth
pursuing. Under sample-stratified five-fold CV across 3,470 RaDaR samples,
Candidate-Oracle achieved 0.9781 mean F1, compared with 0.9090 for fold-local
2SMaRT and 0.9134 for the best global beam-search subset, recovering 436 of the
565 2SMaRT errors (a 77.2% error reduction). We then re-ran the full pipeline
with family-disjoint GroupKFold splits, which we treat as the primary
publication-grade evaluation. Under fair splits the supervised baselines fall by
roughly 3.5 F1 points (2SMaRT to 0.8732, Global-Beam to 0.8820) while the oracle
is essentially unchanged (0.9770), so the oracle's gap over 2SMaRT widens from
6.9 to 10.4 F1 points; on the hardest unseen family it widens to 16.8 points
(2SMaRT 0.80 versus oracle 0.97). However, this is a theoretical ceiling, not a
deployable detector. It validates the existence of conditional headroom; it does
not yet prove that a static-binary recommender can learn to realize that
headroom. A capped-pool analysis shows that a top-25 candidate oracle achieves
0.9502 F1, while the full approximately 1,200-candidate pool reaches 0.9781 F1;
therefore, part of the ceiling comes from post-hoc selection over a large pool.
The correct next step is a controlled binary-tracing pilot that builds
per-binary utility targets from multiple runs, then tests whether static binary
features predict those targets.

## Executive Verdict

Yes: the detection-aware oracle provides enough confidence to start the running
binaries and recording phase.

But the next phase should be treated as a **pilot recommender-data construction
phase**, not as an immediate full-scale data collection sprint. The oracle result
justifies investment because it shows a large gap between global event selection
and perfect conditional routing. The pilot must now answer the missing question:
whether that routing decision is predictable from static binary features.

The decision is:

> Proceed to binary tracing and recommender-target generation, with explicit
> safeguards that prevent training the recommender on labels or runtime
> information unavailable at deployment.

The main methodological risk is not whether conditional event selection has
headroom. The oracle shows that it does. The main risk is whether the
per-sample oracle labels can be converted into stable, learnable **per-binary**
targets.

## 1. Problem Statement

Hardware performance counters expose low-level execution signals such as cache
misses, branch behavior, instruction retirement, micro-op behavior, and memory
hierarchy activity. These signals are useful for malware and anomaly detection
because attacks often perturb low-level execution behavior before they are
visible to higher-level software monitors.

The bottleneck is hardware capacity. A system may expose dozens or hundreds of
event names, but only a few physical PMU registers can be monitored reliably at
the same time. In our RaDaR proof-of-concept, the candidate event universe has
55 events and the PMU budget is four events:

\[
\text{choose } S \subset E, \quad |S| = 4, \quad |E| = 55.
\]

An exact four-event search has

\[
\binom{55}{4} = 341{,}055
\]

possible subsets. Exhaustively training and validating a model for every subset
inside each cross-validation fold would be computationally expensive and, more
importantly, would still not answer whether static binary features can predict
the best subset.

HPC-Boost proposes a different deployment path:

1. Build event-selection targets offline.
2. Extract static features from a binary.
3. Train a recommender that maps binary features to event preferences.
4. At deployment, monitor only the recommended four PMU events.

The key research question is therefore:

> Is there enough performance headroom in binary-specific event selection to
> justify building the static-binary recommender?

## 2. Why the Earlier Oracle Was Inconclusive

The earlier per-sample ranking oracle used the HPC-Boost statistical ranking
module. It fixed leakage in the 2SMaRT baseline and corrected several scoring
bugs. The final reported F1 values were:

| Method | Mean F1 |
|---|---:|
| 2SMaRT fold-local Pearson | 0.7401 |
| HPC-Boost statistical ranking oracle | 0.7058 |
| PCA using all events | 0.6668 |
| Global-Fixed | 0.6593 |

That result was useful but not decisive. The term "oracle" was too strong:
the experiment simulated a perfect recommender for the **hand-designed
statistical ranking**, not a perfect recommender for downstream malware
detection. The ranking criteria rewarded properties such as spread, trend,
stationarity, and low redundancy. Those properties can be useful, but they do
not necessarily maximize class separation.

The 2SMaRT baseline, by contrast, directly selected events correlated with the
malware label on the training fold. It was therefore better aligned with the
evaluation metric. The earlier result showed that the original statistical
ranking target needed revision; it did not show that binary-specific event
selection was unpromising.

## 3. Detection-Aware Candidate-Pool Oracle

To test the strategy itself, we implemented a detection-aware candidate-pool
oracle. The goal was not to create a deployable detector. The goal was to
estimate whether a perfect per-sample or per-binary router could beat global
event selection if given a strong but computationally feasible pool of
four-event subsets.

### 3.1 Candidate Generation

For each outer fold:

1. Split RaDaR into outer training and outer test folds.
2. Run beam search only on the outer training fold.
3. Score candidate subsets by mean AUCPR over three inner folds.
4. Expand subsets from depth 1 to depth 4.
5. Retain the best 25 partial subsets at depths 1, 2, and 3.
6. At depth 4, retain all final four-event expansions, capped at 1,500.

The search objective was inner-fold AUCPR rather than F1. AUCPR is a better
search signal under class imbalance and avoids overfitting candidate generation
to a single fixed decision threshold.

### 3.2 Compared Strategies

The final experiment compared three strategies:

| Strategy | Description | Deployable? |
|---|---|---|
| 2SMaRT | Fold-local global event selection by Pearson correlation with the training labels | Yes, if labels are available during training |
| Global-Beam | Best four-event subset found by training-only beam search | Yes |
| Candidate-Oracle | Post-hoc test-label-assisted routing to a retained candidate that predicts the sample correctly | No |

Candidate-Oracle is intentionally non-deployable. It uses the true outer-test
label after prediction to decide whether a retained candidate is correct for a
sample. It is a ceiling:

\[
\hat{y}_{oracle}(x_i) =
\begin{cases}
y_i, & \exists S_j \in \mathcal{C}: f_{S_j}(x_i) = y_i \\
f_{S^*}(x_i), & \text{otherwise}
\end{cases}
\]

where \(\mathcal{C}\) is the retained candidate pool and \(S^*\) is the best
global beam-search subset. This asks:

> If a perfect router could choose among training-generated candidate subsets,
> how much performance is available beyond one global event set?

### 3.3 Leakage Boundary

The leakage boundary is important:

- Candidate generation used only outer-training data.
- Inner AUCPR scoring used only training-fold splits.
- Global-Beam selection used only training-fold candidate scores.
- XGBoost models were trained on outer-training data.
- Outer-test labels were not used to generate, score, prune, or train candidate
  models.
- Outer-test labels were used only by Candidate-Oracle for post-hoc routing.

Thus, Global-Beam and 2SMaRT are deployable baselines under their respective
training assumptions. Candidate-Oracle is a non-deployable upper ceiling.

## 4. Experimental Setup

The full run used:

| Parameter | Value |
|---|---:|
| Dataset | RaDaR |
| Samples | 3,470 |
| Malware samples | 2,884 |
| Benign samples | 586 |
| Event universe | 55 |
| Aggregated features | 330, from 55 events x 6 statistics |
| PMU budget | 4 events |
| Outer folds | 5 |
| Inner folds | 3 |
| Beam width | 25 |
| Candidate cap | 1,500 |
| Detector | XGBoost |
| XGBoost trees | 100 |
| XGBoost max depth | 4 |
| Parallel workers | 14 |

The full experiment evaluated 51,594 inner-CV model fits and 6,089 final
candidate fits, for 57,683 total XGBoost fits. It completed in approximately
924 seconds, or 15.4 minutes, because each model used only 6 to 24 features and
XGBoost's histogram tree method.

## 5. Main Results

### 5.1 Primary Result: Family-Disjoint GroupKFold (Publication-Grade)

The aggregate numbers were first computed under sample-stratified five-fold CV.
Because RaDaR contains many same-family malware samples, stratified splits can
place near-duplicate family members in both the train and test folds and inflate
the supervised baselines. We therefore re-ran the full pipeline with
family-disjoint GroupKFold splits (grouping by `family_gene`, five outer folds,
three inner folds). This grouped run is the primary, publication-grade result;
the stratified run in 5.2 is retained only as a secondary comparison that
isolates the effect of the split protocol.

| Strategy | F1 Mean | F1 Std | Precision | Recall | Accuracy | FPR |
|---|---:|---:|---:|---:|---:|---:|
| 2SMaRT | 0.8732 | 0.0515 | 0.8267 | 0.9306 | 0.7852 | 0.8413 |
| Global-Beam | 0.8820 | 0.0423 | 0.8383 | 0.9360 | 0.8002 | 0.7748 |
| Candidate-Oracle | 0.9770 | 0.0126 | 0.9563 | 0.9989 | 0.9628 | 0.1862 |

Three things change under fair evaluation, and all of them strengthen the
thesis:

1. The supervised baselines fall by roughly 3.5 F1 points (2SMaRT 0.9090 to
   0.8732; Global-Beam 0.9134 to 0.8820), confirming that the stratified numbers
   were inflated by family leakage.
2. The oracle is essentially unchanged (0.9781 to 0.9770, a 0.1-point drop),
   confirming that the conditional-selection headroom is not a leakage artifact.
3. The oracle's advantage over 2SMaRT therefore widens from 6.9 to 10.4 F1
   points (0.9770 - 0.8732 = 0.1038).

The most persuasive evidence is the hardest held-out family. GroupKFold over
`family_gene` produces three folds that each hold out one large unseen family
and two folds that hold out clusters of smaller families:

| Fold | Held-out test families | 2SMaRT F1 | Global-Beam F1 | Candidate-Oracle F1 |
|---:|---:|---:|---:|---:|
| 1 | 1 family | 0.9165 | 0.9212 | 0.9935 |
| 2 | 1 family | 0.9311 | 0.9283 | 0.9848 |
| 3 | 1 family | 0.8035 | 0.8280 | 0.9714 |
| 4 | 8 families | 0.8546 | 0.8724 | 0.9608 |
| 5 | 7 families | 0.8605 | 0.8601 | 0.9744 |

On fold 3 the global selectors collapse on an unseen family (2SMaRT 0.80,
Global-Beam 0.83) while the oracle holds at 0.97 - a 16.8-point gap. This is the
clearest demonstration that one global four-event set generalizes poorly to a
novel family, while a conditional router able to pick a family-appropriate subset
does not. This unseen-family behavior is the strongest single piece of evidence
for the recommender thesis.

### 5.2 Sample-Stratified Aggregate Metrics (Secondary)

The table below and the detailed analyses in 5.3-5.6 come from the original
sample-stratified run. They remain useful for understanding the oracle's internal
behavior (confusion structure, capped-pool sensitivity, candidate flatness), but
the headline F1 values here are leakage-inflated and should not be quoted as the
primary result; use 5.1 instead.

| Strategy | F1 Mean | F1 Std | Precision | Recall | Accuracy | AUCPR |
|---|---:|---:|---:|---:|---:|---:|
| 2SMaRT | 0.9090 | 0.0059 | 0.8485 | 0.9789 | 0.8372 | 0.9350 |
| Global-Beam | 0.9134 | 0.0034 | 0.8583 | 0.9761 | 0.8461 | 0.9418 |
| Candidate-Oracle | 0.9781 | 0.0040 | 0.9572 | 1.0000 | 0.9628 | 0.9722 |

The key comparison is not Global-Beam versus 2SMaRT. That difference is small:

\[
0.9134 - 0.9090 = 0.0043 \text{ F1}.
\]

The key comparison is Candidate-Oracle versus global selection:

\[
0.9781 - 0.9090 = 0.0691 \text{ F1 over 2SMaRT},
\]

and

\[
0.9781 - 0.9134 = 0.0648 \text{ F1 over Global-Beam}.
\]

This pattern is exactly what HPC-Boost needs to justify the recommender:
better global search gives only a small gain, while conditional routing gives a
large gain.

### 5.3 Pooled Confusion Counts (Stratified Run)

Across all 3,470 samples:

| Strategy | TP | FP | TN | FN |
|---|---:|---:|---:|---:|
| 2SMaRT | 2,823 | 504 | 82 | 61 |
| Global-Beam | 2,815 | 465 | 121 | 69 |
| Candidate-Oracle | 2,884 | 129 | 457 | 0 |

Candidate-Oracle eliminated all false negatives and reduced false positives
from 504 to 129 relative to 2SMaRT.

The baseline false-positive rate is high because the dataset is malware-heavy:

| Strategy | FPR | Balanced Accuracy |
|---|---:|---:|
| 2SMaRT | 0.8601 | 0.5594 |
| Global-Beam | 0.7935 | 0.5913 |
| Candidate-Oracle | 0.2201 | 0.8899 |

This means the oracle's practical gain is mostly a false-positive reduction
gain on benign samples, while preserving malware recall. This is operationally
valuable, but it also means future experiments must report FPR, balanced
accuracy, and TPR at fixed FPR rather than relying only on malware-weighted F1.

The total error count changed as follows:

| Strategy | Errors | Error Rate |
|---|---:|---:|
| 2SMaRT | 565 | 16.28% |
| Global-Beam | 534 | 15.39% |
| Candidate-Oracle | 129 | 3.72% |

Relative to 2SMaRT, Candidate-Oracle recovered 436 of 565 errors:

\[
\frac{436}{565} = 77.2\% \text{ error reduction}.
\]

Relative to Global-Beam, Candidate-Oracle recovered 405 of 534 errors:

\[
\frac{405}{534} = 75.8\% \text{ error reduction}.
\]

### 5.4 Capped-Pool Oracle Analysis (Stratified Run)

The full Candidate-Oracle can select among roughly 1,200 retained candidates per
fold. This makes it an intentionally optimistic ceiling. To understand how much
of the gain comes from a large candidate pool, we recomputed the oracle by
allowing routing only within the top \(P\) candidates ordered by training
inner-CV AUCPR.

| Pool Size \(P\) | F1 | Precision | Recall | Accuracy | FPR | Errors |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.9134 | 0.8582 | 0.9761 | 0.8461 | 0.7935 | 534 |
| 2 | 0.9213 | 0.8660 | 0.9840 | 0.8602 | 0.7491 | 485 |
| 5 | 0.9351 | 0.8822 | 0.9948 | 0.8853 | 0.6536 | 398 |
| 10 | 0.9406 | 0.8901 | 0.9972 | 0.8954 | 0.6058 | 363 |
| 25 | 0.9502 | 0.9060 | 0.9990 | 0.9130 | 0.5102 | 302 |
| 50 | 0.9570 | 0.9181 | 0.9993 | 0.9254 | 0.4386 | 259 |
| 100 | 0.9620 | 0.9267 | 1.0000 | 0.9343 | 0.3891 | 228 |
| 500 | 0.9733 | 0.9481 | 1.0000 | 0.9545 | 0.2696 | 158 |
| Full pool | 0.9781 | 0.9572 | 1.0000 | 0.9628 | 0.2201 | 129 |

This table materially changes the interpretation. The full-pool number should
be presented as the most optimistic ceiling, not as the likely performance of a
learned recommender. The top-25 result, 0.9502 F1, is a more conservative and
more learnable ceiling because it restricts routing to the highest-ranked
candidates. It still beats 2SMaRT by 0.0412 F1 and Global-Beam by 0.0368 F1.

The improvement from \(P=25\) to the full pool may reflect real conditional
utility among many near-equivalent subsets, but it may also reflect post-hoc
multiple-comparison effects. Importantly, the deeper candidates are not
obviously poor models: their inner-CV AUCPR remains high, typically around
0.93-0.94 at the bottom of the retained pool. The correct criticism is therefore
not that these are necessarily random low-performing models, but that a
label-assisted router over a large pool will overestimate deployable
performance.

### 5.5 AUCPR Plateau (Stratified Run)

A deeper audit of the candidate files shows that the retained candidate pools
are extremely flat under the training objective:

| Fold | Pool Size | Min Inner AUCPR | Max Inner AUCPR | Spread |
|---:|---:|---:|---:|---:|
| 1 | 1,242 | 0.9355 | 0.9505 | 0.0150 |
| 2 | 1,215 | 0.9352 | 0.9497 | 0.0144 |
| 3 | 1,213 | 0.9369 | 0.9475 | 0.0106 |
| 4 | 1,219 | 0.9327 | 0.9468 | 0.0142 |
| 5 | 1,200 | 0.9340 | 0.9485 | 0.0144 |

This is the hardest issue for the recommender thesis. The oracle can separate
near-equivalent candidates using the outer-test label; a deployable recommender
cannot. Therefore, the pilot must test whether per-binary utility is more
differentiated than fold-level inner AUCPR. If per-binary utility is also flat,
the recommender should be trained to predict an FPR-safe candidate class or
utility band rather than a precise top subset.

### 5.6 Per-Fold Stability (Stratified Run)

Candidate-Oracle beat 2SMaRT on every fold:

| Fold | 2SMaRT F1 | Global-Beam F1 | Candidate-Oracle F1 |
|---:|---:|---:|---:|
| 1 | 0.9101 | 0.9107 | 0.9829 |
| 2 | 0.9091 | 0.9180 | 0.9813 |
| 3 | 0.9066 | 0.9117 | 0.9738 |
| 4 | 0.9177 | 0.9161 | 0.9780 |
| 5 | 0.9016 | 0.9104 | 0.9747 |

The paired fold difference for Candidate-Oracle minus 2SMaRT was:

\[
\Delta F1 = 0.0691 \pm 0.0068
\]

using a rough 95% t-interval over the five folds. In contrast, Global-Beam
minus 2SMaRT was:

\[
\Delta F1 = 0.0043 \pm 0.0059.
\]

This indicates that the oracle gain is much larger and more stable than the
gain from merely replacing Pearson event selection with a stronger global
beam-search selector.

## 6. What the Oracle Actually Found

### 6.1 Candidate Coverage

Candidate-Oracle coverage was 96.28%. This means that for 3,341 of 3,470
samples, at least one retained candidate subset predicted the sample correctly.
For 129 samples, no retained candidate predicted correctly.

The class-level pattern is asymmetric:

| Class | Samples | Zero Correct Candidates | Mean Correct Candidates |
|---|---:|---:|---:|
| Benign | 586 | 129 | 218.4 |
| Malware | 2,884 | 0 | 1,190.6 |

Every malware sample had at least one correct candidate, and most malware
samples were correctly predicted by almost the entire candidate pool. The hard
cases were benign samples that many candidate subsets falsely classified as
malware.

This is crucial for the next phase. The recommender should not merely learn
"which subset detects malware"; most subsets already detect malware. It should
learn which subset reduces false positives while preserving malware recall.

### 6.2 Routing Diversity

Across all routed samples, Candidate-Oracle used 223 distinct four-event subsets.
However, most samples used the fold's best global candidate because it was
already correct. The global candidate index was used for 3,065 of 3,470 samples.

For the 405 samples where Global-Beam was wrong but Candidate-Oracle was
correct, the oracle used 219 distinct subsets. This matters because the gain is
not explained by a single alternative global subset. The correction set is
distributed across many subsets.

This is the strongest evidence for binary-conditional routing:

> The performance gain appears when the selector can choose different subsets
> for different samples, not when it simply chooses a stronger single global
> subset.

### 6.3 Event Patterns

The most common events in candidate pools included:

| Rank | Event | Candidate-Pool Occurrences |
|---:|---|---:|
| 1 | L3_LAT_C_Miss | 4,028 |
| 2 | L2ReqPFms | 3,071 |
| 3 | M_Ld_Ret_L2Hit | 1,863 |
| 4 | M_Ld_LLCH.XS_M | 1,340 |
| 5 | M_Ld_LLCH.XS_N | 887 |

The most common events in oracle-routed subsets included:

| Rank | Event | Routed Occurrences |
|---:|---|---:|
| 1 | L3_LAT_C_Miss | 2,085 |
| 2 | res.stl. | 1,933 |
| 3 | L2ReqPFms | 1,489 |
| 4 | M_Ld_Ret_L2Hit | 1,374 |
| 5 | M_Ld_LLCH.XS_M | 1,359 |

The candidate pool was memory-hierarchy heavy. This does not mean these events
are universally optimal, but it suggests that the RaDaR dataset's detection
signal is strongly tied to cache, memory-load, and pipeline-stall behavior.

### 6.4 Fold-to-Fold Candidate Stability

The union of retained candidate subsets across folds contained 4,382 unique
subsets, while the intersection across all five folds contained only 6 subsets.
Pairwise Jaccard overlap ranged from approximately 0.03 to 0.31.

This has two interpretations:

1. There is meaningful instability in the exact selected subset pool.
2. There are many near-equivalent subsets with similar detection utility.

For recommender training, this argues against using a single hard top-4 label as
the only target. A better target is a utility distribution or ranked candidate
list.

## 7. What This Supports

The experiment supports five claims.

### Claim 1: There is real headroom for conditional event selection

Candidate-Oracle improves F1 from 0.9090 to 0.9781 over 2SMaRT. This is too
large to dismiss as noise in this proof-of-concept setting.

### Claim 2: The headroom is not merely better global event selection

Global-Beam improves only from 0.9090 to 0.9134. If the gain came only from
finding a better global subset, Global-Beam would have captured it. It did not.

### Claim 3: The useful signal is sample-conditional

Candidate-Oracle recovered 405 samples that Global-Beam missed, using 219
distinct routed subsets for those recoveries. That pattern is consistent with
the HPC-Boost thesis that different binaries or runtime cases benefit from
different event subsets.

### Claim 4: The main practical opportunity is false-positive reduction

All malware samples had at least one correct candidate. The unrecoverable cases
were benign. Therefore, the recommender should be optimized not only for malware
recall but for choosing subsets that avoid benign false positives.

### Claim 5: The next phase is justified

The result justifies collecting our own binary traces and training a recommender
because it demonstrates a large gap between global selection and ideal
conditional selection. Without such a gap, recommender training would be
unjustified. With this gap, the recommender becomes a meaningful research
question.

However, the AUCPR plateau means the next phase must be framed as a learnability
test. The current result justifies asking whether static binary features can
predict useful conditional choices; it does not yet show that the ranking signal
is sufficiently differentiated for a recommender.

## 8. What This Does Not Establish

This experiment should not be oversold. A strong reviewer would object to any
of the following claims.

### It does not prove deployable HPC-Boost performance

Candidate-Oracle uses the true test label after prediction. A deployed system
will not have this label. The result is an upper ceiling, not a deployable
detector.

### It does not prove that static binary features can predict the route

The oracle chooses a subset after seeing candidate predictions and the true
label. The recommender will see only static binary features. The next phase must
test whether static features contain enough information to approximate the
oracle's choice.

### It does not prove per-binary stability

RaDaR provides samples, but the future recommender is binary-driven. If multiple
runs of the same binary require different event subsets due to input,
scheduling, or phase behavior, then one static top-4 label per binary may be
unstable. The new data collection must measure this.

### Trace-level split leakage: now tested with GroupKFold

The original run used stratified random folds over RaDaR samples, which can let
family-level near-duplicates cross the train and test folds. We have since
addressed this directly with family-disjoint GroupKFold (Section 5.1). The result
is reassuring rather than damaging: the supervised baselines were indeed
leakage-inflated and drop by about 3.5 F1 points under grouped splits, but the
oracle is essentially unchanged, so the conditional-selection headroom is not a
split artifact - it grows from 6.9 to 10.4 F1 points. A residual caveat remains
for the eventual custom dataset: `family_gene` grouping prevents family leakage
but not necessarily repeated-run or collection-session leakage, which the new
collection must control explicitly by grouping on binary and session as well.

### It does not show that Global-Beam is meaningfully better than 2SMaRT

Global-Beam's mean F1 is higher than 2SMaRT's by 0.0043, but the paired
fold-level difference is not statistically significant at the conventional
0.05 level. The memo should therefore treat Global-Beam as approximately tied
with 2SMaRT, not as a clear improvement.

### It does not prove cross-machine generalization

The oracle is tied to a dataset and event universe. A recommender trained on one
processor, kernel, workload mix, or PMU naming convention may not transfer
without adaptation.

### It does not validate AUCPR for the oracle as a calibrated metric

Candidate-Oracle's probabilities come from different candidate models selected
with label assistance. Therefore, oracle AUCPR should be treated cautiously.
F1, accuracy, error recovery, and coverage are the more meaningful oracle
statistics.

## 9. Decision: Start Binary Tracing, But As a Controlled Pilot

The result gives enough confidence to start running binaries and recording HPC
traces for recommender training. It does **not** justify blindly collecting a
large dataset without first validating the target-construction protocol.

The recommended next phase is a pilot with explicit go/no-go criteria.

The RaDaR oracle has now been rerun with family-disjoint GroupKFold splits
(Section 5.1), which serve as the primary publication-grade numbers; the
stratified run is reported only as a secondary split-methodology comparison. The
custom collection should extend this discipline to grouping by binary and
collection session, not only by family.

### 9.1 Unit of Recommendation

The recommender input is a static binary. Therefore, the target should be
defined per binary, not per individual trace.

For a binary \(b\), collect multiple runs:

\[
\mathcal{R}_b = \{r_{b,1}, r_{b,2}, \dots, r_{b,n}\}.
\]

For each candidate event subset \(S\), estimate utility over runs:

\[
U(b, S) =
\text{DetectionUtility}(S, \mathcal{R}_b)
- \lambda \cdot \text{FalsePositivePenalty}(S, \mathcal{R}_b)
- \gamma \cdot \text{InstabilityPenalty}(S, \mathcal{R}_b).
\]

The recommender should learn either:

- a ranked list of event subsets,
- a utility vector over events or candidate subsets,
- or a soft distribution over near-optimal choices.

It should not be trained only on one arbitrary correct subset.

### 9.2 Pilot Collection Design

The pilot should collect:

| Component | Recommendation |
|---|---|
| Benign binaries | 30-50 diverse programs |
| Malware or attack cases | 30-50 controlled attack/malware executions |
| Runs per condition | At least 5, preferably 10 |
| PMU events | Full event catalog, collected in four-event groups without multiplexing |
| Core control | Pin to fixed P-core; disable migration |
| Frequency control | Performance governor; disable turbo if possible |
| Split discipline | Split by binary, family, and collection session |

The pilot's purpose is not to maximize dataset size. Its purpose is to answer:

1. Are event-utility rankings stable across repeated runs of the same binary?
2. Do different binaries have meaningfully different optimal subsets?
3. Can static binary features predict those utilities better than a global
   baseline?

### 9.3 Go/No-Go Criteria

Proceed to full-scale recommender training if the pilot shows:

| Criterion | Minimum Evidence |
|---|---|
| Oracle gap | Per-binary detection-aware oracle beats best global subset by at least 3-5 F1 points |
| Stability | Top-k event utility is stable across repeated runs |
| Learnability | A simple recommender beats global and random routing on held-out binaries |
| Generalization | Held-out binaries do not collapse to global-event behavior |
| Overhead | Static feature extraction and recommender inference are lightweight |

Pause or redesign if:

- per-binary oracle gain disappears on our own traces,
- event utility is unstable across repeated runs,
- static features fail to predict the utility target,
- or the gain comes only from benign/malware label leakage rather than binary
  structure.

For the pilot, a useful initial target is not the full-pool \(P \approx 1200\)
oracle. Use a capped pool such as \(P=25\) or \(P=50\) for the primary
recommender target, and report regret against both the capped oracle and the
full oracle. This prevents the recommender from being judged against a target
that may be dominated by post-hoc pool-size effects.

## 10. Recommended Recommender Target

The new result argues for a utility-aware recommender. The target should not be
the original unsupervised statistical ranking. It should be detection-aware.

A practical target construction is:

1. Generate a candidate pool per training fold using beam search.
2. Evaluate each candidate subset on repeated runs of each binary.
3. Assign each binary a utility score for each candidate:

\[
U_{b,S} = \alpha \cdot \text{TPR}_{b,S}
- \beta \cdot \text{FPR}_{b,S}
+ \eta \cdot \text{AUCPR}_{b,S}
- \gamma \cdot \text{Variance}_{b,S}.
\]

4. Train the recommender with a listwise ranking loss or soft-label objective:

\[
P(S \mid b) =
\frac{\exp(U_{b,S}/\tau)}
{\sum_{S'} \exp(U_{b,S'}/\tau)}.
\]

5. Evaluate top-1 and top-k regret:

\[
\text{Regret}(b) = U(b, S^*_b) - U(b, \hat{S}_b).
\]

This formulation handles the fact that many subsets are near-equivalent. It is
also more defensible than forcing the model to imitate one oracle-selected
subset when many candidates would work.

## 11. Reviewer-Risk Checklist

Before using this result in a paper, the following risks should be addressed.

### Risk 1: "The oracle cheats"

Correct response: yes, Candidate-Oracle is explicitly non-deployable. Its role
is to measure headroom. Deployable performance must be reported separately for
Global-Beam, 2SMaRT, and the learned recommender.

### Risk 2: "The recommender cannot see test labels"

Correct response: the recommender will not be trained to use test labels. The
oracle motivates target construction. The recommender must be evaluated on
held-out binaries using only static features.

### Risk 3: "The gain may reflect dataset artifacts"

Correct response: the custom data collection must use grouped splits by binary,
family, and collection session; cross-session and cross-machine tests should be
added where possible.

### Risk 3a: "The oracle gain is inflated by selecting over too many candidates"

Correct response: report capped-pool oracle curves. The full-pool Candidate-
Oracle is a ceiling. The top-25 capped oracle still shows a meaningful gain
over 2SMaRT, but the full-pool number should not be used as the expected
recommender performance.

### Risk 3b: "The candidate pool is too flat for a recommender to learn"

Correct response: this is an open risk. The pilot must measure per-binary
utility separation and repeated-run stability. If candidate utility remains
flat at the binary level, the target should be FPR-safe subset classification or
coarse utility-band prediction rather than exact listwise ranking.

### Risk 4: "The oracle is mostly fixing benign false positives"

Correct response: that is a valid and important finding. In high-recall malware
detection, false-positive reduction is operationally valuable. The recommender
objective should explicitly optimize FPR at high TPR.

### Risk 5: "The selected subsets are unstable"

Correct response: exact subsets are unstable, but utility may be stable over a
family of near-equivalent subsets. The recommender should predict utility
distributions or event preferences, not only one hard subset.

## 12. Conclusion

The detection-aware candidate-pool oracle changes the project status. The
earlier statistical-ranking oracle was inconclusive because it tested a
hand-designed ranking target rather than detection-aware selection. The new
oracle directly asks whether conditional four-event selection has enough
performance headroom to justify a recommender.

The answer is yes. Candidate-Oracle improves mean F1 from 0.9090 to 0.9781 over
2SMaRT and reduces total errors by 77.2%. Global-Beam, a stronger global
selector, improves only slightly over 2SMaRT. Therefore, the performance gain is
not primarily from better global event selection; it comes from conditional
routing among multiple event subsets.

This is sufficient confidence to begin running binaries and recording traces for
the recommender phase. The next step should be a controlled pilot that builds
detection-aware, per-binary utility targets from repeated runs. If static binary
features can predict those targets on held-out binaries, HPC-Boost will have a
credible path from theoretical oracle headroom to deployable binary-driven PMU
event recommendation.

## Appendix A: Artifact Provenance

Primary local artifacts used:

- `oracle_results/data/processed/results/detection_aware_beam_oracle/aggregate_metrics.csv`
- `oracle_results/data/processed/results/detection_aware_beam_oracle/fold_metrics.csv`
- `oracle_results/data/processed/results/detection_aware_beam_oracle/config.json`
- `oracle_results/data/processed/results/detection_aware_beam_oracle/fold_*/summary.json`
- `oracle_results/data/processed/results/detection_aware_beam_oracle/fold_*/predictions.csv`
- `oracle_results/data/processed/results/detection_aware_beam_oracle/fold_*/candidates.csv`
- `oracle_results/logs/detection_aware_beam_oracle.log`
- `hpc_boost_v2/experiments/exp3_detection_aware_oracle/run_beam_oracle.py`
- `hpc_boost_v2/experiments/exp3_detection_aware_oracle/analyze_pool_size.py`

Family-disjoint GroupKFold run (primary, Section 5.1):

- `oracle_results/data/processed/results/group_kfold_oracle/aggregate_metrics.csv`
- `oracle_results/data/processed/results/group_kfold_oracle/fold_metrics.csv`
- `oracle_results/data/processed/results/group_kfold_oracle/summary.json`
- `oracle_results/data/processed/results/group_kfold_oracle/all_predictions.csv`
- `hpc_boost_v2/experiments/exp3_detection_aware_oracle/run_group_kfold_oracle.py`

No external web sources were used in this memo. All quantitative claims are
derived from the local experiment artifacts listed above.
