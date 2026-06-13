# Detection-Aware Beam Oracle Design

## Objective

Measure a computationally feasible upper ceiling for per-binary four-PMU event
selection. Candidate event subsets are generated without outer-test information.
Outer-test labels are used only by an explicitly non-deployable oracle router to
measure whether a perfect recommender could choose a correct candidate per
sample.

## Evaluation Protocol

- Use five stratified outer folds.
- Run beam search using only each outer training fold.
- Score search candidates by mean three-fold inner-validation AUCPR.
- Expand subsets from one to four events.
- Retain the best 25 subsets after depths one, two, and three.
- At depth four, retain all unique expansions as the candidate pool, subject to
  an optional cap ordered by inner AUCPR.
- Fit every final candidate on the full outer training fold and predict the
  untouched outer test fold.
- The best inner-CV candidate is the deployable `Global-Beam` result.
- The `Candidate-Oracle` chooses the highest-inner-score candidate that predicts
  each outer-test sample correctly. If none does, it uses the Global-Beam
  prediction.
- Fit a fold-local 2SMaRT baseline for comparison.

## Leakage Boundary

Outer-test samples and labels do not participate in feature aggregation
normalization, beam expansion, candidate scoring, pruning, model fitting, or
global-candidate selection. Test labels participate only in post-hoc
Candidate-Oracle routing and must be described as non-deployable.

## Outputs

Each outer fold writes:

- candidate subset scores,
- per-sample Global-Beam, 2SMaRT, and Candidate-Oracle predictions,
- oracle-routed subset identities,
- fold metrics and runtimes,
- a completion checkpoint.

The final run writes CSV and JSON summaries. A resume flag skips completed
folds.

## Operational Controls

The CLI controls dataset path, sample limit, outer and inner folds, beam width,
candidate cap, parallel jobs, XGBoost estimators/depth, random seed, output
directory, and resume behavior. XGBoost always uses one thread per fit so joblib
controls parallelism.

## Validation

Unit tests use synthetic data to verify deterministic subset expansion,
training-only beam selection, oracle routing, and metric behavior. A server
smoke test uses a small sample limit and reduced search settings before the full
run.
