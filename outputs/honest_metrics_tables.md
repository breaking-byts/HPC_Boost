# HPC-Boost Honest Metrics


F1 is majority-class inflated on this ~83% positive data; balanced accuracy (mean of recall and specificity) is the honest summary metric. The trivial always-malware baseline scores F1 ~0.91 and balanced accuracy exactly 0.50.


## (a) Stratified 5-fold CV: pooled metrics

_Caption: Pooled (sample-level) confusion and metrics across all 5 folds for the stratified run. F1 is **majority-class inflated** (the trivial always-malware row scores F1 ~0.91); **balanced accuracy** and MCC are the honest metrics. Positive class = malware._

| Strategy | TP | FP | TN | FN | Precision | Recall | F1 | FPR | Bal. Acc | MCC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2SMaRT | 2823 | 504 | 82 | 61 | 0.8485 | 0.9788 | 0.9090 | 0.8601 | 0.5594 | 0.2239 |
| Global-Beam | 2815 | 465 | 121 | 69 | 0.8582 | 0.9761 | 0.9134 | 0.7935 | 0.5913 | 0.3006 |
| Candidate-Oracle | 2884 | 129 | 457 | 0 | 0.9572 | 1.0000 | 0.9781 | 0.2201 | 0.8899 | 0.8640 |
| _Trivial (always malware)_ | 2884 | 586 | 0 | 0 | 0.8311 | 1.0000 | 0.9078 | 1.0000 | 0.5000 | 0.0000 |


## (b) GroupKFold (family-disjoint) CV: pooled metrics

_Caption: Pooled metrics for the family-disjoint GroupKFold run, where test families never appear in training. Same caveat: F1 is inflated by the ~83% malware prior; read **balanced accuracy** and MCC. Positive class = malware._

| Strategy | TP | FP | TN | FN | Precision | Recall | F1 | FPR | Bal. Acc | MCC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2SMaRT | 2685 | 493 | 93 | 199 | 0.8449 | 0.9310 | 0.8858 | 0.8413 | 0.5449 | 0.1211 |
| Global-Beam | 2692 | 454 | 132 | 192 | 0.8557 | 0.9334 | 0.8929 | 0.7747 | 0.5793 | 0.2043 |
| Candidate-Oracle | 2881 | 109 | 477 | 3 | 0.9635 | 0.9990 | 0.9809 | 0.1860 | 0.9065 | 0.8822 |
| _Trivial (always malware)_ | 2884 | 586 | 0 | 0 | 0.8311 | 1.0000 | 0.9078 | 1.0000 | 0.5000 | 0.0000 |


## (c) Fold-mean vs pooled F1 reconciliation

_Caption: Fold-averaged F1 (mean of per-fold F1) vs pooled (sample-level) F1. Differences arise because folds have unequal sizes and difficulty; the unweighted fold mean over-weights small or easy folds. Pooled F1 is the sample-level ground truth._

| Run | Strategy | Fold-mean F1 | Pooled F1 | Difference |
|---|---|---:|---:|---:|
| Stratified | 2SMaRT | 0.9090 | 0.9090 | -0.0000 |
| Stratified | Global-Beam | 0.9134 | 0.9134 | +0.0000 |
| Stratified | Candidate-Oracle | 0.9781 | 0.9781 | +0.0000 |
| GroupKFold | 2SMaRT | 0.8732 | 0.8858 | -0.0126 |
| GroupKFold | Global-Beam | 0.8820 | 0.8929 | -0.0108 |
| GroupKFold | Candidate-Oracle | 0.9770 | 0.9809 | -0.0039 |


## (d) GroupKFold family structure

_Caption: Family composition driving the GroupKFold splits. All benign samples belong to a single family, so family-disjoint CV places that entire benign family in one test fold; the other folds see very few or zero benign test samples, which is why per-fold specificity (and thus balanced accuracy) is volatile._

| Quantity | Value |
|---|---:|
| Total samples | 3470 |
| Malware samples | 2884 |
| Benign samples | 586 |
| Total distinct families | 19 |
| Distinct benign (y=0) families | 1 |
| Distinct malware (y=1) families | 18 |
| Families with both labels (mixed) | 0 |

| Fold | Test samples | Test malware | Test benign | Test families |
|---:|---:|---:|---:|---:|
| 1 | 1041 | 923 | 118 | 1 |
| 2 | 862 | 745 | 117 | 1 |
| 3 | 647 | 530 | 117 | 1 |
| 4 | 460 | 343 | 117 | 8 |
| 5 | 460 | 343 | 117 | 7 |


## (e) Capped candidate-pool oracle curve (stratified run)

_Caption: Candidate-Oracle metrics when the oracle may route only among the top-P retained candidates (ordered by training score). P=1 reproduces the Global-Beam operating point; the curve rises to the full oracle as P grows. F1 is inflated throughout; track balanced accuracy and errors._

| Pool size P | F1 | Precision | Recall | Accuracy | FPR | Bal. Acc | Errors | TP | FP | TN | FN |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.9134 | 0.8582 | 0.9761 | 0.8461 | 0.7935 | 0.5913 | 534 | 2815 | 465 | 121 | 69 |
| 2 | 0.9213 | 0.8660 | 0.9840 | 0.8602 | 0.7491 | 0.6175 | 485 | 2838 | 439 | 147 | 46 |
| 5 | 0.9351 | 0.8822 | 0.9948 | 0.8853 | 0.6536 | 0.6706 | 398 | 2869 | 383 | 203 | 15 |
| 10 | 0.9406 | 0.8901 | 0.9972 | 0.8954 | 0.6058 | 0.6957 | 363 | 2876 | 355 | 231 | 8 |
| 25 | 0.9502 | 0.9060 | 0.9990 | 0.9130 | 0.5102 | 0.7444 | 302 | 2881 | 299 | 287 | 3 |
| 50 | 0.9570 | 0.9181 | 0.9993 | 0.9254 | 0.4386 | 0.7804 | 259 | 2882 | 257 | 329 | 2 |
| 100 | 0.9620 | 0.9267 | 1.0000 | 0.9343 | 0.3891 | 0.8055 | 228 | 2884 | 228 | 358 | 0 |
| 250 | 0.9689 | 0.9397 | 1.0000 | 0.9467 | 0.3157 | 0.8422 | 185 | 2884 | 185 | 401 | 0 |
| 500 | 0.9733 | 0.9481 | 1.0000 | 0.9545 | 0.2696 | 0.8652 | 158 | 2884 | 158 | 428 | 0 |
| 1000 | 0.9768 | 0.9547 | 1.0000 | 0.9605 | 0.2338 | 0.8831 | 137 | 2884 | 137 | 449 | 0 |
| 1500 | 0.9781 | 0.9572 | 1.0000 | 0.9628 | 0.2201 | 0.8899 | 129 | 2884 | 129 | 457 | 0 |
