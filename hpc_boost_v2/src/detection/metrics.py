from __future__ import annotations
from typing import Dict
import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix,
    balanced_accuracy_score, matthews_corrcoef, roc_curve,
)


def tpr_at_fpr(y_true: np.ndarray, y_proba: np.ndarray, fpr_target: float) -> float:
    """Highest TPR achievable at a false-positive rate <= fpr_target.

    Computed from the ROC curve. Returns 0.0 when the curve cannot be built
    (e.g. only one class present in y_true).
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba, dtype=float)
    if len(np.unique(y_true)) < 2:
        return 0.0
    try:
        fpr, tpr, _ = roc_curve(y_true, y_proba)
    except ValueError:
        return 0.0
    allowed = fpr <= fpr_target
    if not np.any(allowed):
        return 0.0
    return float(np.max(tpr[allowed]))


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray,
                         y_proba: np.ndarray | None = None) -> Dict[str, float]:
    """Compute all detection metrics."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    results = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'f1': float(f1_score(y_true, y_pred, zero_division=0)),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall_tpr': float(recall_score(y_true, y_pred, zero_division=0)),
        'fpr': float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0,
        'balanced_accuracy': float(balanced_accuracy_score(y_true, y_pred)),
        'mcc': float(matthews_corrcoef(y_true, y_pred)),
        'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn),
    }
    if y_proba is not None:
        try:
            results['auc_roc'] = float(roc_auc_score(y_true, y_proba))
        except ValueError:
            results['auc_roc'] = 0.0
    return results
