"""Classification metrics, threshold tuning and calibration helpers."""

from __future__ import annotations

import numpy as np
from scipy.stats import chi2
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float | int]:
    """Precision/recall/F1/accuracy/AUC/confusion counts for positive class 1."""
    y_true = np.asarray(y_true).astype(int).ravel()
    y_prob = np.asarray(y_prob, dtype=float).ravel()
    if y_true.shape != y_prob.shape:
        raise ValueError("y_true and y_prob must have the same shape")
    if len(y_true) == 0:
        raise ValueError("cannot compute metrics on an empty array")

    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    both_classes = len(np.unique(y_true)) > 1
    roc_auc = float(roc_auc_score(y_true, y_prob)) if both_classes else float("nan")
    pr_auc = float(average_precision_score(y_true, y_prob)) if both_classes else float("nan")

    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "n": int(len(y_true)),
    }


def tune_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    grid: np.ndarray | None = None,
) -> tuple[float, float]:
    """Return ``(best_threshold, best_f1)``; ties prefer the threshold nearest 0.5."""
    if grid is None:
        grid = np.arange(0.05, 0.96, 0.01)
    y_true = np.asarray(y_true).astype(int).ravel()
    y_prob = np.asarray(y_prob, dtype=float).ravel()

    scored: list[tuple[float, float]] = []
    for threshold in grid:
        pred = (y_prob >= threshold).astype(int)
        scored.append((float(threshold), float(f1_score(y_true, pred, zero_division=0))))

    best_f1 = max(f1 for _, f1 in scored)
    candidates = [thr for thr, f1 in scored if np.isclose(f1, best_f1)]
    best_threshold = min(candidates, key=lambda thr: abs(thr - 0.5))
    return float(best_threshold), float(best_f1)


def mcnemar(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> dict[str, float | int]:
    """McNemar test with continuity correction for two paired predictions."""
    y_true = np.asarray(y_true).astype(int).ravel()
    correct_a = np.asarray(pred_a).astype(int).ravel() == y_true
    correct_b = np.asarray(pred_b).astype(int).ravel() == y_true

    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))
    if b + c == 0:
        statistic, p_value = 0.0, 1.0
    else:
        statistic = (abs(b - c) - 1) ** 2 / (b + c)
        p_value = float(chi2.sf(statistic, df=1))
    return {"b": b, "c": c, "chi2": float(statistic), "p_value": p_value}


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Mean squared error between predicted probability and outcome."""
    return float(brier_score_loss(np.asarray(y_true).astype(int).ravel(), y_prob))


def reliability_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (mean predicted probability, empirical positive rate) per bin."""
    y_true = np.asarray(y_true).astype(int).ravel()
    y_prob = np.asarray(y_prob, dtype=float).ravel()
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.clip(np.digitize(y_prob, edges[1:-1], right=True), 0, n_bins - 1)

    mean_pred = np.full(n_bins, np.nan)
    empirical = np.full(n_bins, np.nan)
    for index in range(n_bins):
        mask = bin_ids == index
        if mask.any():
            mean_pred[index] = float(y_prob[mask].mean())
            empirical[index] = float(y_true[mask].mean())
    return mean_pred, empirical
