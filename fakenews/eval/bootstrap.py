"""Bootstrap confidence intervals for F1 and paired F1 differences."""

from __future__ import annotations

import numpy as np


def _f1(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> float:
    pred = y_prob >= threshold
    tp = np.sum(pred & (y_true == 1))
    fp = np.sum(pred & (y_true == 0))
    fn = np.sum(~pred & (y_true == 1))
    denominator = 2 * tp + fp + fn
    return float(2 * tp / denominator) if denominator else 0.0


def _resample_indices(y_true: np.ndarray, rng: np.random.Generator, stratified: bool) -> np.ndarray:
    if not stratified:
        return rng.integers(0, len(y_true), len(y_true))
    indices = []
    for label in (0, 1):
        label_idx = np.flatnonzero(y_true == label)
        if label_idx.size:
            indices.append(rng.choice(label_idx, label_idx.size, replace=True))
    if not indices:
        return rng.integers(0, len(y_true), len(y_true))
    return np.concatenate(indices)


def bootstrap_f1(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    n: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
    stratified: bool = True,
) -> tuple[float, float]:
    """Percentile bootstrap CI for F1 (positive class = 1)."""
    y_true = np.asarray(y_true).astype(int).ravel()
    y_prob = np.asarray(y_prob, dtype=float).ravel()
    if len(y_true) == 0:
        raise ValueError("empty input")
    rng = np.random.default_rng(seed)
    values = np.empty(n)
    for i in range(n):
        idx = _resample_indices(y_true, rng, stratified)
        values[i] = _f1(y_true[idx], y_prob[idx], threshold)
    return (
        float(np.percentile(values, 100 * alpha / 2)),
        float(np.percentile(values, 100 * (1 - alpha / 2))),
    )


def paired_bootstrap_delta(
    y_true: np.ndarray,
    prob_a: np.ndarray,
    prob_b: np.ndarray,
    threshold: float = 0.5,
    n: int = 1000,
    seed: int = 42,
    stratified: bool = True,
) -> dict[str, float]:
    """Paired bootstrap of ``F1(a) - F1(b)`` on the same rows.

    Returns the point estimate, percentile CI, and the share of replicates with
    ``delta <= 0`` (a one-sided bootstrap p-value for "a is not better").
    """
    y_true = np.asarray(y_true).astype(int).ravel()
    prob_a = np.asarray(prob_a, dtype=float).ravel()
    prob_b = np.asarray(prob_b, dtype=float).ravel()
    if not (len(y_true) == len(prob_a) == len(prob_b)):
        raise ValueError("arrays must have the same length")

    point = _f1(y_true, prob_a, threshold) - _f1(y_true, prob_b, threshold)
    rng = np.random.default_rng(seed)
    deltas = np.empty(n)
    for i in range(n):
        idx = _resample_indices(y_true, rng, stratified)
        deltas[i] = _f1(y_true[idx], prob_a[idx], threshold) - _f1(
            y_true[idx], prob_b[idx], threshold
        )
    return {
        "delta": float(point),
        "ci_low": float(np.percentile(deltas, 2.5)),
        "ci_high": float(np.percentile(deltas, 97.5)),
        "p_delta_le_0": float(np.mean(deltas <= 0)),
    }
