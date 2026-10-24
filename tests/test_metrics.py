"""Tests for metrics, threshold tuning and McNemar."""

from __future__ import annotations

import numpy as np

from fakenews.eval.metrics import (
    brier_score,
    compute_metrics,
    mcnemar,
    reliability_curve,
    tune_threshold,
)


def test_compute_metrics_perfect_classifier():
    y = np.array([0, 0, 1, 1])
    prob = np.array([0.1, 0.2, 0.8, 0.9])
    metrics = compute_metrics(y, prob, threshold=0.5)
    assert metrics["tp"] == 2
    assert metrics["tn"] == 2
    assert metrics["fp"] == 0
    assert metrics["fn"] == 0
    assert metrics["f1"] == 1.0
    assert metrics["roc_auc"] == 1.0
    assert metrics["n"] == 4


def test_single_class_auc_is_nan():
    y = np.zeros(4, dtype=int)
    prob = np.array([0.1, 0.2, 0.3, 0.4])
    metrics = compute_metrics(y, prob)
    assert np.isnan(metrics["roc_auc"])
    assert np.isnan(metrics["pr_auc"])


def test_compute_metrics_known_confusion_matrix():
    y = np.array([0, 0, 0, 1, 1, 1])
    prob = np.array([0.2, 0.6, 0.4, 0.7, 0.3, 0.8])
    metrics = compute_metrics(y, prob, threshold=0.5)
    # predictions: 0, 1, 0, 1, 0, 1 -> tp=2, fp=1, fn=1, tn=2
    assert (metrics["tp"], metrics["fp"], metrics["fn"], metrics["tn"]) == (2, 1, 1, 2)
    assert metrics["precision"] == 2 / 3
    assert metrics["recall"] == 2 / 3


def test_tune_threshold_returns_best_f1():
    y = np.array([0, 0, 1, 1])
    prob = np.array([0.05, 0.35, 0.65, 0.95])
    threshold, f1 = tune_threshold(y, prob)
    assert f1 == 1.0
    assert 0.05 <= threshold <= 0.65


def test_mcnemar_counts_discordant_pairs():
    y = np.array([0, 0, 1, 1])
    perfect = np.array([0, 0, 1, 1])
    inverted = np.array([1, 1, 0, 0])
    result = mcnemar(y, perfect, inverted)
    assert sorted([result["b"], result["c"]]) == [0, 4]
    assert 0 <= result["p_value"] <= 1


def test_brier_and_reliability_shapes():
    y = np.array([0, 1, 0, 1])
    prob = np.array([0.0, 1.0, 0.0, 1.0])
    assert brier_score(y, prob) == 0.0
    mean_pred, empirical = reliability_curve(y, prob, n_bins=5)
    assert mean_pred.shape == (5,)
    assert empirical.shape == (5,)
