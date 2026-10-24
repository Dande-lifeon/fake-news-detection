"""Tests for bootstrap confidence intervals."""

from __future__ import annotations

import numpy as np

from fakenews.eval.bootstrap import bootstrap_f1, paired_bootstrap_delta


def test_bootstrap_f1_contains_point_estimate():
    y = np.array([0, 0, 1, 1] * 20)
    prob = np.where(y == 1, 0.9, 0.1)
    low, high = bootstrap_f1(y, prob, n=200, seed=42)
    assert 0.0 <= low <= high <= 1.0
    assert low > 0.9


def test_paired_delta_is_zero_for_identical_predictions():
    y = np.array([0, 1, 0, 1] * 20)
    prob = np.where(y == 1, 0.8, 0.2)
    result = paired_bootstrap_delta(y, prob, prob, n=200, seed=42)
    assert result["delta"] == 0.0
    assert result["ci_low"] == 0.0
    assert result["ci_high"] == 0.0
    assert result["p_delta_le_0"] == 1.0


def test_bootstrap_is_deterministic():
    y = np.array([0, 0, 1, 1] * 20)
    prob = np.where(y == 1, 0.8, 0.2)
    first = bootstrap_f1(y, prob, n=100, seed=7)
    second = bootstrap_f1(y, prob, n=100, seed=7)
    assert first == second
