"""Tests for probability fusion and threshold selection."""

from __future__ import annotations

import numpy as np
import pytest

from fakenews.eval.metrics import tune_threshold
from fakenews.models.classical import combine_probs


def test_fusion_alpha_extremes_degenerate_to_single_branch():
    p_title = np.array([0.1, 0.9, 0.4])
    p_body = np.array([0.8, 0.2, 0.6])
    assert np.allclose(combine_probs(p_title, p_body, 1.0), p_title)
    assert np.allclose(combine_probs(p_title, p_body, 0.0), p_body)


def test_fusion_is_convex_combination():
    p_title = np.array([0.2, 0.8])
    p_body = np.array([0.8, 0.2])
    fused = combine_probs(p_title, p_body, 0.25)
    assert np.allclose(fused, [0.65, 0.35])


def test_fusion_rejects_out_of_range_alpha():
    with pytest.raises(ValueError):
        combine_probs(np.array([0.5]), np.array([0.5]), 1.5)


def test_tune_threshold_prefers_separating_threshold():
    y = np.array([0, 0, 1, 1])
    prob = np.array([0.1, 0.2, 0.8, 0.9])
    threshold, f1 = tune_threshold(y, prob)
    assert 0.2 <= threshold <= 0.8
    assert f1 == 1.0


def test_tuned_threshold_ties_prefer_0_5():
    y = np.array([0, 1])
    prob = np.array([0.3, 0.7])
    threshold, f1 = tune_threshold(y, prob)
    assert f1 == 1.0
    assert abs(threshold - 0.5) < 1e-9
