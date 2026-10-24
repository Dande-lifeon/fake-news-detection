"""Tests for the 18 shortcut / format features."""

from __future__ import annotations

import numpy as np
import pandas as pd

from fakenews.features.shortcut import FEATURES, extract_shortcut_features


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "title": ["WATCH: Is this real?", "Calm title"],
            "text_raw": [
                "WASHINGTON (Reuters) - Hello https://x.com via Getty Images follow us!",
                "No source cue here.",
            ],
            "label": [1, 0],
        },
        index=[10, 20],
    )


def test_feature_columns_and_index():
    features = extract_shortcut_features(_frame())
    assert list(features.columns) == FEATURES
    assert features.index.tolist() == [10, 20]
    assert not features.isna().any().any()
    assert all(np.issubdtype(dtype, np.floating) for dtype in features.dtypes)


def test_cue_flags():
    features = extract_shortcut_features(_frame())
    assert features.loc[10, "has_reuters_dateline"] == 1
    assert features.loc[10, "has_url"] == 1
    assert features.loc[10, "has_getty"] == 1
    assert features.loc[10, "has_via"] == 1
    assert features.loc[10, "has_follow_us"] == 1
    assert features.loc[10, "starts_with_city_dateline"] == 1
    assert features.loc[10, "title_has_question"] == 1

    assert features.loc[20, "has_reuters_dateline"] == 0
    assert features.loc[20, "has_url"] == 0
    assert features.loc[20, "starts_with_city_dateline"] == 0


def test_counts_and_lengths():
    features = extract_shortcut_features(_frame())
    assert features.loc[10, "title_len"] == len("WATCH: Is this real?")
    assert features.loc[10, "text_len"] == len(
        "WASHINGTON (Reuters) - Hello https://x.com via Getty Images follow us!"
    )
    assert features.loc[10, "exclamation_count"] == 1
    assert features.loc[10, "question_count"] == 0
    assert features.loc[20, "question_count"] == 0
    assert features.loc[10, "all_caps_word_count"] >= 1


def test_missing_text_raw_raises():
    frame = pd.DataFrame({"title": ["a"], "label": [0]})
    try:
        extract_shortcut_features(frame)
    except KeyError as exc:
        assert "text_raw" in str(exc)
    else:
        raise AssertionError("expected KeyError for missing text_raw")
