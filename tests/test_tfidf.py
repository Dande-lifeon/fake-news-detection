"""Tests for TF-IDF building: fit on train only, consistent dimensions."""

from __future__ import annotations

import pandas as pd

from fakenews.config import Config
from fakenews.features.tfidf import build_tfidf, concat_branch


def _cfg() -> Config:
    return Config(
        {
            "features": {
                "title": {
                    "ngram_max": 2,
                    "min_df": 1,
                    "max_features": 100,
                    "stop_words": None,
                    "sublinear_tf": True,
                },
                "body": {
                    "ngram_max": 2,
                    "min_df": 1,
                    "max_features": 100,
                    "stop_words": None,
                    "sublinear_tf": True,
                },
            }
        }
    )


def _frame(texts: list[str], titles: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "title": titles,
            "text_clean": texts,
            "label": [index % 2 for index in range(len(texts))],
        }
    )


def test_vectorizer_is_fit_on_train_only():
    train = _frame(
        ["alpha beta gamma", "delta epsilon zeta"],
        ["train title one", "train title two"],
    )
    val = _frame(["alpha unseenword", "delta unseenword"], ["val title", "val title"])
    matrices = build_tfidf(train, val, None, _cfg())

    assert "unseenword" not in matrices.title_vec.vocabulary_
    assert "unseenword" not in matrices.body_vec.vocabulary_
    assert matrices.title_train.shape[0] == 2
    assert matrices.title_val.shape[0] == 2
    assert matrices.title_train.shape[1] == matrices.title_val.shape[1]
    assert matrices.body_train.shape[1] == matrices.body_val.shape[1]
    assert matrices.title_test is None


def test_concat_branch_stacks_title_and_body():
    train = _frame(
        ["alpha beta", "gamma delta"],
        ["one two", "three four"],
    )
    val = _frame(["alpha", "gamma"], ["one", "three"])
    matrices = build_tfidf(train, val, None, _cfg())
    train_x, val_x, test_x = concat_branch(matrices)

    assert train_x.shape[0] == 2
    assert val_x.shape[0] == 2
    assert test_x is None
    assert train_x.shape[1] == matrices.title_train.shape[1] + matrices.body_train.shape[1]


def test_custom_text_column_is_used():
    train = _frame(["clean body", "clean body two"], ["t1", "t2"])
    train["text_masked"] = ["MASKED one", "MASKED two"]
    val = _frame(["clean body"], ["t1"])
    val["text_masked"] = ["MASKED one"]
    matrices = build_tfidf(train, val, None, _cfg(), text_col="text_masked")
    assert "masked" in matrices.body_vec.vocabulary_
