"""Smoke tests for the dual-branch classical model (LR path)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from fakenews.config import Config
from fakenews.models.classical import CONDITIONS, DualTextModel


def _cfg() -> Config:
    return Config(
        {
            "classical": {
                "C_grid": [1.0],
                "alpha_grid": [0.0, 0.5, 1.0],
                "cv_folds": 2,
                "threshold": 0.5,
            },
            "features": {
                "title": {
                    "ngram_max": 2,
                    "min_df": 1,
                    "max_features": 500,
                    "stop_words": None,
                    "sublinear_tf": True,
                },
                "body": {
                    "ngram_max": 2,
                    "min_df": 1,
                    "max_features": 500,
                    "stop_words": None,
                    "sublinear_tf": True,
                },
            },
        }
    )


def _frame(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    labels = np.array([0, 1] * (n // 2), dtype=np.int8)
    rng.shuffle(labels)
    rows = []
    for index, label in enumerate(labels):
        marker = "realword" if label == 0 else "fakeword"
        rows.append(
            {
                "title": f"{marker} title number {index}",
                "text_clean": f"{marker} body sentence about topic {index % 7}",
                "label": int(label),
                "id": index,
            }
        )
    return pd.DataFrame(rows)


def test_lr_fit_produces_four_conditions():
    train = _frame(160, seed=1)
    val = _frame(80, seed=2)
    model = DualTextModel("lr", _cfg()).fit(train, val)
    probs = model.predict_proba(val)

    assert set(probs) == set(CONDITIONS)
    for values in probs.values():
        assert values.shape == (len(val),)
        assert np.all((values >= 0) & (values <= 1))
    assert 0.0 <= model.alpha <= 1.0
    assert model.alpha in {0.0, 0.5, 1.0}
    assert set(model.thresholds) == set(CONDITIONS)


def test_refit_full_keeps_alpha_and_predicts():
    train = _frame(160, seed=3)
    val = _frame(80, seed=4)
    model = DualTextModel("lr", _cfg()).fit(train, val)
    alpha_before = model.alpha
    model.refit_full(pd.concat([train, val], ignore_index=True))
    probs = model.predict_proba(val)

    assert model.alpha == alpha_before
    assert probs["fusion"].shape == (len(val),)


def test_save_and_load_roundtrip(tmp_path: Path):
    train = _frame(160, seed=5)
    val = _frame(80, seed=6)
    model = DualTextModel("lr", _cfg()).fit(train, val)
    model.save(tmp_path)

    loaded = DualTextModel.load(tmp_path)
    assert loaded.alpha == model.alpha
    assert np.allclose(loaded.predict_proba(val)["fusion"], model.predict_proba(val)["fusion"])
