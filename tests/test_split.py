"""Tests for ``fakenews.data.split``."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fakenews.data.split import assert_split_invariants, save_splits, split_dataset


def _fake_frame(n: int = 1000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    labels = np.array([0, 1] * (n // 2), dtype=np.int8)
    rng.shuffle(labels)
    return pd.DataFrame(
        {
            "id": np.arange(n, dtype=np.int64),
            "title": [f"title {i}" for i in range(n)],
            "text_raw": [f"raw body {i}" for i in range(n)],
            "text_clean": [f"clean body text number {i}" for i in range(n)],
            "label": labels,
            "subject": ["news"] * n,
            "date": ["2017-01-01"] * n,
            "dup_key": [f"key {i}" for i in range(n)],
        }
    )


def test_split_sizes_and_disjoint_ids():
    frame = _fake_frame(1000)
    splits = split_dataset(frame, val_size=0.10, test_size=0.20, seed=42)
    assert len(splits["train"]) == 700
    assert len(splits["val"]) == 100
    assert len(splits["test"]) == 200
    assert_split_invariants(**splits)


def test_tiny_split_tolerates_small_count_noise():
    # 40 rows cannot satisfy the 0.5 % rate tolerance exactly; the validator
    # should allow a two-row absolute slack instead of failing the whole split.
    frame = _fake_frame(40)
    splits = split_dataset(frame, val_size=0.10, test_size=0.20, seed=42)
    assert_split_invariants(**splits)


def test_split_is_stratified():
    frame = _fake_frame(1000)
    splits = split_dataset(frame, val_size=0.10, test_size=0.20, seed=42)
    overall = (frame["label"] == 1).mean()
    for name, part in splits.items():
        assert abs((part["label"] == 1).mean() - overall) < 0.01, name


def test_overlap_is_detected():
    frame = _fake_frame(100)
    splits = split_dataset(frame, val_size=0.10, test_size=0.20, seed=0)
    splits["val"] = pd.concat([splits["val"], splits["train"].head(1)], ignore_index=True)
    with pytest.raises(AssertionError, match="shares"):
        assert_split_invariants(**splits)


def test_save_splits_writes_manifest(tmp_path: Path):
    frame = _fake_frame(200)
    splits = split_dataset(frame, val_size=0.10, test_size=0.20, seed=7)
    manifest_path = tmp_path / "split_manifest.json"
    manifest = save_splits(splits, tmp_path, manifest_path, seed=7)

    assert manifest_path.exists()
    assert (tmp_path / "train.parquet").exists()
    assert (tmp_path / "val.parquet").exists()
    assert (tmp_path / "test.parquet").exists()
    assert manifest["total"] == 200
    assert sum(manifest["sizes"].values()) == 200
    assert manifest["seed"] == 7
    assert set(manifest["sha256"]) == {"train", "val", "test"}

    loaded = pd.read_parquet(tmp_path / "test.parquet")
    assert len(loaded) == 40
