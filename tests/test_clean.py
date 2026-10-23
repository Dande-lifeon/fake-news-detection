"""Tests for ``fakenews.data.clean``."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fakenews.data.clean import (
    CLEAN_COLUMNS,
    clean_dataset,
    clean_text,
    load_single_csv,
    make_dup_key,
    save_clean,
)


def test_clean_text_removes_html_and_collapses_whitespace():
    assert clean_text("<p>Hello <b>world</b></p>\n\n  test") == "Hello world test"


def test_clean_text_keeps_urls_and_datelines():
    raw = "WASHINGTON (Reuters) - Read more at https://example.com/a?b=1 now."
    cleaned = clean_text(raw)
    assert "(Reuters)" in cleaned
    assert "https://example.com/a?b=1" in cleaned


@pytest.mark.parametrize("value", [None, np.nan, 42, ""])
def test_clean_text_non_string_returns_empty(value):
    assert clean_text(value) == ""


def test_make_dup_key_is_case_and_whitespace_insensitive():
    assert make_dup_key("Hello  World", "Body\n\nText") == make_dup_key(
        " hello world ", "body text"
    )


def test_clean_dataset_bookkeeping(tiny_csv: Path):
    raw = load_single_csv(tiny_csv)
    cleaned, stats = clean_dataset(raw, min_len=50, drop_conflicts=True)

    assert stats["loaded_total"] == 40
    assert stats["loaded_true"] == 20
    assert stats["loaded_fake"] == 20
    assert stats["short_text_dropped"] == 1
    assert stats["conflict_keys"] == 1
    assert stats["conflict_rows_dropped"] == 2
    assert stats["duplicate_rows_dropped"] == 2
    assert stats["remaining_total"] == 35
    assert stats["remaining_true"] == 18
    assert stats["remaining_fake"] == 17
    assert len(cleaned) == stats["remaining_total"]

    dropped = (
        stats["short_text_dropped"]
        + stats["conflict_rows_dropped"]
        + stats["duplicate_rows_dropped"]
    )
    assert stats["loaded_total"] - dropped == stats["remaining_total"]


def test_conflict_rows_are_removed_not_kept(tiny_csv: Path):
    raw = load_single_csv(tiny_csv)
    cleaned, _ = clean_dataset(raw, min_len=50, drop_conflicts=True)

    # The conflicting story must not survive in either class.
    assert "Shared dateline story about the budget" not in set(cleaned["title"])
    assert cleaned["dup_key"].is_unique
    assert not (cleaned["text_clean"].str.len() <= 50).any()


def test_duplicates_removed(tiny_csv: Path):
    raw = load_single_csv(tiny_csv)
    cleaned, _ = clean_dataset(raw, min_len=50, drop_conflicts=True)
    assert not cleaned["dup_key"].duplicated().any()
    # 35 rows: 40 - 1 short - 2 conflict - 2 duplicate
    assert len(cleaned) == 35


def test_ids_are_stable_and_unique(tiny_csv: Path):
    raw = load_single_csv(tiny_csv)
    cleaned, _ = clean_dataset(raw, min_len=50, drop_conflicts=True)
    assert cleaned["id"].is_unique
    assert set(cleaned["id"]).issubset(set(raw["id"]))
    assert list(cleaned.columns) == CLEAN_COLUMNS


def test_save_clean_roundtrip(tiny_csv: Path, tmp_path: Path):
    raw = load_single_csv(tiny_csv)
    cleaned, _ = clean_dataset(raw, min_len=50, drop_conflicts=True)
    parquet = tmp_path / "clean.parquet"
    csv = tmp_path / "clean.csv"
    save_clean(cleaned, parquet, csv_path=csv)

    assert parquet.exists() and csv.exists()
    loaded = pd.read_parquet(parquet)
    assert list(loaded.columns) == CLEAN_COLUMNS
    assert loaded["id"].dtype == np.int64
    assert loaded["label"].dtype == np.int8
    assert len(loaded) == 35

    csv_loaded = pd.read_csv(csv)
    assert len(csv_loaded) == 35
    assert set(csv_loaded.columns) == set(CLEAN_COLUMNS)
