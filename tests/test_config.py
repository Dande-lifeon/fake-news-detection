"""Tests for config loading/merging."""

from __future__ import annotations

from pathlib import Path

import pytest

from fakenews.config import Config, apply_override, load_config, merge


def test_merge_is_recursive_and_override_wins():
    base = {"a": 1, "nested": {"x": 1, "y": 2}, "items": [1, 2]}
    override = {"nested": {"y": 20, "z": 30}, "items": [3]}
    merged = merge(base, override)
    assert merged == {"a": 1, "nested": {"x": 1, "y": 20, "z": 30}, "items": [3]}
    # inputs untouched
    assert base["nested"]["y"] == 2


def test_attribute_access_is_recursive():
    cfg = Config({"data": {"min_text_len": 50}})
    assert cfg.data.min_text_len == 50
    with pytest.raises(AttributeError):
        _ = cfg.missing


def test_load_config_merges_overrides(tmp_path: Path):
    base = tmp_path / "base.yaml"
    extra = tmp_path / "extra.yaml"
    base.write_text("seed: 42\ndata:\n  min_text_len: 50\n", encoding="utf-8")
    extra.write_text("data:\n  min_text_len: 80\n", encoding="utf-8")
    cfg = load_config(base, extra)
    assert cfg.seed == 42
    assert cfg.data.min_text_len == 80


def test_apply_override_keeps_yaml_types():
    cfg = Config({"a": {"b": 1}, "flag": False})
    cfg = apply_override(cfg, "a.b=3")
    assert cfg.a.b == 3
    cfg = apply_override(cfg, "flag=true")
    assert cfg.flag is True
    with pytest.raises(ValueError):
        apply_override(cfg, "not-a-dotted-pair")
