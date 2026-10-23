"""YAML configuration loading with recursive overrides.

Usage::

    cfg = load_config("configs/base.yaml", "configs/bilstm.yaml")
    cfg.data.min_text_len          # attribute access, nested dicts become Config
    cfg["data"]["min_text_len"]    # plain mapping access still works
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

PathLike = str | Path


class Config(dict):
    """A ``dict`` that also supports attribute access at every nesting level."""

    def __getattr__(self, name: str) -> Any:
        try:
            value = self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc
        return Config(value) if isinstance(value, dict) else value

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def to_dict(self) -> dict[str, Any]:
        """Return a plain (deep) ``dict`` copy."""
        return _to_plain(self)


def _to_plain(obj: Any) -> Any:
    if isinstance(obj, Mapping):
        return {key: _to_plain(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_to_plain(value) for value in obj]
    return obj


def load_config(path: PathLike, *overrides: PathLike) -> Config:
    """Load ``path`` and merge every extra YAML file on top of it, in order."""
    with open(path, encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"config root must be a mapping: {path}")
    for extra in overrides:
        with open(extra, encoding="utf-8") as handle:
            data = merge(data, yaml.safe_load(handle) or {})
    return Config(data)


def merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into ``base`` (override wins).

    Lists are replaced, not concatenated. The inputs are not mutated.
    """
    result: dict[str, Any] = {key: _to_plain(value) for key, value in base.items()}
    for key, value in override.items():
        if isinstance(result.get(key), Mapping) and isinstance(value, Mapping):
            result[key] = merge(result[key], value)  # type: ignore[arg-type]
        else:
            result[key] = _to_plain(value)
    return result


def apply_override(cfg: Config, dotted: str) -> Config:
    """Apply one ``a.b.c=value`` override parsed from the command line.

    Values are parsed as YAML, so ``true``/``1.0``/``[1, 2]`` keep their types.
    """
    if "=" not in dotted:
        raise ValueError(f"override must look like key.path=value, got: {dotted!r}")
    key_path, raw_value = dotted.split("=", 1)
    value = yaml.safe_load(raw_value)
    data = cfg.to_dict()
    node: dict[str, Any] = data
    keys = key_path.strip().split(".")
    for key in keys[:-1]:
        node = node.setdefault(key, {})
        if not isinstance(node, dict):
            raise TypeError(f"cannot descend into non-mapping key: {key}")
    node[keys[-1]] = value
    return Config(data)


def save_config(cfg: Mapping[str, Any], path: PathLike) -> Path:
    """Dump a config to YAML (used to snapshot every run)."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as handle:
        yaml.safe_dump(_to_plain(cfg), handle, sort_keys=False, allow_unicode=True)
    return target
