"""Shared pytest fixtures."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def tiny_csv(repo_root: Path) -> Path:
    path = repo_root / "tests" / "fixtures" / "tiny.csv"
    assert path.exists(), f"missing fixture: {path}"
    return path
