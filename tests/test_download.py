"""Tests for the raw-data download helpers (no network)."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from types import SimpleNamespace

from fakenews.data import download


def test_cli_fallback_copies_before_tempdir_is_removed(monkeypatch, tmp_path: Path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True)

    def fake_run(cmd, **kwargs):
        target = Path(cmd[cmd.index("-p") + 1])
        (target / "True.csv").write_text(
            "title,text,subject,date\nreal,body,worldnews,2017-01-01\n",
            encoding="utf-8",
        )
        (target / "Fake.csv").write_text(
            "title,text,subject,date\nfake,body,News,2017-01-02\n",
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(download.shutil, "which", lambda name: "kaggle")
    monkeypatch.setattr(subprocess, "run", fake_run)

    result = download._download_with_cli(raw_dir, logging.getLogger("test"))

    assert result == raw_dir
    assert (raw_dir / "True.csv").exists()
    assert (raw_dir / "Fake.csv").exists()
