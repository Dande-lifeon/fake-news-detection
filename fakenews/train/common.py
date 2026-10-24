"""Shared plumbing for training runs: run ids, predictions, metrics, snapshots."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fakenews.config import Config, save_config
from fakenews.utils import ensure_dir, sha256_file

METRIC_COLUMNS = [
    "run_id",
    "model",
    "condition",
    "seed",
    "split",
    "threshold",
    "precision",
    "recall",
    "f1",
    "accuracy",
    "roc_auc",
    "pr_auc",
    "tp",
    "fp",
    "fn",
    "tn",
    "n",
    "alpha",
    "gate_mean",
    "train_time_sec",
    "params",
    "git_commit",
    "config_hash",
    "created_at",
]

PREDICTION_COLUMNS = [
    "id",
    "y_true",
    "y_prob",
    "split",
    "model",
    "condition",
    "seed",
]


def make_run_id(model: str, condition: str, seed: int, timestamp: str | None = None) -> str:
    stamp = timestamp or f"{datetime.now():%Y%m%d_%H%M%S}"
    return f"{model}_{condition}_seed{seed}_{stamp}"


def prediction_frame(
    df: pd.DataFrame,
    y_prob: np.ndarray,
    split: str,
    model: str,
    condition: str,
    seed: int,
) -> pd.DataFrame:
    if len(df) != len(y_prob):
        raise ValueError(f"row/probability mismatch: {len(df)} vs {len(y_prob)}")
    return pd.DataFrame(
        {
            "id": df["id"].to_numpy(dtype=np.int64),
            "y_true": df["label"].to_numpy(dtype=np.int8),
            "y_prob": np.asarray(y_prob, dtype=np.float32),
            "split": split,
            "model": model,
            "condition": condition,
            "seed": int(seed),
        }
    )


def save_predictions(frame: pd.DataFrame, path: Path) -> Path:
    """Write per-sample predictions as Parquet; column order is fixed."""
    target = Path(path)
    ensure_dir(target.parent)
    frame[PREDICTION_COLUMNS].to_parquet(target, index=False)
    return target


def append_metrics(row: dict[str, Any], path: Path) -> pd.DataFrame:
    """Append one metrics row, creating the CSV with a stable header if needed."""
    target = Path(path)
    ensure_dir(target.parent)
    if target.exists():
        existing = pd.read_csv(target)
        combined = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
    else:
        combined = pd.DataFrame([row])
    combined = combined.reindex(columns=METRIC_COLUMNS)
    combined.to_csv(target, index=False, encoding="utf-8")
    return combined


def compute_git_commit(cwd: Path | None = None) -> str:
    """Return the current git commit hash, or ``unknown`` outside a repository."""
    root = cwd or Path(__file__).resolve().parents[2]
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def snapshot_config(cfg: Config, out_dir: Path) -> str:
    """Write ``config.yaml`` into the run directory and return its SHA-256."""
    target = Path(out_dir) / "config.yaml"
    save_config(cfg, target)
    return sha256_file(target)


def write_environment(out_dir: Path) -> Path:
    """Record the interpreter and key package versions for reproducibility."""
    target = ensure_dir(out_dir) / "environment.txt"
    import matplotlib
    import scipy
    import sklearn

    lines = [
        f"python: {sys.version.split()[0]}",
        f"executable: {sys.executable}",
        f"platform: {platform.platform()}",
        f"pandas: {pd.__version__}",
        f"numpy: {np.__version__}",
        f"scipy: {scipy.__version__}",
        f"scikit-learn: {sklearn.__version__}",
        f"matplotlib: {matplotlib.__version__}",
        f"created_at: {datetime.now().isoformat(timespec='seconds')}",
    ]
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


def params_json(params: dict[str, Any]) -> str:
    return json.dumps(params, sort_keys=True, ensure_ascii=False, default=str)
