"""Bootstrap confidence intervals for each prediction file.

This complements ``collect_metrics.py``: the collector writes point estimates,
while this module resamples the saved per-sample predictions to produce 95%
intervals for accuracy, precision, recall, and F1.

Usage:
    python -m fakenews.eval.ci --config configs/base.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from fakenews.config import load_config
from fakenews.utils import ensure_dir, get_logger

METRICS = ("accuracy", "precision", "recall", "f1")


def _binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict[str, float]:
    pred = y_prob >= threshold
    tp = int(np.sum(pred & (y_true == 1)))
    fp = int(np.sum(pred & (y_true == 0)))
    fn = int(np.sum(~pred & (y_true == 1)))
    tn = int(np.sum(~pred & (y_true == 0)))
    n = tp + fp + fn + tn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "accuracy": (tp + tn) / n if n else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _resample(y_true: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    parts = []
    for label in (0, 1):
        idx = np.flatnonzero(y_true == label)
        if idx.size:
            parts.append(rng.choice(idx, idx.size, replace=True))
    return np.concatenate(parts) if parts else rng.integers(0, len(y_true), len(y_true))


def _split_condition(condition: str) -> tuple[str, str]:
    if "_masked_" in condition:
        base, mask = condition.rsplit("_masked_", 1)
        return base, mask
    return condition, "none"


def collect_intervals(
    predictions_dir: Path,
    n_boot: int = 1000,
    seed: int = 42,
    threshold: float = 0.5,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for path in sorted(predictions_dir.glob("*.parquet")):
        frame = pd.read_parquet(path)
        for (model, condition, seed_value, split), group in frame.groupby(
            ["model", "condition", "seed", "split"], sort=False
        ):
            y = group["y_true"].to_numpy(dtype=int)
            p = group["y_prob"].to_numpy(dtype=float)
            point = _binary_metrics(y, p, threshold)
            samples = {metric: [] for metric in METRICS}
            for _ in range(n_boot):
                idx = _resample(y, rng)
                values = _binary_metrics(y[idx], p[idx], threshold)
                for metric in METRICS:
                    samples[metric].append(values[metric])
            base, mask = _split_condition(str(condition))
            for metric in METRICS:
                values = np.asarray(samples[metric])
                rows.append(
                    {
                        "run_id": path.stem,
                        "model": model,
                        "condition": condition,
                        "base_condition": base,
                        "mask": mask,
                        "seed": int(seed_value),
                        "split": split,
                        "threshold": threshold,
                        "metric": metric,
                        "point": point[metric],
                        "ci_low": float(np.percentile(values, 2.5)),
                        "ci_high": float(np.percentile(values, 97.5)),
                        "n_boot": n_boot,
                    }
                )
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Bootstrap CIs from prediction files")
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args(argv)

    logger = get_logger("fakenews.ci")
    cfg = load_config(args.config)
    results = Path(cfg.paths.results_dir)
    table = collect_intervals(results / "predictions", n_boot=args.n_boot, threshold=args.threshold)
    target = ensure_dir(results) / "cis.csv"
    table.to_csv(target, index=False, encoding="utf-8")
    logger.info("wrote %s (%d rows)", target, len(table))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
