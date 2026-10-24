"""Collect per-sample predictions into one comparable metrics table.

Every saved prediction file is read at threshold 0.5; per (model, condition,
split) we compute accuracy / precision / recall / F1 / ROC-AUC / PR-AUC and the
TP/FP/FN/TN confusion matrix. Masked Stage 8 conditions are split into
``base_condition`` and ``mask`` for easier pivoting.

Usage::

    python -m fakenews.eval.collect_metrics --config configs/base.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from fakenews.config import load_config
from fakenews.eval.metrics import compute_metrics
from fakenews.utils import ensure_dir, get_logger

COLUMNS = [
    "run_id",
    "model",
    "condition",
    "base_condition",
    "mask",
    "seed",
    "split",
    "threshold",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "pr_auc",
    "tp",
    "fp",
    "fn",
    "tn",
    "n",
]


def _split_condition(condition: str) -> tuple[str, str]:
    if "_masked_" in condition:
        base, mask = condition.rsplit("_masked_", 1)
        return base, mask
    return condition, "none"


def collect(predictions_dir: Path, threshold: float = 0.5) -> pd.DataFrame:
    rows = []
    for path in sorted(predictions_dir.glob("*.parquet")):
        frame = pd.read_parquet(path)
        run_id = path.stem
        for (model, condition, seed, split), group in frame.groupby(
            ["model", "condition", "seed", "split"], sort=False
        ):
            metrics = compute_metrics(
                group["y_true"].to_numpy(), group["y_prob"].to_numpy(), threshold
            )
            base_condition, mask = _split_condition(str(condition))
            rows.append(
                {
                    "run_id": run_id,
                    "model": model,
                    "condition": condition,
                    "base_condition": base_condition,
                    "mask": mask,
                    "seed": int(seed),
                    "split": split,
                    "threshold": threshold,
                    **{key: metrics[key] for key in metrics},
                }
            )
    if not rows:
        raise FileNotFoundError(f"no prediction files under {predictions_dir}")
    table = pd.DataFrame(rows).reindex(columns=COLUMNS)
    order = {"val": 0, "test": 1}
    table["_order"] = table["split"].map(order).fillna(2)
    table = table.sort_values(["model", "base_condition", "mask", "split", "_order"]).drop(
        columns="_order"
    )
    return table.reset_index(drop=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Collect metrics from predictions")
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args(argv)

    logger = get_logger("fakenews.collect_metrics")
    cfg = load_config(args.config)
    results = Path(cfg.paths.results_dir)
    table = collect(Path(results) / "predictions", threshold=args.threshold)
    target = ensure_dir(results) / "all_metrics.csv"
    table.to_csv(target, index=False, encoding="utf-8")
    logger.info("wrote %s (%d rows)", target, len(table))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
