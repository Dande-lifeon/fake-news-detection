"""Write a small Markdown table of false positives and false negatives.

Usage:
    python -m fakenews.eval.error_examples --config configs/base.yaml --model lr
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import pandas as pd

from fakenews.config import load_config
from fakenews.utils import get_logger


def _latest(results: Path, model: str, condition: str, seed: int) -> Path:
    pattern = str(results / "predictions" / f"{model}_{condition}_seed{seed}_*.parquet")
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"no prediction file for {model}/{condition}/seed{seed}")
    return Path(matches[-1])


def _snippet(text: str, limit: int = 260) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Extract FP/FN examples")
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    parser.add_argument("--model", choices=["lr", "svm"], default="lr")
    parser.add_argument("--condition", default="fusion")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--per-type", type=int, default=2)
    args = parser.parse_args(argv)

    logger = get_logger("fakenews.error_examples")
    cfg = load_config(args.config)
    results = Path(cfg.paths.results_dir)
    processed = Path(cfg.paths.processed_dir)

    pred = pd.read_parquet(_latest(results, args.model, args.condition, args.seed))
    pred = pred[pred["split"] == "test"].copy()
    test = pd.read_parquet(processed / "test.parquet")[
        ["id", "title", "text_clean", "subject", "date", "label"]
    ]
    frame = pred.merge(test, on="id", how="inner", suffixes=("", "_gold"))
    if len(frame) != len(pred):
        raise RuntimeError("prediction/test merge lost rows")

    frame["pred"] = (frame["y_prob"] >= 0.5).astype(int)
    fp = frame[(frame["y_true"] == 0) & (frame["pred"] == 1)].nlargest(args.per_type, "y_prob")
    fn = frame[(frame["y_true"] == 1) & (frame["pred"] == 0)].nsmallest(args.per_type, "y_prob")

    lines = [
        f"# FP/FN examples: {args.model.upper()} {args.condition} (test)",
        "",
        "Threshold 0.5. Snippets are the first 260 characters of `text_clean`.",
        "",
    ]
    for title, part in (("False positives (real predicted fake)", fp),
                        ("False negatives (fake predicted real)", fn)):
        lines += [f"## {title}", "", "| id | p(fake) | title | snippet |", "|---:|---:|---|---|"]
        for row in part.itertuples():
            title_text = str(row.title).replace("|", "\\|")
            body = _snippet(row.text_clean).replace("|", "\\|")
            lines.append(f"| {row.id} | {row.y_prob:.4f} | {title_text} | {body} |")
        lines.append("")

    target = results / "error_examples.md"
    target.write_text("\n".join(lines), encoding="utf-8")
    logger.info("wrote %s (%d FP, %d FN)", target, len(fp), len(fn))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
