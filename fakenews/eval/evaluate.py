"""Run the full post-hoc evaluation from saved predictions.

This is the single entry point for the steps that used to be separate:
point metrics, bootstrap CIs, result figures, and FP/FN examples.

Usage:
    python -m fakenews.eval.evaluate --config configs/base.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

from fakenews.eval.ci import main as ci_main
from fakenews.eval.collect_metrics import main as collect_main
from fakenews.eval.error_examples import main as error_main
from fakenews.viz.figures import main as figures_main


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the full evaluation pipeline")
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    parser.add_argument("--model", choices=["lr", "svm"], default="lr",
                        help="model used for the FP/FN table")
    parser.add_argument("--skip-figures", action="store_true")
    parser.add_argument("--skip-errors", action="store_true")
    args = parser.parse_args(argv)

    config = ["--config", str(args.config)]
    collect_main(config)
    ci_main(config)
    if not args.skip_figures:
        figures_main(config)
    if not args.skip_errors:
        error_main([*config, "--model", args.model])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
