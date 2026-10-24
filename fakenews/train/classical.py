"""Train LR / LinearSVM on TF-IDF and evaluate the four conditions.

Usage::

    python -m fakenews.train.classical --config configs/base.yaml --model lr --final
    python -m fakenews.train.classical --config configs/base.yaml --model svm --final
    python -m fakenews.train.classical --config configs/base.yaml --model lr --smoke

The test split is read only with ``--final`` and only after the model has been
re-fitted on ``train + val``; ``alpha`` and thresholds always come from ``val``.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from fakenews.config import Config, apply_override, load_config, merge
from fakenews.eval.metrics import compute_metrics
from fakenews.models.classical import CONDITIONS, DualTextModel
from fakenews.train.common import (
    append_metrics,
    compute_git_commit,
    make_run_id,
    params_json,
    prediction_frame,
    save_predictions,
    snapshot_config,
    write_environment,
)
from fakenews.utils import ensure_dir, get_logger, load_json, save_json, seed_everything, timer

SMOKE_OVERRIDES = {
    "classical": {
        "C_grid": [1.0],
        "alpha_grid": [0.5],
        "cv_folds": 2,
        "threshold": 0.5,
    }
}


def _sample(frame: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if len(frame) <= n:
        return frame.reset_index(drop=True)
    part, _ = train_test_split(
        frame, train_size=n, stratify=frame["label"], random_state=seed
    )
    return part.reset_index(drop=True)


def _metric_row(
    *,
    run_id: str,
    model: str,
    condition: str,
    seed: int,
    split: str,
    threshold: float,
    metrics: dict,
    alpha: float,
    train_time_sec: float,
    params: str,
    git_commit: str,
    config_hash: str,
    created_at: str,
) -> dict:
    row = {
        "run_id": run_id,
        "model": model,
        "condition": condition,
        "seed": int(seed),
        "split": split,
        "threshold": float(threshold),
        "alpha": alpha,
        "gate_mean": np.nan,
        "train_time_sec": float(train_time_sec),
        "params": params,
        "git_commit": git_commit,
        "config_hash": config_hash,
        "created_at": created_at,
    }
    row.update({key: value for key, value in metrics.items() if key != "n"})
    row["n"] = metrics["n"]
    return row


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a classical fake-news model")
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    parser.add_argument("--model", choices=["lr", "svm"], required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--final", action="store_true", help="also evaluate the test split")
    parser.add_argument("--smoke", action="store_true", help="tiny subset, 1 C, 1 alpha")
    parser.add_argument("--override", action="append", default=[], metavar="KEY=VALUE")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logger = get_logger("fakenews.train.classical")
    cfg: Config = load_config(args.config)
    for dotted in args.override:
        cfg = apply_override(cfg, dotted)

    seed = args.seed if args.seed is not None else int(cfg.seed)
    seed_everything(seed)

    results_dir = Path(cfg.paths.results_dir)
    models_root = Path(cfg.paths.models_dir)
    processed_dir = Path(cfg.paths.processed_dir)
    train = pd.read_parquet(processed_dir / "train.parquet")
    val = pd.read_parquet(processed_dir / "val.parquet")

    if args.smoke:
        cfg = Config(merge(cfg.to_dict(), SMOKE_OVERRIDES))
        train = _sample(train, 400, seed)
        val = _sample(val, 200, seed)
        results_dir = results_dir / "smoke"
        models_root = models_root / "smoke"
        logger.info("smoke mode: train=%d val=%d", len(train), len(val))

    test = None
    if args.final and not args.smoke:
        test = pd.read_parquet(processed_dir / "test.parquet")
        logger.info("test split loaded once: %d rows", len(test))

    timestamp = f"{datetime.now():%Y%m%d_%H%M%S}"
    job_id = f"{args.model}_seed{seed}_{timestamp}"
    job_dir = ensure_dir(results_dir / "logs" / job_id)
    config_hash = snapshot_config(cfg, job_dir)
    write_environment(job_dir)
    git_commit = compute_git_commit()
    (job_dir / "git_commit.txt").write_text(git_commit + "\n", encoding="utf-8")

    logger.info("model=%s seed=%d job=%s", args.model, seed, job_id)
    model = DualTextModel(args.model, cfg)
    with timer() as elapsed:
        model.fit(train, val)
    train_time = float(elapsed[0])
    logger.info("fit done in %.1fs; alpha=%.2f", train_time, model.alpha)
    logger.info("best C: %s", model.best_c)

    params = params_json(
        {
            "C_title": model.best_c.get("title"),
            "C_body": model.best_c.get("body"),
            "C_concat": model.best_c.get("concat"),
            "alpha": model.alpha,
            "text_col": model.text_col,
            "cv_folds": model.cv_folds,
        }
    )
    created_at = datetime.now().isoformat(timespec="seconds")

    val_probs = model.predict_proba(val)
    conditions = [condition for condition in CONDITIONS if condition in val_probs]
    run_ids = {
        condition: make_run_id(args.model, condition, seed, timestamp)
        for condition in conditions
    }
    prediction_frames: dict[str, list[pd.DataFrame]] = {
        condition: [] for condition in conditions
    }
    summary: dict[str, dict] = {}
    rows: list[dict] = []

    for condition in conditions:
        probs = val_probs[condition]
        alpha_value = float(model.alpha) if condition == "fusion" else float("nan")
        prediction_frames[condition].append(
            prediction_frame(val, probs, "val", args.model, condition, seed)
        )
        scored = [(0.5, "0.5")]
        tuned = float(model.thresholds[condition])
        if abs(tuned - 0.5) > 1e-9:
            scored.append((tuned, "tuned"))

        summary[condition] = {}
        for threshold, tag in scored:
            metrics = compute_metrics(val["label"].to_numpy(), probs, threshold)
            summary[condition][f"val_{tag}"] = metrics
            rows.append(
                _metric_row(
                    run_id=run_ids[condition],
                    model=args.model,
                    condition=condition,
                    seed=seed,
                    split="val",
                    threshold=threshold,
                    metrics=metrics,
                    alpha=alpha_value,
                    train_time_sec=train_time,
                    params=params,
                    git_commit=git_commit,
                    config_hash=config_hash,
                    created_at=created_at,
                )
            )

    if test is not None:
        logger.info("re-fitting on train+val (%d rows)", len(train) + len(val))
        model.refit_full(pd.concat([train, val], ignore_index=True))
        test_probs = model.predict_proba(test)
        for condition in conditions:
            probs = test_probs[condition]
            threshold = float(model.thresholds[condition])
            metrics = compute_metrics(test["label"].to_numpy(), probs, threshold)
            summary[condition]["test_tuned"] = metrics
            prediction_frames[condition].append(
                prediction_frame(test, probs, "test", args.model, condition, seed)
            )
            rows.append(
                _metric_row(
                    run_id=run_ids[condition],
                    model=args.model,
                    condition=condition,
                    seed=seed,
                    split="test",
                    threshold=threshold,
                    metrics=metrics,
                    alpha=float(model.alpha) if condition == "fusion" else float("nan"),
                    train_time_sec=train_time,
                    params=params,
                    git_commit=git_commit,
                    config_hash=config_hash,
                    created_at=created_at,
                )
            )

    for condition, frames in prediction_frames.items():
        target = results_dir / "predictions" / f"{run_ids[condition]}.parquet"
        save_predictions(pd.concat(frames, ignore_index=True), target)

    for row in rows:
        append_metrics(row, results_dir / "metrics.csv")

    model_dir = models_root / args.model / job_id
    model.save(model_dir)
    logger.info("model saved to %s", model_dir)

    fusion_path = results_dir / "fusion.json"
    fusion = load_json(fusion_path) if fusion_path.exists() else {}
    fusion[args.model] = {
        "job_id": job_id,
        "seed": seed,
        "alpha": model.alpha,
        "alpha_curve": model.alpha_curve,
        "best_c": model.best_c,
        "thresholds": model.thresholds,
        "val_f1": {
            condition: summary[condition]["val_0.5"]["f1"] for condition in conditions
        },
        "test_f1": {
            condition: summary[condition]["test_tuned"]["f1"]
            for condition in conditions
            if "test_tuned" in summary[condition]
        },
    }
    save_json(fusion, fusion_path)

    save_json(
        {
            "job_id": job_id,
            "model": args.model,
            "seed": seed,
            "params": json.loads(params),
            "summary": {
                condition: {
                    key: {metric: value for metric, value in metrics.items()}
                    for key, metrics in values.items()
                }
                for condition, values in summary.items()
            },
            "git_commit": git_commit,
            "config_hash": config_hash,
            "train_time_sec": train_time,
        },
        job_dir / "metrics.json",
    )

    header = f"{'condition':12s} {'val F1@0.5':>10s} {'val F1@tuned':>12s} {'test F1':>8s}"
    print(header)
    print("-" * len(header))
    for condition in conditions:
        val_f1 = summary[condition]["val_0.5"]["f1"]
        tuned_f1 = summary[condition].get("val_tuned", summary[condition]["val_0.5"])["f1"]
        test_f1 = summary[condition].get("test_tuned", {}).get("f1", float("nan"))
        print(f"{condition:12s} {val_f1:10.4f} {tuned_f1:12.4f} {test_f1:8.4f}")
    print(f"\nalpha={model.alpha:.2f}; metrics -> {results_dir / 'metrics.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
