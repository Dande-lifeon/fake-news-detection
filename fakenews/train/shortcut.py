"""Stage 4: shortcut and format-feature baseline (LR on 18 surface features).

Usage::

    python -m fakenews.train.shortcut --config configs/shortcut.yaml --final
    python -m fakenews.train.shortcut --config configs/shortcut.yaml --smoke

It measures how far a model can get without reading the article: datelines,
media credits, URLs, casing, and lengths only.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from fakenews.config import Config, apply_override, load_config, merge
from fakenews.eval.metrics import compute_metrics, tune_threshold
from fakenews.features.shortcut import FEATURES, extract_shortcut_features
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
from fakenews.utils import ensure_dir, get_logger, save_json, seed_everything, timer

SMOKE_OVERRIDES = {"classical": {"C_grid": [1.0], "cv_folds": 2}}


def _load_config(path: Path) -> Config:
    base = Path("configs/base.yaml")
    if path.name == "base.yaml" or not base.exists():
        return load_config(path)
    return load_config(base, path)


def _make_pipeline(c_value: float, seed: int) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(C=c_value, max_iter=2000, solver="liblinear",
                                      random_state=seed)),
        ]
    )


def _rule_metrics(features: pd.DataFrame, y: np.ndarray) -> dict[str, dict]:
    rules = {
        "no_reuters": (features["has_reuters_dateline"].to_numpy() == 0).astype(float),
        "has_url": features["has_url"].to_numpy().astype(float),
        "has_getty": features["has_getty"].to_numpy().astype(float),
        "no_reuters_or_url_or_getty": (
            (features["has_reuters_dateline"].to_numpy() == 0)
            | (features["has_url"].to_numpy() == 1)
            | (features["has_getty"].to_numpy() == 1)
        ).astype(float),
    }
    return {name: compute_metrics(y, prob, threshold=0.5) for name, prob in rules.items()}


def _metric_row(
    *,
    run_id: str,
    seed: int,
    split: str,
    threshold: float,
    metrics: dict,
    train_time: float,
    params: str,
    git_commit: str,
    config_hash: str,
    created_at: str,
) -> dict:
    return {
        "run_id": run_id,
        "model": "shortcut",
        "condition": "format",
        "seed": int(seed),
        "split": split,
        "threshold": float(threshold),
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "accuracy": metrics["accuracy"],
        "roc_auc": metrics["roc_auc"],
        "pr_auc": metrics["pr_auc"],
        "tp": metrics["tp"],
        "fp": metrics["fp"],
        "fn": metrics["fn"],
        "tn": metrics["tn"],
        "n": metrics["n"],
        "alpha": np.nan,
        "gate_mean": np.nan,
        "train_time_sec": float(train_time),
        "params": params,
        "git_commit": git_commit,
        "config_hash": config_hash,
        "created_at": created_at,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Shortcut / format-feature baseline")
    parser.add_argument("--config", type=Path, default=Path("configs/shortcut.yaml"))
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--final", action="store_true", help="also evaluate test once")
    parser.add_argument("--smoke", action="store_true", help="small subset, 1 C, 2 folds")
    parser.add_argument("--override", action="append", default=[], metavar="KEY=VALUE")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logger = get_logger("fakenews.train.shortcut")
    cfg = _load_config(args.config)
    for dotted in args.override:
        cfg = apply_override(cfg, dotted)

    seed = args.seed if args.seed is not None else int(cfg.seed)
    seed_everything(seed)

    processed_dir = Path(cfg.paths.processed_dir)
    results_dir = Path(cfg.paths.results_dir)
    train = pd.read_parquet(processed_dir / "train.parquet")
    val = pd.read_parquet(processed_dir / "val.parquet")

    if args.smoke:
        cfg = Config(merge(cfg.to_dict(), SMOKE_OVERRIDES))
        train = pd.concat(
            [
                train[train.label == 0].head(250),
                train[train.label == 1].head(250),
            ]
        )
        val = pd.concat([val[val.label == 0].head(100), val[val.label == 1].head(100)])
        results_dir = results_dir / "smoke"
        logger.info("smoke mode: train=%d val=%d", len(train), len(val))

    test = None
    if args.final and not args.smoke:
        test = pd.read_parquet(processed_dir / "test.parquet")
        logger.info("test split loaded once: %d rows", len(test))

    x_train = extract_shortcut_features(train)
    y_train = train["label"].to_numpy().astype(int)
    x_val = extract_shortcut_features(val)
    y_val = val["label"].to_numpy().astype(int)
    logger.info("features: %d columns, train=%d val=%d", x_train.shape[1], len(x_train), len(x_val))

    folds = max(2, min(int(cfg.classical.cv_folds), int(np.bincount(y_train).min())))
    search = GridSearchCV(
        _make_pipeline(1.0, seed),
        {"lr__C": [float(c) for c in cfg.classical.C_grid]},
        scoring="f1",
        cv=folds,
        n_jobs=1,
        refit=True,
    )
    with timer() as elapsed:
        search.fit(x_train, y_train)
    train_time = float(elapsed[0])
    best_c = float(search.best_params_["lr__C"])
    logger.info("CV done in %.1fs; best C=%.4g", train_time, best_c)

    val_prob = search.best_estimator_.predict_proba(x_val)[:, 1]
    tuned_threshold, tuned_f1 = tune_threshold(y_val, val_prob)
    logger.info("val threshold tuned: %.2f (F1 %.4f)", tuned_threshold, tuned_f1)

    final_model = _make_pipeline(best_c, seed)
    final_model.fit(
        pd.concat([x_train, x_val], ignore_index=True),
        np.concatenate([y_train, y_val]),
    )

    timestamp = f"{datetime.now():%Y%m%d_%H%M%S}"
    job_id = f"shortcut_format_seed{seed}_{timestamp}"
    run_id = make_run_id("shortcut", "format", seed, timestamp)
    job_dir = ensure_dir(results_dir / "logs" / job_id)
    config_hash = snapshot_config(cfg, job_dir)
    write_environment(job_dir)
    git_commit = compute_git_commit()
    (job_dir / "git_commit.txt").write_text(git_commit + "\n", encoding="utf-8")
    created_at = datetime.now().isoformat(timespec="seconds")

    params = params_json(
        {"C": best_c, "cv_folds": folds, "features": FEATURES, "tuned_threshold": tuned_threshold}
    )

    summary: dict[str, dict] = {
        "val_0.5": compute_metrics(y_val, val_prob, 0.5),
        "val_tuned": compute_metrics(y_val, val_prob, tuned_threshold),
    }
    rows = [
        _metric_row(
            run_id=run_id, seed=seed, split="val", threshold=0.5, metrics=summary["val_0.5"],
            train_time=train_time, params=params, git_commit=git_commit,
            config_hash=config_hash, created_at=created_at,
        )
    ]
    if abs(tuned_threshold - 0.5) > 1e-9:
        rows.append(
            _metric_row(
                run_id=run_id, seed=seed, split="val", threshold=tuned_threshold,
                metrics=summary["val_tuned"], train_time=train_time, params=params,
                git_commit=git_commit, config_hash=config_hash, created_at=created_at,
            )
        )

    prediction_frames = [
        prediction_frame(val, val_prob, "val", "shortcut", "format", seed)
    ]

    test_at_half = None
    if test is not None:
        x_test = extract_shortcut_features(test)
        y_test = test["label"].to_numpy().astype(int)
        test_prob = final_model.predict_proba(x_test)[:, 1]
        summary["test_tuned"] = compute_metrics(y_test, test_prob, tuned_threshold)
        summary["test_0.5"] = compute_metrics(y_test, test_prob, 0.5)
        test_at_half = summary["test_0.5"]
        rows.append(
            _metric_row(
                run_id=run_id, seed=seed, split="test", threshold=tuned_threshold,
                metrics=summary["test_tuned"], train_time=train_time, params=params,
                git_commit=git_commit, config_hash=config_hash, created_at=created_at,
            )
        )
        prediction_frames.append(
            prediction_frame(test, test_prob, "test", "shortcut", "format", seed)
        )
        rule_metrics = _rule_metrics(x_test, y_test)
    else:
        rule_metrics = {}

    save_predictions(
        pd.concat(prediction_frames, ignore_index=True),
        results_dir / "predictions" / f"{run_id}.parquet",
    )
    for row in rows:
        append_metrics(row, results_dir / "metrics.csv")

    model_dir = ensure_dir(Path(cfg.paths.models_dir) / "shortcut" / job_id)
    joblib.dump(final_model, model_dir / "model.joblib")

    coef = search.best_estimator_.named_steps["lr"].coef_.ravel()
    top_features = [
        {"feature": FEATURES[index], "coef": float(coef[index])}
        for index in np.argsort(np.abs(coef))[::-1]
    ]

    payload = {
        "model": "shortcut",
        "condition": "format",
        "seed": seed,
        "job_id": job_id,
        "run_id": run_id,
        "features": FEATURES,
        "best_C": best_c,
        "thresholds": {"default": 0.5, "tuned": tuned_threshold},
        "val": {"f1_at_0.5": summary["val_0.5"], "f1_at_tuned": summary["val_tuned"]},
        "test": {
            "f1_at_tuned": summary.get("test_tuned"),
            "f1_at_0.5": test_at_half,
        },
        "top_features": top_features,
        "rule_baselines": rule_metrics,
        "train_time_sec": train_time,
        "git_commit": git_commit,
        "config_hash": config_hash,
        "created_at": created_at,
    }
    save_json(payload, results_dir / "shortcut.json")

    print(f"{'split':6s} {'thr':>5s} {'precision':>9s} {'recall':>7s} {'f1':>7s} {'roc_auc':>8s}")
    print("-" * 50)
    for name, metrics in (
        ("val", summary["val_0.5"]),
        ("val*", summary["val_tuned"]),
        ("test", summary.get("test_tuned")),
    ):
        if metrics is None:
            continue
        print(
            f"{name:6s} {0.5 if name == 'val' else tuned_threshold:5.2f} "
            f"{metrics['precision']:9.4f} {metrics['recall']:7.4f} "
            f"{metrics['f1']:7.4f} {metrics['roc_auc']:8.4f}"
        )
    print(f"\nshortcut -> {results_dir / 'shortcut.json'}")
    if test_at_half is not None:
        print(f"test@0.5 F1={test_at_half['f1']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
