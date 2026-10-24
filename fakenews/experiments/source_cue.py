"""Stage 8 — source-cue masking experiment.

Retrain the classical model on bodies with the Reuters dateline and URLs removed
(and optionally CMS/media-credit artefacts), then compare test F1 against the
unmasked model on the **same rows and threshold**. Hyper-parameters (C, alpha)
are copied from the original run; nothing is tuned on the masked data.
"""

from __future__ import annotations

import argparse
import glob
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from fakenews.config import Config, apply_override, load_config
from fakenews.eval.bootstrap import paired_bootstrap_delta
from fakenews.eval.metrics import compute_metrics, mcnemar
from fakenews.features.masking import MASK_MODES, count_hits, mask_dataframe
from fakenews.models.classical import DualTextModel
from fakenews.train.common import prediction_frame, save_predictions
from fakenews.utils import get_logger, load_json, save_json, seed_everything

CONDITIONS = ("title_only", "body_only", "concat", "fusion")


def _latest_predictions(results_dir: Path, model: str, condition: str) -> pd.DataFrame:
    pattern = str(results_dir / "predictions" / f"{model}_{condition}_seed42_*.parquet")
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"no original predictions for {model}/{condition}: {pattern}")
    return pd.read_parquet(matches[-1])


def _load_original_model(models_dir: Path, model: str, job_id: str) -> DualTextModel:
    path = models_dir / model / job_id / "model.joblib"
    if not path.exists():
        matches = sorted(glob.glob(str(models_dir / model / "*" / "model.joblib")))
        if not matches:
            raise FileNotFoundError(f"no saved model under {models_dir / model}")
        path = Path(matches[-1])
    return joblib.load(path)


def _coef(clf) -> np.ndarray:
    if hasattr(clf, "coef_"):
        return np.asarray(clf.coef_).ravel()
    coefs = [np.asarray(c.estimator.coef_).ravel() for c in clf.calibrated_classifiers_]
    return np.mean(coefs, axis=0)


def _audit_rows(model: DualTextModel, model_name: str, mask: str, k: int = 50):
    rows = []
    for branch, clf, vectorizer in (
        ("body", model.body_clf, model.matrices.body_vec),
        ("title", model.title_clf, model.matrices.title_vec),
    ):
        coef = _coef(clf)
        names = np.asarray(vectorizer.get_feature_names_out())
        for index in np.argsort(np.abs(coef))[::-1][:k]:
            rows.append(
                {
                    "model": model_name,
                    "mask": mask,
                    "branch": branch,
                    "feature": names[index],
                    "coef": float(coef[index]),
                    "category": "",
                }
            )
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stage 8 source-cue masking")
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    parser.add_argument("--model", choices=["lr", "svm"], default="lr")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--masks", default=",".join(MASK_MODES))
    parser.add_argument("--bootstrap-n", type=int, default=1000)
    parser.add_argument("--override", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args(argv)

    logger = get_logger("fakenews.source_cue")
    cfg: Config = load_config(args.config)
    for dotted in args.override:
        cfg = apply_override(cfg, dotted)
    seed_everything(args.seed)

    processed = Path(cfg.paths.processed_dir)
    results = Path(cfg.paths.results_dir)
    masks = [mask.strip() for mask in args.masks.split(",") if mask.strip()]
    for mask in masks:
        if mask not in MASK_MODES:
            raise ValueError(f"unknown mask mode: {mask}")

    train = pd.read_parquet(processed / "train.parquet")
    val = pd.read_parquet(processed / "val.parquet")
    test = pd.read_parquet(processed / "test.parquet")

    fusion = load_json(results / "fusion.json")
    if args.model not in fusion:
        raise KeyError(f"no original run for model {args.model!r} in fusion.json")
    original = fusion[args.model]
    alpha = float(original["alpha"])
    best_c = {key: float(value) for key, value in original["best_c"].items()}
    logger.info("original %s: alpha=%.2f C=%s", args.model, alpha, best_c)

    hit_counts = count_hits(pd.concat([train, val, test], ignore_index=True))
    logger.info(
        "cue hits: %s",
        {k: {"real": v["real"], "fake": v["fake"]} for k, v in hit_counts.items()},
    )

    timestamp = f"{datetime.now():%Y%m%d_%H%M%S}"
    source_cue_path = results / "source_cue.json"
    payload = load_json(source_cue_path) if source_cue_path.exists() else {}
    payload["updated_at"] = datetime.now().isoformat(timespec="seconds")
    payload["hit_counts"] = hit_counts
    payload["hit_counts_by_split"] = {
        name: count_hits(frame) for name, frame in (("train", train), ("val", val), ("test", test))
    }
    payload.setdefault("models", {})
    payload["models"].setdefault(args.model, {})

    models_dir = Path(cfg.paths.models_dir)
    audit_rows: list[dict] = []
    original_model = _load_original_model(models_dir, args.model, original["job_id"])
    audit_rows.extend(_audit_rows(original_model, args.model, "original"))

    for mask in masks:
        logger.info("mask=%s: masking train/val/test", mask)
        train_m = mask_dataframe(train, mask)
        val_m = mask_dataframe(val, mask)
        test_m = mask_dataframe(test, mask)

        model = DualTextModel(args.model, cfg, text_col="text_masked")
        model.fit(train_m, val_m, fixed_c=best_c, fixed_alpha=alpha)
        model.thresholds = {condition: 0.5 for condition in CONDITIONS}
        val_probs = model.predict_proba(val_m)
        model.refit_full(pd.concat([train_m, val_m], ignore_index=True))
        test_probs = model.predict_proba(test_m)
        audit_rows.extend(_audit_rows(model, args.model, mask))

        mask_result = {
            "alpha": alpha,
            "best_c": best_c,
            "conditions": {},
        }

        prediction_frames = []
        for condition in CONDITIONS:
            if condition not in test_probs:
                continue
            original_pred = _latest_predictions(results, args.model, condition)
            original_pred = original_pred[original_pred["split"] == "test"].reset_index(drop=True)
            ids = test["id"].to_numpy()
            assert (original_pred["id"].to_numpy() == ids).all(), f"id mismatch {condition}"

            y = test["label"].to_numpy().astype(int)
            p_original = original_pred["y_prob"].to_numpy()
            p_masked = test_probs[condition]
            original_f1 = compute_metrics(y, p_original, 0.5)["f1"]
            masked_f1 = compute_metrics(y, p_masked, 0.5)["f1"]
            delta = paired_bootstrap_delta(
                y, p_masked, p_original, threshold=0.5, n=args.bootstrap_n, seed=args.seed
            )
            mc = mcnemar(y, (p_masked >= 0.5).astype(int), (p_original >= 0.5).astype(int))
            mask_result["conditions"][condition] = {
                "original_f1_at_0.5": original_f1,
                "masked_f1_at_0.5": masked_f1,
                "delta_f1": delta["delta"],
                "ci_low": delta["ci_low"],
                "ci_high": delta["ci_high"],
                "p_delta_le_0": delta["p_delta_le_0"],
                "mcnemar_b": mc["b"],
                "mcnemar_c": mc["c"],
                "mcnemar_p": mc["p_value"],
                "original_threshold": float(original["thresholds"][condition]),
                "masked_f1_at_original_threshold": compute_metrics(
                    y, p_masked, float(original["thresholds"][condition])
                )["f1"],
            }
            prediction_frames.append(
                prediction_frame(
                    test, p_masked, "test", args.model, f"{condition}_masked_{mask}", args.seed
                )
            )
            prediction_frames.append(
                prediction_frame(
                    val, val_probs[condition], "val", args.model,
                    f"{condition}_masked_{mask}", args.seed,
                )
            )

        run_id = f"{args.model}_source_cue_{mask}_seed{args.seed}_{timestamp}"
        save_predictions(
            pd.concat(prediction_frames, ignore_index=True),
            results / "predictions" / f"{run_id}.parquet",
        )
        mask_result["run_id"] = run_id
        payload["models"][args.model][mask] = mask_result

        print(f"\n[{args.model} / {mask}]")
        print(f"{'condition':11s} {'orig@0.5':>9s} {'masked@0.5':>11s} {'delta':>8s} "
              f"{'95% CI':>18s} {'p':>7s}")
        for condition, values in mask_result["conditions"].items():
            print(
                f"{condition:11s} {values['original_f1_at_0.5']:9.4f} "
                f"{values['masked_f1_at_0.5']:11.4f} {values['delta_f1']:+8.4f} "
                f"[{values['ci_low']:+.4f},{values['ci_high']:+.4f}] "
                f"{values['mcnemar_p']:7.4f}"
            )

    audit = pd.DataFrame(audit_rows)
    audit["abs_coef"] = audit["coef"].abs()
    audit = audit.sort_values(
        ["model", "mask", "branch", "abs_coef"], ascending=[True, True, True, False]
    ).drop(columns="abs_coef")
    audit_path = results / "feature_audit.csv"
    if audit_path.exists():
        existing = pd.read_csv(audit_path)
        existing = existing[existing["model"] != args.model]
        audit = pd.concat([existing, audit], ignore_index=True)
    audit.to_csv(audit_path, index=False, encoding="utf-8")
    payload["feature_audit"] = "results/feature_audit.csv"
    save_json(payload, source_cue_path)
    logger.info("wrote %s and %s", source_cue_path, results / "feature_audit.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
