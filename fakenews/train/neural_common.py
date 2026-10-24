"""Shared evaluation and logging helpers for the neural models."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from fakenews.eval.metrics import compute_metrics
from fakenews.train.common import METRIC_COLUMNS, append_metrics, prediction_frame, save_predictions


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    probs, labels, ids, gates = [], [], [], []
    total_loss = 0.0
    criterion = nn.BCEWithLogitsLoss(reduction="sum")
    for batch in loader:
        ids_batch = batch["id"].cpu().numpy()
        features = {
            key: value.to(device)
            for key, value in batch.items()
            if key not in ("id", "label")
        }
        outputs = model(**features)
        loss = criterion(outputs["logit"], batch["label"].to(device))
        total_loss += float(loss)
        probs.append(torch.sigmoid(outputs["logit"]).cpu().numpy())
        labels.append(batch["label"].cpu().numpy())
        ids.append(ids_batch)
        gates.append(outputs["gate"].cpu().numpy())
    return {
        "probs": np.concatenate(probs),
        "labels": np.concatenate(labels).astype(int),
        "ids": np.concatenate(ids).astype(int) if ids else np.array([], dtype=int),
        "gate_mean": float(np.mean(np.concatenate(gates))),
        "loss": total_loss / max(len(loader.dataset), 1),
    }


def save_run(
    *,
    model_name: str,
    condition: str,
    seed: int,
    split: str,
    result: dict,
    run_id: str,
    results_dir: Path,
    train_time: float,
    params: int,
    git_commit: str,
    config_hash: str,
    alpha: float | None = None,
) -> dict:
    probs = result["probs"]
    labels = result["labels"]
    ids = result["ids"]
    if len(ids) == 0:
        raise ValueError("prediction results are missing ids")
    frame = pd.DataFrame({"id": ids, "label": labels})
    pred = prediction_frame(frame, probs, split, model_name, condition, seed)
    save_predictions(pred, results_dir / "predictions" / f"{run_id}_{split}.parquet")

    metrics = compute_metrics(labels, probs, threshold=0.5)
    row = {
        "run_id": run_id,
        "model": model_name,
        "condition": condition,
        "seed": int(seed),
        "split": split,
        "threshold": 0.5,
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "roc_auc": metrics["roc_auc"],
        "pr_auc": metrics["pr_auc"],
        "tp": metrics["tp"],
        "fp": metrics["fp"],
        "fn": metrics["fn"],
        "tn": metrics["tn"],
        "n": metrics["n"],
        "alpha": alpha if alpha is not None else np.nan,
        "gate_mean": result["gate_mean"],
        "train_time_sec": float(train_time),
        "params": int(params),
        "git_commit": git_commit,
        "config_hash": config_hash,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    row = {key: row.get(key) for key in METRIC_COLUMNS}
    append_metrics(row, results_dir / "metrics.csv")
    return row
