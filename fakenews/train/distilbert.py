"""Fine-tune the shared-weight dual-encoder DistilBERT."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from fakenews.config import Config, load_config
from fakenews.data.neural import DistilBertDataset, collate_bert
from fakenews.eval.metrics import compute_metrics
from fakenews.models.distilbert import DualDistilBert
from fakenews.train.common import (
    compute_git_commit,
    make_run_id,
    snapshot_config,
    write_environment,
)
from fakenews.train.neural_common import evaluate, save_run
from fakenews.utils import ensure_dir, get_logger, seed_everything

CONDITIONS = ("title_only", "body_only", "fusion")


def _load(path: Path) -> Config:
    base = Path("configs/base.yaml")
    return load_config(path) if path.name == "base.yaml" else load_config(base, path)


def _loader(frame: pd.DataFrame, tokenizer, cfg: Config, shuffle: bool) -> DataLoader:
    dataset = DistilBertDataset(
        frame, tokenizer,
        max_title=int(cfg.distilbert.max_title_len),
        max_body=int(cfg.distilbert.max_body_len),
        head=int(cfg.distilbert.head_tokens),
        tail=int(cfg.distilbert.tail_tokens),
    )
    return DataLoader(
        dataset,
        batch_size=int(cfg.distilbert.batch_size),
        shuffle=shuffle,
        collate_fn=collate_bert,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fine-tune a dual-encoder DistilBERT")
    parser.add_argument("--config", type=Path, default=Path("configs/distilbert.yaml"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--condition", choices=CONDITIONS, default="fusion")
    parser.add_argument("--final", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)

    logger = get_logger("fakenews.distilbert")
    cfg = _load(args.config)
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(cfg.distilbert.fp16) and device.type == "cuda"
    logger.info("device=%s amp=%s", device, use_amp)

    processed = Path(cfg.paths.processed_dir)
    results = Path(cfg.paths.results_dir)
    models = Path(cfg.paths.models_dir)
    train = pd.read_parquet(processed / "train.parquet")
    val = pd.read_parquet(processed / "val.parquet")
    if args.smoke:
        train = pd.concat([train[train.label == 0].head(400), train[train.label == 1].head(400)])
        val = pd.concat([val[val.label == 0].head(100), val[val.label == 1].head(100)])
        cfg = Config({**cfg.to_dict(), "distilbert": {**cfg.distilbert.to_dict(), "epochs": 1}})
        logger.info("smoke mode: train=%d val=%d", len(train), len(val))

    tokenizer = AutoTokenizer.from_pretrained(cfg.distilbert.model_name)
    train_loader = _loader(train, tokenizer, cfg, shuffle=True)
    val_loader = _loader(val, tokenizer, cfg, shuffle=False)

    model = DualDistilBert(cfg.distilbert.model_name, condition=args.condition).to(device)
    params = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.distilbert.lr),
        weight_decay=float(cfg.distilbert.weight_decay),
    )
    total_steps = len(train_loader) * int(cfg.distilbert.epochs)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * float(cfg.distilbert.warmup_ratio)),
        num_training_steps=total_steps,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    criterion = nn.BCEWithLogitsLoss()
    run_id = make_run_id("distilbert", args.condition, args.seed)
    run_dir = ensure_dir(results / "logs" / run_id)
    config_hash = snapshot_config(cfg, run_dir)
    write_environment(run_dir)
    start = time.time()
    best_f1, best_state = -1.0, None

    for epoch in range(1, int(cfg.distilbert.epochs) + 1):
        model.train()
        losses = []
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            features = {key: value.to(device) for key, value in batch.items()
                        if key not in ("id", "label")}
            labels = batch["label"].to(device)
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(**features)["logit"]
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            losses.append(float(loss.detach()))
        val_result = evaluate(model, val_loader, device)
        val_f1 = compute_metrics(val_result["labels"], val_result["probs"], 0.5)["f1"]
        logger.info("epoch %d: loss=%.4f val_f1=%.4f", epoch, np.mean(losses), val_f1)
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }

    if best_state is not None:
        model.load_state_dict(best_state)
    elapsed = time.time() - start

    model_dir = ensure_dir(models / "distilbert" / run_id)
    torch.save(model.state_dict(), model_dir / "state_dict.pt")
    tokenizer.save_pretrained(model_dir / "tokenizer")
    snapshot_config(cfg, model_dir)

    git_commit = compute_git_commit()
    val_result = evaluate(model, val_loader, device)
    save_run(
        model_name="distilbert", condition=args.condition, seed=args.seed, split="val",
        result=val_result, run_id=run_id, results_dir=results, train_time=elapsed,
        params=params, git_commit=git_commit, config_hash=config_hash,
    )

    if args.final and not args.smoke:
        test = pd.read_parquet(processed / "test.parquet")
        test_loader = _loader(test, tokenizer, cfg, shuffle=False)
        test_result = evaluate(model, test_loader, device)
        save_run(
            model_name="distilbert", condition=args.condition, seed=args.seed, split="test",
            result=test_result, run_id=run_id, results_dir=results, train_time=elapsed,
            params=params, git_commit=git_commit, config_hash=config_hash,
        )
        logger.info(
            "test F1=%.4f",
            compute_metrics(test_result["labels"], test_result["probs"], 0.5)["f1"],
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
