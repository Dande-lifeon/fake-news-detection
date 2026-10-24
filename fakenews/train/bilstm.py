"""Train and evaluate the dual-encoder Bi-LSTM."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from fakenews.config import Config, load_config
from fakenews.data.neural import BiLSTMDataset, Vocab, collate_bilstm
from fakenews.models.bilstm import DualBiLSTM
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


def _loader(frame: pd.DataFrame, vocab: Vocab, cfg: Config, shuffle: bool) -> DataLoader:
    dataset = BiLSTMDataset(
        frame, vocab,
        max_title=int(cfg.bilstm.max_title_len),
        max_body=int(cfg.bilstm.max_body_len),
    )
    return DataLoader(
        dataset,
        batch_size=int(cfg.bilstm.batch_size),
        shuffle=shuffle,
        collate_fn=collate_bilstm,
        num_workers=0,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train a dual-encoder Bi-LSTM")
    parser.add_argument("--config", type=Path, default=Path("configs/bilstm.yaml"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--condition", choices=CONDITIONS, default="fusion")
    parser.add_argument("--final", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)

    logger = get_logger("fakenews.bilstm")
    cfg = _load(args.config)
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("device=%s", device)

    processed = Path(cfg.paths.processed_dir)
    results = Path(cfg.paths.results_dir)
    models = Path(cfg.paths.models_dir)
    train = pd.read_parquet(processed / "train.parquet")
    val = pd.read_parquet(processed / "val.parquet")
    if args.smoke:
        train = pd.concat([train[train.label == 0].head(300), train[train.label == 1].head(300)])
        val = pd.concat([val[val.label == 0].head(100), val[val.label == 1].head(100)])
        cfg = Config({**cfg.to_dict(), "bilstm": {**cfg.bilstm.to_dict(), "epochs": 1}})
        logger.info("smoke mode: train=%d val=%d", len(train), len(val))

    vocab = Vocab.build(
        pd.concat([train["title"], train["text_clean"]]),
        min_freq=int(cfg.bilstm.min_freq),
        max_size=int(cfg.bilstm.vocab_size),
    )
    train_loader = _loader(train, vocab, cfg, shuffle=True)
    val_loader = _loader(val, vocab, cfg, shuffle=False)

    model = DualBiLSTM(len(vocab.stoi), cfg.bilstm, condition=args.condition).to(device)
    params = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg.bilstm.lr))
    criterion = nn.BCEWithLogitsLoss()
    patience = int(cfg.bilstm.patience)
    best_f1, best_state, stale = -1.0, None, 0
    run_id = make_run_id("bilstm", args.condition, args.seed)
    run_dir = ensure_dir(results / "logs" / run_id)
    config_hash = snapshot_config(cfg, run_dir)
    write_environment(run_dir)
    start = time.time()

    for epoch in range(1, int(cfg.bilstm.epochs) + 1):
        model.train()
        losses = []
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            features = {key: value.to(device) for key, value in batch.items()
                        if key not in ("id", "label")}
            logits = model(**features)["logit"]
            loss = criterion(logits, batch["label"].to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        val_result = evaluate(model, val_loader, device)
        from fakenews.eval.metrics import compute_metrics
        val_f1 = compute_metrics(val_result["labels"], val_result["probs"], 0.5)["f1"]
        logger.info("epoch %d: loss=%.4f val_f1=%.4f", epoch, np.mean(losses), val_f1)
        if val_f1 > best_f1:
            best_f1, stale = val_f1, 0
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
        else:
            stale += 1
            if stale >= patience:
                logger.info("early stop at epoch %d", epoch)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    elapsed = time.time() - start

    model_dir = ensure_dir(models / "bilstm" / run_id)
    torch.save(model.state_dict(), model_dir / "state_dict.pt")
    vocab.save(model_dir / "vocab.tsv")
    snapshot_config(cfg, model_dir)

    git_commit = compute_git_commit()
    val_result = evaluate(model, val_loader, device)
    save_run(
        model_name="bilstm", condition=args.condition, seed=args.seed, split="val",
        result=val_result, run_id=run_id, results_dir=results, train_time=elapsed,
        params=params, git_commit=git_commit, config_hash=config_hash,
    )

    if args.final and not args.smoke:
        test = pd.read_parquet(processed / "test.parquet")
        test_loader = _loader(test, vocab, cfg, shuffle=False)
        test_result = evaluate(model, test_loader, device)
        save_run(
            model_name="bilstm", condition=args.condition, seed=args.seed, split="test",
            result=test_result, run_id=run_id, results_dir=results, train_time=elapsed,
            params=params, git_commit=git_commit, config_hash=config_hash,
        )
        from fakenews.eval.metrics import compute_metrics
        logger.info(
            "test F1=%.4f",
            compute_metrics(test_result["labels"], test_result["probs"], 0.5)["f1"],
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
