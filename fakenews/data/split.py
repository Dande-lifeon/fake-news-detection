"""Stratified train/val/test split with invariant checks and a manifest.

Split before any feature extraction; the test file is made read-only and must be
read exactly once for a final evaluation (see ``docs/protocol.md``).
"""

from __future__ import annotations

import argparse
import os
import stat
from datetime import datetime
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from fakenews.config import Config, apply_override, load_config
from fakenews.utils import ensure_dir, get_logger, save_json, sha256_file

SPLIT_NAMES = ("train", "val", "test")


def split_dataset(
    df: pd.DataFrame,
    val_size: float,
    test_size: float,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """Stratified 70/10/20-style split: take test first, then val from the rest."""
    if "label" not in df.columns:
        raise ValueError("dataframe must contain a 'label' column")
    if not 0 < test_size < 1 or not 0 < val_size < 1 or test_size + val_size >= 1:
        raise ValueError("val_size and test_size must be in (0, 1) and sum to < 1")

    train_tmp, test = train_test_split(
        df,
        test_size=test_size,
        stratify=df["label"],
        random_state=seed,
    )
    relative_val = val_size / (1.0 - test_size)
    train, val = train_test_split(
        train_tmp,
        test_size=relative_val,
        stratify=train_tmp["label"],
        random_state=seed,
    )
    return {
        "train": train.reset_index(drop=True),
        "val": val.reset_index(drop=True),
        "test": test.reset_index(drop=True),
    }


def assert_split_invariants(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    tolerance: float = 0.005,
) -> None:
    """Assert no id overlap, full coverage, class balance and non-empty bodies."""
    splits = {"train": train, "val": val, "test": test}
    id_sets = {name: set(frame["id"]) for name, frame in splits.items()}
    for name, frame in splits.items():
        if frame["id"].duplicated().any():
            raise AssertionError(f"duplicate ids inside {name}")
        if len(id_sets[name]) != len(frame):
            raise AssertionError(f"id count mismatch in {name}")

    overlaps = {
        "train∩val": id_sets["train"] & id_sets["val"],
        "train∩test": id_sets["train"] & id_sets["test"],
        "val∩test": id_sets["val"] & id_sets["test"],
    }
    for pair, shared in overlaps.items():
        if shared:
            raise AssertionError(f"{pair} shares {len(shared)} ids")

    total = len(train) + len(val) + len(test)
    union = id_sets["train"] | id_sets["val"] | id_sets["test"]
    if len(union) != total:
        raise AssertionError(f"union has {len(union)} ids but splits have {total} rows")

    overall = pd.concat(splits.values(), ignore_index=True)
    overall_rate = float((overall["label"] == 1).mean())
    for name, frame in splits.items():
        rate = float((frame["label"] == 1).mean())
        if abs(rate - overall_rate) > tolerance:
            raise AssertionError(
                f"{name} positive rate {rate:.4f} deviates from {overall_rate:.4f} "
                f"by more than {tolerance}"
            )
        if (frame["text_clean"].str.len() == 0).any():
            raise AssertionError(f"{name} contains empty text_clean")


def _set_readonly(path: Path, readonly: bool) -> None:
    """Toggle the read-only flag on a file (Windows attribute / POSIX mode)."""
    if not path.exists():
        return
    if readonly:
        os.chmod(path, stat.S_IREAD)
    else:
        os.chmod(path, stat.S_IWRITE | stat.S_IREAD)


def save_splits(
    splits: dict[str, pd.DataFrame],
    out_dir: Path,
    manifest_path: Path,
    seed: int = 42,
) -> dict:
    """Write the three parquet files, lock ``test`` and write the manifest."""
    out_dir = ensure_dir(out_dir)
    manifest: dict = {
        "seed": int(seed),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "pandas_version": pd.__version__,
        "sizes": {},
        "label_counts": {},
        "sha256": {},
    }

    for name in SPLIT_NAMES:
        frame = splits[name]
        target = out_dir / f"{name}.parquet"
        _set_readonly(target, False)
        frame.to_parquet(target, index=False)
        manifest["sizes"][name] = int(len(frame))
        manifest["label_counts"][name] = {
            "real": int((frame["label"] == 0).sum()),
            "fake": int((frame["label"] == 1).sum()),
        }
        manifest["sha256"][name] = sha256_file(target)
        if name == "test":
            _set_readonly(target, True)

    manifest["total"] = int(sum(manifest["sizes"].values()))
    save_json(manifest, manifest_path)
    return manifest


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Split cleaned data into train/val/test")
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--force", action="store_true", help="overwrite existing splits")
    parser.add_argument("--override", action="append", default=[], metavar="KEY=VALUE")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logger = get_logger("fakenews.split")
    cfg: Config = load_config(args.config)
    for dotted in args.override:
        cfg = apply_override(cfg, dotted)

    processed_dir = Path(cfg.paths.processed_dir)
    source = args.input or (processed_dir / "articles_clean.parquet")
    manifest_path = processed_dir / "split_manifest.json"
    seed = args.seed if args.seed is not None else int(cfg.seed)

    if manifest_path.exists() and not args.force:
        raise SystemExit(
            f"{manifest_path} already exists; pass --force to regenerate the splits"
        )

    frame = pd.read_parquet(source)
    splits = split_dataset(
        frame,
        val_size=float(cfg.data.val_size),
        test_size=float(cfg.data.test_size),
        seed=seed,
    )
    assert_split_invariants(*splits.values())
    manifest = save_splits(splits, args.out_dir or processed_dir, manifest_path, seed=seed)

    for name in SPLIT_NAMES:
        counts = manifest["label_counts"][name]
        logger.info(
            "%s: %d rows (real=%d, fake=%d)",
            name,
            manifest["sizes"][name],
            counts["real"],
            counts["fake"],
        )
    logger.info("manifest → %s", manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
