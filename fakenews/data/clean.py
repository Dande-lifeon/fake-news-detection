"""Cleaning of the raw ISOT CSVs.

Pipeline (order matters, see ``docs/coding_plan.md`` §5.4):

1. ``text_clean = clean_text(text_raw)`` — HTML tags/control chars removed,
   whitespace collapsed; URLs and ``(Reuters)`` are kept.
2. Drop rows whose cleaned body is ``<= min_len`` characters.
3. Build ``dup_key`` from ``title`` + ``text_clean``.
4. Drop **all** rows belonging to a ``dup_key`` seen with both labels.
5. Exact duplicate removal (``keep="first"``) on ``dup_key``.

The cleaned table is written both as Parquet (canonical, used by training) and
as CSV (for inspection / handing the data to someone else).
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from fakenews.config import Config, apply_override, load_config
from fakenews.utils import ensure_dir, get_logger, save_json

HTML_RE = re.compile(r"<[^>]+>")
WS_RE = re.compile(r"\s+")
CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

CLEAN_COLUMNS = [
    "id",
    "title",
    "text_raw",
    "text_clean",
    "label",
    "subject",
    "date",
    "dup_key",
]


def clean_text(text: str) -> str:
    """Strip HTML tags and control characters, collapse whitespace, strip.

    URLs and datelines such as ``WASHINGTON (Reuters) -`` are intentionally
    kept: masking them is a separate, auditable experiment.
    """
    if not isinstance(text, str):
        return ""
    without_tags = HTML_RE.sub(" ", text)
    without_control = CONTROL_RE.sub(" ", without_tags)
    return WS_RE.sub(" ", without_control).strip()


def _normalize_key(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return WS_RE.sub(" ", text).strip().lower()


def make_dup_key(title: str, text_clean: str) -> str:
    """Normalized ``title + "\\n" + text_clean`` used for duplicate auditing."""
    return f"{_normalize_key(title)}\n{_normalize_key(text_clean)}"


def _read_csv(path: Path) -> pd.DataFrame:
    # keep_default_na=False keeps literal tokens such as "NA" inside articles.
    return pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8")


def load_raw(raw_dir: Path) -> pd.DataFrame:
    """Read ``True.csv`` (label 0) and ``Fake.csv`` (label 1), concatenated.

    Adds a stable ``id`` (row number in the concatenated, uncleaned table) and a
    ``source`` audit column. ``id`` is the join key for every later experiment.
    """
    raw_dir = Path(raw_dir)
    frames: list[pd.DataFrame] = []
    for filename, label, source in (
        ("True.csv", 0, "reuters"),
        ("Fake.csv", 1, "fake"),
    ):
        path = raw_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"missing raw file: {path}")
        part = _read_csv(path)
        part["label"] = np.int8(label)
        part["source"] = source
        frames.append(part)

    frame = pd.concat(frames, ignore_index=True)
    frame["id"] = np.arange(len(frame), dtype=np.int64)
    for column in ("title", "text", "subject", "date"):
        if column not in frame.columns:
            frame[column] = ""
        frame[column] = frame[column].fillna("").astype(str)
    ordered = ["id", "title", "text", "subject", "date", "label", "source"]
    return frame[ordered]


def load_single_csv(path: Path, text_col: str = "text", title_col: str = "title") -> pd.DataFrame:
    """Load a single labelled CSV (fixtures, smoke tests); adds ``id`` if absent."""
    frame = _read_csv(Path(path))
    if "label" not in frame.columns:
        raise ValueError(f"{path} must contain a 'label' column")
    if text_col not in frame.columns:
        raise ValueError(f"{path} must contain a {text_col!r} column")
    if title_col not in frame.columns:
        frame[title_col] = ""
    frame["label"] = frame["label"].astype(int).astype(np.int8)
    if "id" not in frame.columns:
        frame["id"] = np.arange(len(frame), dtype=np.int64)
    frame["id"] = frame["id"].astype(np.int64)
    frame["source"] = np.where(frame["label"] == 1, "fake", "reuters")
    return frame


def clean_dataset(
    df: pd.DataFrame,
    min_len: int = 50,
    drop_conflicts: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """Clean a concatenated raw frame and return ``(cleaned_df, stats)``."""
    frame = df.copy()

    if "text_raw" not in frame.columns:
        frame["text_raw"] = frame["text"].fillna("").astype(str)
    else:
        frame["text_raw"] = frame["text_raw"].fillna("").astype(str)
    if "title" not in frame.columns:
        frame["title"] = ""
    frame["title"] = frame["title"].fillna("").astype(str)

    stats: dict[str, int] = {
        "loaded_total": int(len(frame)),
        "loaded_true": int((frame["label"] == 0).sum()),
        "loaded_fake": int((frame["label"] == 1).sum()),
        "missing_title": int((frame["title"].str.strip() == "").sum()),
        "missing_text": int((frame["text_raw"].str.strip() == "").sum()),
        "short_text_dropped": 0,
        "conflict_keys": 0,
        "conflict_rows_dropped": 0,
        "duplicate_rows_dropped": 0,
        "remaining_total": 0,
        "remaining_true": 0,
        "remaining_fake": 0,
    }

    frame["text_clean"] = frame["text_raw"].map(clean_text)

    short_mask = frame["text_clean"].str.len() <= min_len
    stats["short_text_dropped"] = int(short_mask.sum())
    frame = frame.loc[~short_mask].copy()

    frame["dup_key"] = [
        make_dup_key(title, body)
        for title, body in zip(
            frame["title"].to_numpy(), frame["text_clean"].to_numpy(), strict=True
        )
    ]

    label_counts = frame.groupby("dup_key", sort=False)["label"].nunique()
    conflict_keys = label_counts[label_counts > 1].index
    stats["conflict_keys"] = int(len(conflict_keys))
    conflict_mask = frame["dup_key"].isin(conflict_keys)
    stats["conflict_rows_dropped"] = int(conflict_mask.sum())
    if drop_conflicts and stats["conflict_rows_dropped"]:
        frame = frame.loc[~conflict_mask].copy()

    duplicate_mask = frame.duplicated(subset="dup_key", keep="first")
    stats["duplicate_rows_dropped"] = int(duplicate_mask.sum())
    frame = frame.loc[~duplicate_mask].copy()

    for column in ("subject", "date"):
        if column not in frame.columns:
            frame[column] = ""
        frame[column] = frame[column].fillna("").astype(str)

    frame = frame[CLEAN_COLUMNS].reset_index(drop=True)
    stats["remaining_total"] = int(len(frame))
    stats["remaining_true"] = int((frame["label"] == 0).sum())
    stats["remaining_fake"] = int((frame["label"] == 1).sum())

    dropped = (
        stats["short_text_dropped"]
        + stats["conflict_rows_dropped"]
        + stats["duplicate_rows_dropped"]
    )
    if stats["loaded_total"] - dropped != stats["remaining_total"]:
        raise AssertionError(
            "cleaning bookkeeping mismatch: "
            f"{stats['loaded_total']} - {dropped} != {stats['remaining_total']}"
        )
    return frame, stats


def _to_contract_dtypes(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["id"] = frame["id"].astype(np.int64)
    frame["label"] = frame["label"].astype(np.int8)
    for column in ("title", "text_raw", "text_clean", "subject", "date", "dup_key"):
        frame[column] = frame[column].fillna("").astype("string")
    return frame[CLEAN_COLUMNS]


def save_clean(df: pd.DataFrame, path: Path, csv_path: Path | None = None) -> Path:
    """Write the cleaned table as Parquet (and optionally CSV); return Parquet path."""
    if df["id"].duplicated().any():
        raise ValueError("duplicate id values in cleaned dataframe")
    frame = _to_contract_dtypes(df)
    target = Path(path)
    ensure_dir(target.parent)
    frame.to_parquet(target, index=False)
    if csv_path is not None:
        ensure_dir(Path(csv_path).parent)
        frame.to_csv(csv_path, index=False, encoding="utf-8")
    return target


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Clean the ISOT raw CSVs")
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    parser.add_argument("--raw-dir", type=Path, default=None)
    parser.add_argument("--input", type=Path, default=None, help="single labelled CSV")
    parser.add_argument("--output", type=Path, default=None, help="output .parquet path")
    parser.add_argument("--csv", action=argparse.BooleanOptionalAction, default=True,
                        help="also write a .csv next to the parquet (default: true)")
    parser.add_argument("--min-len", type=int, default=None)
    parser.add_argument("--drop-conflicts", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--override", action="append", default=[], metavar="KEY=VALUE")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logger = get_logger("fakenews.clean")
    cfg: Config = load_config(args.config)
    for dotted in args.override:
        cfg = apply_override(cfg, dotted)

    min_len = args.min_len if args.min_len is not None else int(cfg.data.min_text_len)
    drop_conflicts = (
        args.drop_conflicts
        if args.drop_conflicts is not None
        else bool(cfg.data.drop_cross_label_conflicts)
    )

    if args.input is not None:
        raw = load_single_csv(args.input)
        source_name = args.input.name
    else:
        raw_dir = args.raw_dir or Path(cfg.paths.raw_dir)
        raw = load_raw(raw_dir)
        source_name = str(raw_dir)

    logger.info("loaded %d rows from %s", len(raw), source_name)
    cleaned, stats = clean_dataset(raw, min_len=min_len, drop_conflicts=drop_conflicts)

    processed_dir = Path(cfg.paths.processed_dir)
    output = args.output or (processed_dir / "articles_clean.parquet")
    csv_path = output.with_suffix(".csv") if args.csv else None
    save_clean(cleaned, output, csv_path=csv_path)

    stats_path = processed_dir / "clean_log.json"
    save_json({"source": source_name, "min_len": min_len, "stats": stats}, stats_path)

    logger.info(
        "clean: %d → %d rows (short=%d, conflicts=%d, duplicates=%d)",
        stats["loaded_total"],
        stats["remaining_total"],
        stats["short_text_dropped"],
        stats["conflict_rows_dropped"],
        stats["duplicate_rows_dropped"],
    )
    logger.info("wrote %s%s", output, f" and {csv_path}" if csv_path else "")
    logger.info("stats → %s", stats_path)

    print("\nCleaning summary")
    for key, value in stats.items():
        print(f"  {key:24s} {value:>8d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
