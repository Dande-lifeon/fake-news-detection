"""Small shared helpers: seeding, logging, hashing, JSON, timing."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np

PathLike = str | Path


def seed_everything(seed: int = 42, deterministic_torch: bool = True) -> None:
    """Fix ``random`` / ``numpy`` / ``torch`` (and CUDA) seeds."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
    except ImportError:  # torch is an optional (neural) dependency
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic_torch:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_logger(name: str, log_file: PathLike | None = None) -> logging.Logger:
    """Return a logger that writes to stdout and, optionally, to ``log_file``."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.handlers:  # keep repeated calls from duplicating output
        return logger

    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(fmt)
    logger.addHandler(stream)

    if log_file is not None:
        target = Path(log_file)
        target.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(target, encoding="utf-8")
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

    return logger


def sha256_file(path: PathLike, chunk_size: int = 1 << 20) -> str:
    """Streaming SHA-256 of a file."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_dir(path: PathLike) -> Path:
    """Create ``path`` (including parents) and return it."""
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def save_json(obj: Any, path: PathLike, indent: int = 2) -> None:
    """Write JSON atomically-ish (UTF-8, ``ensure_ascii=False``)."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as handle:
        json.dump(obj, handle, ensure_ascii=False, indent=indent)
        handle.write("\n")


def load_json(path: PathLike) -> dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


@contextmanager
def timer() -> Iterator[list[float]]:
    """``with timer() as elapsed: ...`` → ``elapsed[0]`` holds seconds."""
    elapsed: list[float] = []
    start = time.perf_counter()
    try:
        yield elapsed
    finally:
        elapsed.append(time.perf_counter() - start)
