"""Download and validate the ISOT raw CSVs.

``True.csv`` → ``label=0`` (real), ``Fake.csv`` → ``label=1`` (fake).

Authentication (in order of precedence):

1. ``KAGGLE_API_TOKEN`` environment variable (new-style ``KGAT_...`` token) —
   used through ``kagglehub``.
2. Classic ``~/.kaggle/kaggle.json`` (``username`` + ``key``) — used through
   ``kagglehub`` or the ``kaggle`` CLI.
3. Manual download: drop ``True.csv`` / ``Fake.csv`` into ``data/raw/`` or pass
   ``--archive path/to/archive.zip``.

The raw CSV files are read-only inputs: this module never rewrites them.
"""

from __future__ import annotations

import argparse
import os
import shutil
import tempfile
import zipfile
from pathlib import Path

import pandas as pd

from fakenews.config import Config, apply_override, load_config
from fakenews.utils import ensure_dir, get_logger, save_json, sha256_file

DATASET_SLUG = "clmentbisaillon/fake-and-real-news-dataset"
REQUIRED_COLUMNS = {"title", "text", "subject", "date"}
EXPECTED_FILES = ("True.csv", "Fake.csv")


def _candidate_cache_dir(raw_dir: Path) -> Path:
    """Keep kagglehub's cache inside the (git-ignored) project tree."""
    return raw_dir.parent / ".cache" / "kagglehub"


def _stream_to(url: str, target: Path, logger) -> None:
    """Stream ``url`` to ``target`` with a lightweight progress log."""
    import requests  # noqa: PLC0415 - imported lazily to keep CLI startup fast

    response = requests.get(url, stream=True, timeout=(15, 30))
    response.raise_for_status()
    total = int(response.headers.get("content-length") or 0)
    written = 0
    next_report = 10 * 1024 * 1024
    with open(target, "wb") as handle:
        for chunk in response.iter_content(chunk_size=1 << 20):
            if not chunk:
                continue
            handle.write(chunk)
            written += len(chunk)
            if written >= next_report:
                logger.info("downloaded %.1f MB%s", written / 1e6,
                            f" / {total / 1e6:.1f} MB" if total else "")
                next_report += 10 * 1024 * 1024
    if total and written != total:
        raise IOError(f"incomplete download: {written} of {total} bytes")
    logger.info("downloaded %.1f MB", written / 1e6)


def _download_with_api(raw_dir: Path, logger) -> bool:
    """Download ``archive.zip`` through the Kaggle API and extract the CSVs.

    Some networks reset TLS connections to ``storage.googleapis.com`` while
    allowing plain HTTP. In that case the signed URL is retried over HTTP; this
    is only a transport fallback and is logged loudly.
    """
    token = os.environ.get("KAGGLE_API_TOKEN") or os.environ.get("KAGGLE_KEY")
    if not token:
        return False

    try:
        import requests  # noqa: PLC0415
    except ImportError:
        logger.warning("requests is not installed; skipping the Kaggle API path")
        return False

    url = f"https://www.kaggle.com/api/v1/datasets/download/{DATASET_SLUG}"
    logger.info("requesting a signed download URL from the Kaggle API")
    try:
        response = requests.get(
            url,
            headers={"Authorization": f"Bearer {token}", "User-Agent": "fakenews/0.1"},
            allow_redirects=False,
            timeout=60,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Kaggle API request failed: %s", exc)
        return False

    if response.status_code not in (301, 302, 303, 307, 308):
        logger.warning("unexpected Kaggle API status %s: %s", response.status_code,
                       response.text[:200])
        return False

    signed_url = response.headers.get("Location")
    if not signed_url:
        logger.warning("Kaggle API did not return a download location")
        return False

    archive = raw_dir / "archive.zip"
    try:
        _stream_to(signed_url, archive, logger)
    except Exception as exc:  # noqa: BLE001
        logger.warning("download via HTTPS failed: %s", exc)
        if signed_url.startswith("https://storage.googleapis.com"):
            insecure = "http://" + signed_url[len("https://"):]
            logger.warning("retrying the same signed URL over plain HTTP (TLS blocked)")
            try:
                _stream_to(insecure, archive, logger)
            except Exception as exc2:  # noqa: BLE001
                logger.warning("plain-HTTP fallback failed: %s", exc2)
                archive.unlink(missing_ok=True)
                return False
        else:
            archive.unlink(missing_ok=True)
            return False

    try:
        _extract_archive(archive, raw_dir)
    finally:
        archive.unlink(missing_ok=True)
    return True


def _download_with_kagglehub(raw_dir: Path, logger) -> Path | None:
    try:
        import kagglehub  # noqa: PLC0415 - optional dependency, imported lazily
    except ImportError:
        logger.warning("kagglehub is not installed; skipping the Kaggle API path")
        return None

    cache = Path(os.environ.get("KAGGLEHUB_CACHE", _candidate_cache_dir(raw_dir))).resolve()
    ensure_dir(cache)
    # kagglehub reads the cache location from the environment at call time.
    os.environ["KAGGLEHUB_CACHE"] = str(cache)

    logger.info("downloading %s via kagglehub (cache: %s)", DATASET_SLUG, cache)
    try:
        source = Path(kagglehub.dataset_download(DATASET_SLUG))
    except Exception as exc:  # noqa: BLE001 - surface any auth/network failure as a hint
        logger.warning("kagglehub download failed: %s", exc)
        return None
    return source


def _download_with_cli(raw_dir: Path, logger) -> Path | None:
    import subprocess  # noqa: PLC0415

    kaggle = shutil.which("kaggle")
    if kaggle is None:
        logger.warning("kaggle CLI not found; skipping `kaggle datasets download`")
        return None

    with tempfile.TemporaryDirectory(dir=raw_dir.parent) as tmp:
        cmd = [kaggle, "datasets", "download", "-d", DATASET_SLUG, "-p", tmp, "--unzip"]
        logger.info("running: %s", " ".join(cmd))
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        except OSError as exc:  # sandboxed environments may forbid child pipes
            logger.warning("could not launch kaggle CLI: %s", exc)
            return None
        if result.returncode != 0:
            logger.warning("kaggle CLI failed: %s", (result.stderr or result.stdout).strip())
            return None
        return Path(tmp)


def _copy_from(source_dir: Path, raw_dir: Path) -> None:
    """Copy ``True.csv`` / ``Fake.csv`` from ``source_dir`` (recursively) to raw_dir."""
    for name in EXPECTED_FILES:
        matches = sorted(source_dir.rglob(name))
        if not matches:
            raise FileNotFoundError(f"{name} not found under {source_dir}")
        target = raw_dir / name
        if target.exists() and target.stat().st_size > 0:
            continue
        shutil.copy2(matches[0], target)


def _extract_archive(archive: Path, raw_dir: Path) -> None:
    with zipfile.ZipFile(archive) as zf:
        names = [n for n in zf.namelist() if Path(n).name in EXPECTED_FILES]
        if not names:
            raise FileNotFoundError(f"archive {archive} does not contain True.csv / Fake.csv")
        for name in names:
            with zf.open(name) as src, open(raw_dir / Path(name).name, "wb") as dst:
                shutil.copyfileobj(src, dst)


def validate_raw(true_path: Path, fake_path: Path) -> dict:
    """Validate columns/rows of both CSVs and return a summary with SHA-256."""
    summary: dict[str, dict] = {}
    for label_name, path, label in (("true", true_path, 0), ("fake", fake_path, 1)):
        if not path.exists():
            raise FileNotFoundError(path)
        if path.stat().st_size == 0:
            raise ValueError(f"empty file: {path}")
        frame = pd.read_csv(path)
        missing_cols = REQUIRED_COLUMNS - set(frame.columns)
        if missing_cols:
            raise ValueError(f"{path.name} is missing columns: {sorted(missing_cols)}")
        if len(frame) == 0:
            raise ValueError(f"no rows in {path.name}")
        summary[label_name] = {
            "file": path.name,
            "label": label,
            "rows": int(len(frame)),
            "columns": [str(c) for c in frame.columns],
            "dtypes": {str(k): str(v) for k, v in frame.dtypes.items()},
            "missing": {str(k): int(v) for k, v in frame.isna().sum().items()},
            "sha256": sha256_file(path),
            "bytes": int(path.stat().st_size),
        }
    return summary


def write_checksums(summary: dict, raw_dir: Path) -> Path:
    """Write ``data/raw/CHECKSUMS.txt`` (``sha256  filename`` per line)."""
    target = raw_dir / "CHECKSUMS.txt"
    lines = ["# ISOT Fake and Real News dataset — SHA-256 of raw files\n"]
    for key in ("true", "fake"):
        lines.append(f"{summary[key]['sha256']}  {summary[key]['file']}\n")
    target.write_text("".join(lines), encoding="utf-8")
    return target


def download_dataset(raw_dir: Path, force: bool = False, archive: Path | None = None) -> dict:
    """Ensure both raw CSVs exist in ``raw_dir``; return the validation summary."""
    logger = get_logger("fakenews.download")
    raw_dir = ensure_dir(raw_dir)

    present = [name for name in EXPECTED_FILES if (raw_dir / name).exists()]
    if force:
        for name in present:
            (raw_dir / name).unlink()
        present = []

    if len(present) < len(EXPECTED_FILES):
        if archive is not None:
            logger.info("extracting %s", archive)
            _extract_archive(Path(archive), raw_dir)
        elif _download_with_api(raw_dir, logger):
            logger.info("Kaggle API download complete")
        else:
            source = _download_with_kagglehub(raw_dir, logger)
            if source is None:
                source = _download_with_cli(raw_dir, logger)
            if source is None:
                raise RuntimeError(
                    "Could not download the dataset automatically.\n"
                    "Do one of the following:\n"
                    "  1) set KAGGLE_API_TOKEN (KGAT_...) or ~/.kaggle/kaggle.json, then rerun;\n"
                    "  2) download archive.zip from "
                    f"https://www.kaggle.com/datasets/{DATASET_SLUG} and run\n"
                    "     python -m fakenews.data.download --archive path/to/archive.zip;\n"
                    f"  3) copy True.csv and Fake.csv into {raw_dir} manually."
                )
            logger.info("copying CSVs from %s", source)
            _copy_from(source, raw_dir)

    summary = validate_raw(raw_dir / "True.csv", raw_dir / "Fake.csv")
    checksums = write_checksums(summary, raw_dir)
    save_json(summary, raw_dir / "raw_summary.json")

    logger.info(
        "raw data ready: True=%d rows, Fake=%d rows; checksums → %s",
        summary["true"]["rows"],
        summary["fake"]["rows"],
        checksums,
    )
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download/validate the ISOT raw CSVs")
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    parser.add_argument("--raw-dir", type=Path, default=None)
    parser.add_argument("--archive", type=Path, default=None, help="local archive.zip")
    parser.add_argument("--force", action="store_true", help="re-download even if files exist")
    parser.add_argument("--override", action="append", default=[], metavar="KEY=VALUE")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    cfg: Config = load_config(args.config)
    for dotted in args.override:
        cfg = apply_override(cfg, dotted)

    raw_dir = args.raw_dir or Path(cfg.paths.raw_dir)
    summary = download_dataset(raw_dir, force=args.force, archive=args.archive)
    total = summary["true"]["rows"] + summary["fake"]["rows"]
    print(f"\nTotal rows: {total} (True={summary['true']['rows']}, Fake={summary['fake']['rows']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
