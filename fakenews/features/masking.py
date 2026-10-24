"""Source-cue masking for the Stage 8 experiment.

Two mask levels are provided:

``dateline_url``
    Remove the ``CITY (Reuters) -`` dateline and URLs (the pre-registered
    experiment from the coding plan).
``format``
    Also removes CMS and media-credit artefacts that are almost exclusive to
    the fake source: ``featured image``, ``Photo by ... / Getty Images``,
    ``Image via ...``, ``via``, ``follow us`` / ``subscribe``, and ``read more``.

Only the body is masked. Titles are left untouched so the title-only branch is a
sanity check (its score should barely move).
"""

from __future__ import annotations

import re

import pandas as pd

DATELINE_RE = re.compile(r"[A-Za-z .'\-/]{0,60}\(Reuters\)\s*[-–—]\s*")
URL_RE = re.compile(r"https?://\S+|www\.\S+")

FEATURED_RE = re.compile(r"featured image[^.\n]{0,180}", re.I)
PHOTO_CREDIT_RE = re.compile(r"photo(?:s)?\s+(?:by|via|courtesy of)[^.\n]{0,180}", re.I)
IMAGE_CREDIT_RE = re.compile(r"image(?:s)?\s+(?:via|by|source|caption|credit)[^.\n]{0,160}", re.I)
GETTY_RE = re.compile(r"\bgetty images?\b", re.I)
VIA_RE = re.compile(r"\bvia\b", re.I)
FOLLOW_RE = re.compile(r"(?:follow us|subscribe)[^.\n]{0,140}", re.I)
READ_MORE_RE = re.compile(r"read more[^.\n]{0,120}", re.I)
WS_RE = re.compile(r"\s+")

MASK_MODES = ("dateline_url", "format")


def mask_body(text: str, mode: str = "dateline_url") -> str:
    """Strip source/format cues from a body while keeping the content."""
    if not isinstance(text, str):
        return ""
    if mode not in MASK_MODES:
        raise ValueError(f"unknown mask mode: {mode}")

    masked = DATELINE_RE.sub(" ", text)
    masked = URL_RE.sub(" ", masked)
    if mode == "format":
        for pattern in (
            FEATURED_RE,
            PHOTO_CREDIT_RE,
            IMAGE_CREDIT_RE,
            GETTY_RE,
            VIA_RE,
            FOLLOW_RE,
            READ_MORE_RE,
        ):
            masked = pattern.sub(" ", masked)
    return WS_RE.sub(" ", masked).strip()


def mask_dataframe(df: pd.DataFrame, mode: str = "dateline_url") -> pd.DataFrame:
    """Return a copy of ``df`` with an extra ``text_masked`` column.

    Mask the cleaned body, because that is what the baseline models were
    trained on.  Falling back to ``text_raw`` keeps the helper usable on
    fixtures that only provide raw text.
    """
    frame = df.copy()
    if "text_clean" in frame.columns:
        source = frame["text_clean"]
    elif "text_raw" in frame.columns:
        source = frame["text_raw"]
    else:
        source = pd.Series([""] * len(frame), index=frame.index)
    frame["text_masked"] = source.fillna("").astype(str).map(lambda text: mask_body(text, mode))
    return frame


def count_hits(df: pd.DataFrame) -> dict[str, dict[str, int]]:
    """Count cue hits by class (``real`` / ``fake``) for sanity checks."""
    body = df["text_raw"].fillna("").astype(str)
    labels = df["label"].astype(int)
    patterns = {
        "reuters_dateline": r"\(Reuters\)",
        "url": r"https?://|www\.",
        "getty": r"getty",
        "featured_image": r"featured image",
        "via": r"\bvia\b",
        "follow_us": r"follow us|subscribe",
        "our_standards": r"our standards",
    }
    hits: dict[str, dict[str, int]] = {}
    for name, pattern in patterns.items():
        hit = body.str.contains(pattern, case=False, regex=True)
        hits[name] = {
            "real": int(hit[labels == 0].sum()),
            "fake": int(hit[labels == 1].sum()),
            "real_total": int((labels == 0).sum()),
            "fake_total": int((labels == 1).sum()),
        }
    return hits
