"""Surface features used to measure the source shortcut.

Everything here comes from ``title`` and ``text_raw``, never from
``text_clean``, ``subject`` or ``date``. The features cover the Reuters
dateline, CMS and media credits, URLs, and surface style that leak the source
label in ISOT.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FEATURES = [
    "has_reuters_dateline",
    "has_url",
    "has_via",
    "has_getty",
    "has_follow_us",
    "has_our_standards",
    "starts_with_city_dateline",
    "title_has_question",
    "title_has_exclamation",
    "exclamation_count",
    "question_count",
    "uppercase_ratio",
    "all_caps_word_count",
    "title_len",
    "text_len",
    "digit_ratio",
    "quote_count",
    "body_has_url",
]


def _text_column(df: pd.DataFrame, column: str, *, required: bool = True) -> pd.Series:
    if column not in df.columns:
        if required:
            raise KeyError(f"shortcut features require a {column!r} column")
        return pd.Series([""] * len(df), index=df.index, dtype="object")
    return df[column].fillna("").astype(str)


def extract_shortcut_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return the 18 fixed-width numeric format features, aligned to ``df.index``."""
    title = _text_column(df, "title", required=False)
    body = _text_column(df, "text_raw")
    combined = title + " " + body
    body_len = body.str.len().clip(lower=1)

    features = pd.DataFrame(index=df.index)
    features["has_reuters_dateline"] = body.str.contains(
        r"\(Reuters\)", case=False, regex=True
    )
    features["has_url"] = combined.str.contains(
        r"https?://|www\.", case=False, regex=True
    )
    features["has_via"] = body.str.contains(r"\bvia\b", case=False, regex=True)
    features["has_getty"] = body.str.contains(r"getty", case=False, regex=True)
    features["has_follow_us"] = body.str.contains(
        r"follow us|subscribe", case=False, regex=True
    )
    features["has_our_standards"] = body.str.contains(
        r"our standards", case=False, regex=True
    )
    features["starts_with_city_dateline"] = body.str.contains(
        r"^\s*[A-Z][A-Za-z .'\-]{1,40}\(Reuters\)", regex=True
    )
    features["title_has_question"] = title.str.contains("?", regex=False)
    features["title_has_exclamation"] = title.str.contains("!", regex=False)
    features["exclamation_count"] = body.str.count(r"!")
    features["question_count"] = body.str.count(r"\?")
    features["uppercase_ratio"] = body.str.count(r"[A-Z]") / body.str.count(
        r"[A-Za-z]"
    ).clip(lower=1)
    features["all_caps_word_count"] = body.str.count(r"\b[A-Z]{2,}\b")
    features["title_len"] = title.str.len()
    features["text_len"] = body.str.len()
    features["digit_ratio"] = body.str.count(r"\d") / body_len
    features["quote_count"] = body.str.count(r'["“”]')
    features["body_has_url"] = body.str.contains(
        r"https?://|www\.", case=False, regex=True
    )

    return features[FEATURES].astype(np.float32)


def feature_importance(model, top_k: int = 15) -> list[dict[str, float | str]]:
    """Return signed LR coefficients sorted by absolute value (for inspection)."""
    coef = model.coef_.ravel()
    order = np.argsort(np.abs(coef))[::-1][:top_k]
    return [
        {"feature": FEATURES[index], "coef": float(coef[index])} for index in order
    ]
