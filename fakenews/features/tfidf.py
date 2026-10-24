"""TF-IDF feature builders for the title and body branches.

Both vectorisers are fit **only** on the training split; validation and test are
only transformed. This is the main guard against feature leakage.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer

from fakenews.config import Config


@dataclass
class TfidfMatrices:
    """Sparse matrices for the title/body branches (test is optional)."""

    title_train: csr_matrix
    title_val: csr_matrix
    title_test: csr_matrix | None
    body_train: csr_matrix
    body_val: csr_matrix
    body_test: csr_matrix | None
    title_vec: TfidfVectorizer
    body_vec: TfidfVectorizer


def _make_vectorizer(section: Config) -> TfidfVectorizer:
    return TfidfVectorizer(
        ngram_range=(1, int(section.ngram_max)),
        min_df=int(section.min_df),
        max_features=int(section.max_features),
        stop_words=section.stop_words,
        sublinear_tf=bool(section.sublinear_tf),
        dtype=np.float32,
    )


def build_tfidf(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame | None,
    cfg: Config,
    text_col: str = "text_clean",
) -> TfidfMatrices:
    """Fit title/body TF-IDF on ``train`` and transform ``val``/``test``."""
    if "title" not in train.columns or text_col not in train.columns:
        raise KeyError(f"train must contain 'title' and {text_col!r}")

    title_vec = _make_vectorizer(cfg.features.title)
    body_vec = _make_vectorizer(cfg.features.body)

    title_train = title_vec.fit_transform(train["title"].fillna("").astype(str))
    body_train = body_vec.fit_transform(train[text_col].fillna("").astype(str))

    title_val = title_vec.transform(val["title"].fillna("").astype(str))
    body_val = body_vec.transform(val[text_col].fillna("").astype(str))

    title_test = body_test = None
    if test is not None:
        title_test = title_vec.transform(test["title"].fillna("").astype(str))
        body_test = body_vec.transform(test[text_col].fillna("").astype(str))

    return TfidfMatrices(
        title_train=title_train,
        title_val=title_val,
        title_test=title_test,
        body_train=body_train,
        body_val=body_val,
        body_test=body_test,
        title_vec=title_vec,
        body_vec=body_vec,
    )


def concat_branch(
    matrices: TfidfMatrices,
) -> tuple[csr_matrix, csr_matrix, csr_matrix | None]:
    """Early-concat condition: ``[title_tfidf | body_tfidf]``."""
    train = hstack([matrices.title_train, matrices.body_train]).tocsr()
    val = hstack([matrices.title_val, matrices.body_val]).tocsr()
    test = None
    if matrices.title_test is not None and matrices.body_test is not None:
        test = hstack([matrices.title_test, matrices.body_test]).tocsr()
    return train, val, test
