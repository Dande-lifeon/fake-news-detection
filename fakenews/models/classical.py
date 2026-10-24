"""Dual-branch classical models: TF-IDF with LR or a calibrated LinearSVC.

One fit produces four conditions:

* ``title_only``: title branch only
* ``body_only``: body branch only
* ``concat``: early concatenation of both TF-IDF blocks, one classifier
* ``fusion``: late fusion ``alpha * p_title + (1 - alpha) * p_body``

``alpha`` and the decision thresholds are selected on the validation split only.
``refit_full`` refits the classifiers on ``train + val`` but keeps the vectorisers
fitted on train, so the selected ``alpha`` and ``C`` remain valid.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC

from fakenews.config import Config
from fakenews.eval.metrics import tune_threshold
from fakenews.features.tfidf import TfidfMatrices, build_tfidf

Condition = Literal["title_only", "body_only", "concat", "fusion"]
CONDITIONS: tuple[Condition, ...] = ("title_only", "body_only", "concat", "fusion")


def combine_probs(p_title: np.ndarray, p_body: np.ndarray, alpha: float) -> np.ndarray:
    """Late fusion; ``alpha=0`` degenerates to body-only, ``alpha=1`` to title-only."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    return alpha * np.asarray(p_title) + (1.0 - alpha) * np.asarray(p_body)


@dataclass
class BranchModel:
    """A vectoriser + classifier pair for one text field."""

    vectorizer: object
    clf: object
    name: str

    def predict_proba(self, texts: pd.Series) -> np.ndarray:
        matrix = self.vectorizer.transform(texts.fillna("").astype(str))
        return self.clf.predict_proba(matrix)[:, 1]


class DualTextModel:
    """Title/body TF-IDF model with title-only, body-only, concat and fusion heads."""

    def __init__(
        self,
        model_type: Literal["lr", "svm"],
        cfg: Config,
        text_col: str = "text_clean",
    ) -> None:
        if model_type not in ("lr", "svm"):
            raise ValueError(f"unknown model_type: {model_type}")
        self.model_type = model_type
        self.cfg = cfg
        self.text_col = text_col
        self.c_grid = [float(c) for c in cfg.classical.C_grid]
        self.alpha_grid = [float(a) for a in cfg.classical.alpha_grid]
        self.cv_folds = int(cfg.classical.cv_folds)
        self.default_threshold = float(cfg.classical.threshold)

        self.matrices: TfidfMatrices | None = None
        self.title_clf = None
        self.body_clf = None
        self.concat_clf = None
        self.best_c: dict[str, float] = {}
        self.alpha: float = 0.5
        self.alpha_curve: list[dict[str, float]] = []
        self.thresholds: dict[str, float] = {}
        self.threshold_tuned: dict[str, float] = {}
        self._fixed_c: dict[str, float] = {}

    # ------------------------------------------------------------------ fit
    def fit(
        self,
        train: pd.DataFrame,
        val: pd.DataFrame,
        fixed_c: dict[str, float] | None = None,
        fixed_alpha: float | None = None,
    ) -> DualTextModel:
        """Fit vectorisers on train, classifiers on train, select alpha/threshold on val.

        ``fixed_c`` / ``fixed_alpha`` are used by pre-registered analyses (e.g. the
        Stage 8 masked retrain) to reuse the original model's hyper-parameters
        instead of tuning them again.
        """
        self._fixed_c = dict(fixed_c or {})
        self.matrices = build_tfidf(train, val, None, self.cfg, text_col=self.text_col)
        y_train = train["label"].to_numpy().astype(int)
        y_val = val["label"].to_numpy().astype(int)

        title_x = self.matrices.title_train
        body_x = self.matrices.body_train
        self.title_clf, self.best_c["title"] = self._fit_branch(title_x, y_train, "title")
        self.body_clf, self.best_c["body"] = self._fit_branch(body_x, y_train, "body")

        concat_train = hstack([title_x, body_x]).tocsr()
        self.concat_clf, self.best_c["concat"] = self._fit_branch(concat_train, y_train, "concat")

        p_title = self._predict(self.title_clf, self.matrices.title_val)
        p_body = self._predict(self.body_clf, self.matrices.body_val)

        if fixed_alpha is not None:
            self.alpha = float(fixed_alpha)
            probs = combine_probs(p_title, p_body, self.alpha)
            f1 = float(f1_score(y_val, (probs >= 0.5).astype(int), zero_division=0))
            self.alpha_curve = [{"alpha": self.alpha, "val_f1": f1}]
        else:
            self.alpha, self.alpha_curve = self._select_alpha(p_title, p_body, y_val)
        p_fusion = combine_probs(p_title, p_body, self.alpha)

        concat_val = hstack([self.matrices.title_val, self.matrices.body_val]).tocsr()
        p_concat = self.concat_clf.predict_proba(concat_val)[:, 1]

        for condition, probs in (
            ("title_only", p_title),
            ("body_only", p_body),
            ("concat", p_concat),
            ("fusion", p_fusion),
        ):
            threshold, _ = tune_threshold(y_val, probs)
            self.thresholds[condition] = threshold
            self.threshold_tuned[condition] = threshold
        return self

    def _fit_branch(self, matrix: csr_matrix, y: np.ndarray, name: str) -> tuple[object, float]:
        fixed = self._fixed_c.get(name)
        if fixed is not None:
            clf = self._make_estimator(float(fixed)).fit(matrix, y)
            return clf, float(fixed)

        folds = self._safe_folds(y)
        if self.model_type == "lr":
            estimator = LogisticRegression(max_iter=1000, solver="liblinear")
            grid = {"C": self.c_grid}
        else:
            estimator = CalibratedClassifierCV(
                estimator=LinearSVC(), cv=folds, method="sigmoid"
            )
            grid = {"estimator__C": self.c_grid}

        search = GridSearchCV(
            estimator,
            grid,
            scoring="f1",
            cv=folds,
            n_jobs=1,
            refit=True,
        )
        search.fit(matrix, y)
        best = search.best_estimator_
        best_c = float(search.best_params_.get("C", search.best_params_.get("estimator__C", 1.0)))
        return best, best_c

    def _safe_folds(self, y: np.ndarray) -> int:
        counts = np.bincount(np.asarray(y).astype(int), minlength=2)
        smallest = int(counts[counts > 0].min()) if (counts > 0).any() else 2
        return max(2, min(self.cv_folds, smallest))

    def _make_estimator(self, c_value: float) -> object:
        if self.model_type == "lr":
            return LogisticRegression(C=c_value, max_iter=1000, solver="liblinear")
        return CalibratedClassifierCV(
            estimator=LinearSVC(C=c_value), cv=self.cv_folds, method="sigmoid"
        )

    def _select_alpha(
        self,
        p_title: np.ndarray,
        p_body: np.ndarray,
        y_val: np.ndarray,
    ) -> tuple[float, list[dict[str, float]]]:
        curve: list[dict[str, float]] = []
        for alpha in self.alpha_grid:
            probs = combine_probs(p_title, p_body, alpha)
            pred = (probs >= 0.5).astype(int)
            f1 = float(f1_score(y_val, pred, zero_division=0))
            curve.append({"alpha": float(alpha), "val_f1": f1})

        best_f1 = max(item["val_f1"] for item in curve)
        candidates = [item["alpha"] for item in curve if np.isclose(item["val_f1"], best_f1)]
        best_alpha = min(candidates, key=lambda alpha: abs(alpha - 0.5))
        return float(best_alpha), curve

    def _predict(self, clf: object, matrix: csr_matrix) -> np.ndarray:
        return clf.predict_proba(matrix)[:, 1]

    # ----------------------------------------------------------- predict/refit
    def predict_proba(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        """Return probabilities for every available condition."""
        if self.matrices is None or self.title_clf is None or self.body_clf is None:
            raise RuntimeError("call fit() before predict_proba()")
        title_x = self.matrices.title_vec.transform(df["title"].fillna("").astype(str))
        body_x = self.matrices.body_vec.transform(df[self.text_col].fillna("").astype(str))

        p_title = self._predict(self.title_clf, title_x)
        p_body = self._predict(self.body_clf, body_x)
        out: dict[str, np.ndarray] = {
            "title_only": p_title,
            "body_only": p_body,
            "fusion": combine_probs(p_title, p_body, self.alpha),
        }
        if self.concat_clf is not None:
            concat_x = hstack([title_x, body_x]).tocsr()
            out["concat"] = self._predict(self.concat_clf, concat_x)
        return out

    def refit_full(self, train_val: pd.DataFrame) -> DualTextModel:
        """Re-fit classifiers on train+val with the vectorisers and hyper-parameters fixed."""
        if self.matrices is None:
            raise RuntimeError("call fit() before refit_full()")
        y = train_val["label"].to_numpy().astype(int)
        title_x = self.matrices.title_vec.transform(train_val["title"].fillna("").astype(str))
        body_x = self.matrices.body_vec.transform(
            train_val[self.text_col].fillna("").astype(str)
        )

        self.title_clf = self._make_estimator(self.best_c["title"]).fit(title_x, y)
        self.body_clf = self._make_estimator(self.best_c["body"]).fit(body_x, y)
        concat_x = hstack([title_x, body_x]).tocsr()
        self.concat_clf = self._make_estimator(self.best_c["concat"]).fit(concat_x, y)
        return self

    # -------------------------------------------------------------- persistence
    def save(self, out_dir: Path) -> Path:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, out_dir / "model.joblib")
        metadata = {
            "model_type": self.model_type,
            "text_col": self.text_col,
            "best_c": self.best_c,
            "alpha": self.alpha,
            "alpha_curve": self.alpha_curve,
            "thresholds": self.thresholds,
        }
        (out_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        return out_dir

    @classmethod
    def load(cls, out_dir: Path) -> DualTextModel:
        return joblib.load(Path(out_dir) / "model.joblib")
