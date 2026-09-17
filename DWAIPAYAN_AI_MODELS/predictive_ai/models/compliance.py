"""
MODEL 3 - COMPLIANCE INDEX
==========================

Question answered: *"Does the money trail for this work look like the money
trail for every other work, or does it look strange?"*

Why unsupervised? Nobody hands us a labelled list of fraudulent MPLADS works.
Fraud is rare, evolving, and mostly undetected - so a supervised classifier has
nothing to learn from. Isolation Forest instead learns what *normal* spending
looks like and flags whatever is easy to isolate from the crowd.

Score direction (read this carefully)
-------------------------------------
`IsolationForest.score_samples()` returns a negative number where **higher means
more normal**. That is the opposite of what a risk dashboard wants, so:

    compliance_index = (max_train - s) / (max_train - min_train)

    -> 0.0 = the most routine spending pattern seen in training
    -> 1.0 = the most irregular spending pattern seen in training

`max_train` and `min_train` are frozen at fit time and any unseen extreme is
clipped, which guarantees the strict [0, 1] contract at inference.

A convenience field `compliance_health = 1 - compliance_index` is also returned
for anyone who prefers "higher is better".
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from predictive_ai.config import (
    IFOREST_PARAMS,
    SHAP_BACKGROUND_SIZE,
    SHAP_ENABLED,
    SHAP_MAX_EVALS,
    SHAP_TOP_DRIVERS,
)
from predictive_ai.data.mock_generator import COMPLIANCE_FEATURES
from predictive_ai.models.base import (
    BaseIndexModel,
    Driver,
    IndexResult,
    min_max_scale,
    rank_drivers,
)

logger = logging.getLogger(__name__)

# Plain-English labels so the dashboard never shows a raw column name.
FEATURE_LABELS: dict[str, str] = {
    "utilisation_ratio": "Funds utilised vs funds released",
    "days_to_first_payment": "Days from sanction to first payment",
    "num_payments": "Number of payment instalments",
    "avg_payment_size_lakh": "Average instalment size (lakh)",
    "round_number_payment_ratio": "Share of suspiciously round-figure payments",
    "vendor_concentration_ratio": "Share of district funds to a single vendor",
    "uc_submission_lag_days": "Utilisation Certificate submission delay",
    "fy_end_disbursal_ratio": "Share disbursed in the last month of the FY",
    "cost_revision_count": "Number of cost revisions",
    "payment_interval_cv": "Irregularity of payment timing",
}


class ComplianceIndexModel(BaseIndexModel):
    name = "compliance_index"
    version = "1.1.0"
    artifact_filename = "compliance_index.joblib"

    def __init__(self, features: list[str] | None = None) -> None:
        self.features = list(features or COMPLIANCE_FEATURES)
        self.model = None                      # sklearn IsolationForest
        self.score_min: float = 0.0            # frozen at fit time
        self.score_max: float = 1.0
        self.defaults: dict[str, float] = {}   # training medians, for imputation
        self.background: np.ndarray | None = None   # SHAP reference sample
        self._explainer = None
        self.metrics: dict[str, float] = {}

    # -- training ----------------------------------------------------------- #

    def fit(self, df: pd.DataFrame, eval_label: str | None = "is_seeded_anomaly") -> "ComplianceIndexModel":
        from sklearn.ensemble import IsolationForest

        missing = [c for c in self.features if c not in df.columns]
        if missing:
            raise KeyError(f"Missing columns in financial frame: {missing}")

        X = df[self.features].astype(float)
        X_arr = X.to_numpy()   # fit on the array so sklearn records no feature names,
                               # which keeps single-row inference warning-free

        self.model = IsolationForest(**IFOREST_PARAMS)
        self.model.fit(X_arr)                    # unsupervised: no labels used

        raw = self.model.score_samples(X_arr)    # higher = more normal
        self.score_min = float(raw.min())
        self.score_max = float(raw.max())
        self.defaults = {c: float(X[c].median()) for c in self.features}

        rng = np.random.default_rng(IFOREST_PARAMS["random_state"])
        take = min(SHAP_BACKGROUND_SIZE, len(X))
        self.background = X.to_numpy()[rng.choice(len(X), size=take, replace=False)]

        index = self._to_index(raw)
        self.metrics = {
            "n_train": int(len(X)),
            "raw_score_min": self.score_min,
            "raw_score_max": self.score_max,
            "flagged_rate_at_0_5": float((index >= 0.5).mean()),
            "flagged_rate_at_0_75": float((index >= 0.75).mean()),
        }

        # Optional sanity check against seeded anomalies - evaluation only.
        if eval_label and eval_label in df.columns:
            from sklearn.metrics import average_precision_score, roc_auc_score

            y = df[eval_label].astype(int).to_numpy()
            if y.sum() > 0:
                self.metrics["eval_roc_auc"] = float(roc_auc_score(y, index))
                self.metrics["eval_average_precision"] = float(average_precision_score(y, index))

        logger.info("Compliance model trained: %s", self.metrics)
        return self

    # -- scaling ------------------------------------------------------------ #

    def _to_index(self, raw: np.ndarray | float) -> np.ndarray | float:
        """
        Strict Min-Max mapping of the raw anomaly score onto [0, 1], inverted so
        that 1.0 = most anomalous.
        """
        if np.isscalar(raw):
            return 1.0 - min_max_scale(float(raw), self.score_min, self.score_max)
        arr = np.asarray(raw, dtype=float)
        span = max(self.score_max - self.score_min, 1e-12)
        return 1.0 - np.clip((arr - self.score_min) / span, 0.0, 1.0)

    # -- persistence -------------------------------------------------------- #

    def save(self) -> Path:
        if self.model is None:
            raise RuntimeError("Call fit() before save()")
        joblib.dump(
            {
                "version": self.version,
                "model": self.model,
                "features": self.features,
                "score_min": self.score_min,
                "score_max": self.score_max,
                "defaults": self.defaults,
                "background": self.background,
                "metrics": self.metrics,
            },
            self.artifact_path,
            compress=3,
        )
        logger.info("Saved compliance artifact -> %s", self.artifact_path)
        return self.artifact_path

    @classmethod
    def load(cls) -> "ComplianceIndexModel":
        obj = cls()
        blob = joblib.load(obj.artifact_path)
        obj.model = blob["model"]
        obj.features = blob["features"]
        obj.score_min = blob["score_min"]
        obj.score_max = blob["score_max"]
        obj.defaults = blob.get("defaults", {})
        obj.background = blob.get("background")
        obj.metrics = blob.get("metrics", {})
        return obj

    def is_ready(self) -> bool:
        return self.model is not None

    # -- feature assembly --------------------------------------------------- #

    def _vectorise(self, payload: dict[str, Any]) -> tuple[np.ndarray, list[str]]:
        row, imputed = [], []
        for feat in self.features:
            val = payload.get(feat)
            if val is None or (isinstance(val, float) and not np.isfinite(val)):
                val = self.defaults.get(feat, 0.0)
                imputed.append(feat)
            row.append(float(val))
        return np.array([row], dtype=float), imputed

    # -- inference ---------------------------------------------------------- #

    def score(self, payload: dict[str, Any], explain: bool = True) -> IndexResult:
        if not self.is_ready():
            raise RuntimeError("ComplianceIndexModel is not fitted/loaded")

        X, imputed = self._vectorise(payload)
        raw = float(self.model.score_samples(X)[0])
        index = float(self._to_index(raw))
        is_outlier = bool(self.model.predict(X)[0] == -1)

        drivers = self._explain(X) if (explain and SHAP_ENABLED) else []

        return IndexResult(
            index_name=self.name,
            score=index,
            model_version=self.version,
            drivers=drivers,
            narrative=self._narrative(index, drivers),
            meta={
                "raw_anomaly_score": round(raw, 6),
                "scaler": {"min": self.score_min, "max": self.score_max, "method": "min_max_inverted"},
                "isolation_forest_flag": is_outlier,
                "compliance_health": round(1.0 - index, 4),
                "imputed_features": imputed,
                "training_metrics": self.metrics,
            },
        )

    def score_batch(self, df: pd.DataFrame) -> np.ndarray:
        X = df.reindex(columns=self.features).astype(float)
        for feat in self.features:
            X[feat] = X[feat].fillna(self.defaults.get(feat, 0.0))
        return np.asarray(self._to_index(self.model.score_samples(X.to_numpy())))

    # -- explainability ----------------------------------------------------- #

    def _explain(self, X: np.ndarray) -> list[Driver]:
        """
        SHAP over the Isolation Forest.

        TreeExplainer works directly on the forest and is fast. Its values
        explain `score_samples`, where *higher = more normal*, so we negate them
        to express contributions to the compliance **risk** index. If the tree
        path fails on a given sklearn/shap combination we fall back to
        KernelExplainer over the index function itself, which is slower but
        always correct.
        """
        labels = [FEATURE_LABELS.get(f, f) for f in self.features]
        values = X[0].tolist()

        try:
            import shap

            if self._explainer is None:
                self._explainer = shap.TreeExplainer(self.model)
            sv = np.asarray(self._explainer.shap_values(X, check_additivity=False))
            if sv.ndim == 3:
                sv = sv[..., -1]
            return rank_drivers(labels, values, -sv[0], SHAP_TOP_DRIVERS)
        except Exception as exc:
            logger.warning("TreeSHAP unavailable for IsolationForest (%s); using KernelSHAP", exc)

        try:
            import shap

            def f(data: np.ndarray) -> np.ndarray:
                return np.asarray(self._to_index(self.model.score_samples(np.atleast_2d(data))))

            bg = self.background if self.background is not None else X
            kernel = shap.KernelExplainer(f, shap.kmeans(bg, min(10, len(bg))))
            sv = np.asarray(kernel.shap_values(X[0], nsamples=SHAP_MAX_EVALS, silent=True))
            return rank_drivers(labels, values, sv.ravel(), SHAP_TOP_DRIVERS)
        except Exception as exc:  # pragma: no cover
            logger.warning("KernelSHAP also failed for compliance model: %s", exc)
            return []

    @staticmethod
    def _narrative(score: float, drivers: list[Driver]) -> str:
        top = [d.feature for d in drivers[:2] if d.direction == "increases_risk"]
        tail = f" Most unusual: {', '.join(top)}." if top else ""
        if score >= 0.75:
            return f"Spending pattern is a strong statistical outlier - recommend a physical audit.{tail}"
        if score >= 0.50:
            return f"Spending pattern deviates noticeably from comparable works.{tail}"
        if score >= 0.25:
            return f"Minor deviations from the norm; no immediate action needed.{tail}"
        return "Financial behaviour is consistent with comparable MPLADS works."
