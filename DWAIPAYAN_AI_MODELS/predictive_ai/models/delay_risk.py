"""
MODEL 2 - DELAY / COST-OVERRUN INDEX
====================================

Question answered: *"Given what we know at sanction time, what is the
probability that this work finishes late or over budget?"*

Method
------
Gradient-boosted trees (XGBoost) trained as a binary classifier on historical
works. The index is simply `predict_proba(x)[:, 1]`, which is already a
0.0-1.0 quantity - no rescaling needed, and that is exactly why a probabilistic
classifier is the right tool for this sub-problem.

One refinement over the naive approach: raw boosted-tree probabilities are
often over-confident. We fit an isotonic regression on a held-out split so that
"0.70" really does mean roughly 70 out of 100 such works went bad. Auditors act
on these numbers, so calibration matters.

Explainability
--------------
`shap.TreeExplainer` gives exact Shapley values for tree ensembles in
milliseconds - no sampling, no approximation. For each work item the API can
state "vendor_past_overrun_rate contributed +0.9 log-odds, 34% of the total
push towards HIGH risk".
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from predictive_ai.config import (
    RANDOM_SEED,
    SHAP_ENABLED,
    SHAP_TOP_DRIVERS,
    XGB_PARAMS,
)
from predictive_ai.data.mock_generator import DELAY_FEATURES
from predictive_ai.models.base import (
    BaseIndexModel,
    Driver,
    IndexResult,
    clip01,
    rank_drivers,
)

logger = logging.getLogger(__name__)

# Sensible neutral defaults so a partially-filled request never crashes the API.
FEATURE_DEFAULTS: dict[str, float] = {
    "sanctioned_cost_lakh": 12.0,
    "planned_duration_days": 180.0,
    "sanction_to_start_lag_days": 45.0,
    "num_bidders": 4.0,
    "vendor_past_overrun_rate": 0.25,
    "vendor_completed_projects": 12.0,
    "district_past_delay_rate": 0.35,
    "monsoon_overlap_months": 1.0,
    "fund_release_tranches": 2.0,
    "rate_deviation_pct": 0.0,
    "is_election_year": 0.0,
    "category_code": 0.0,
}


class DelayRiskIndexModel(BaseIndexModel):
    name = "delay_cost_overrun_index"
    version = "1.1.0"
    artifact_filename = "delay_risk_index.joblib"

    def __init__(self, features: list[str] | None = None, calibrate: bool = True) -> None:
        self.features = list(features or DELAY_FEATURES)
        self.calibrate = calibrate
        self.model = None                 # xgboost.XGBClassifier
        self.calibrator = None            # sklearn IsotonicRegression | None
        self._explainer = None            # shap.TreeExplainer (lazy)
        self.metrics: dict[str, float] = {}

    # -- training ----------------------------------------------------------- #

    def fit(
        self,
        df: pd.DataFrame,
        target: str = "is_delayed_or_overrun",
        test_size: float = 0.2,
    ) -> "DelayRiskIndexModel":
        from sklearn.metrics import (
            average_precision_score,
            brier_score_loss,
            roc_auc_score,
        )
        from sklearn.model_selection import train_test_split
        import xgboost as xgb

        missing = [c for c in self.features + [target] if c not in df.columns]
        if missing:
            raise KeyError(f"Missing columns in training frame: {missing}")

        X = df[self.features].astype(float)
        y = df[target].astype(int)

        X_tr, X_va, y_tr, y_va = train_test_split(
            X, y, test_size=test_size, random_state=RANDOM_SEED, stratify=y
        )

        # Handle class imbalance without resampling.
        pos = max(int(y_tr.sum()), 1)
        neg = max(int(len(y_tr) - pos), 1)
        params = dict(XGB_PARAMS)
        params["scale_pos_weight"] = neg / pos

        self.model = xgb.XGBClassifier(**params)
        self.model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

        raw_va = self.model.predict_proba(X_va)[:, 1]

        if self.calibrate:
            from sklearn.isotonic import IsotonicRegression

            self.calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            self.calibrator.fit(raw_va, y_va)
            cal_va = self.calibrator.predict(raw_va)
        else:
            cal_va = raw_va

        self.metrics = {
            "roc_auc": float(roc_auc_score(y_va, raw_va)),
            "average_precision": float(average_precision_score(y_va, raw_va)),
            "brier_raw": float(brier_score_loss(y_va, raw_va)),
            "brier_calibrated": float(brier_score_loss(y_va, cal_va)),
            "n_train": int(len(X_tr)),
            "n_valid": int(len(X_va)),
            "positive_rate": float(y.mean()),
        }
        logger.info("Delay model trained: %s", self.metrics)
        return self

    # -- persistence -------------------------------------------------------- #

    def save(self) -> Path:
        if self.model is None:
            raise RuntimeError("Call fit() before save()")
        joblib.dump(
            {
                "version": self.version,
                "model": self.model,
                "calibrator": self.calibrator,
                "features": self.features,
                "metrics": self.metrics,
            },
            self.artifact_path,
            compress=3,
        )
        logger.info("Saved delay-risk artifact -> %s", self.artifact_path)
        return self.artifact_path

    @classmethod
    def load(cls) -> "DelayRiskIndexModel":
        obj = cls()
        blob = joblib.load(obj.artifact_path)
        obj.model = blob["model"]
        obj.calibrator = blob.get("calibrator")
        obj.features = blob["features"]
        obj.metrics = blob.get("metrics", {})
        return obj

    def is_ready(self) -> bool:
        return self.model is not None

    # -- feature assembly --------------------------------------------------- #

    def _vectorise(self, payload: dict[str, Any]) -> tuple[pd.DataFrame, list[str]]:
        """Build a single-row frame in the exact training column order."""
        row, imputed = {}, []
        for feat in self.features:
            val = payload.get(feat)
            if val is None or (isinstance(val, float) and not np.isfinite(val)):
                val = FEATURE_DEFAULTS.get(feat, 0.0)
                imputed.append(feat)
            row[feat] = float(val)
        return pd.DataFrame([row], columns=self.features), imputed

    # -- inference ---------------------------------------------------------- #

    def score(self, payload: dict[str, Any], explain: bool = True) -> IndexResult:
        if not self.is_ready():
            raise RuntimeError("DelayRiskIndexModel is not fitted/loaded")

        X, imputed = self._vectorise(payload)

        raw_prob = float(self.model.predict_proba(X)[0, 1])
        final = float(self.calibrator.predict([raw_prob])[0]) if self.calibrator else raw_prob
        final = clip01(final)

        drivers = self._explain(X) if (explain and SHAP_ENABLED) else []

        return IndexResult(
            index_name=self.name,
            score=final,
            model_version=self.version,
            drivers=drivers,
            narrative=self._narrative(final, drivers),
            meta={
                "raw_probability": round(raw_prob, 4),
                "calibrated": self.calibrator is not None,
                "imputed_features": imputed,
                "training_metrics": self.metrics,
            },
        )

    def score_batch(self, df: pd.DataFrame) -> np.ndarray:
        """Vectorised scoring for the nightly Celery re-scoring job."""
        X = df.reindex(columns=self.features)
        for feat in self.features:
            X[feat] = X[feat].astype(float).fillna(FEATURE_DEFAULTS.get(feat, 0.0))
        raw = self.model.predict_proba(X)[:, 1]
        return np.clip(self.calibrator.predict(raw) if self.calibrator else raw, 0.0, 1.0)

    # -- explainability ----------------------------------------------------- #

    def _get_explainer(self):
        if self._explainer is None:
            import shap

            self._explainer = shap.TreeExplainer(self.model)
        return self._explainer

    def _explain(self, X: pd.DataFrame) -> list[Driver]:
        try:
            explainer = self._get_explainer()
            sv = explainer.shap_values(X)
            sv = np.asarray(sv)
            if sv.ndim == 3:                      # (n, features, classes)
                sv = sv[..., -1]
            return rank_drivers(
                feature_names=list(X.columns),
                values=X.iloc[0].tolist(),
                shap_values=sv[0],
                top_k=SHAP_TOP_DRIVERS,
            )
        except Exception as exc:  # pragma: no cover
            logger.warning("TreeSHAP failed for delay model: %s", exc)
            return []

    def global_importance(self, df: pd.DataFrame, sample: int = 500) -> dict[str, float]:
        """Mean |SHAP| per feature - the global view for the 'Model Insights' tab."""
        import shap

        X = df[self.features].astype(float).head(sample)
        sv = np.asarray(shap.TreeExplainer(self.model).shap_values(X))
        if sv.ndim == 3:
            sv = sv[..., -1]
        mean_abs = np.abs(sv).mean(axis=0)
        return dict(
            sorted(
                {f: float(v) for f, v in zip(self.features, mean_abs)}.items(),
                key=lambda kv: -kv[1],
            )
        )

    @staticmethod
    def _narrative(score: float, drivers: list[Driver]) -> str:
        pct = int(round(score * 100))
        top = ", ".join(d.feature for d in drivers[:2] if d.direction == "increases_risk")
        tail = f" Main pressure from: {top}." if top else ""
        if score >= 0.75:
            return f"About {pct}% modelled chance of delay or cost overrun - critical.{tail}"
        if score >= 0.50:
            return f"About {pct}% modelled chance of slippage - schedule a mid-term review.{tail}"
        if score >= 0.25:
            return f"Moderate slippage risk (~{pct}%); routine monitoring is sufficient.{tail}"
        return f"Low slippage risk (~{pct}%) on the evidence available at sanction."
