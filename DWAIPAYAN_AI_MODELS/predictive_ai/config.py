"""
Central configuration for the MPLADS Predictive AI stage.

Everything that a deployment might want to change (model names, artifact
locations, composite weights, risk thresholds) lives here so that no other
module hard-codes a magic number.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #

BASE_DIR = Path(os.getenv("PAI_BASE_DIR", Path(__file__).resolve().parent.parent))
ARTIFACT_DIR = Path(os.getenv("PAI_ARTIFACT_DIR", BASE_DIR / "artifacts"))
DATA_DIR = Path(os.getenv("PAI_DATA_DIR", BASE_DIR / "data"))

ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Global determinism
# --------------------------------------------------------------------------- #

RANDOM_SEED = int(os.getenv("PAI_SEED", "42"))

# --------------------------------------------------------------------------- #
# Model 1 - Duplication Index (Sentence-BERT)
# --------------------------------------------------------------------------- #

SBERT_MODEL_NAME = os.getenv("PAI_SBERT_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
SBERT_BATCH_SIZE = int(os.getenv("PAI_SBERT_BATCH", "64"))

# How many nearest historical works to return alongside the score.
DUPLICATION_TOP_K = int(os.getenv("PAI_DUP_TOP_K", "5"))

# Below this cosine similarity we treat the pair as "unrelated" and floor the
# index to 0.0. Sentence-BERT gives ~0.2-0.3 similarity even to random text,
# so calibrating the floor stops every work item from looking mildly duplicated.
DUPLICATION_FLOOR = float(os.getenv("PAI_DUP_FLOOR", "0.30"))

# At or above this similarity the index saturates at 1.0.
DUPLICATION_CEILING = float(os.getenv("PAI_DUP_CEILING", "0.95"))

# --------------------------------------------------------------------------- #
# Model 2 - Delay / Cost-Overrun Index (XGBoost)
# --------------------------------------------------------------------------- #

XGB_PARAMS = {
    "n_estimators": 400,
    "max_depth": 5,
    "learning_rate": 0.06,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "min_child_weight": 3,
    "reg_lambda": 1.5,
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "tree_method": "hist",
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
}

# --------------------------------------------------------------------------- #
# Model 3 - Compliance Index (Isolation Forest)
# --------------------------------------------------------------------------- #

IFOREST_PARAMS = {
    "n_estimators": 300,
    "max_samples": "auto",
    "contamination": 0.06,   # our prior belief about how much of MPLADS spend is irregular
    "max_features": 1.0,
    "bootstrap": False,
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
}

# --------------------------------------------------------------------------- #
# Composite Fraud Index
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class CompositeWeights:
    """Weights used to fuse the three indices into one 0.0-1.0 score."""

    duplication: float = 0.35
    delay_overrun: float = 0.35
    compliance: float = 0.30

    def __post_init__(self) -> None:
        total = self.duplication + self.delay_overrun + self.compliance
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Composite weights must sum to 1.0, got {total:.4f}")

    def as_dict(self) -> dict[str, float]:
        return {
            "duplication": self.duplication,
            "delay_overrun": self.delay_overrun,
            "compliance": self.compliance,
        }


COMPOSITE_WEIGHTS = CompositeWeights(
    duplication=float(os.getenv("PAI_W_DUP", "0.35")),
    delay_overrun=float(os.getenv("PAI_W_DELAY", "0.35")),
    compliance=float(os.getenv("PAI_W_COMP", "0.30")),
)

# --------------------------------------------------------------------------- #
# Risk bands - shared by every index so the dashboard colours stay consistent
# --------------------------------------------------------------------------- #

RISK_BANDS: tuple[tuple[float, str], ...] = (
    (0.25, "LOW"),
    (0.50, "MODERATE"),
    (0.75, "HIGH"),
    (1.01, "CRITICAL"),
)

# --------------------------------------------------------------------------- #
# Explainability
# --------------------------------------------------------------------------- #

SHAP_ENABLED = os.getenv("PAI_SHAP_ENABLED", "1") == "1"
SHAP_TOP_DRIVERS = int(os.getenv("PAI_SHAP_TOP_K", "5"))

# KernelExplainer / Text explainer are slow. Cap the work they are allowed to do
# so a single API request never blocks the event loop for minutes.
SHAP_MAX_EVALS = int(os.getenv("PAI_SHAP_MAX_EVALS", "200"))
SHAP_BACKGROUND_SIZE = int(os.getenv("PAI_SHAP_BG", "100"))
