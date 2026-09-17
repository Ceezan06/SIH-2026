"""
Shared contracts for every index model.

The whole point of this file: the FastAPI layer should not care *which*
algorithm produced a score. Every model returns the same `IndexResult`
object, so `/score/duplication`, `/score/delay` and `/score/compliance`
all serialise identically and the React dashboard can render one component
three times.
"""

from __future__ import annotations

import abc
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from predictive_ai.config import ARTIFACT_DIR, RISK_BANDS

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Small numeric helpers
# --------------------------------------------------------------------------- #


def clip01(value: float) -> float:
    """Force any float into the closed interval [0.0, 1.0]."""
    if not np.isfinite(value):
        return 0.0
    return float(min(1.0, max(0.0, value)))


def min_max_scale(value: float, lo: float, hi: float) -> float:
    """
    Strict Min-Max scaling into [0, 1], safe against a degenerate range.

    `lo` and `hi` are learned at *fit* time and frozen. At inference an unseen
    extreme value is clipped rather than allowed to escape the range - this is
    what keeps the API contract "always between 0.0 and 1.0" true.
    """
    if hi - lo < 1e-12:
        return 0.0
    return clip01((value - lo) / (hi - lo))


def band_for(score: float) -> str:
    """Map a 0-1 score onto LOW / MODERATE / HIGH / CRITICAL."""
    for threshold, label in RISK_BANDS:
        if score < threshold:
            return label
    return RISK_BANDS[-1][1]


# --------------------------------------------------------------------------- #
# Output contract
# --------------------------------------------------------------------------- #


@dataclass
class Driver:
    """One explainability row: 'why did this item score the way it did?'"""

    feature: str
    value: Any
    contribution: float          # signed SHAP value, in the model's native units
    contribution_pct: float      # |shap| share of total |shap|, 0-100
    direction: str               # "increases_risk" | "reduces_risk"

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # numpy scalars are not JSON serialisable
        if isinstance(d["value"], (np.generic,)):
            d["value"] = d["value"].item()
        return d


@dataclass
class IndexResult:
    """Uniform response object returned by all three models."""

    index_name: str
    score: float                              # ALWAYS in [0.0, 1.0]
    band: str = ""
    model_version: str = "0.0.0"
    drivers: list[Driver] = field(default_factory=list)
    narrative: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.score = clip01(self.score)
        if not self.band:
            self.band = band_for(self.score)

    def to_dict(self) -> dict[str, Any]:
        return {
            "index_name": self.index_name,
            "score": round(self.score, 4),
            "band": self.band,
            "model_version": self.model_version,
            "narrative": self.narrative,
            "drivers": [d.to_dict() for d in self.drivers],
            "meta": _jsonify(self.meta),
        }


def _jsonify(obj: Any) -> Any:
    """Recursively convert numpy types so FastAPI can serialise the payload."""
    if isinstance(obj, dict):
        return {k: _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def rank_drivers(
    feature_names: list[str],
    values: list[Any],
    shap_values: np.ndarray,
    top_k: int,
) -> list[Driver]:
    """Turn a raw SHAP vector into the top-k human-readable drivers."""
    shap_values = np.asarray(shap_values, dtype=float).ravel()
    total = float(np.abs(shap_values).sum())
    total = total if total > 1e-12 else 1.0

    order = np.argsort(-np.abs(shap_values))[:top_k]
    return [
        Driver(
            feature=feature_names[i],
            value=values[i],
            contribution=round(float(shap_values[i]), 6),
            contribution_pct=round(100.0 * abs(float(shap_values[i])) / total, 2),
            direction="increases_risk" if shap_values[i] > 0 else "reduces_risk",
        )
        for i in order
    ]


# --------------------------------------------------------------------------- #
# Abstract model
# --------------------------------------------------------------------------- #


class BaseIndexModel(abc.ABC):
    """
    Contract every index model implements.

    Lifecycle:
        model = SomeIndexModel()
        model.fit(training_dataframe)      # offline, run by train.py
        model.save()                       # writes to artifacts/
        ...
        model = SomeIndexModel.load()      # at FastAPI startup
        result = model.score(payload)      # per request, returns IndexResult
    """

    name: str = "base_index"
    version: str = "1.0.0"
    artifact_filename: str = "base.joblib"

    # -- persistence -------------------------------------------------------- #

    @property
    def artifact_path(self) -> Path:
        return ARTIFACT_DIR / self.artifact_filename

    @abc.abstractmethod
    def fit(self, *args: Any, **kwargs: Any) -> "BaseIndexModel":
        ...

    @abc.abstractmethod
    def save(self) -> Path:
        ...

    @classmethod
    @abc.abstractmethod
    def load(cls) -> "BaseIndexModel":
        ...

    # -- inference ---------------------------------------------------------- #

    @abc.abstractmethod
    def score(self, payload: dict[str, Any], explain: bool = True) -> IndexResult:
        """Return an IndexResult whose `.score` is guaranteed to be in [0, 1]."""

    # -- health ------------------------------------------------------------- #

    def is_ready(self) -> bool:
        return True

    def health(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "ready": self.is_ready(),
            "artifact": str(self.artifact_path),
            "artifact_exists": self.artifact_path.exists(),
        }
