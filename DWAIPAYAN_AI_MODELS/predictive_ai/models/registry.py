"""
Model registry.

FastAPI workers must load heavy artifacts **once at startup**, never per
request. Sentence-BERT alone is ~90 MB and takes a couple of seconds to load;
doing that inside a request handler would make the API unusable.

This module owns the single in-process copy of all three models.
"""

from __future__ import annotations

import logging
from typing import Any

from predictive_ai.models.compliance import ComplianceIndexModel
from predictive_ai.models.delay_risk import DelayRiskIndexModel
from predictive_ai.models.duplication import DuplicationIndexModel

logger = logging.getLogger(__name__)


class ModelRegistry:
    def __init__(self) -> None:
        self.duplication: DuplicationIndexModel | None = None
        self.delay: DelayRiskIndexModel | None = None
        self.compliance: ComplianceIndexModel | None = None
        self.errors: dict[str, str] = {}

    def load_all(self, strict: bool = False) -> "ModelRegistry":
        """
        Load every artifact from disk.

        `strict=False` lets the API come up with two of three models working -
        useful during a hackathon demo where one artifact may still be training.
        The `/health` endpoint reports exactly what is missing.
        """
        for attr, cls in (
            ("duplication", DuplicationIndexModel),
            ("delay", DelayRiskIndexModel),
            ("compliance", ComplianceIndexModel),
        ):
            try:
                setattr(self, attr, cls.load())
                logger.info("Loaded %s", cls.name)
            except Exception as exc:
                self.errors[attr] = f"{type(exc).__name__}: {exc}"
                logger.error("Failed to load %s: %s", cls.name, exc)
                if strict:
                    raise
        return self

    def require(self, attr: str):
        model = getattr(self, attr, None)
        if model is None or not model.is_ready():
            raise RuntimeError(
                f"Model '{attr}' is not available. "
                f"Run `python train.py` to build the artifacts. "
                f"Details: {self.errors.get(attr, 'not loaded')}"
            )
        return model

    def health(self) -> dict[str, Any]:
        out: dict[str, Any] = {"status": "ok", "models": {}, "errors": self.errors}
        for attr in ("duplication", "delay", "compliance"):
            model = getattr(self, attr)
            out["models"][attr] = model.health() if model else {"ready": False}
        if not all(out["models"][a].get("ready") for a in out["models"]):
            out["status"] = "degraded"
        return out


REGISTRY = ModelRegistry()
