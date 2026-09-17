"""
COMPOSITE FRAUD INDEX
=====================

Fuses the three independent indices into the single 0.0-1.0 number the React
dashboard shows as an 0-100 Risk Index.

Two rules make the fusion defensible to an auditor:

1. **Weighted mean** - transparent, tunable per state/scheme, and every weight
   is printed in the API response. No black box on top of the black boxes.

2. **Critical override** - a weighted mean can hide one catastrophic signal
   behind two calm ones. If any single index is >= 0.90 the composite is floored
   so the item can never drop out of the HIGH band. Real audit teams would
   rather chase a false positive than miss a certain duplicate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from predictive_ai.config import COMPOSITE_WEIGHTS, CompositeWeights
from predictive_ai.models.base import IndexResult, band_for, clip01

CRITICAL_SINGLE_INDEX = 0.90
CRITICAL_FLOOR = 0.75


@dataclass
class CompositeResult:
    score: float
    band: str
    risk_index_100: int
    weights: dict[str, float]
    components: dict[str, dict[str, Any]]
    override_applied: bool = False
    narrative: str = ""
    top_drivers: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "composite_fraud_index": round(self.score, 4),
            "risk_index_100": self.risk_index_100,
            "band": self.band,
            "weights": self.weights,
            "critical_override_applied": self.override_applied,
            "narrative": self.narrative,
            "top_drivers": self.top_drivers,
            "components": self.components,
        }


def fuse(
    duplication: IndexResult,
    delay: IndexResult,
    compliance: IndexResult,
    weights: CompositeWeights = COMPOSITE_WEIGHTS,
) -> CompositeResult:
    w = weights.as_dict()
    parts = {
        "duplication": duplication,
        "delay_overrun": delay,
        "compliance": compliance,
    }

    weighted = sum(w[k] * r.score for k, r in parts.items())
    score = clip01(weighted)

    override = any(r.score >= CRITICAL_SINGLE_INDEX for r in parts.values())
    if override:
        score = max(score, CRITICAL_FLOOR)

    # Merge per-model drivers, scaling each contribution by its model's weight so
    # the dashboard can show one ranked "why is this flagged" list.
    merged: list[dict[str, Any]] = []
    for key, res in parts.items():
        for d in res.drivers:
            merged.append(
                {
                    "source_model": key,
                    "feature": d.feature,
                    "value": d.value,
                    "direction": d.direction,
                    "weighted_share_pct": round(d.contribution_pct * w[key], 2),
                }
            )
    merged.sort(key=lambda x: -x["weighted_share_pct"])

    return CompositeResult(
        score=score,
        band=band_for(score),
        risk_index_100=int(round(score * 100)),
        weights=w,
        components={k: r.to_dict() for k, r in parts.items()},
        override_applied=override,
        narrative=_narrative(score, parts, override),
        top_drivers=merged[:6],
    )


def _narrative(score: float, parts: dict[str, IndexResult], override: bool) -> str:
    worst_key, worst = max(parts.items(), key=lambda kv: kv[1].score)
    pretty = worst_key.replace("_", " ")
    head = f"Composite risk {int(round(score * 100))}/100 ({band_for(score)})."
    body = f" Dominant signal: {pretty} at {worst.score:.2f}."
    tail = (
        " One sub-index exceeded the critical threshold, so the composite was "
        "floored to HIGH regardless of the other two."
        if override
        else ""
    )
    return head + body + tail
