"""
Pydantic v2 request / response contracts.

These double as the OpenAPI docs the React team codes against: run the API and
open http://localhost:8000/docs.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


# --------------------------------------------------------------------------- #
# Requests
# --------------------------------------------------------------------------- #


class DuplicationRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    description: str = Field(..., min_length=3, description="New work description text")
    restrict_to: dict[str, str] | None = Field(
        default=None,
        description="Optional blocking filter, e.g. {'district': 'Nadia'}",
        examples=[{"district": "Nadia"}],
    )
    top_k: int = Field(default=5, ge=1, le=25)
    explain: bool = True


class DelayFeatures(BaseModel):
    """Everything below is knowable at sanction time. Omitted fields are imputed."""

    model_config = ConfigDict(extra="ignore")

    sanctioned_cost_lakh: float | None = Field(default=None, ge=0)
    planned_duration_days: float | None = Field(default=None, ge=0)
    sanction_to_start_lag_days: float | None = Field(default=None, ge=0)
    num_bidders: float | None = Field(default=None, ge=0)
    vendor_past_overrun_rate: float | None = Field(default=None, ge=0, le=1)
    vendor_completed_projects: float | None = Field(default=None, ge=0)
    district_past_delay_rate: float | None = Field(default=None, ge=0, le=1)
    monsoon_overlap_months: float | None = Field(default=None, ge=0, le=12)
    fund_release_tranches: float | None = Field(default=None, ge=0)
    rate_deviation_pct: float | None = None
    is_election_year: float | None = Field(default=None, ge=0, le=1)
    category_code: float | None = Field(default=None, ge=0)


class DelayRequest(DelayFeatures):
    explain: bool = True


class ComplianceFeatures(BaseModel):
    model_config = ConfigDict(extra="ignore")

    utilisation_ratio: float | None = Field(default=None, ge=0)
    days_to_first_payment: float | None = Field(default=None, ge=0)
    num_payments: float | None = Field(default=None, ge=0)
    avg_payment_size_lakh: float | None = Field(default=None, ge=0)
    round_number_payment_ratio: float | None = Field(default=None, ge=0, le=1)
    vendor_concentration_ratio: float | None = Field(default=None, ge=0, le=1)
    uc_submission_lag_days: float | None = Field(default=None, ge=0)
    fy_end_disbursal_ratio: float | None = Field(default=None, ge=0, le=1)
    cost_revision_count: float | None = Field(default=None, ge=0)
    payment_interval_cv: float | None = Field(default=None, ge=0)


class ComplianceRequest(ComplianceFeatures):
    explain: bool = True


class CompositeRequest(BaseModel):
    """One work item, scored end-to-end through all three models."""

    model_config = ConfigDict(extra="ignore")

    work_id: str | None = None
    description: str = Field(..., min_length=3)
    restrict_to: dict[str, str] | None = None
    project_metrics: DelayFeatures = Field(default_factory=DelayFeatures)
    financials: ComplianceFeatures = Field(default_factory=ComplianceFeatures)
    explain: bool = True


# --------------------------------------------------------------------------- #
# Responses
# --------------------------------------------------------------------------- #


class DriverOut(BaseModel):
    feature: str
    value: Any
    contribution: float
    contribution_pct: float
    direction: str


class IndexResponse(BaseModel):
    index_name: str
    score: float = Field(..., ge=0.0, le=1.0)
    band: str
    model_version: str
    narrative: str
    drivers: list[DriverOut] = []
    meta: dict[str, Any] = {}


class CompositeResponse(BaseModel):
    work_id: str | None = None
    composite_fraud_index: float = Field(..., ge=0.0, le=1.0)
    risk_index_100: int
    band: str
    weights: dict[str, float]
    critical_override_applied: bool
    narrative: str
    top_drivers: list[dict[str, Any]] = []
    components: dict[str, IndexResponse]
