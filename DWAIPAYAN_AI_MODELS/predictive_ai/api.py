"""
FastAPI service for the MPLADS Predictive AI stage.

Run:
    uvicorn predictive_ai.api:app --reload --port 8000

Endpoints:
    GET  /health                 - per-model readiness
    POST /score/duplication      - Model 1
    POST /score/delay            - Model 2
    POST /score/compliance       - Model 3
    POST /score/composite        - all three + weighted Composite Fraud Index
    GET  /explain/global/delay   - global SHAP importance for the insights tab

Concurrency note
----------------
SHAP and Sentence-BERT are CPU-bound and block the event loop. Every handler
therefore runs the model call in a worker thread via `run_in_threadpool`. In the
full architecture the heavy bulk jobs go to Celery + Redis instead; these
endpoints stay for interactive, single-item scoring.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware

from predictive_ai.models.composite import fuse
from predictive_ai.models.registry import REGISTRY
from predictive_ai.schemas import (
    ComplianceRequest,
    CompositeRequest,
    CompositeResponse,
    DelayRequest,
    DuplicationRequest,
    IndexResponse,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading model artifacts...")
    REGISTRY.load_all(strict=False)
    logger.info("Startup complete: %s", REGISTRY.health()["status"])
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="MPLADS Predictive AI Stage",
    description=(
        "Duplication, Delay/Cost-Overrun and Compliance indices for MPLADS works, "
        "fused into a single explainable Composite Fraud Index."
    ),
    version="1.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # tighten before any public deployment
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------- #
# Health
# --------------------------------------------------------------------------- #


@app.get("/health", tags=["ops"])
async def health() -> dict:
    return REGISTRY.health()


# --------------------------------------------------------------------------- #
# Individual indices
# --------------------------------------------------------------------------- #


@app.post("/score/duplication", response_model=IndexResponse, tags=["scoring"])
async def score_duplication(req: DuplicationRequest) -> dict:
    model = _require("duplication")
    result = await run_in_threadpool(
        model.score,
        {"description": req.description, "restrict_to": req.restrict_to, "top_k": req.top_k},
        req.explain,
    )
    return result.to_dict()


@app.post("/score/delay", response_model=IndexResponse, tags=["scoring"])
async def score_delay(req: DelayRequest) -> dict:
    model = _require("delay")
    payload = req.model_dump(exclude={"explain"}, exclude_none=True)
    result = await run_in_threadpool(model.score, payload, req.explain)
    return result.to_dict()


@app.post("/score/compliance", response_model=IndexResponse, tags=["scoring"])
async def score_compliance(req: ComplianceRequest) -> dict:
    model = _require("compliance")
    payload = req.model_dump(exclude={"explain"}, exclude_none=True)
    result = await run_in_threadpool(model.score, payload, req.explain)
    return result.to_dict()


# --------------------------------------------------------------------------- #
# Composite
# --------------------------------------------------------------------------- #


@app.post("/score/composite", response_model=CompositeResponse, tags=["scoring"])
async def score_composite(req: CompositeRequest) -> dict:
    dup_model = _require("duplication")
    delay_model = _require("delay")
    comp_model = _require("compliance")

    def _run():
        dup = dup_model.score(
            {"description": req.description, "restrict_to": req.restrict_to},
            req.explain,
        )
        delay = delay_model.score(req.project_metrics.model_dump(exclude_none=True), req.explain)
        comp = comp_model.score(req.financials.model_dump(exclude_none=True), req.explain)
        return fuse(dup, delay, comp)

    composite = await run_in_threadpool(_run)
    payload = composite.to_dict()
    payload["work_id"] = req.work_id
    return payload


# --------------------------------------------------------------------------- #
# Global explainability
# --------------------------------------------------------------------------- #


@app.get("/explain/global/delay", tags=["explainability"])
async def global_delay_importance(sample: int = 300) -> dict:
    from predictive_ai.data.mock_generator import generate_project_metrics

    model = _require("delay")
    df = await run_in_threadpool(generate_project_metrics, sample)
    importance = await run_in_threadpool(model.global_importance, df, sample)
    return {
        "model": model.name,
        "metric": "mean_absolute_shap_value",
        "note": "Higher = the feature moves the delay prediction more, on average.",
        "importance": importance,
        "training_metrics": model.metrics,
    }


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _require(attr: str):
    try:
        return REGISTRY.require(attr)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
