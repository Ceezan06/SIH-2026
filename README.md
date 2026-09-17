# MPLADS Predictive AI Stage — Team EXPERIMENTALISTS (SIH 2026)

Three independent models score every MPLADS work item on a **0.0 – 1.0** scale,
and a weighted fusion turns them into one explainable **Composite Fraud Index**.

| # | Index | Algorithm | Learning type | How 0–1 is produced |
|---|-------|-----------|---------------|---------------------|
| 1 | Duplication | Sentence-BERT (`all-MiniLM-L6-v2`) + cosine similarity | Pre-trained embeddings, no training needed | Cosine similarity, calibrated from the informative band `[0.30, 0.95]` onto `[0, 1]` |
| 2 | Delay / Cost Overrun | XGBoost binary classifier | Supervised | `predict_proba()[:, 1]`, then isotonic calibration |
| 3 | Compliance | Isolation Forest | Unsupervised anomaly detection | Strict Min-Max of `score_samples()`, inverted so 1.0 = most irregular |

Every model returns the same `IndexResult` object, so the React dashboard renders
one component three times.

---

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python train.py --save-mock        # generates mock data, trains all 3, writes ./artifacts
python test_pipeline.py            # 21 assertions, end-to-end
uvicorn predictive_ai.api:app --reload --port 8000
# open http://localhost:8000/docs
```

**No internet at the venue?** Sentence-BERT downloads ~90 MB on first run. Use the
offline fallback encoder instead — same API, same SHAP output, weaker paraphrase
detection:

```bash
python train.py --offline-encoder
python test_pipeline.py --offline-encoder
```

The encoder is saved inside the duplication artifact, so whatever you trained
with is what the API loads. Corpus vectors and query vectors always come from the
same encoder — mixing them would make every cosine similarity meaningless.

---

## Project layout

```
predictive_ai/
├── config.py                  # every tunable: weights, thresholds, hyperparameters
├── schemas.py                 # Pydantic v2 request/response contracts (= your OpenAPI docs)
├── registry.py                # loads artifacts once at FastAPI startup, never per request
├── composite.py               # weighted fusion + critical override
├── api.py                     # FastAPI routes
├── data/mock_generator.py     # 3 synthetic tables with realistic failure modes injected
├── models/
│   ├── base.py                # IndexResult / Driver / clip01 / min_max_scale / band_for
│   ├── duplication.py         # Model 1
│   ├── delay_risk.py          # Model 2
│   └── compliance.py          # Model 3
└── utils/offline_encoder.py   # TF-IDF + SVD fallback encoder
train.py                       # offline training entrypoint
test_pipeline.py               # end-to-end smoke test
```

---

## API

| Method | Route | Purpose |
|--------|-------|---------|
| GET | `/health` | Per-model readiness; returns `degraded` if any artifact is missing |
| POST | `/score/duplication` | Model 1 |
| POST | `/score/delay` | Model 2 |
| POST | `/score/compliance` | Model 3 |
| POST | `/score/composite` | All three + Composite Fraud Index |
| GET | `/explain/global/delay` | Mean \|SHAP\| per feature, for the Model Insights tab |

```bash
curl -X POST localhost:8000/score/composite -H 'Content-Type: application/json' -d '{
  "work_id": "MPLAD-2026-100777",
  "description": "Building of RCC road from Nokari to Fatepur at Dhantala under MPLADS, Ward No. 7, Chakdaha block, Nadia district",
  "restrict_to": {"district": "Nadia"},
  "project_metrics": {
    "sanctioned_cost_lakh": 38.0, "num_bidders": 1,
    "vendor_past_overrun_rate": 0.88, "monsoon_overlap_months": 4
  },
  "financials": {
    "vendor_concentration_ratio": 0.97, "fy_end_disbursal_ratio": 0.96,
    "uc_submission_lag_days": 380, "cost_revision_count": 5
  }
}'
```

Response shape:

```jsonc
{
  "work_id": "MPLAD-2026-100777",
  "composite_fraud_index": 0.6421,
  "risk_index_100": 64,
  "band": "HIGH",
  "weights": {"duplication": 0.35, "delay_overrun": 0.35, "compliance": 0.30},
  "critical_override_applied": false,
  "narrative": "Composite risk 64/100 (HIGH). Dominant signal: compliance at 0.78.",
  "top_drivers": [ /* merged, weight-scaled SHAP drivers across all 3 models */ ],
  "components": { "duplication": {...}, "delay_overrun": {...}, "compliance": {...} }
}
```

Missing feature fields are imputed from training medians and listed back in
`meta.imputed_features` — the API never 500s on a partial record, and the
dashboard can show which numbers were guessed.

---

## Design decisions worth defending

**Why three models instead of one.** The three questions have different data
shapes. Duplication is a *text* problem with no labels. Delay is a *supervised
tabular* problem with real historical outcomes. Compliance is an *unlabelled
rare-event* problem. One model cannot be the right tool for all three, and
keeping them separate means one can be retrained or switched off without
touching the others.

**Why the duplication score is calibrated.** Sentence-BERT returns ~0.2–0.3
cosine similarity even between unrelated government text. Reporting the raw
number would make every work look mildly suspicious. We stretch the informative
band `[DUPLICATION_FLOOR, DUPLICATION_CEILING]` across the full 0–1 range.

**Why the delay probability is isotonically calibrated.** Raw boosted-tree
probabilities are over-confident. Auditors act on these numbers, so "0.70" must
mean roughly 70 in 100. Calibration is fitted on a held-out split; the Brier
score before and after is in `meta.training_metrics`.

**Why the compliance index is inverted.** `IsolationForest.score_samples()`
returns higher values for *more normal* points. `min` and `max` are frozen at fit
time and unseen extremes are clipped, which is what guarantees the strict `[0, 1]`
contract. `meta.compliance_health = 1 - index` is provided for the
"higher is better" view.

**Why there is a critical override in the fusion.** A weighted mean can hide one
catastrophic signal behind two calm ones. Any single index ≥ 0.90 floors the
composite at 0.75, so a certain duplicate can never be averaged away.

**The model does not accuse anyone.** Every response is a prioritisation signal
with a plain-English `narrative` and a SHAP driver list. A human auditor decides.

---

## Honest limitations (say these before a judge finds them)

- **Mock data is generated from a known process**, so the reported metrics are an
  upper bound. The Isolation Forest's 1.00 ROC-AUC against seeded anomalies means
  the seeded anomalies are cleanly separable *by construction*; expect ~0.7–0.85
  on real MoSPI data.
- **The delay model has no real outcome labels yet.** MoSPI publishes sanction and
  completion data; the target has to be derived from it before the numbers mean
  anything.
- **Anomaly ≠ fraud.** An unusual spending pattern can be a genuine emergency
  repair. That is exactly why the output is a ranked queue, not a verdict.
- **Text similarity has a legitimate false-positive mode**: two genuinely
  different roads in the same village will read almost identically. District and
  category blocking (`restrict_to`) plus the `exclude_work_ids` self-match guard
  reduce but do not eliminate this.

---

## Where this plugs into the wider architecture

```
MoSPI / Dataful  ->  MongoDB (raw)  ->  PostgreSQL (clean)
                                              |
                                    Celery worker + Redis
                                              |
                                  predictive_ai (this repo)
                                              |
                        FastAPI  ->  React role-based audit dashboard
```

Interactive single-item scoring uses the FastAPI routes directly; bulk nightly
re-scoring should call `score_batch()` on each model from a Celery task, which
skips SHAP and is orders of magnitude faster.
