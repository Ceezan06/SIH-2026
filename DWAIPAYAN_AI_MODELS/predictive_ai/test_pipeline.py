#!/usr/bin/env python3
"""
End-to-end smoke test - run this before the demo, and run it in front of the
judges if they ask "does it actually work?".

    python test_pipeline.py                  # uses Sentence-BERT
    python test_pipeline.py --offline-encoder  # no model download needed

It asserts the property the whole design rests on: **every index is a finite
float in [0.0, 1.0]**, for normal inputs, empty inputs and absurd inputs alike.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from predictive_ai.models.composite import fuse
from predictive_ai.data.mock_generator import (
    generate_financial_records,
    generate_historical_works,
    generate_project_metrics,
)
from predictive_ai.models.compliance import ComplianceIndexModel
from predictive_ai.models.delay_risk import DelayRiskIndexModel
from predictive_ai.models.duplication import DuplicationIndexModel

PASS, FAIL = "PASS", "FAIL"


def check(label: str, condition: bool) -> bool:
    print(f"  [{PASS if condition else FAIL}] {label}")
    return condition


def assert_index(result) -> bool:
    ok = isinstance(result.score, float) and np.isfinite(result.score) and 0.0 <= result.score <= 1.0
    return check(f"{result.index_name}: score={result.score:.4f} band={result.band}", ok)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline-encoder", action="store_true")
    ap.add_argument("--rows", type=int, default=2000)
    args = ap.parse_args()

    results: list[bool] = []

    # ================= MODEL 1 ================= #
    print("\n--- MODEL 1: Duplication Index ---")
    works = generate_historical_works(n_rows=600)
    if args.offline_encoder:
        from predictive_ai.models.offline_encoder import TfidfSvdEncoder

        dup = DuplicationIndexModel(encoder=TfidfSvdEncoder())
    else:
        dup = DuplicationIndexModel()
    dup.fit(works)

    seeded = works[works["is_seeded_duplicate"]].iloc[0]
    original = works[works["work_id"] == seeded["duplicate_of"]].iloc[0]

    # Exclude the work from its own corpus row, otherwise it self-matches at 1.00.
    # What we want to prove is that it finds the *paraphrased* re-sanction.
    r_dup = dup.score({
        "description": original["description"],
        "exclude_work_ids": [original["work_id"]],
    })
    results.append(assert_index(r_dup))
    results.append(check("known duplicate scores above 0.30", r_dup.score > 0.30))
    matched = r_dup.meta["top_matches"][0]["work_id"]
    results.append(check(f"found the seeded paraphrase ({matched})", matched == seeded["work_id"]))
    print(f"      matched -> {matched}")
    print(f"      why     -> {[d.feature for d in r_dup.drivers]}")

    r_novel = dup.score({"description": "Procurement of laboratory oscilloscopes for a polytechnic"})
    results.append(assert_index(r_novel))
    results.append(check("unrelated text scores below the duplicate", r_novel.score < r_dup.score))

    r_empty = dup.score({"description": ""})
    results.append(check("empty description -> 0.0, no crash", r_empty.score == 0.0))

    # ================= MODEL 2 ================= #
    print("\n--- MODEL 2: Delay / Cost-Overrun Index ---")
    metrics_df = generate_project_metrics(n_rows=args.rows)
    delay = DelayRiskIndexModel().fit(metrics_df)
    print(f"      validation ROC-AUC = {delay.metrics['roc_auc']:.3f}")
    results.append(check("ROC-AUC beats random (>0.65)", delay.metrics["roc_auc"] > 0.65))
    results.append(
        check(
            "isotonic calibration did not worsen Brier",
            delay.metrics["brier_calibrated"] <= delay.metrics["brier_raw"] + 1e-3,
        )
    )

    high_risk = {
        "sanctioned_cost_lakh": 44.0, "planned_duration_days": 500,
        "sanction_to_start_lag_days": 190, "num_bidders": 1,
        "vendor_past_overrun_rate": 0.93, "vendor_completed_projects": 0,
        "district_past_delay_rate": 0.88, "monsoon_overlap_months": 4,
        "fund_release_tranches": 4, "rate_deviation_pct": 42.0,
        "is_election_year": 1, "category_code": 8,
    }
    low_risk = {
        "sanctioned_cost_lakh": 3.0, "planned_duration_days": 60,
        "sanction_to_start_lag_days": 8, "num_bidders": 8,
        "vendor_past_overrun_rate": 0.02, "vendor_completed_projects": 55,
        "district_past_delay_rate": 0.05, "monsoon_overlap_months": 0,
        "fund_release_tranches": 1, "rate_deviation_pct": 0.5,
        "is_election_year": 0, "category_code": 2,
    }
    r_hi = delay.score(high_risk)
    r_lo = delay.score(low_risk)
    results.append(assert_index(r_hi))
    results.append(assert_index(r_lo))
    results.append(check("risky project scores above safe project", r_hi.score > r_lo.score))
    results.append(check("SHAP returned drivers", len(r_hi.drivers) > 0))
    print(f"      top driver -> {r_hi.drivers[0].feature} "
          f"({r_hi.drivers[0].contribution_pct}% of the decision)")

    r_partial = delay.score({"sanctioned_cost_lakh": 20.0})
    results.append(check("partial payload imputes and still scores", 0.0 <= r_partial.score <= 1.0))

    # ================= MODEL 3 ================= #
    print("\n--- MODEL 3: Compliance Index ---")
    fin = generate_financial_records(n_rows=args.rows)
    comp = ComplianceIndexModel().fit(fin)
    if "eval_roc_auc" in comp.metrics:
        print(f"      seeded-anomaly ROC-AUC = {comp.metrics['eval_roc_auc']:.3f}")
        results.append(check("separates seeded anomalies (>0.75)", comp.metrics["eval_roc_auc"] > 0.75))

    batch = comp.score_batch(fin)
    results.append(check("batch scores stay inside [0,1]", bool((batch >= 0).all() and (batch <= 1).all())))

    dirty = {
        "utilisation_ratio": 1.45, "days_to_first_payment": 1, "num_payments": 1,
        "avg_payment_size_lakh": 38.0, "round_number_payment_ratio": 0.98,
        "vendor_concentration_ratio": 0.99, "uc_submission_lag_days": 420,
        "fy_end_disbursal_ratio": 0.97, "cost_revision_count": 6,
        "payment_interval_cv": 3.4,
    }
    clean = {
        "utilisation_ratio": 0.9, "days_to_first_payment": 40, "num_payments": 5,
        "avg_payment_size_lakh": 3.5, "round_number_payment_ratio": 0.05,
        "vendor_concentration_ratio": 0.15, "uc_submission_lag_days": 30,
        "fy_end_disbursal_ratio": 0.1, "cost_revision_count": 0,
        "payment_interval_cv": 0.4,
    }
    r_dirty = comp.score(dirty)
    r_clean = comp.score(clean)
    results.append(assert_index(r_dirty))
    results.append(assert_index(r_clean))
    results.append(check("irregular ledger scores above clean ledger", r_dirty.score > r_clean.score))
    results.append(check("SHAP returned drivers", len(r_dirty.drivers) > 0))
    print(f"      top driver -> {r_dirty.drivers[0].feature}")

    extreme = comp.score({k: v * 1000 for k, v in dirty.items()})
    results.append(check("absurd out-of-range input still clipped to <=1.0", extreme.score <= 1.0))

    # ================= COMPOSITE ================= #
    print("\n--- COMPOSITE FRAUD INDEX ---")
    composite = fuse(r_dup, r_hi, r_dirty)
    results.append(check(
        f"composite={composite.score:.4f} ({composite.risk_index_100}/100, {composite.band})",
        0.0 <= composite.score <= 1.0,
    ))
    print("\n" + json.dumps(
        {k: v for k, v in composite.to_dict().items() if k != "components"},
        indent=2, default=float,
    ))

    # ================= VERDICT ================= #
    print("\n" + "=" * 60)
    print(f"{sum(results)}/{len(results)} checks passed")
    print("=" * 60)
    raise SystemExit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
