#!/usr/bin/env python3
"""
Offline training entrypoint.

    python train.py                 # train all three on freshly generated mock data
    python train.py --offline-encoder   # skip the Sentence-BERT download
    python train.py --only delay        # retrain a single model
    python train.py --data-dir ./data   # train on real CSVs instead of mock data

Artifacts land in ./artifacts and are what `uvicorn predictive_ai.api:app` loads.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import pandas as pd

from predictive_ai.config import ARTIFACT_DIR, DATA_DIR
from predictive_ai.data.mock_generator import (
    generate_financial_records,
    generate_historical_works,
    generate_project_metrics,
)
from predictive_ai.models.compliance import ComplianceIndexModel
from predictive_ai.models.delay_risk import DelayRiskIndexModel
from predictive_ai.models.duplication import DuplicationIndexModel

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("train")


def _load_or_generate(data_dir: Path | None, name: str, generator) -> pd.DataFrame:
    if data_dir is not None:
        path = data_dir / f"{name}.csv"
        if path.exists():
            log.info("Loading real data: %s", path)
            return pd.read_csv(path)
        log.warning("%s not found; falling back to mock data", path)
    return generator()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["duplication", "delay", "compliance"], default=None)
    ap.add_argument("--offline-encoder", action="store_true",
                    help="Use the TF-IDF+SVD encoder instead of downloading Sentence-BERT")
    ap.add_argument("--data-dir", type=Path, default=None,
                    help="Directory containing real CSVs; falls back to mock data per-table")
    ap.add_argument("--save-mock", action="store_true", help="Also write the mock CSVs to disk")
    args = ap.parse_args()

    summary: dict[str, dict] = {}
    targets = [args.only] if args.only else ["duplication", "delay", "compliance"]

    # ---------------- Model 1 ---------------- #
    if "duplication" in targets:
        t0 = time.time()
        works = _load_or_generate(args.data_dir, "historical_works", generate_historical_works)
        if args.save_mock:
            works.to_csv(DATA_DIR / "historical_works.csv", index=False)

        if args.offline_encoder:
            from predictive_ai.utils.offline_encoder import TfidfSvdEncoder

            dup = DuplicationIndexModel(encoder=TfidfSvdEncoder())
        else:
            dup = DuplicationIndexModel()

        dup.fit(works).save()
        summary["duplication"] = {
            "rows": int(len(works)),
            "encoder": getattr(dup.encoder, "model_name", "custom"),
            "embedding_dim": int(dup.embeddings.shape[1]),
            "seconds": round(time.time() - t0, 2),
        }

    # ---------------- Model 2 ---------------- #
    if "delay" in targets:
        t0 = time.time()
        metrics_df = _load_or_generate(args.data_dir, "project_metrics", generate_project_metrics)
        if args.save_mock:
            metrics_df.to_csv(DATA_DIR / "project_metrics.csv", index=False)

        delay = DelayRiskIndexModel().fit(metrics_df)
        delay.save()
        summary["delay"] = {"rows": int(len(metrics_df)), **delay.metrics,
                            "seconds": round(time.time() - t0, 2)}

    # ---------------- Model 3 ---------------- #
    if "compliance" in targets:
        t0 = time.time()
        fin = _load_or_generate(args.data_dir, "financial_records", generate_financial_records)
        if args.save_mock:
            fin.to_csv(DATA_DIR / "financial_records.csv", index=False)

        comp = ComplianceIndexModel().fit(fin)
        comp.save()
        summary["compliance"] = {"rows": int(len(fin)), **comp.metrics,
                                 "seconds": round(time.time() - t0, 2)}

    report_path = ARTIFACT_DIR / "training_report.json"
    report_path.write_text(json.dumps(summary, indent=2, default=float))

    print("\n" + "=" * 66)
    print("TRAINING COMPLETE")
    print("=" * 66)
    print(json.dumps(summary, indent=2, default=float))
    print(f"\nArtifacts: {ARTIFACT_DIR}")
    print("Next: uvicorn predictive_ai.api:app --reload --port 8000")


if __name__ == "__main__":
    main()
