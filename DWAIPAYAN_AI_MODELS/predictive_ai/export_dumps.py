import json
import shutil
from pathlib import Path

# 1. Set your name here (folder will be created as model_dumps/<YOUR_NAME>)
USER_NAME = "YOUR_NAME"  # <-- Change this to your name / GitHub handle

dump_dir = Path(f"model_dumps/{USER_NAME}")
dump_dir.mkdir(parents=True, exist_ok=True)

artifacts_dir = Path("artifacts")

# 2. Copy existing JSON reports from artifacts if present
for file_name in ["training_parameters.json", "training_report.json"]:
    src = artifacts_dir / file_name
    if src.exists():
        shutil.copy(src, dump_dir / file_name)
        print(f"Copied {file_name} to {dump_dir}")

# 3. Dump live model configurations & metadata
summary_data = {
    "developer_name": USER_NAME,
    "system": "SIH MPLAD Predictive AI Pipeline",
    "models": {
        "duplication_index": {
            "type": "Sentence-BERT (all-MiniLM-L6-v2)",
            "embedding_dim": 384,
            "metric": "Cosine Similarity"
        },
        "delay_risk_index": {
            "type": "XGBoost Classifier",
            "n_estimators": 100,
            "objective": "binary:logistic",
            "eval_metric": "logloss"
        },
        "compliance_index": {
            "type": "Isolation Forest",
            "n_estimators": 100,
            "contamination": 0.08
        }
    }
}

with open(dump_dir / "parameter_dumps.json", "w") as f:
    json.dump(summary_data, f, indent=4)

print(f"\n=======================================================")
print(f"✅ Model parameter dumps created in: {dump_dir.resolve()}")
print(f"=======================================================\n")