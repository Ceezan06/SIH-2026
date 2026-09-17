import json
from pathlib import Path

# Metadata and parameters matching your training configuration
training_parameters = {
    "project": "SIH MPLAD Fraud Detection System",
    "global_config": {
        "random_seed": 42,
        "dataset_rows": {
            "historical_works": 1296,
            "project_metrics": 6000,
            "financial_records": 4000
        }
    },
    "model_hyperparameters": {
        "duplication_model": {
            "algorithm": "Sentence-BERT (all-MiniLM-L6-v2)",
            "fallback_encoder": "TF-IDF + TruncatedSVD",
            "embedding_dimensions": 384,
            "distance_metric": "Cosine Similarity"
        },
        "delay_risk_model": {
            "algorithm": "XGBoost Classifier (XGBClassifier)",
            "n_estimators": 100,
            "objective": "binary:logistic",
            "validation_split": "80/20 Train-Test",
            "target": "is_delayed"
        },
        "compliance_model": {
            "algorithm": "Isolation Forest",
            "n_estimators": 100,
            "contamination_rate": 0.08,
            "random_state": 42
        }
    }
}

# Save output inside artifacts
output_path = Path("artifacts/training_parameters.json")
output_path.parent.mkdir(exist_ok=True)

with open(output_path, "w") as f:
    json.dump(training_parameters, f, indent=4)

print(f"✅ Training parameters successfully saved to {output_path}")