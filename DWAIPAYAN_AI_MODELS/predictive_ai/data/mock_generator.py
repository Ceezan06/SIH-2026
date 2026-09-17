"""
Synthetic MPLADS-style dataset generator.

Real MoSPI / Dataful extracts will replace this, but the column contract stays
identical, so swapping in real data means changing only the loader - not the
models.

Three tables are produced:

1. `generate_historical_works()`   -> corpus for the Duplication Index (Model 1)
2. `generate_project_metrics()`    -> labelled training set for the Delay /
                                      Cost-Overrun Index (Model 2)
3. `generate_financial_records()`  -> unlabelled ledger for the Compliance
                                      Index (Model 3)

The generators deliberately inject realistic failure modes:
  * near-duplicate work descriptions across adjacent villages
  * delay risk driven by vendor history, monsoon overlap and thin bidding
  * end-of-financial-year "budget dumping" and vendor concentration
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from predictive_ai.config import RANDOM_SEED

# --------------------------------------------------------------------------- #
# Vocabulary
# --------------------------------------------------------------------------- #

DISTRICTS = [
    "Nadia", "Hooghly", "Bardhaman", "Murshidabad", "Bankura",
    "Purulia", "Birbhum", "Malda", "Jalpaiguri", "Howrah",
]

BLOCKS = ["Ranaghat-I", "Haringhata", "Chakdaha", "Santipur", "Krishnanagar-II",
          "Nakashipara", "Tehatta-I", "Kaliganj", "Karimpur-II", "Hanskhali"]

VILLAGES = ["Bahirgachi", "Nokari", "Fatepur", "Rautari", "Dhantala",
            "Panchberia", "Mamjoan", "Barasat", "Jaguli", "Aistala",
            "Subarnapur", "Kalinagar", "Debagram", "Muragachha", "Bhatjangla"]

WORK_TEMPLATES = [
    ("Road", "Construction of {length} m concrete road from {a} to {b} at {village}"),
    ("Road", "Improvement and metalling of village approach road at {village}, {block}"),
    ("Water", "Installation of {n} deep tube wells with submersible pump at {village}"),
    ("Water", "Construction of overhead water reservoir of {n}000 litre capacity at {village}"),
    ("Education", "Construction of additional classroom block at {village} Primary School"),
    ("Education", "Supply of benches and desks to {village} High School under MPLADS"),
    ("Health", "Construction of waiting shed at {village} Sub-Centre"),
    ("Health", "Supply of ambulance to {block} Rural Hospital under MPLADS"),
    ("Sanitation", "Construction of {n} community sanitary complex at {village}"),
    ("Electrification", "Installation of {n} solar street lights at {village}, {block}"),
    ("Electrification", "Providing LED high mast lighting at {village} market area"),
    ("Community", "Construction of community hall with toilet block at {village}"),
    ("Community", "Renovation of crematorium boundary wall at {village}"),
    ("Sports", "Development of playground with boundary fencing at {village}"),
    ("Irrigation", "Re-excavation of village pond for irrigation at {village}"),
]

# Paraphrases used to create *semantic* duplicates - same work, different words.
# This is exactly the case keyword search misses and Sentence-BERT catches.
PARAPHRASE_MAP = {
    "Construction of": "Building of",
    "Installation of": "Setting up of",
    "Improvement and metalling of": "Upgradation and black-topping of",
    "Supply of": "Procurement and supply of",
    "concrete road": "RCC road",
    "deep tube wells": "borewells",
    "solar street lights": "solar-powered street lamps",
    "community hall": "community centre",
    "additional classroom block": "extra class rooms",
}


def _paraphrase(text: str, rng: np.random.Generator) -> str:
    """Rewrite a description so it means the same thing but shares few words."""
    out = text
    for src, dst in PARAPHRASE_MAP.items():
        if src in out and rng.random() < 0.8:
            out = out.replace(src, dst)
    return out


# --------------------------------------------------------------------------- #
# 1. Historical works corpus  (Model 1 - Duplication)
# --------------------------------------------------------------------------- #


def generate_historical_works(
    n_rows: int = 1200,
    duplicate_fraction: float = 0.08,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Build the historical MPLADS works register.

    Columns
    -------
    work_id, mp_name, district, block, village, category, description,
    sanctioned_cost_lakh, sanction_year, status
    """
    rng = np.random.default_rng(seed)
    rows: list[dict] = []

    for i in range(n_rows):
        category, template = WORK_TEMPLATES[rng.integers(len(WORK_TEMPLATES))]
        district = DISTRICTS[rng.integers(len(DISTRICTS))]
        block = BLOCKS[rng.integers(len(BLOCKS))]
        village = VILLAGES[rng.integers(len(VILLAGES))]

        # A ward number plus the administrative suffix makes each record
        # realistically unique, so an accidental exact collision between two
        # unrelated works is rare - only the deliberately seeded duplicates
        # below should score near 1.0.
        ward = int(rng.integers(1, 25))
        description = template.format(
            length=int(rng.integers(150, 1800)),
            n=int(rng.integers(1, 9)),
            a=VILLAGES[rng.integers(len(VILLAGES))],
            b=VILLAGES[rng.integers(len(VILLAGES))],
            village=village,
            block=block,
        ) + f" under MPLADS, Ward No. {ward}, {block} block, {district} district"

        rows.append(
            {
                "work_id": f"MPLAD-{2019 + i % 7}-{100000 + i}",
                "mp_name": f"MP_{rng.integers(1, 40):02d}",
                "district": district,
                "block": block,
                "village": village,
                "category": category,
                "description": description,
                "sanctioned_cost_lakh": round(float(rng.uniform(1.5, 45.0)), 2),
                "sanction_year": int(2019 + i % 7),
                "status": rng.choice(
                    ["Completed", "Ongoing", "Sanctioned"], p=[0.62, 0.27, 0.11]
                ),
            }
        )

    df = pd.DataFrame(rows)

    # ---- inject semantic duplicates -------------------------------------- #
    n_dupes = int(len(df) * duplicate_fraction)
    dupe_sources = rng.choice(len(df), size=n_dupes, replace=False)
    dupes = []
    for j, src_idx in enumerate(dupe_sources):
        src = df.iloc[int(src_idx)].to_dict()
        src = dict(src)
        src["work_id"] = f"MPLAD-DUP-{200000 + j}"
        src["description"] = _paraphrase(src["description"], rng)
        src["sanction_year"] = min(2025, src["sanction_year"] + 1)
        # duplicated works are usually re-sanctioned at a slightly higher cost
        src["sanctioned_cost_lakh"] = round(
            src["sanctioned_cost_lakh"] * float(rng.uniform(1.02, 1.30)), 2
        )
        src["is_seeded_duplicate"] = True
        src["duplicate_of"] = df.iloc[int(src_idx)]["work_id"]
        dupes.append(src)

    df["is_seeded_duplicate"] = False
    df["duplicate_of"] = None
    out = pd.concat([df, pd.DataFrame(dupes)], ignore_index=True)
    return out.sample(frac=1.0, random_state=seed).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 2. Project metrics  (Model 2 - Delay / Cost overrun)
# --------------------------------------------------------------------------- #

DELAY_FEATURES = [
    "sanctioned_cost_lakh",
    "planned_duration_days",
    "sanction_to_start_lag_days",
    "num_bidders",
    "vendor_past_overrun_rate",
    "vendor_completed_projects",
    "district_past_delay_rate",
    "monsoon_overlap_months",
    "fund_release_tranches",
    "rate_deviation_pct",
    "is_election_year",
    "category_code",
]

CATEGORY_CODES = {
    "Road": 0, "Water": 1, "Education": 2, "Health": 3, "Sanitation": 4,
    "Electrification": 5, "Community": 6, "Sports": 7, "Irrigation": 8,
}


def generate_project_metrics(
    n_rows: int = 6000,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Labelled training table for the supervised delay / cost-overrun classifier.

    Target
    ------
    `is_delayed_or_overrun` : 1 if the work finished >20% late OR >10% over cost.

    Note every feature is knowable *at sanction time*. Nothing here leaks the
    outcome, which is what makes the model genuinely predictive rather than
    descriptive.
    """
    rng = np.random.default_rng(seed)
    n = n_rows

    sanctioned_cost = rng.gamma(shape=2.2, scale=6.0, size=n) + 1.0        # lakh
    planned_duration = rng.integers(45, 540, size=n).astype(float)          # days
    start_lag = rng.gamma(shape=2.0, scale=28.0, size=n)                    # days
    num_bidders = rng.integers(1, 9, size=n).astype(float)
    vendor_overrun_rate = np.clip(rng.beta(2.0, 6.0, size=n), 0, 1)
    vendor_completed = rng.integers(0, 60, size=n).astype(float)
    district_delay_rate = np.clip(rng.beta(3.0, 5.0, size=n), 0, 1)
    monsoon_overlap = rng.integers(0, 5, size=n).astype(float)
    tranches = rng.integers(1, 5, size=n).astype(float)
    rate_deviation = rng.normal(0.0, 12.0, size=n)                          # % vs district SoR
    is_election_year = rng.binomial(1, 0.22, size=n).astype(float)
    category_code = rng.integers(0, len(CATEGORY_CODES), size=n).astype(float)

    # ---- latent risk process (the "ground truth" the model must recover) --- #
    logit = (
        -4.05
        + 0.030 * sanctioned_cost
        + 0.0016 * planned_duration
        + 0.0115 * start_lag
        - 0.170 * num_bidders
        + 2.300 * vendor_overrun_rate
        - 0.019 * vendor_completed
        + 1.850 * district_delay_rate
        + 0.230 * monsoon_overlap
        + 0.145 * tranches
        + 0.026 * np.abs(rate_deviation)
        + 0.420 * is_election_year
        + 0.045 * (category_code == CATEGORY_CODES["Irrigation"]).astype(float)
        # interaction: an inexperienced vendor on a big-ticket work is far worse
        + 0.060 * sanctioned_cost * (vendor_completed < 5).astype(float)
        + rng.normal(0, 0.55, size=n)
    )
    prob = 1.0 / (1.0 + np.exp(-logit))
    label = rng.binomial(1, prob)

    df = pd.DataFrame(
        {
            "project_id": [f"PRJ-{300000 + i}" for i in range(n)],
            "sanctioned_cost_lakh": np.round(sanctioned_cost, 2),
            "planned_duration_days": planned_duration,
            "sanction_to_start_lag_days": np.round(start_lag, 1),
            "num_bidders": num_bidders,
            "vendor_past_overrun_rate": np.round(vendor_overrun_rate, 4),
            "vendor_completed_projects": vendor_completed,
            "district_past_delay_rate": np.round(district_delay_rate, 4),
            "monsoon_overlap_months": monsoon_overlap,
            "fund_release_tranches": tranches,
            "rate_deviation_pct": np.round(rate_deviation, 2),
            "is_election_year": is_election_year,
            "category_code": category_code,
            "is_delayed_or_overrun": label,
        }
    )
    return df


# --------------------------------------------------------------------------- #
# 3. Financial records  (Model 3 - Compliance)
# --------------------------------------------------------------------------- #

COMPLIANCE_FEATURES = [
    "utilisation_ratio",
    "days_to_first_payment",
    "num_payments",
    "avg_payment_size_lakh",
    "round_number_payment_ratio",
    "vendor_concentration_ratio",
    "uc_submission_lag_days",
    "fy_end_disbursal_ratio",
    "cost_revision_count",
    "payment_interval_cv",
]


def generate_financial_records(
    n_rows: int = 4000,
    anomaly_fraction: float = 0.05,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Unlabelled financial ledger, one row per work.

    Isolation Forest is unsupervised, so no target column is used for training.
    `is_seeded_anomaly` exists only so we can *evaluate* the model - it is never
    fed to `fit()`.
    """
    rng = np.random.default_rng(seed + 7)
    n_anom = int(n_rows * anomaly_fraction)
    n_norm = n_rows - n_anom

    def _normal(m: int) -> dict[str, np.ndarray]:
        return {
            "utilisation_ratio": np.clip(rng.normal(0.88, 0.10, m), 0.05, 1.05),
            "days_to_first_payment": np.clip(rng.gamma(2.6, 14.0, m), 1, None),
            "num_payments": rng.integers(2, 9, m).astype(float),
            "avg_payment_size_lakh": np.clip(rng.gamma(2.2, 1.8, m), 0.1, None),
            "round_number_payment_ratio": np.clip(rng.beta(1.5, 8.0, m), 0, 1),
            "vendor_concentration_ratio": np.clip(rng.beta(2.0, 6.0, m), 0, 1),
            "uc_submission_lag_days": np.clip(rng.gamma(2.4, 18.0, m), 0, None),
            "fy_end_disbursal_ratio": np.clip(rng.beta(1.8, 6.0, m), 0, 1),
            "cost_revision_count": rng.poisson(0.35, m).astype(float),
            "payment_interval_cv": np.clip(rng.gamma(2.0, 0.22, m), 0, None),
        }

    def _anomalous(m: int) -> dict[str, np.ndarray]:
        """Fraud-like signatures: budget dumping, single-vendor capture, UC delay."""
        return {
            "utilisation_ratio": np.clip(rng.normal(1.06, 0.16, m), 0.05, 1.6),
            "days_to_first_payment": np.clip(rng.gamma(1.1, 5.0, m), 0, None),
            "num_payments": rng.integers(1, 3, m).astype(float),
            "avg_payment_size_lakh": np.clip(rng.gamma(5.0, 4.5, m), 0.1, None),
            "round_number_payment_ratio": np.clip(rng.beta(7.0, 1.6, m), 0, 1),
            "vendor_concentration_ratio": np.clip(rng.beta(9.0, 1.4, m), 0, 1),
            "uc_submission_lag_days": np.clip(rng.gamma(6.0, 38.0, m), 0, None),
            "fy_end_disbursal_ratio": np.clip(rng.beta(8.0, 1.5, m), 0, 1),
            "cost_revision_count": rng.poisson(2.6, m).astype(float) + 1,
            "payment_interval_cv": np.clip(rng.gamma(5.0, 0.45, m), 0, None),
        }

    normal_df = pd.DataFrame(_normal(n_norm))
    normal_df["is_seeded_anomaly"] = 0
    anom_df = pd.DataFrame(_anomalous(n_anom))
    anom_df["is_seeded_anomaly"] = 1

    df = pd.concat([normal_df, anom_df], ignore_index=True)
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    df.insert(0, "work_id", [f"MPLAD-FIN-{400000 + i}" for i in range(len(df))])
    for col in df.columns:
        if col not in ("work_id", "is_seeded_anomaly"):
            df[col] = df[col].round(4)
    return df


# --------------------------------------------------------------------------- #
# Convenience
# --------------------------------------------------------------------------- #


def generate_all(seed: int = RANDOM_SEED) -> dict[str, pd.DataFrame]:
    """Generate all three tables in one call."""
    return {
        "historical_works": generate_historical_works(seed=seed),
        "project_metrics": generate_project_metrics(seed=seed),
        "financial_records": generate_financial_records(seed=seed),
    }


if __name__ == "__main__":  # pragma: no cover
    from predictive_ai.config import DATA_DIR

    tables = generate_all()
    for name, frame in tables.items():
        path = DATA_DIR / f"{name}.csv"
        frame.to_csv(path, index=False)
        print(f"{name:20s} {frame.shape}  ->  {path}")
