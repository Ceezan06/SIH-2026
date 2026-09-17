"""
MODEL 1 - DUPLICATION INDEX
===========================

Question answered: *"Has this exact work already been sanctioned before, under
slightly different wording?"*

Method
------
1. Every historical work description is encoded once into a 384-dimensional
   vector by Sentence-BERT (`all-MiniLM-L6-v2`).
2. A new description is encoded the same way.
3. Cosine similarity against the whole corpus gives a 0-1 closeness to every
   past work. The maximum is the duplication evidence.
4. That raw similarity is calibrated onto a clean 0.0-1.0 index.

Why not keyword / fuzzy matching? Because "Construction of concrete road at
Dhantala" and "Building of RCC road at Dhantala" share almost no tokens but are
the same work. Embeddings compare *meaning*, not spelling.

Explainability
--------------
SHAP's Partition explainer with a Text masker perturbs the input sentence word
by word and measures how the similarity score moves. The output is a per-word
attribution: exactly which phrases made the system flag the work as a duplicate.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Protocol, Sequence

import joblib
import numpy as np
import pandas as pd

from predictive_ai.config import (
    DUPLICATION_CEILING,
    DUPLICATION_FLOOR,
    DUPLICATION_TOP_K,
    SBERT_BATCH_SIZE,
    SBERT_MODEL_NAME,
    SHAP_ENABLED,
    SHAP_TOP_DRIVERS,
)
from predictive_ai.models.base import BaseIndexModel, Driver, IndexResult, clip01

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Encoder abstraction - keeps Sentence-BERT swappable and unit-testable
# --------------------------------------------------------------------------- #


class TextEncoder(Protocol):
    def encode(self, texts: Sequence[str], **kwargs: Any) -> np.ndarray: ...


class SentenceBertEncoder:
    """Thin lazy wrapper around `sentence_transformers.SentenceTransformer`."""

    def __init__(self, model_name: str = SBERT_MODEL_NAME) -> None:
        self.model_name = model_name
        self._model = None

    def _ensure(self) -> None:
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # lazy: heavy import

            logger.info("Loading Sentence-BERT model: %s", self.model_name)
            self._model = SentenceTransformer(self.model_name)

    def encode(self, texts: Sequence[str], **kwargs: Any) -> np.ndarray:
        self._ensure()
        kwargs.setdefault("batch_size", SBERT_BATCH_SIZE)
        kwargs.setdefault("convert_to_numpy", True)
        kwargs.setdefault("normalize_embeddings", True)   # -> cosine == dot product
        kwargs.setdefault("show_progress_bar", False)
        return np.asarray(self._model.encode(list(texts), **kwargs), dtype=np.float32)


def _l2_normalise(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return mat / norms


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #


class DuplicationIndexModel(BaseIndexModel):
    name = "duplication_index"
    version = "1.1.0"
    artifact_filename = "duplication_index.joblib"

    META_COLUMNS = ("work_id", "district", "block", "village", "category",
                    "sanctioned_cost_lakh", "sanction_year", "status")

    def __init__(
        self,
        encoder: TextEncoder | None = None,
        floor: float = DUPLICATION_FLOOR,
        ceiling: float = DUPLICATION_CEILING,
    ) -> None:
        self.encoder: TextEncoder = encoder or SentenceBertEncoder()
        self.floor = floor
        self.ceiling = ceiling
        self.embeddings: np.ndarray | None = None      # (N, d), L2-normalised
        self.corpus_meta: pd.DataFrame | None = None
        self.corpus_texts: list[str] = []

    # -- training ----------------------------------------------------------- #

    def fit(
        self,
        historical_df: pd.DataFrame,
        text_column: str = "description",
    ) -> "DuplicationIndexModel":
        if text_column not in historical_df.columns:
            raise KeyError(f"'{text_column}' not found in historical dataframe")

        df = historical_df.reset_index(drop=True).copy()
        texts = df[text_column].astype(str).tolist()

        logger.info("Encoding %d historical work descriptions...", len(texts))
        emb = self.encoder.encode(texts)
        self.embeddings = _l2_normalise(np.asarray(emb, dtype=np.float32))

        keep = [c for c in self.META_COLUMNS if c in df.columns]
        self.corpus_meta = df[keep].copy()
        self.corpus_texts = texts
        return self

    # -- persistence -------------------------------------------------------- #

    def save(self) -> Path:
        if self.embeddings is None:
            raise RuntimeError("Call fit() before save()")

        # The query at inference time MUST be encoded by the same encoder that
        # built the corpus, otherwise the two live in different vector spaces
        # and every cosine similarity is meaningless. So the encoder travels
        # with the artifact. Sentence-BERT's multi-hundred-MB torch module is
        # detached first - it is reloadable from its name alone.
        heavy = getattr(self.encoder, "_model", None)
        if heavy is not None:
            self.encoder._model = None  # type: ignore[attr-defined]
        try:
            joblib.dump(
                {
                    "version": self.version,
                    "embeddings": self.embeddings,
                    "corpus_meta": self.corpus_meta,
                    "corpus_texts": self.corpus_texts,
                    "floor": self.floor,
                    "ceiling": self.ceiling,
                    "encoder": self.encoder,
                    "encoder_name": getattr(self.encoder, "model_name", "custom"),
                },
                self.artifact_path,
                compress=3,
            )
        finally:
            if heavy is not None:
                self.encoder._model = heavy  # type: ignore[attr-defined]

        logger.info("Saved duplication artifact -> %s", self.artifact_path)
        return self.artifact_path

    @classmethod
    def load(cls, encoder: TextEncoder | None = None) -> "DuplicationIndexModel":
        model = cls(encoder=encoder)
        blob = joblib.load(model.artifact_path)
        if encoder is None and blob.get("encoder") is not None:
            model.encoder = blob["encoder"]        # same vector space as the corpus
        model.embeddings = blob["embeddings"]
        model.corpus_meta = blob["corpus_meta"]
        model.corpus_texts = blob["corpus_texts"]
        model.floor = blob.get("floor", DUPLICATION_FLOOR)
        model.ceiling = blob.get("ceiling", DUPLICATION_CEILING)
        return model

    def is_ready(self) -> bool:
        return self.embeddings is not None and len(self.corpus_texts) > 0

    # -- core maths --------------------------------------------------------- #

    def _calibrate(self, raw_cosine: float) -> float:
        """
        Map raw cosine similarity onto the reported index.

        Sentence-BERT returns ~0.2-0.3 even for unrelated government text, so a
        raw score would make everything look slightly suspicious. We stretch the
        informative band [floor, ceiling] across the full [0, 1] range.
        """
        span = max(self.ceiling - self.floor, 1e-9)
        return clip01((raw_cosine - self.floor) / span)

    def _candidate_mask(self, payload: dict[str, Any]) -> np.ndarray:
        """
        Optional blocking. If the caller passes `restrict_to`, only compare
        against historical works in the same district / category. This both
        speeds things up and removes false positives across unrelated regions.
        """
        n = len(self.corpus_texts)
        mask = np.ones(n, dtype=bool)
        if self.corpus_meta is None:
            return mask

        for col, val in (payload.get("restrict_to") or {}).items():
            if col in self.corpus_meta.columns and val is not None:
                mask &= (self.corpus_meta[col].astype(str).values == str(val))

        # Re-scoring an already-stored work must not match the work against
        # itself - that would report a perfect 1.00 duplicate for every row in
        # the nightly Celery job.
        exclude = payload.get("exclude_work_ids") or []
        if exclude and "work_id" in self.corpus_meta.columns:
            ids = self.corpus_meta["work_id"].astype(str).values
            mask &= ~np.isin(ids, [str(x) for x in exclude])

        # never let blocking empty the corpus
        return mask if mask.any() else np.ones(n, dtype=bool)

    def _similarities(self, text: str, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (similarities over the masked corpus, indices into the corpus)."""
        query = self.encoder.encode([text])
        query = _l2_normalise(np.asarray(query, dtype=np.float32))[0]
        idx = np.flatnonzero(mask)
        sims = self.embeddings[idx] @ query        # cosine, both sides normalised
        return sims, idx

    # -- inference ---------------------------------------------------------- #

    def score(self, payload: dict[str, Any], explain: bool = True) -> IndexResult:
        """
        payload
        -------
        {
          "description": "Construction of RCC road at Dhantala ...",  # required
          "restrict_to": {"district": "Nadia"},        # optional blocking filter
          "exclude_work_ids": ["MPLAD-2022-100514"],   # optional self-match guard
          "top_k": 5                                   # optional
        }
        """
        if not self.is_ready():
            raise RuntimeError("DuplicationIndexModel is not fitted/loaded")

        text = str(payload.get("description", "")).strip()
        if not text:
            return IndexResult(
                index_name=self.name,
                score=0.0,
                model_version=self.version,
                narrative="No work description supplied; duplication could not be assessed.",
                meta={"status": "empty_input"},
            )

        top_k = int(payload.get("top_k", DUPLICATION_TOP_K))
        mask = self._candidate_mask(payload)
        sims, idx = self._similarities(text, mask)

        order = np.argsort(-sims)[:top_k]
        best_local = int(order[0])
        best_global = int(idx[best_local])
        raw_best = float(sims[best_local])
        index_score = self._calibrate(raw_best)

        matches = []
        for o in order:
            g = int(idx[int(o)])
            row = {} if self.corpus_meta is None else self.corpus_meta.iloc[g].to_dict()
            matches.append(
                {
                    "work_id": row.get("work_id"),
                    "district": row.get("district"),
                    "village": row.get("village"),
                    "category": row.get("category"),
                    "sanction_year": row.get("sanction_year"),
                    "sanctioned_cost_lakh": row.get("sanctioned_cost_lakh"),
                    "description": self.corpus_texts[g],
                    "raw_cosine": round(float(sims[int(o)]), 4),
                    "calibrated_index": round(self._calibrate(float(sims[int(o)])), 4),
                }
            )

        drivers: list[Driver] = []
        if explain and SHAP_ENABLED:
            drivers = self._explain(text, best_global)

        return IndexResult(
            index_name=self.name,
            score=index_score,
            model_version=self.version,
            drivers=drivers,
            narrative=self._narrative(index_score, matches[0]),
            meta={
                "raw_cosine_similarity": round(raw_best, 4),
                "calibration": {"floor": self.floor, "ceiling": self.ceiling},
                "corpus_size": len(self.corpus_texts),
                "candidates_compared": int(mask.sum()),
                "top_matches": matches,
            },
        )

    def score_batch(self, descriptions: Sequence[str]) -> list[float]:
        """Vectorised path for bulk ingestion - no SHAP, maximum throughput."""
        if not self.is_ready():
            raise RuntimeError("DuplicationIndexModel is not fitted/loaded")
        q = _l2_normalise(np.asarray(self.encoder.encode(list(descriptions)), dtype=np.float32))
        sims = q @ self.embeddings.T                 # (B, N)
        return [self._calibrate(float(s)) for s in sims.max(axis=1)]

    # -- explainability ----------------------------------------------------- #

    def _explain(self, text: str, reference_idx: int) -> list[Driver]:
        """
        Word-level SHAP attribution for the similarity score.

        We freeze the best-matching historical work as the reference, then let
        SHAP mask out words of the *new* description and observe the drop in
        similarity. Words whose removal collapses the score are the words that
        caused the duplicate flag.
        """
        try:
            import shap
        except ImportError:  # pragma: no cover
            logger.warning("shap not installed; skipping duplication explanation")
            return self._occlusion_fallback(text, reference_idx)

        reference_vec = self.embeddings[reference_idx]

        def f(texts: np.ndarray) -> np.ndarray:
            cleaned = [t if str(t).strip() else " " for t in list(texts)]
            emb = _l2_normalise(np.asarray(self.encoder.encode(cleaned), dtype=np.float32))
            sims = emb @ reference_vec
            return np.array([self._calibrate(float(s)) for s in sims])

        try:
            masker = shap.maskers.Text(r"\W+")
            explainer = shap.Explainer(f, masker, silent=True)
            sv = explainer([text])
            tokens = list(sv.data[0])
            values = np.asarray(sv.values[0], dtype=float).ravel()
        except Exception as exc:  # pragma: no cover - SHAP text path is fragile
            logger.warning("SHAP text explainer failed (%s); using occlusion", exc)
            return self._occlusion_fallback(text, reference_idx)

        return self._to_drivers(tokens, values)

    def _occlusion_fallback(self, text: str, reference_idx: int) -> list[Driver]:
        """
        Leave-one-word-out attribution: a fast, dependency-free approximation of
        the same idea SHAP implements exactly.
        """
        words = [w for w in re.split(r"\s+", text) if w]
        if not words:
            return []
        reference_vec = self.embeddings[reference_idx]

        variants = [text] + [" ".join(words[:i] + words[i + 1:]) for i in range(len(words))]
        emb = _l2_normalise(np.asarray(self.encoder.encode(variants), dtype=np.float32))
        scores = np.array([self._calibrate(float(v)) for v in emb @ reference_vec])
        contributions = scores[0] - scores[1:]      # drop when the word is removed
        return self._to_drivers(words, contributions)

    @staticmethod
    def _to_drivers(tokens: list[str], values: np.ndarray) -> list[Driver]:
        total = float(np.abs(values).sum()) or 1.0
        order = np.argsort(-np.abs(values))[:SHAP_TOP_DRIVERS]
        out = []
        for i in order:
            tok = str(tokens[int(i)]).strip()
            if not tok:
                continue
            v = float(values[int(i)])
            out.append(
                Driver(
                    feature=tok,
                    value="token",
                    contribution=round(v, 6),
                    contribution_pct=round(100.0 * abs(v) / total, 2),
                    direction="increases_risk" if v > 0 else "reduces_risk",
                )
            )
        return out

    @staticmethod
    def _narrative(score: float, best: dict[str, Any]) -> str:
        wid = best.get("work_id", "an earlier work")
        year = best.get("sanction_year", "?")
        if score >= 0.75:
            return (
                f"Very high textual overlap with {wid} sanctioned in {year}. "
                f"Treat as a probable re-sanction of the same work until the "
                f"implementing agency proves otherwise."
            )
        if score >= 0.50:
            return (
                f"Substantial overlap with {wid} ({year}). Manual verification of "
                f"scope and location is recommended before fund release."
            )
        if score >= 0.25:
            return f"Some resemblance to {wid} ({year}), most likely a similar but distinct work."
        return "No meaningful overlap with previously sanctioned works."
