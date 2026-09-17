"""
Offline fallback encoder.

`SentenceBertEncoder` needs to download ~90 MB of weights the first time it
runs. On a locked-down demo laptop, a hackathon venue Wi-Fi, or a CI runner
that has no internet, that download fails and the whole pipeline dies.

`TfidfSvdEncoder` is a drop-in replacement built only from scikit-learn:
character + word TF-IDF reduced by truncated SVD to a dense vector. It is
genuinely worse at paraphrases than Sentence-BERT - that is the honest
trade-off - but it keeps the API, the SHAP explanations and the demo alive.

Use it explicitly:

    from predictive_ai.utils.offline_encoder import TfidfSvdEncoder
    model = DuplicationIndexModel(encoder=TfidfSvdEncoder())
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


class TfidfSvdEncoder:
    """Sentence encoder with the same `.encode()` signature as Sentence-BERT."""

    model_name = "tfidf-svd-offline"

    def __init__(self, n_components: int = 128, seed: int = 42) -> None:
        self.n_components = n_components
        self.seed = seed
        self._pipeline = None

    def _build(self):
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.pipeline import FeatureUnion, Pipeline
        from sklearn.preprocessing import Normalizer

        union = FeatureUnion(
            [
                ("word", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=1,
                                         sublinear_tf=True, stop_words="english")),
                ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2,
                                         sublinear_tf=True)),
            ]
        )
        return Pipeline(
            [
                ("tfidf", union),
                ("svd", TruncatedSVD(n_components=self.n_components, random_state=self.seed)),
                ("norm", Normalizer(copy=False)),
            ]
        )

    def fit(self, corpus: Sequence[str]) -> "TfidfSvdEncoder":
        self._pipeline = self._build()
        n_comp = min(self.n_components, max(2, len(set(corpus)) - 1))
        self._pipeline.named_steps["svd"].n_components = n_comp
        self._pipeline.fit(list(corpus))
        return self

    def encode(self, texts: Sequence[str], **kwargs: Any) -> np.ndarray:
        texts = list(texts)
        if self._pipeline is None:
            # First call is the corpus fit, exactly how DuplicationIndexModel.fit uses it.
            self.fit(texts)
        return np.asarray(self._pipeline.transform(texts), dtype=np.float32)
