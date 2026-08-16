"""
Diff-content embeddings — the actual differentiator this project was built
around. Uses a small pretrained sentence-embedding model (no training
needed, runs fine on CPU) to turn raw diff text into dense vectors, then
reduces dimensionality with PCA.

Leakage discipline: PCA is fit ONLY on training-set diff embeddings, same
rule as job_base_historical_fail_rate elsewhere in this project. The
embedding model itself is pretrained and frozen — using it isn't leakage,
but fitting PCA on test data would be.
"""

import logging

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)

_model = None


def get_embedder():
    global _model
    if _model is None:
        # Code-aware model (trained on code search pairs), not a general
        # English sentence model — a fairer test of whether diff CONTENT
        # helps, rather than surface-level English text patterns in comments.
        logger.info("Loading code-aware embedding model (first call downloads, then cached)...")
        _model = SentenceTransformer("flax-sentence-embeddings/st-codesearch-distilroberta-base")
    return _model


def embed_diffs(diff_texts: pd.Series, batch_size: int = 64) -> np.ndarray:
    model = get_embedder()
    texts = diff_texts.fillna("(no diff available)").astype(str).tolist()
    return model.encode(texts, batch_size=batch_size, show_progress_bar=True)


class DiffFeatureReducer:
    """Fit PCA on TRAIN diff embeddings only; apply the same transform to test."""

    def __init__(self, n_components: int = 12):
        self.pca = PCA(n_components=n_components, random_state=42)
        self._fitted = False

    def fit_transform(self, train_diff_texts: pd.Series) -> pd.DataFrame:
        logger.info("Embedding %d training diffs...", len(train_diff_texts))
        embeddings = embed_diffs(train_diff_texts)
        reduced = self.pca.fit_transform(embeddings)
        self._fitted = True
        explained = self.pca.explained_variance_ratio_.sum()
        logger.info("PCA fit on train diffs: %d components explain %.1f%% of variance",
                    reduced.shape[1], explained * 100)
        cols = [f"diff_pc_{i}" for i in range(reduced.shape[1])]
        return pd.DataFrame(reduced, columns=cols, index=train_diff_texts.index)

    def transform(self, diff_texts: pd.Series) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("Call fit_transform on training data before transform.")
        logger.info("Embedding %d test diffs...", len(diff_texts))
        embeddings = embed_diffs(diff_texts)
        reduced = self.pca.transform(embeddings)
        cols = [f"diff_pc_{i}" for i in range(reduced.shape[1])]
        return pd.DataFrame(reduced, columns=cols, index=diff_texts.index)