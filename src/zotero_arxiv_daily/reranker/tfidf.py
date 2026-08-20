from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import numpy as np

from .base import BaseReranker, register_reranker


@register_reranker("tfidf")
class TfidfReranker(BaseReranker):
    """Fast local similarity without a downloaded embedding model."""

    def get_similarity_score(self, s1: list[str], s2: list[str]) -> np.ndarray:
        texts = [text or "" for text in [*s1, *s2]]
        if not s1 or not s2:
            return np.zeros((len(s1), len(s2)), dtype=float)
        try:
            features = TfidfVectorizer(
                stop_words="english",
                ngram_range=(1, 2),
                max_features=50_000,
                sublinear_tf=True,
            ).fit_transform(texts)
        except ValueError:
            features = TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(3, 5),
                max_features=50_000,
                sublinear_tf=True,
            ).fit_transform(texts)
        return np.asarray(linear_kernel(features[:len(s1)], features[len(s1):]))
