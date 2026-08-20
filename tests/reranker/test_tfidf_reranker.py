from datetime import datetime

import numpy as np

from zotero_arxiv_daily.protocol import CorpusPaper, Paper
from zotero_arxiv_daily.reranker.tfidf import TfidfReranker


def test_tfidf_similarity_is_local_and_semantic(config):
    reranker = TfidfReranker(config)
    similarity = reranker.get_similarity_score(
        ["robot manipulation policy learning", "medical image segmentation"],
        ["robot policy for dexterous manipulation", "language model alignment"],
    )

    assert similarity.shape == (2, 2)
    assert similarity[0, 0] > similarity[0, 1]


def test_tfidf_rerank_assigns_scores_without_model_download(config):
    reranker = TfidfReranker(config)
    corpus = [
        CorpusPaper(
            title="Robot policy",
            abstract="dexterous robot manipulation policy learning",
            added_date=datetime(2026, 8, 1),
            paths=["robotics"],
        )
    ]
    candidates = [
        Paper(
            source="arxiv",
            title="Relevant",
            authors=["A"],
            abstract="robot manipulation policy",
            url="https://arxiv.org/abs/2608.00001v1",
        ),
        Paper(
            source="arxiv",
            title="Unrelated",
            authors=["B"],
            abstract="medical image segmentation",
            url="https://arxiv.org/abs/2608.00002v1",
        ),
    ]

    ranked = reranker.rerank(candidates, corpus)

    assert [paper.title for paper in ranked] == ["Relevant", "Unrelated"]
    assert all(isinstance(paper.score, (float, np.floating)) for paper in ranked)
