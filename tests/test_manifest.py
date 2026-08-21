from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import pytest

from zotero_arxiv_daily.manifest import (
    ManifestValidationError,
    build_manifest,
    validate_manifest,
    write_manifest,
)
from zotero_arxiv_daily.protocol import Paper


GENERATED_AT = datetime(2026, 8, 17, 4, 0, tzinfo=timezone.utc)
PRODUCER = {
    "repository": "zhuhu00/zotero-arxiv-daily",
    "workflow_run_id": 31846447439,
    "head_sha": "0123456789abcdef0123456789abcdef01234567",
}


def _ranked_papers(count: int) -> list[Paper]:
    return [
        Paper(
            source="arxiv",
            title=f"Paper {index}",
            authors=["Author One", "Author Two"],
            abstract=f"Abstract {index}",
            url=f"https://arxiv.org/abs/2608.{index:05d}v1",
            pdf_url=f"https://arxiv.org/pdf/2608.{index:05d}v1",
            tldr=f"论文 {index} 的中文摘要",
            score=1.0 / index,
            source_id=f"2608.{index:05d}v1",
            base_id=f"2608.{index:05d}",
            version=1,
            published_at=datetime(2026, 8, 16, tzinfo=timezone.utc),
        )
        for index in range(1, count + 1)
    ]


@pytest.mark.parametrize("candidate_count", [0, 2, 10])
def test_build_manifest_handles_empty_fewer_and_top_ten(candidate_count):
    papers = _ranked_papers(candidate_count)
    payload = build_manifest(
        papers,
        candidate_count=candidate_count,
        categories=["cs.RO", "cs.CV"],
        producer=PRODUCER,
        generated_at=GENERATED_AT,
    )

    assert payload["limit"] == 10
    assert [paper["rank"] for paper in payload["papers"]] == list(
        range(1, candidate_count + 1)
    )
    assert [paper["queue_key"] for paper in payload["papers"]] == [
        f"arxiv:2608.{index:05d}v1" for index in range(1, candidate_count + 1)
    ]


def test_build_manifest_selects_exactly_ten_from_larger_candidate_set():
    payload = build_manifest(
        _ranked_papers(10),
        candidate_count=12,
        categories=["cs.RO", "cs.CV"],
        producer=PRODUCER,
        generated_at=GENERATED_AT,
    )
    assert len(payload["papers"]) == 10


@pytest.mark.parametrize("limit", [3, 25])
def test_build_manifest_accepts_configurable_positive_limit(limit):
    payload = build_manifest(
        _ranked_papers(limit),
        candidate_count=limit + 2,
        categories=["cs.RO", "cs.CV"],
        producer=PRODUCER,
        generated_at=GENERATED_AT,
        limit=limit,
    )

    assert payload["limit"] == limit
    assert len(payload["papers"]) == limit
    validate_manifest(payload)


@pytest.mark.parametrize("limit", [0, -1, True])
def test_build_manifest_rejects_non_positive_or_boolean_limit(limit):
    with pytest.raises(ManifestValidationError, match="positive integer"):
        build_manifest(
            [],
            candidate_count=0,
            categories=["cs.RO", "cs.CV"],
            producer=PRODUCER,
            generated_at=GENERATED_AT,
            limit=limit,
        )


def test_manifest_serialization_is_deterministic_and_refuses_overwrite(tmp_path):
    payload = build_manifest(
        _ranked_papers(2),
        candidate_count=2,
        categories=["cs.RO", "cs.CV"],
        producer=PRODUCER,
        generated_at=GENERATED_AT,
    )
    first = write_manifest(tmp_path / "first.json", payload)
    second = write_manifest(tmp_path / "second.json", payload)
    assert first.read_bytes() == second.read_bytes()
    assert json.loads(first.read_text(encoding="utf-8")) == payload
    with pytest.raises(FileExistsError):
        write_manifest(first, payload)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda payload: payload.update(schema="wrong"), "schema"),
        (lambda payload: payload["papers"][0].update(rank=2), "rank"),
        (lambda payload: payload["papers"][0].update(source_id="2608.00001"), "source_id"),
        (lambda payload: payload["papers"][0].update(pdf_url="http://example.com/a.pdf"), "pdf_url"),
        (lambda payload: payload["papers"][0].update(score=float("nan")), "score"),
        (lambda payload: payload["papers"][0].update(published_at="yesterday"), "published_at"),
    ],
)
def test_validate_manifest_rejects_invalid_fields(mutate, message):
    payload = build_manifest(
        _ranked_papers(1),
        candidate_count=1,
        categories=["cs.RO", "cs.CV"],
        producer=PRODUCER,
        generated_at=GENERATED_AT,
    )
    invalid = deepcopy(payload)
    mutate(invalid)
    with pytest.raises(ManifestValidationError, match=message):
        validate_manifest(invalid)


def test_build_manifest_rejects_missing_ranked_paper():
    with pytest.raises(ManifestValidationError, match="exactly"):
        build_manifest(
            _ranked_papers(1),
            candidate_count=2,
            categories=["cs.RO", "cs.CV"],
            producer=PRODUCER,
            generated_at=GENERATED_AT,
        )


def test_consumer_fixture_is_valid_and_checksum_is_recorded():
    fixture = Path(__file__).parent / "fixtures" / "daily_recommendations_v1.json"
    checksum_file = fixture.with_suffix(fixture.suffix + ".sha256")
    data = fixture.read_bytes()
    payload = json.loads(data)

    validate_manifest(payload)
    assert len(payload["papers"]) == 10
    assert hashlib.sha256(data).hexdigest() == checksum_file.read_text().split()[0]
