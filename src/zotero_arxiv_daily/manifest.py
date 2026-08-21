from __future__ import annotations

from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
from typing import Any
from urllib.parse import urlparse

from .protocol import Paper


SCHEMA = "paper2video.daily-recommendations/v1"
ARXIV_ID_PATTERN = re.compile(
    r"^(?P<base>(?:\d{4}\.\d{4,5}|[A-Za-z.-]+/\d{7}))v(?P<version>[1-9]\d*)$"
)
SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")
ROOT_KEYS = {
    "schema",
    "producer",
    "generated_at",
    "categories",
    "limit",
    "candidate_count",
    "papers",
}
PRODUCER_KEYS = {"repository", "workflow_run_id", "head_sha"}
PAPER_KEYS = {
    "queue_key",
    "rank",
    "source",
    "source_id",
    "base_id",
    "version",
    "title",
    "authors",
    "abstract",
    "entry_url",
    "pdf_url",
    "published_at",
    "score",
    "tldr",
}


class ManifestValidationError(ValueError):
    pass


def _utc_timestamp(value: datetime, field: str) -> str:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ManifestValidationError(f"{field} must be a timezone-aware datetime")
    return value.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def producer_from_environment() -> dict[str, Any]:
    repository = os.environ.get("GITHUB_REPOSITORY", "")
    run_id = os.environ.get("GITHUB_RUN_ID", "")
    head_sha = os.environ.get("GITHUB_SHA", "")
    if not repository or not run_id or not head_sha:
        raise ManifestValidationError(
            "manifest output requires GITHUB_REPOSITORY, GITHUB_RUN_ID, and GITHUB_SHA"
        )
    try:
        parsed_run_id = int(run_id)
    except ValueError as exc:
        raise ManifestValidationError("GITHUB_RUN_ID must be an integer") from exc
    return {
        "repository": repository,
        "workflow_run_id": parsed_run_id,
        "head_sha": head_sha,
    }


def build_manifest(
    papers: list[Paper],
    *,
    candidate_count: int,
    categories: list[str],
    producer: dict[str, Any],
    generated_at: datetime,
    limit: int = 10,
) -> dict[str, Any]:
    if not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0:
        raise ManifestValidationError("limit must be a positive integer")
    if not isinstance(candidate_count, int) or isinstance(candidate_count, bool) or candidate_count < 0:
        raise ManifestValidationError("candidate_count must be a non-negative integer")
    if len(papers) != min(limit, candidate_count):
        raise ManifestValidationError(
            "papers must contain exactly min(limit, candidate_count) ranked papers"
        )

    manifest_papers = []
    for rank, paper in enumerate(papers, start=1):
        if paper.source != "arxiv":
            raise ManifestValidationError(f"papers[{rank - 1}].source must be arxiv")
        source_id = paper.source_id or ""
        match = ARXIV_ID_PATTERN.fullmatch(source_id)
        if match is None:
            raise ManifestValidationError(f"papers[{rank - 1}].source_id is invalid")
        if paper.base_id != match.group("base"):
            raise ManifestValidationError(f"papers[{rank - 1}].base_id does not match source_id")
        if paper.version != int(match.group("version")):
            raise ManifestValidationError(f"papers[{rank - 1}].version does not match source_id")
        if not isinstance(paper.score, (int, float)) or isinstance(paper.score, bool):
            raise ManifestValidationError(f"papers[{rank - 1}].score must be numeric")
        score = float(paper.score)
        if not math.isfinite(score):
            raise ManifestValidationError(f"papers[{rank - 1}].score must be finite")

        manifest_papers.append(
            {
                "queue_key": f"arxiv:{source_id}",
                "rank": rank,
                "source": "arxiv",
                "source_id": source_id,
                "base_id": paper.base_id,
                "version": paper.version,
                "title": paper.title,
                "authors": list(paper.authors),
                "abstract": paper.abstract,
                "entry_url": paper.url,
                "pdf_url": paper.pdf_url,
                "published_at": _utc_timestamp(paper.published_at, f"papers[{rank - 1}].published_at"),
                "score": score,
                "tldr": paper.tldr,
            }
        )

    payload = {
        "schema": SCHEMA,
        "producer": dict(producer),
        "generated_at": _utc_timestamp(generated_at, "generated_at"),
        "categories": list(categories),
        "limit": limit,
        "candidate_count": candidate_count,
        "papers": manifest_papers,
    }
    validate_manifest(payload)
    return payload


def _require_non_empty_string(value: Any, field: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ManifestValidationError(f"{field} must be a non-empty string")


def _validate_https_url(value: Any, field: str, expected_path: str) -> None:
    _require_non_empty_string(value, field)
    parsed = urlparse(value)
    if parsed.scheme != "https" or parsed.netloc != "arxiv.org" or parsed.path != expected_path:
        raise ManifestValidationError(f"{field} must be the canonical arxiv.org URL")
    if parsed.params or parsed.query or parsed.fragment:
        raise ManifestValidationError(f"{field} must not contain URL parameters")


def _validate_timestamp(value: Any, field: str) -> None:
    _require_non_empty_string(value, field)
    if not value.endswith("Z"):
        raise ManifestValidationError(f"{field} must be UTC and end in Z")
    try:
        datetime.fromisoformat(value.removesuffix("Z") + "+00:00")
    except ValueError as exc:
        raise ManifestValidationError(f"{field} must be an ISO-8601 timestamp") from exc


def validate_manifest(payload: dict[str, Any]) -> None:
    if not isinstance(payload, dict) or set(payload) != ROOT_KEYS:
        raise ManifestValidationError("manifest root fields do not match the v1 schema")
    if payload["schema"] != SCHEMA:
        raise ManifestValidationError(f"schema must be {SCHEMA}")

    producer = payload["producer"]
    if not isinstance(producer, dict) or set(producer) != PRODUCER_KEYS:
        raise ManifestValidationError("producer fields do not match the v1 schema")
    repository = producer["repository"]
    _require_non_empty_string(repository, "producer.repository")
    if repository.count("/") != 1:
        raise ManifestValidationError("producer.repository must use owner/repository format")
    run_id = producer["workflow_run_id"]
    if not isinstance(run_id, int) or isinstance(run_id, bool) or run_id <= 0:
        raise ManifestValidationError("producer.workflow_run_id must be a positive integer")
    if not isinstance(producer["head_sha"], str) or SHA_PATTERN.fullmatch(producer["head_sha"]) is None:
        raise ManifestValidationError("producer.head_sha must be a 40-character lowercase Git SHA")

    _validate_timestamp(payload["generated_at"], "generated_at")
    categories = payload["categories"]
    if not isinstance(categories, list) or not categories or any(
        not isinstance(category, str) or not category for category in categories
    ):
        raise ManifestValidationError("categories must be a non-empty list of strings")
    limit = payload["limit"]
    if not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0:
        raise ManifestValidationError("limit must be a positive integer")
    candidate_count = payload["candidate_count"]
    if not isinstance(candidate_count, int) or isinstance(candidate_count, bool) or candidate_count < 0:
        raise ManifestValidationError("candidate_count must be a non-negative integer")
    papers = payload["papers"]
    if not isinstance(papers, list) or len(papers) != min(limit, candidate_count):
        raise ManifestValidationError("papers length must equal min(limit, candidate_count)")

    for index, paper in enumerate(papers):
        prefix = f"papers[{index}]"
        if not isinstance(paper, dict) or set(paper) != PAPER_KEYS:
            raise ManifestValidationError(f"{prefix} fields do not match the v1 schema")
        if paper["rank"] != index + 1:
            raise ManifestValidationError(f"{prefix}.rank must be contiguous from 1")
        if paper["source"] != "arxiv":
            raise ManifestValidationError(f"{prefix}.source must be arxiv")
        source_id = paper["source_id"]
        if not isinstance(source_id, str):
            raise ManifestValidationError(f"{prefix}.source_id must be a string")
        match = ARXIV_ID_PATTERN.fullmatch(source_id)
        if match is None:
            raise ManifestValidationError(f"{prefix}.source_id is invalid")
        if paper["queue_key"] != f"arxiv:{source_id}":
            raise ManifestValidationError(f"{prefix}.queue_key does not match source_id")
        if paper["base_id"] != match.group("base"):
            raise ManifestValidationError(f"{prefix}.base_id does not match source_id")
        if paper["version"] != int(match.group("version")):
            raise ManifestValidationError(f"{prefix}.version does not match source_id")
        for field in ("title", "abstract", "tldr"):
            _require_non_empty_string(paper[field], f"{prefix}.{field}")
        authors = paper["authors"]
        if not isinstance(authors, list) or not authors or any(
            not isinstance(author, str) or not author.strip() for author in authors
        ):
            raise ManifestValidationError(f"{prefix}.authors must be a non-empty list of strings")
        _validate_https_url(paper["entry_url"], f"{prefix}.entry_url", f"/abs/{source_id}")
        _validate_https_url(paper["pdf_url"], f"{prefix}.pdf_url", f"/pdf/{source_id}")
        _validate_timestamp(paper["published_at"], f"{prefix}.published_at")
        score = paper["score"]
        if not isinstance(score, (int, float)) or isinstance(score, bool) or not math.isfinite(score):
            raise ManifestValidationError(f"{prefix}.score must be finite and numeric")


def write_manifest(path: str | Path, payload: dict[str, Any]) -> Path:
    validate_manifest(payload)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    with output_path.open("x", encoding="utf-8") as output:
        output.write(serialized)
        output.flush()
        os.fsync(output.fileno())
    return output_path
