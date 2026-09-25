from .base import BaseRetriever, register_retriever
from arxiv import Result as ArxivResult
from ..protocol import Paper
from ..utils import extract_markdown_from_pdf, extract_tex_code_from_tar
from tempfile import TemporaryDirectory
import feedparser
import multiprocessing
import os
from queue import Empty
from time import sleep
from typing import Any, Callable, TypeVar
from loguru import logger
import requests
import re
import calendar
from datetime import datetime, timezone

T = TypeVar("T")

ARXIV_ID_PATTERN = re.compile(
    r"^(?P<base>(?:\d{4}\.\d{4,5}|[A-Za-z.-]+/\d{7}))v(?P<version>[1-9]\d*)$"
)

DOWNLOAD_TIMEOUT = (10, 60)
PDF_EXTRACT_TIMEOUT = 180
TAR_EXTRACT_TIMEOUT = 180


def _download_file(url: str, path: str) -> None:
    with requests.get(url, stream=True, timeout=DOWNLOAD_TIMEOUT) as response:
        response.raise_for_status()
        with open(path, "wb") as file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    file.write(chunk)


def _run_in_subprocess(
    result_queue: Any,
    func: Callable[..., T | None],
    args: tuple[Any, ...],
) -> None:
    try:
        result_queue.put(("ok", func(*args)))
    except Exception as exc:
        result_queue.put(("error", f"{type(exc).__name__}: {exc}"))


def _run_with_hard_timeout(
    func: Callable[..., T | None],
    args: tuple[Any, ...],
    *,
    timeout: float,
    operation: str,
    paper_title: str,
) -> T | None:
    start_methods = multiprocessing.get_all_start_methods()
    context = multiprocessing.get_context("fork" if "fork" in start_methods else start_methods[0])
    result_queue = context.Queue()
    process = context.Process(target=_run_in_subprocess, args=(result_queue, func, args))
    process.start()

    try:
        status, payload = result_queue.get(timeout=timeout)
    except Empty:
        if process.is_alive():
            process.kill()
        process.join(5)
        result_queue.close()
        result_queue.join_thread()
        logger.warning(f"{operation} timed out for {paper_title} after {timeout} seconds")
        return None

    process.join(5)
    result_queue.close()
    result_queue.join_thread()

    if status == "ok":
        return payload

    logger.warning(f"{operation} failed for {paper_title}: {payload}")
    return None


def _extract_text_from_pdf_worker(pdf_url: str) -> str:
    with TemporaryDirectory() as temp_dir:
        path = os.path.join(temp_dir, "paper.pdf")
        _download_file(pdf_url, path)
        return extract_markdown_from_pdf(path)


def _extract_text_from_html_worker(html_url: str) -> str | None:
    import trafilatura

    downloaded = trafilatura.fetch_url(html_url)
    if downloaded is None:
        raise ValueError(f"Failed to download HTML from {html_url}")
    text = trafilatura.extract(downloaded, include_comments=False, include_tables=False)
    if not text:
        raise ValueError(f"No text extracted from {html_url}")
    return text


def _extract_text_from_tar_worker(source_url: str, paper_id: str, paper_title: str | None = None) -> str | None:
    with TemporaryDirectory() as temp_dir:
        path = os.path.join(temp_dir, "paper.tar.gz")
        _download_file(source_url, path)
        file_contents = extract_tex_code_from_tar(path, paper_id, paper_title=paper_title)
        if not file_contents or "all" not in file_contents:
            raise ValueError("Main tex file not found.")
        return file_contents["all"]


# Adapted from TideDra/zotero-arxiv-daily#301. Keep versioned identity:
# RSS alternate links omit the version even when the OAI identifier includes it.
HTML_TAG_PATTERN = re.compile(
    r"</?(?:p|a|span|div|b|i|strong|em|br|hr|h[1-6]|ul|ol|li|sub|sup|table|tr|td|th)\b(?:\s+[^>]*)?/?>",
    flags=re.IGNORECASE,
)
ARXIV_HEADER_PATTERN = re.compile(
    r"^(?:arxiv:\s*\S+(?:\s+\[[^\]]*\])?)?\s*(?:announce\s+type:\s*[\w-]+)?\s*(?:abstract:\s*)?",
    flags=re.IGNORECASE,
)


def _clean_abstract(summary: str) -> str:
    cleaned = HTML_TAG_PATTERN.sub(" ", summary).strip()
    return " ".join(ARXIV_HEADER_PATTERN.sub("", cleaned).split())


def _parse_entry_time(value: Any) -> datetime | None:
    if value is None:
        return None
    return datetime.fromtimestamp(calendar.timegm(value), tz=timezone.utc)


def _entry_to_arxiv_result(entry: Any) -> ArxivResult:
    paper_id = entry.get("id", "").removeprefix("oai:arXiv.org:")
    if not ARXIV_ID_PATTERN.fullmatch(paper_id):
        raise ValueError(f"Invalid versioned arXiv RSS ID: {paper_id}")
    published = _parse_entry_time(entry.get("published_parsed"))
    if published is None:
        raise ValueError(f"Missing publication date in arXiv RSS entry: {paper_id}")
    title = " ".join(entry.get("title", "").split())
    summary = _clean_abstract(entry.get("summary", ""))
    if not title or not summary:
        raise ValueError(f"Missing title or abstract in arXiv RSS entry: {paper_id}")
    # Atom RSS packs the complete comma-separated author list into one name.
    names = [a.get("name", "") for a in entry.get("authors", [])]
    if not names:
        names = [entry.get("author", "")]
    authors = [ArxivResult.Author(name.strip()) for group in names
               for name in group.split(",") if name.strip()]
    if not authors:
        raise ValueError(f"Missing authors in arXiv RSS entry: {paper_id}")
    categories = [tag["term"] for tag in entry.get("tags", []) if tag.get("term")]
    primary = entry.get("arxiv_primary_category", {})
    primary = primary.get("term", "") if isinstance(primary, dict) else primary
    link = f"https://arxiv.org/abs/{paper_id}"
    return ArxivResult(
        entry_id=link,
        published=published,
        updated=_parse_entry_time(entry.get("updated_parsed")),
        title=title,
        summary=summary,
        authors=authors,
        categories=categories,
        primary_category=primary or (categories[0] if categories else ""),
        comment=entry.get("arxiv_comment", ""),
        journal_ref=entry.get("arxiv_journal_reference", entry.get("arxiv_journal_ref", "")),
        doi=entry.get("arxiv_doi", ""),
        links=[
            ArxivResult.Link(href=link, rel="alternate"),
            ArxivResult.Link(href=f"https://arxiv.org/pdf/{paper_id}", title="pdf", rel="related"),
        ],
    )


@register_retriever("arxiv")
class ArxivRetriever(BaseRetriever):
    def __init__(self, config):
        super().__init__(config)
        if self.config.source.arxiv.category is None:
            raise ValueError("category must be specified for arxiv.")

    def _retrieve_raw_papers(self) -> list[ArxivResult]:
        query = '+'.join(self.config.source.arxiv.category)
        include_cross_list = self.config.source.arxiv.get("include_cross_list", False)
        rss_url = f"https://rss.arxiv.org/atom/{query}"
        for attempt in range(5):
            feed = feedparser.parse(rss_url)
            title = feed.get("feed", {}).get("title", "")
            if "Feed error for query" in title:
                raise ValueError(f"Invalid ARXIV_QUERY: {query}.")
            status = feed.get("status", 200)
            # Even a partly parsed feed must not silently lose papers.
            if (200 <= status < 300 and not feed.get("bozo", False)
                    and feed.get("version") in {"atom10", "rss20"} and title):
                break
            if attempt == 4:
                raise RuntimeError(
                    f"Invalid arXiv RSS feed: {rss_url} "
                    f"(status={status}, error={feed.get('bozo_exception')})"
                )
            logger.warning(f"Invalid arXiv RSS feed; retrying in 5s: {rss_url}")
            sleep(5)

        allowed_announce_types = {"new", "cross"} if include_cross_list else {"new"}
        raw_papers = []
        seen = set()
        for entry in feed.entries:
            if entry.get("arxiv_announce_type", "new") not in allowed_announce_types:
                continue
            paper = _entry_to_arxiv_result(entry)
            if paper.entry_id in seen:
                continue
            seen.add(paper.entry_id)
            raw_papers.append(paper)
            if self.config.executor.debug and len(raw_papers) == 10:
                break

        return raw_papers

    def convert_to_paper(self, raw_paper: ArxivResult) -> Paper:
        source_id = _extract_arxiv_id(raw_paper)
        match = ARXIV_ID_PATTERN.fullmatch(source_id)
        if match is None:
            raise ValueError(f"Invalid versioned arXiv ID: {source_id}")

        title = raw_paper.title
        authors = [a.name for a in raw_paper.authors]
        abstract = raw_paper.summary
        full_text = None
        if self.config.source.arxiv.get("extract_full_text", True):
            full_text = extract_text_from_tar(raw_paper)
            if full_text is None:
                full_text = extract_text_from_html(raw_paper)
            if full_text is None:
                full_text = extract_text_from_pdf(raw_paper)
        return Paper(
            source=self.name,
            title=title,
            authors=authors,
            abstract=abstract,
            url=f"https://arxiv.org/abs/{source_id}",
            pdf_url=f"https://arxiv.org/pdf/{source_id}",
            full_text=full_text,
            source_id=source_id,
            base_id=match.group("base"),
            version=int(match.group("version")),
            published_at=getattr(raw_paper, "published", None),
        )


def _extract_arxiv_id(raw_paper: ArxivResult) -> str:
    get_short_id = getattr(raw_paper, "get_short_id", None)
    if callable(get_short_id):
        source_id = get_short_id()
    else:
        marker = "/abs/"
        if marker not in raw_paper.entry_id:
            raise ValueError(f"Cannot extract arXiv ID from {raw_paper.entry_id}")
        source_id = raw_paper.entry_id.split(marker, 1)[1]
    return source_id.split("?", 1)[0].split("#", 1)[0].rstrip("/")


def extract_text_from_html(paper: ArxivResult) -> str | None:
    html_url = paper.entry_id.replace("/abs/", "/html/")
    try:
        return _extract_text_from_html_worker(html_url)
    except Exception as exc:
        logger.warning(f"HTML extraction failed for {paper.title}: {exc}")
        return None


def extract_text_from_pdf(paper: ArxivResult) -> str | None:
    if paper.pdf_url is None:
        logger.warning(f"No PDF URL available for {paper.title}")
        return None
    return _run_with_hard_timeout(
        _extract_text_from_pdf_worker,
        (paper.pdf_url,),
        timeout=PDF_EXTRACT_TIMEOUT,
        operation="PDF extraction",
        paper_title=paper.title,
    )


def extract_text_from_tar(paper: ArxivResult) -> str | None:
    source_url = paper.source_url()
    if source_url is None:
        logger.warning(f"No source URL available for {paper.title}")
        return None
    return _run_with_hard_timeout(
        _extract_text_from_tar_worker,
        (source_url, paper.entry_id, paper.title),
        timeout=TAR_EXTRACT_TIMEOUT,
        operation="Tar extraction",
        paper_title=paper.title,
    )
