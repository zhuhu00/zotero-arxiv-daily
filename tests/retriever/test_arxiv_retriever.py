"""Tests for ArxivRetriever."""

import time
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from zotero_arxiv_daily.retriever.arxiv_retriever import ArxivRetriever, _run_with_hard_timeout
import zotero_arxiv_daily.retriever.arxiv_retriever as arxiv_retriever


def _sleep_and_return(value: str, delay_seconds: float) -> str:
    time.sleep(delay_seconds)
    return value


def _raise_runtime_error() -> None:
    raise RuntimeError("boom")


def test_arxiv_retriever(config, mock_feedparser, monkeypatch):
    monkeypatch.setattr("zotero_arxiv_daily.retriever.base.sleep", lambda _: None)

    import arxiv

    monkeypatch.setattr(arxiv, "Client", lambda **kw: pytest.fail("API must not be called"))
    new_entries = [e for e in mock_feedparser.entries
                   if e.get("arxiv_announce_type", "new") == "new"]

    # Skip file downloads in convert_to_paper
    monkeypatch.setattr(arxiv_retriever, "extract_text_from_html", lambda paper: None)
    monkeypatch.setattr(arxiv_retriever, "extract_text_from_pdf", lambda paper: None)
    monkeypatch.setattr(arxiv_retriever, "extract_text_from_tar", lambda paper: None)

    retriever = ArxivRetriever(config)
    papers = retriever.retrieve_papers()

    assert len(papers) == len(new_entries)
    assert set(p.title for p in papers) == set(e.title for e in new_entries)
    assert all(p.source_id and p.source_id.endswith("v1") for p in papers)
    assert all(p.base_id == p.source_id.removesuffix("v1") for p in papers)
    assert all(p.version == 1 for p in papers)
    assert all(p.published_at == datetime(2025, 8, 20, 4, tzinfo=timezone.utc) for p in papers)


def test_arxiv_retriever_can_skip_candidate_full_text(config, monkeypatch):
    from omegaconf import open_dict

    with open_dict(config):
        config.source.arxiv.extract_full_text = False

    raw_paper = SimpleNamespace(
        title="Lightweight candidate",
        authors=[SimpleNamespace(name="Test Author")],
        summary="Abstract only",
        pdf_url="https://arxiv.org/pdf/2608.12345v2",
        entry_id="https://arxiv.org/abs/2608.12345v2",
        published=datetime(2026, 8, 17, tzinfo=timezone.utc),
    )
    monkeypatch.setattr(
        arxiv_retriever,
        "extract_text_from_tar",
        lambda paper: pytest.fail("full text extraction must be skipped"),
    )
    paper = ArxivRetriever(config).convert_to_paper(raw_paper)

    assert paper.full_text is None
    assert paper.source_id == "2608.12345v2"
    assert paper.base_id == "2608.12345"
    assert paper.version == 2


def test_run_with_hard_timeout_returns_value():
    result = _run_with_hard_timeout(
        _sleep_and_return, ("done", 0.01), timeout=1, operation="test op", paper_title="paper"
    )
    assert result == "done"


def test_run_with_hard_timeout_returns_none_on_timeout(monkeypatch):
    warnings: list[str] = []
    monkeypatch.setattr(arxiv_retriever, "logger", SimpleNamespace(warning=warnings.append))
    result = _run_with_hard_timeout(
        _sleep_and_return, ("done", 1.0), timeout=0.01, operation="test op", paper_title="paper"
    )
    assert result is None
    assert "timed out" in warnings[0]


def test_run_with_hard_timeout_returns_none_on_failure(monkeypatch):
    warnings: list[str] = []
    monkeypatch.setattr(arxiv_retriever, "logger", SimpleNamespace(warning=warnings.append))
    result = _run_with_hard_timeout(
        _raise_runtime_error, (), timeout=1, operation="test op", paper_title="paper"
    )
    assert result is None
    assert "boom" in warnings[0]


@pytest.mark.parametrize('versioned_id', ['2609.12345v2', 'hep-th/9901001v3'])
def test_rss_preserves_version_and_urls(config, mock_feedparser, versioned_id):
    from copy import deepcopy
    entry = deepcopy(mock_feedparser.entries[0])
    entry['id'] = 'oai:arXiv.org:' + versioned_id
    result = arxiv_retriever._entry_to_arxiv_result(entry)
    assert result.get_short_id() == versioned_id
    assert result.pdf_url == 'https://arxiv.org/pdf/' + versioned_id
    assert result.source_url().endswith('/' + versioned_id)
    assert result.summary.startswith('As large language models')
    assert [a.name for a in result.authors] == ['Chunhua Liu', 'Kabir Manandhar Shrestha', 'Sukai Huang']


def test_missing_publication_date_is_not_fabricated(mock_feedparser):
    entry = mock_feedparser.entries[0]
    del entry['published_parsed']
    with pytest.raises(ValueError, match='Missing publication date'):
        arxiv_retriever._entry_to_arxiv_result(entry)


@pytest.mark.parametrize('paper_id', ['', '2609.12345', 'invalid'])
def test_invalid_rss_identity_fails(mock_feedparser, paper_id):
    entry = mock_feedparser.entries[0]
    entry['id'] = paper_id
    with pytest.raises(ValueError, match='versioned arXiv RSS ID'):
        arxiv_retriever._entry_to_arxiv_result(entry)


@pytest.mark.parametrize('changes', [
    {'status': 406}, {'bozo': True}, {'version': ''}, {'feed': {}},
])
def test_bad_feed_fails_even_with_partial_entries(config, mock_feedparser, monkeypatch, changes):
    mock_feedparser.update(changes)
    monkeypatch.setattr(arxiv_retriever, 'sleep', lambda _: None)
    with pytest.raises(RuntimeError, match='Invalid arXiv RSS feed'):
        ArxivRetriever(config)._retrieve_raw_papers()


def test_valid_empty_feed(config, mock_feedparser):
    mock_feedparser.entries = []
    assert ArxivRetriever(config)._retrieve_raw_papers() == []


def test_cross_list_duplicates_and_debug_limit(config, mock_feedparser):
    from copy import deepcopy
    from omegaconf import open_dict
    with open_dict(config):
        config.source.arxiv.include_cross_list = True
    expected = {e.id for e in mock_feedparser.entries
                if e.get('arxiv_announce_type', 'new') in {'new', 'cross'}}
    mock_feedparser.entries += deepcopy(mock_feedparser.entries)
    retriever = ArxivRetriever(config)
    assert len(retriever._retrieve_raw_papers()) == len(expected)
    seed = deepcopy(mock_feedparser.entries[0])
    seed['arxiv_announce_type'] = 'new'
    mock_feedparser.entries = []
    for index in range(15):
        entry = deepcopy(seed)
        entry['id'] = f'oai:arXiv.org:2609.{index:05d}v1'
        mock_feedparser.entries.append(entry)
    with open_dict(config):
        config.executor.debug = True
    assert len(retriever._retrieve_raw_papers()) == 10


@pytest.mark.parametrize('summary,expected', [
    ('<p>arXiv:2609.12345v1 Announce Type: new<br>Abstract: x < y > 0.</p>', 'x < y > 0.'),
    ('Plain abstract.', 'Plain abstract.'),
    ('arXiv:2609.12345v1 Announce Type: new\nAbstract: Abstract: a benchmark.', 'Abstract: a benchmark.'),
])
def test_abstract_cleaning(summary, expected):
    assert arxiv_retriever._clean_abstract(summary) == expected


def test_feed_retry_recovers(config, mock_feedparser, monkeypatch):
    from copy import deepcopy
    bad = deepcopy(mock_feedparser)
    bad['status'] = 503
    responses = iter([bad, mock_feedparser])
    waits = []
    monkeypatch.setattr(arxiv_retriever.feedparser, 'parse', lambda _: next(responses))
    monkeypatch.setattr(arxiv_retriever, 'sleep', waits.append)
    assert ArxivRetriever(config)._retrieve_raw_papers()
    assert waits == [5]


def test_missing_date_fails_before_conversion(config, mock_feedparser):
    entry = next(e for e in mock_feedparser.entries if e.arxiv_announce_type == 'new')
    del entry['published_parsed']
    with pytest.raises(ValueError, match='Missing publication date'):
        ArxivRetriever(config).retrieve_papers()


@pytest.mark.parametrize('changes,message', [
    ({'title': ''}, 'Missing title or abstract'),
    ({'summary': ''}, 'Missing title or abstract'),
    ({'authors': [], 'author': ''}, 'Missing authors'),
])
def test_missing_rss_metadata_fails_before_conversion(config, mock_feedparser, changes, message):
    entry = next(e for e in mock_feedparser.entries if e.arxiv_announce_type == 'new')
    entry.update(changes)
    with pytest.raises(ValueError, match=message):
        ArxivRetriever(config).retrieve_papers()


def test_optional_atom_metadata_is_preserved(mock_feedparser):
    entry = mock_feedparser.entries[0]
    entry.update({
        'arxiv_primary_category': {'term': 'cs.AI'},
        'arxiv_comment': '12 pages',
        'arxiv_journal_reference': 'Example journal',
        'arxiv_doi': '10.1234/example',
    })
    result = arxiv_retriever._entry_to_arxiv_result(entry)
    assert result.categories == ['cs.CL', 'cs.AI']
    assert result.primary_category == 'cs.AI'
    assert result.comment == '12 pages'
    assert result.journal_ref == 'Example journal'
    assert result.doi == '10.1234/example'
    assert result.updated == datetime(2025, 8, 20, 4, 0, 19, tzinfo=timezone.utc)
