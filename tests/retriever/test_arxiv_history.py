from datetime import date
import pytest
from zotero_arxiv_daily.retriever.arxiv_history import parse_listing, parse_abstract

LIST = '''<dl><h3>Tue, 22 Sep 2026 (showing 3 of 3 entries )</h3>
<dt><a title="Abstract" href="/abs/2609.22267">paper</a></dt><dd></dd>
<dt><a title="Abstract" href="/abs/2609.22268">paper</a> (cross-list from cs.RO)</dt><dd></dd>
<dt><a title="Abstract" href="/abs/2609.22269">paper</a> (replaced)</dt><dd></dd></dl>'''
ABS = '''<html><head>
<meta name="citation_arxiv_id" content="2609.22267">
<meta name="citation_title" content="Example">
<meta name="citation_author" content="Family, Given">
<meta name="citation_date" content="2026/09/08">
<meta name="citation_abstract" content="Abstract body">
</head><body><span class="arxivid">arXiv:2609.22267v1 [cs.CV]</span></body></html>'''


def test_listing_filters_and_missing_day():
    assert parse_listing(LIST, date(2026,9,22), False) == ['2609.22267v1']
    assert parse_listing(LIST, date(2026,9,22), True) == ['2609.22267v1','2609.22268v1']
    assert parse_listing(LIST, date(2026,9,23), True) is None


def test_partial_listing_fails():
    with pytest.raises(ValueError, match='Incomplete'):
        parse_listing(LIST.replace('3 of 3','3 of 4'), date(2026,9,22), True)
    with pytest.raises(ValueError, match='Incomplete'):
        parse_listing(LIST.replace('3 of 3','4 of 4'), date(2026,9,22), True)


def test_abstract_preserves_identity_author_and_real_date():
    result = parse_abstract(ABS, '2609.22267v1')
    assert result.get_short_id() == '2609.22267v1'
    assert result.authors[0].name == 'Family, Given'
    assert result.published.isoformat() == '2026-09-08T00:00:00+00:00'
    assert result.pdf_url.endswith('2609.22267v1')


@pytest.mark.parametrize('replacement', [
    ('citation_date','wrong_date'),('2609.22267v1','2609.22267v2'),
    ('citation_author','wrong_author'),('citation_abstract','wrong_abstract'),
])
def test_bad_abs_fails(replacement):
    with pytest.raises(ValueError):
        parse_abstract(ABS.replace(*replacement), '2609.22267v1')


def test_history_deduplicates_and_throttles(config, monkeypatch):
    from types import SimpleNamespace
    from omegaconf import open_dict
    from zotero_arxiv_daily.retriever import arxiv_history
    with open_dict(config):
        config.source.arxiv.announcement_date = '2026-09-22'
        config.source.arxiv.category = ['cs.CV', 'cs.RO']
    urls, waits = [], []
    def get(url, **kwargs):
        urls.append(url)
        assert kwargs['timeout'] == (10, 60)
        return SimpleNamespace(text=LIST if '/list/' in url else ABS, raise_for_status=lambda: None)
    monkeypatch.setattr(arxiv_history.requests, 'get', get)
    monkeypatch.setattr(arxiv_history, 'monotonic', lambda: 1.0)
    monkeypatch.setattr(arxiv_history, 'sleep', waits.append)
    papers = arxiv_history.retrieve_history(config)
    assert len(papers) == 1
    assert len(urls) == 3
    assert waits == [3.0, 3.0]


def test_history_missing_day_never_becomes_empty_success(config, monkeypatch):
    from types import SimpleNamespace
    from omegaconf import open_dict
    from zotero_arxiv_daily.retriever import arxiv_history
    with open_dict(config):
        config.source.arxiv.announcement_date = '2026-09-23'
    urls = []
    def get(url, **kwargs):
        urls.append(url)
        return SimpleNamespace(text=LIST, raise_for_status=lambda: None)
    monkeypatch.setattr(arxiv_history.requests, 'get', get)
    monkeypatch.setattr(arxiv_history, 'sleep', lambda _: None)
    with pytest.raises(ValueError, match='absent'):
        arxiv_history.retrieve_history(config)
    assert '/2609?show=2000' in urls[-1]


def test_listing_uses_only_matching_day_group():
    other = LIST.replace('22 Sep', '23 Sep').replace('2609.22267', '2609.25017')
    page = '<html><body>' + other + LIST + '</body></html>'
    assert parse_listing(page, date(2026, 9, 22), False) == ['2609.22267v1']
    assert parse_listing(page, date(2026, 9, 23), False) == ['2609.25017v1']


@pytest.mark.parametrize('day', ['2026-02-30', 'not-a-date', '9999-01-01'])
def test_invalid_history_date_fails_before_network(config, monkeypatch, day):
    from omegaconf import open_dict
    from zotero_arxiv_daily.retriever import arxiv_history
    with open_dict(config):
        config.source.arxiv.announcement_date = day
    monkeypatch.setattr(arxiv_history.requests, 'get',
                        lambda *args, **kwargs: pytest.fail('Invalid date must not fetch'))
    with pytest.raises(ValueError):
        arxiv_history.retrieve_history(config)


@pytest.mark.parametrize('failure,status,expected_calls', [
    ('connection', None, 3), ('http', 503, 3), ('http', 429, 3), ('http', 404, 1),
])
def test_history_request_failure_is_bounded(config, monkeypatch, failure, status, expected_calls):
    from omegaconf import open_dict
    import requests
    from zotero_arxiv_daily.retriever import arxiv_history
    with open_dict(config):
        config.source.arxiv.announcement_date = '2026-09-22'
    calls, waits = [], []
    def get(url, **kwargs):
        calls.append(url)
        if failure == 'connection':
            raise requests.ConnectionError('temporary connection failure')
        response = requests.Response()
        response.status_code = status
        return response
    monkeypatch.setattr(arxiv_history.requests, 'get', get)
    monkeypatch.setattr(arxiv_history, 'sleep', waits.append)
    with pytest.raises(requests.RequestException):
        arxiv_history.retrieve_history(config)
    assert len(calls) == expected_calls
    assert len([wait for wait in waits if wait >= 10]) == expected_calls - 1


def test_history_transient_request_recovers(config, monkeypatch):
    from omegaconf import open_dict
    import requests
    from zotero_arxiv_daily.retriever import arxiv_history
    with open_dict(config):
        config.source.arxiv.announcement_date = '2026-09-22'
        config.source.arxiv.category = ['cs.CV']
    calls, waits = [], []
    def get(url, **kwargs):
        calls.append(url)
        response = requests.Response()
        response.status_code = 503 if len(calls) == 1 else 200
        response._content = (LIST if '/list/' in url else ABS).encode()
        return response
    monkeypatch.setattr(arxiv_history.requests, 'get', get)
    monkeypatch.setattr(arxiv_history, 'sleep', waits.append)
    assert len(arxiv_history.retrieve_history(config)) == 1
    assert len(calls) == 3
    assert len([wait for wait in waits if wait >= 10]) == 1
