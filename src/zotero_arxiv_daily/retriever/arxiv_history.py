"""Operator-only recovery from official arXiv announcement lists and abs pages."""
from datetime import date, datetime, timezone
import re
from time import monotonic, sleep

from arxiv import Result
from lxml import html
import requests


def parse_listing(content: str, day: date, include_cross_list: bool) -> list[str] | None:
    document = html.fromstring(content)
    for header in document.xpath('//h3'):
        text = ' '.join(header.text_content().split())
        match = re.fullmatch(r'(\w{3}, \d{1,2} \w{3} \d{4}) \(showing (\d+) of (\d+) entries ?\)', text)
        if not match or datetime.strptime(match[1], '%a, %d %b %Y').date() != day:
            continue
        entries = header.getparent().xpath('./dt')
        if int(match[2]) != int(match[3]) or len(entries) != int(match[3]):
            raise ValueError(f'Incomplete arXiv announcement list for {day}: {text}')
        ids = []
        for entry in entries:
            label = entry.text_content().lower()
            if 'replac' in label or ('cross-list' in label and not include_cross_list):
                continue
            links = entry.xpath('./a[@title="Abstract"]/@href')
            if len(links) != 1 or not re.fullmatch(r'/abs/\d{4}\.\d{4,5}', links[0]):
                raise ValueError(f'Unexpected historical arXiv entry: {entry.text_content()}')
            ids.append(links[0].removeprefix('/abs/') + 'v1')
        return ids
    return None


def parse_abstract(content: str, paper_id: str) -> Result:
    document = html.fromstring(content)
    def values(name):
        return document.xpath(f'//meta[@name="{name}"]/@content')
    def required(name):
        found = values(name)
        if len(found) != 1 or not found[0].strip():
            raise ValueError(f'Missing/ambiguous {name} for {paper_id}')
        return found[0].strip()
    base = paper_id.removesuffix('v1')
    if required('citation_arxiv_id') not in {base, paper_id}:
        raise ValueError(f'Wrong arXiv identity for {paper_id}')
    versions = document.xpath('//span[@class="arxivid"]')
    if not any(re.search(r'\b' + re.escape(paper_id) + r'\b', e.text_content()) for e in versions):
        raise ValueError(f'Cannot verify requested version {paper_id}')
    authors = values('citation_author')
    if not authors or any(not a.strip() for a in authors):
        raise ValueError(f'Missing authors for {paper_id}')
    # The official citation date has day precision. Do not substitute retrieval time.
    published = datetime.strptime(required('citation_date'), '%Y/%m/%d').replace(tzinfo=timezone.utc)
    abstract = values('citation_abstract')
    if not abstract:
        abstract = [e.text_content().strip().removeprefix('Abstract:').strip()
                    for e in document.xpath('//blockquote[contains(@class,"abstract")]')]
    if len(abstract) != 1 or not abstract[0].strip():
        raise ValueError(f'Missing abstract for {paper_id}')
    return Result(entry_id=f'https://arxiv.org/abs/{paper_id}', title=required('citation_title'),
                  authors=[Result.Author(a) for a in authors], summary=abstract[0].strip(),
                  published=published, updated=None,
                  links=[Result.Link(f'https://arxiv.org/pdf/{paper_id}', title='pdf', rel='related')])


def retrieve_history(config) -> list[Result]:
    day = date.fromisoformat(str(config.source.arxiv.announcement_date))
    if day >= datetime.now(timezone.utc).date():
        raise ValueError('announcement_date must be a past arXiv announcement day')
    last_request = None
    def fetch(url):
        nonlocal last_request
        for attempt in range(3):
            if last_request is not None:
                sleep(max(0, 3 - (monotonic() - last_request)))
            last_request = monotonic()
            try:
                response = requests.get(url, timeout=(10, 60))
                response.raise_for_status()
                return response.text
            except requests.HTTPError as exc:
                status = exc.response.status_code if exc.response is not None else None
                if status != 429 and (status is None or status < 500):
                    raise
                if attempt == 2:
                    raise
            except (requests.ConnectionError, requests.Timeout):
                if attempt == 2:
                    raise
            sleep(10 * (attempt + 1))
    ids = []
    for category in config.source.arxiv.category:
        if not re.fullmatch(r'[A-Za-z.-]+', category):
            raise ValueError(f'Invalid arXiv category: {category}')
        entries = None
        for period in ('pastweek', day.strftime('%y%m')):
            entries = parse_listing(fetch(f'https://arxiv.org/list/{category}/{period}?show=2000'),
                                    day, config.source.arxiv.get('include_cross_list', False))
            if entries is not None:
                break
        if entries is None:
            raise ValueError(f'Announcement day {day} absent from official {category} lists')
        ids.extend(entries)
    ids = list(dict.fromkeys(ids))
    if config.executor.debug:
        ids = ids[:10]
    return [parse_abstract(fetch(f'https://arxiv.org/abs/{paper_id}'), paper_id) for paper_id in ids]
