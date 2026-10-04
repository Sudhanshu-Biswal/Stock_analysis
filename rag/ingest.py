"""
rag/ingest.py — pulls news/filings, chunks, embeds, stores

Every Document must carry a published_date -- undated content can't be
temporally filtered later and is a known staleness/hallucination risk
per the project design.
"""

from __future__ import annotations

import logging
import re
from datetime import date, datetime, timezone
from typing import Optional
from urllib.parse import quote

import feedparser
import httpx
from bs4 import BeautifulSoup
from dateutil import parser as dateutil_parser
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# SEC requires a descriptive User-Agent identifying the requester --
# replace with your own contact info before real use.
_SEC_USER_AGENT = "investment-ai-research-tool contact@example.com"
_SEC_HEADERS = {"User-Agent": _SEC_USER_AGENT}

_GOOGLE_NEWS_RSS = "https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
_SEC_TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
_SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:0>10}.json"
_SEC_ARCHIVE_BASE = "https://www.sec.gov/Archives/edgar/data"

_RELEVANT_FORMS = {"10-K", "10-Q"}
_MAX_FILING_CONTENT_CHARS = 5000
_MAX_FILINGS_PER_TICKER = 3

_cik_cache: dict[str, str] = {}


class Document(BaseModel):
    ticker: str
    doc_type: str  # "news" | "filing" | "transcript"
    title: str
    source: str
    url: Optional[str] = None
    published_date: date
    content: str
    ingested_at: datetime


def ingest_news(ticker: str, company_name: Optional[str] = None, max_articles: int = 15) -> list[Document]:
    """Pull recent news via Google News RSS (free, no key required)."""
    query = quote(company_name or ticker)
    feed_url = _GOOGLE_NEWS_RSS.format(query=query)

    try:
        feed = feedparser.parse(feed_url)
    except Exception as exc:
        logger.error(f"[{ticker}] Failed to fetch Google News RSS: {exc}")
        return []

    docs: list[Document] = []
    now = datetime.now(timezone.utc)
    for entry in feed.entries[:max_articles]:
        published_raw = entry.get("published") or entry.get("updated")
        if not published_raw:
            continue  # no date -> skip, per project policy

        try:
            published_dt = dateutil_parser.parse(published_raw)
        except (ValueError, TypeError):
            continue

        title = (entry.get("title") or "").strip()
        summary = (entry.get("summary") or "").strip()
        if not title or not summary:
            continue

        source_field = entry.get("source")
        source = source_field.get("title") if isinstance(source_field, dict) else None

        docs.append(Document(
            ticker=ticker,
            doc_type="news",
            title=title,
            source=source or "Google News",
            url=entry.get("link"),
            published_date=published_dt.date(),
            content=summary,
            ingested_at=now,
        ))

    logger.info(f"[{ticker}] Ingested {len(docs)} news documents")
    return docs


def _lookup_cik(ticker: str) -> Optional[str]:
    """Resolve a bare US ticker (no exchange suffix) to its SEC CIK."""
    key = ticker.upper()
    if key in _cik_cache:
        return _cik_cache[key]

    try:
        resp = httpx.get(_SEC_TICKER_MAP_URL, headers=_SEC_HEADERS, timeout=10.0)
        resp.raise_for_status()
        payload = resp.json()
    except Exception as exc:
        logger.error(f"Failed to fetch SEC ticker map: {exc}")
        return None

    for entry in payload.values():
        _cik_cache[str(entry["ticker"]).upper()] = str(entry["cik_str"])

    return _cik_cache.get(key)


def _strip_html(html: str) -> str:
    text = BeautifulSoup(html, "html.parser").get_text(separator=" ", strip=True)
    return re.sub(r"\s+", " ", text)[:_MAX_FILING_CONTENT_CHARS]


def ingest_filings(ticker: str, market: str) -> list[Document]:
    """
    US: SEC EDGAR submissions API (free, documented, no key) for the most
    recent 10-K/10-Q filings.
    India: not implemented in Phase 2 v1 -- BSE/NSE scraping is too fragile
    for a solo build; this is a known gap, not faked.
    """
    if market != "US":
        logger.info(f"[{ticker}] Filing ingestion skipped: Indian filings unsupported in Phase 2 v1 (known gap).")
        return []

    cik = _lookup_cik(ticker)
    if cik is None:
        logger.warning(f"[{ticker}] Could not resolve SEC CIK; skipping filing ingestion.")
        return []

    try:
        resp = httpx.get(_SEC_SUBMISSIONS_URL.format(cik=cik), headers=_SEC_HEADERS, timeout=10.0)
        resp.raise_for_status()
        submissions = resp.json()
    except Exception as exc:
        logger.error(f"[{ticker}] SEC submissions fetch failed: {exc}")
        return []

    recent = submissions.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    dates = recent.get("filingDate", [])
    accession_numbers = recent.get("accessionNumber", [])
    primary_docs = recent.get("primaryDocument", [])

    docs: list[Document] = []
    now = datetime.now(timezone.utc)
    cik_int = int(cik)

    for form, filed_raw, accession, primary_doc in zip(forms, dates, accession_numbers, primary_docs):
        if form not in _RELEVANT_FORMS:
            continue
        try:
            filed_date = dateutil_parser.parse(filed_raw).date()
        except (ValueError, TypeError):
            continue

        accession_nodash = accession.replace("-", "")
        doc_url = f"{_SEC_ARCHIVE_BASE}/{cik_int}/{accession_nodash}/{primary_doc}"

        try:
            filing_resp = httpx.get(doc_url, headers=_SEC_HEADERS, timeout=15.0)
            filing_resp.raise_for_status()
            content = _strip_html(filing_resp.text)
        except Exception as exc:
            logger.warning(f"[{ticker}] Could not fetch filing document {doc_url}: {exc}")
            continue

        docs.append(Document(
            ticker=ticker,
            doc_type="filing",
            title=f"{form} filed {filed_date.isoformat()}",
            source="SEC EDGAR",
            url=doc_url,
            published_date=filed_date,
            content=content,
            ingested_at=now,
        ))

        if len(docs) >= _MAX_FILINGS_PER_TICKER:
            break

    logger.info(f"[{ticker}] Ingested {len(docs)} filing documents")
    return docs
