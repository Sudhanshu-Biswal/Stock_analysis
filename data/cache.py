"""
data/cache.py — SQLite caching layer

Stores fetched CompanyData as JSON payloads keyed by (ticker, data_type).
Staleness rules:
  - "price"      → 1 hour  (60 min)
  - "financials" → 168 hours (7 days)
  - "full"       → 24 hours  (default)

Usage:
    from data.cache import get_or_fetch
    data = get_or_fetch("AAPL")            # uses "full" staleness (24 h)
    data = get_or_fetch("AAPL", "price")   # uses 1 h staleness
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from sqlalchemy import Column, DateTime, Integer, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from data.fetcher import fetch_company_data
from data.models import CompanyData

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Database setup
# ---------------------------------------------------------------------------

# Default DB path: investment-ai/data/cache.db (next to this file)
_DEFAULT_DB_PATH = Path(__file__).parent / "cache.db"
_DB_URL = f"sqlite:///{_DEFAULT_DB_PATH}"


class _Base(DeclarativeBase):
    pass


class _CacheEntry(_Base):
    __tablename__ = "ticker_cache"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String(32), nullable=False, index=True)
    data_type = Column(String(32), nullable=False)          # "full", "price", "financials"
    fetched_at = Column(DateTime(timezone=True), nullable=False)
    payload_json = Column(Text, nullable=False)


_engine = create_engine(_DB_URL, connect_args={"check_same_thread": False})
_Base.metadata.create_all(_engine)
_SessionLocal = sessionmaker(bind=_engine, autocommit=False, autoflush=False)

# ---------------------------------------------------------------------------
# Staleness configuration
# ---------------------------------------------------------------------------

_STALENESS_HOURS: dict[str, float] = {
    "price": 1.0,
    "financials": 168.0,   # 7 days
    "full": 24.0,
}


def _max_age_for(data_type: str) -> float:
    return _STALENESS_HOURS.get(data_type, 24.0)


def _is_fresh(fetched_at: datetime, max_age_hours: float) -> bool:
    """Returns True if the cached entry is still within the staleness window."""
    # Ensure both datetimes are UTC-aware for comparison
    if fetched_at.tzinfo is None:
        fetched_at = fetched_at.replace(tzinfo=timezone.utc)
    age = datetime.now(timezone.utc) - fetched_at
    return age < timedelta(hours=max_age_hours)


# ---------------------------------------------------------------------------
# Cache read / write
# ---------------------------------------------------------------------------

def _read_cache(
    session: Session,
    ticker: str,
    data_type: str,
    max_age_hours: float,
) -> Optional[CompanyData]:
    entry = (
        session.query(_CacheEntry)
        .filter_by(ticker=ticker, data_type=data_type)
        .order_by(_CacheEntry.fetched_at.desc())
        .first()
    )
    if entry is None:
        return None

    if not _is_fresh(entry.fetched_at, max_age_hours):
        logger.debug(f"[{ticker}] Cache stale for data_type='{data_type}'")
        return None

    logger.info(f"[{ticker}] Cache hit for data_type='{data_type}'")
    try:
        payload = json.loads(entry.payload_json)
        return CompanyData.model_validate(payload)
    except Exception as exc:
        logger.warning(f"[{ticker}] Failed to deserialise cache entry: {exc}")
        return None


def _write_cache(
    session: Session,
    ticker: str,
    data_type: str,
    data: CompanyData,
) -> None:
    # Remove old entries for same ticker+data_type to keep table tidy
    session.query(_CacheEntry).filter_by(ticker=ticker, data_type=data_type).delete()

    entry = _CacheEntry(
        ticker=ticker,
        data_type=data_type,
        fetched_at=datetime.now(timezone.utc),
        payload_json=data.model_dump_json(),
    )
    session.add(entry)
    session.commit()
    logger.info(f"[{ticker}] Written to cache for data_type='{data_type}'")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_or_fetch(
    ticker: str,
    data_type: str = "full",
    max_age_hours: Optional[float] = None,
) -> CompanyData:
    """
    Return CompanyData from cache if fresh, otherwise fetch from yfinance
    and update the cache.

    Parameters
    ----------
    ticker : str
        Stock ticker, e.g. "AAPL" or "RELIANCE.NS".
    data_type : str
        Cache key type. Controls default staleness. One of "full", "price",
        or "financials". Defaults to "full" (24 h).
    max_age_hours : float, optional
        Override the default staleness for data_type. If omitted, uses the
        built-in staleness table.

    Returns
    -------
    CompanyData
        Typed financial data object.
    """
    effective_max_age = max_age_hours if max_age_hours is not None else _max_age_for(data_type)

    with _SessionLocal() as session:
        cached = _read_cache(session, ticker, data_type, effective_max_age)
        if cached is not None:
            return cached

        logger.info(f"[{ticker}] Cache miss — fetching from yfinance")
        data = fetch_company_data(ticker)
        _write_cache(session, ticker, data_type, data)
        return data


def invalidate(ticker: str, data_type: Optional[str] = None) -> None:
    """
    Manually invalidate cache for a ticker.

    Parameters
    ----------
    ticker : str
        Ticker to invalidate.
    data_type : str, optional
        If provided, only invalidate this data_type. Otherwise clears all
        entries for the ticker.
    """
    with _SessionLocal() as session:
        q = session.query(_CacheEntry).filter_by(ticker=ticker)
        if data_type:
            q = q.filter_by(data_type=data_type)
        count = q.delete()
        session.commit()
        logger.info(f"[{ticker}] Invalidated {count} cache entries (data_type={data_type!r})")
