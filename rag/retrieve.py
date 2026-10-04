"""
rag/retrieve.py — hybrid search: semantic similarity + recency weighting

Prevents a slightly-more-similar old chunk from drowning out fresher,
more relevant sources (the mechanism that keeps a 2023 annual report
from being treated as current in 2026).
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Optional

from rag.store import query as store_query

logger = logging.getLogger(__name__)

_RECENCY_HALF_LIFE_DAYS = 30.0  # recency weight halves every 30 days


def _recency_weight(published_date_str: str) -> float:
    try:
        published = date.fromisoformat(published_date_str)
    except (ValueError, TypeError):
        return 0.0
    age_days = max((date.today() - published).days, 0)
    return 0.5 ** (age_days / _RECENCY_HALF_LIFE_DAYS)


def retrieve_context(
    ticker: str,
    query_text: str,
    k: int = 8,
    max_age_days: Optional[int] = None,
    candidate_pool: int = 30,
) -> list[dict]:
    """
    Fetch a candidate pool via semantic search, then re-rank by combining
    semantic similarity with recency. Returns top-k hits, each carrying
    full source metadata.
    """
    candidates = store_query(ticker, query_text, k=candidate_pool, max_age_days=max_age_days)
    if not candidates:
        return []

    # chroma distance is smaller = more similar; convert to a 0..1 similarity score
    max_dist = max((c["distance"] for c in candidates), default=1.0) or 1.0
    for c in candidates:
        similarity = 1.0 - (c["distance"] / max_dist)
        recency = _recency_weight(c["metadata"].get("published_date", ""))
        c["score"] = (0.7 * similarity) + (0.3 * recency)

    candidates.sort(key=lambda c: c["score"], reverse=True)
    return candidates[:k]
