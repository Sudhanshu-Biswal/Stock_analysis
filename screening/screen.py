"""
screening/screen.py — orchestrates: universe -> Phase1 calls -> filter -> rank

Pure deterministic screening (no LLM calls). Fetches each ticker's Phase 1
outputs concurrently (capped at max_workers to avoid hammering yfinance),
runs every criteria_fn against them, and never lets one bad ticker crash
the batch.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Optional

import httpx
from pydantic import BaseModel, Field

from screening.criteria import CriterionResult

logger = logging.getLogger(__name__)

CriteriaFn = Callable[[dict, dict, Optional[dict]], CriterionResult]


class ScreenResult(BaseModel):
    ticker: str
    passed_all: bool
    criteria_results: list[CriterionResult] = Field(default_factory=list)
    pass_count: int = 0
    total_criteria: int = 0
    fetch_error: Optional[str] = None  # set if Phase 1 data couldn't be fetched at all


def _fetch_ticker_data(
    ticker: str, phase1_api_base: str
) -> tuple[Optional[dict], Optional[dict], Optional[dict], Optional[str]]:
    """
    Fetch fundamentals, ratios, reverse-dcf for one ticker from Phase 1's API.

    Returns (fundamentals, ratios, reverse_dcf, error). error is set (with
    the other three None) only if fundamentals/ratios themselves couldn't be
    fetched. reverse_dcf may legitimately be None (e.g. Phase 1's deliberate
    422 "no growth rate justifies this price") without that being a fetch error.
    """
    try:
        with httpx.Client(base_url=phase1_api_base, timeout=30.0) as client:
            fundamentals_resp = client.get(f"/company/{ticker}/fundamentals")
            fundamentals_resp.raise_for_status()
            ratios_resp = client.get(f"/company/{ticker}/ratios")
            ratios_resp.raise_for_status()

            reverse_dcf: Optional[dict] = None
            reverse_dcf_resp = client.get(f"/company/{ticker}/reverse-dcf")
            if reverse_dcf_resp.status_code == 200:
                reverse_dcf = reverse_dcf_resp.json()

            return fundamentals_resp.json(), ratios_resp.json(), reverse_dcf, None
    except httpx.HTTPError as exc:
        logger.warning(f"[{ticker}] Failed to fetch Phase 1 data: {exc}")
        return None, None, None, str(exc)


def _screen_one(ticker: str, criteria_fns: list[CriteriaFn], phase1_api_base: str) -> ScreenResult:
    fundamentals, ratios, reverse_dcf, error = _fetch_ticker_data(ticker, phase1_api_base)
    if error is not None:
        return ScreenResult(
            ticker=ticker, passed_all=False, criteria_results=[],
            pass_count=0, total_criteria=len(criteria_fns), fetch_error=error,
        )

    results = [fn(fundamentals, ratios, reverse_dcf) for fn in criteria_fns]
    pass_count = sum(1 for r in results if r.passed)

    return ScreenResult(
        ticker=ticker,
        passed_all=pass_count == len(results),
        criteria_results=results,
        pass_count=pass_count,
        total_criteria=len(results),
        fetch_error=None,
    )


def run_screen(
    universe: list[str],
    criteria_fns: list[CriteriaFn],
    phase1_api_base: str,
    max_workers: int = 5,
) -> list[ScreenResult]:
    """
    Screens every ticker in universe concurrently (capped at max_workers)
    against every criteria_fn. One ticker's fetch failure never crashes the
    batch -- it's recorded with fetch_error set instead.

    Returns results sorted by pass_count descending; caller applies
    min_pass_count to derive the shortlist view.
    """
    results: list[ScreenResult] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_screen_one, ticker, criteria_fns, phase1_api_base): ticker
            for ticker in universe
        }
        for future in as_completed(futures):
            ticker = futures[future]
            try:
                results.append(future.result())
            except Exception as exc:
                logger.error(f"[{ticker}] Unexpected error during screening: {exc}", exc_info=True)
                results.append(ScreenResult(
                    ticker=ticker, passed_all=False, criteria_results=[],
                    pass_count=0, total_criteria=len(criteria_fns), fetch_error=str(exc),
                ))

    results.sort(key=lambda r: r.pass_count, reverse=True)
    return results
