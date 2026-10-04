"""
screening/universes.py — static ticker lists

Hardcoded samples for v1 -- no live index-constituents API. Expand these
lists once the screening pipeline is proven.
"""

from __future__ import annotations

NIFTY_50_SAMPLE: list[str] = [
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "ITC.NS", "KOTAKBANK.NS", "LT.NS", "SBIN.NS",
    "BHARTIARTL.NS", "BAJFINANCE.NS", "ASIANPAINT.NS", "MARUTI.NS",
    "AXISBANK.NS", "TITAN.NS", "SUNPHARMA.NS", "WIPRO.NS", "ULTRACEMCO.NS",
    "NESTLEIND.NS",
]

US_MEGACAP_SAMPLE: list[str] = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "BRK-B",
    "JPM", "V", "UNH", "XOM", "JNJ", "WMT", "PG",
]

CUSTOM_WATCHLIST: list[str] = []  # user can populate this manually

_UNIVERSES: dict[str, list[str]] = {
    "nifty50_sample": NIFTY_50_SAMPLE,
    "us_megacap_sample": US_MEGACAP_SAMPLE,
    "custom": CUSTOM_WATCHLIST,
}


def get_universe(name: str) -> list[str]:
    """Look up a named universe. Raises ValueError on unknown name."""
    if name not in _UNIVERSES:
        raise ValueError(
            f"Unknown universe '{name}'. Valid options: {sorted(_UNIVERSES.keys())}"
        )
    return list(_UNIVERSES[name])
