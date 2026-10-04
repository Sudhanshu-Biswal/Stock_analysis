"""
tests/test_synthesis.py — Phase 2 synthesis tests

Covers the number-hallucination check directly, and generate_report's
graceful-failure behaviour when Phase 1 is unreachable / RAG returns nothing.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import patch

import httpx
import pytest

from synthesis.report import Phase1Unavailable, find_unsupported_numbers, generate_report
from synthesis.schemas import ResearchReport


def test_find_unsupported_numbers_flags_fabricated_value():
    context = "Revenue grew 12% to $50 billion in the latest quarter."
    thesis = "The company's revenue grew 12% year over year."
    counter_thesis = "However, margins compressed to 45%, a figure absent from the filings."

    flagged = find_unsupported_numbers(thesis, counter_thesis, context)

    assert "45%" in flagged
    assert "12%" not in flagged


def test_find_unsupported_numbers_no_false_positive_when_grounded():
    context = "Revenue grew 12% to $50 billion. Operating margin was 30%."
    thesis = "Revenue grew 12%, and operating margin held at 30%."
    counter_thesis = "Growth of 12% may not be sustainable."

    flagged = find_unsupported_numbers(thesis, counter_thesis, context)

    assert flagged == []


def test_generate_report_raises_clear_error_when_phase1_unreachable():
    with patch("synthesis.report.httpx.Client") as mock_client_cls:
        mock_client = mock_client_cls.return_value.__enter__.return_value
        mock_client.get.side_effect = httpx.ConnectError("connection refused")

        with pytest.raises(Phase1Unavailable):
            generate_report("AAPL")


@patch("synthesis.report._call_llm")
@patch("synthesis.report._gather_rag_context", return_value=([], True))
@patch("synthesis.report._fetch_phase1_data")
def test_generate_report_produces_report_with_zero_rag_documents(
    mock_fetch_phase1, mock_gather_rag, mock_call_llm,
):
    mock_fetch_phase1.return_value = {
        "fundamentals": {"as_of_date": "2026-06-30"},
        "ratios": {},
        "dcf": {},
        "reverse_dcf": {},
    }
    mock_call_llm.return_value = ResearchReport(
        ticker="AAPL",
        generated_at=datetime.now(timezone.utc),
        thesis="Not available in provided data beyond computed metrics.",
        counter_thesis="Not available in provided data beyond computed metrics.",
        catalysts=[],
        risks=[],
        scenarios=[],
        sources_used=[],
        data_as_of="2026-06-30",
        disclaimer="test disclaimer",
    )

    report = generate_report("AAPL")

    assert report.ticker == "AAPL"
    mock_call_llm.assert_called_once()
    # the context blob passed to the LLM call should note no news was available
    context_arg = mock_call_llm.call_args[0][1]
    assert "No recent news" in context_arg
