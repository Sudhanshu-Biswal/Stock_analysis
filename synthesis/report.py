"""
synthesis/report.py — orchestrates: Phase1 data + RAG context -> LLM -> report

Sequence:
1. Call Phase 1's own API over HTTP (never reimplement its logic here).
2. Retrieve recent news + filing context via rag/retrieve.py.
3. Assemble one context blob: COMPUTED (Phase 1) vs RETRIEVED (sourced docs).
4. Send to Claude with the enforced system prompt, request structured output.
5. Post-process: flag any number in thesis/counter_thesis not traceable to
   the input context (logged, not silently shipped).
"""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime, timezone
from typing import Any, Optional

import httpx
from pydantic import ValidationError

from rag.ingest import ingest_filings, ingest_news
from rag.retrieve import retrieve_context
from rag.store import add_documents
from synthesis.prompts import SYSTEM_PROMPT, build_user_message
from synthesis.schemas import ResearchReport

logger = logging.getLogger(__name__)

PHASE1_API_BASE = os.environ.get("PHASE1_API_BASE", "http://127.0.0.1:8000")

# "anthropic" or "openai" -- switch via LLM_PROVIDER in .env, no code change needed.
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "anthropic").lower()
# Verify current model ids at docs.claude.com / platform.openai.com before relying on these.
ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-5.6-terra")

_NUMBER_RE = re.compile(r"\d[\d,]*\.?\d*%?")

FIXED_DISCLAIMER = (
    "This report is generated for personal research purposes only and is not "
    "financial advice. It combines deterministic calculations with LLM-written "
    "narrative synthesis and may contain errors. Always verify numbers against "
    "primary sources before making investment decisions."
)


class Phase1Unavailable(Exception):
    """Raised when Phase 1's API cannot be reached or returns an error."""


def _fetch_phase1_data(ticker: str, dcf_overrides: Optional[dict[str, Any]]) -> dict[str, Any]:
    """Call Phase 1's own API over HTTP -- never reimplement its logic here."""
    dcf_body = {"growth_rate": 0.10, "terminal_growth": 0.03, "years": 5}
    if dcf_overrides:
        dcf_body.update({k: v for k, v in dcf_overrides.items() if v is not None})

    try:
        with httpx.Client(base_url=PHASE1_API_BASE, timeout=30.0) as client:
            fundamentals = client.get(f"/company/{ticker}/fundamentals")
            fundamentals.raise_for_status()
            ratios = client.get(f"/company/{ticker}/ratios")
            ratios.raise_for_status()
            dcf = client.post(f"/company/{ticker}/dcf", json=dcf_body)
            dcf.raise_for_status()

            # reverse-dcf can legitimately 422 (e.g. "no growth rate under 50%
            # justifies this price") -- that's a real, deliberate refusal from
            # Phase 1, not an outage, so don't treat it as Phase1Unavailable.
            reverse_dcf = client.get(f"/company/{ticker}/reverse-dcf")
            if reverse_dcf.status_code == 422:
                reverse_dcf_data: Any = {"unavailable": True, "detail": reverse_dcf.json().get("detail")}
            else:
                reverse_dcf.raise_for_status()
                reverse_dcf_data = reverse_dcf.json()
    except httpx.HTTPError as exc:
        raise Phase1Unavailable(f"Phase 1 API unreachable or errored for {ticker}: {exc}") from exc

    return {
        "fundamentals": fundamentals.json(),
        "ratios": ratios.json(),
        "dcf": dcf.json(),
        "reverse_dcf": reverse_dcf_data,
    }


def _gather_rag_context(ticker: str, market: str, query_text: str) -> tuple[list[dict], bool]:
    """Ingest fresh docs, store them, and retrieve the most relevant chunks."""
    news_docs = ingest_news(ticker)
    filing_docs = ingest_filings(ticker, market)
    all_docs = news_docs + filing_docs

    if all_docs:
        add_documents(all_docs)

    hits = retrieve_context(ticker, query_text, k=8, max_age_days=365)
    return hits, len(all_docs) == 0


def _build_context_blob(
    ticker: str,
    phase1_data: dict[str, Any],
    rag_hits: list[dict],
    no_news_available: bool,
) -> str:
    lines = [f"=== COMPUTED — from deterministic system (Phase 1 API) for {ticker} ==="]
    lines.append(json.dumps(phase1_data, indent=2, default=str))

    lines.append("\n=== RETRIEVED — news/filing excerpts ===")
    if no_news_available:
        lines.append(
            "No recent news or filing documents were retrieved. Use only the "
            "computed data above and say so explicitly in the report."
        )
    for hit in rag_hits:
        meta = hit["metadata"]
        lines.append(
            f"\n[RETRIEVED — from {meta.get('source')}, {meta.get('published_date')}] "
            f"{meta.get('title')} ({meta.get('url')})\n{hit['text']}"
        )
    return "\n".join(lines)


def find_unsupported_numbers(thesis: str, counter_thesis: str, context: str) -> list[str]:
    """
    Scan thesis/counter_thesis for numeric tokens and flag any that don't
    appear anywhere in the source context. Simple substring check -- good
    enough for v1 flagging, not proof against paraphrase/rounding.
    """
    flagged: list[str] = []
    for text in (thesis, counter_thesis):
        for match in _NUMBER_RE.finditer(text):
            token = match.group()
            digits_only = token.rstrip("%").replace(",", "")
            if len(digits_only) < 2:
                continue  # skip single digits, usually structural (e.g. "1 of 3")
            if token not in context:
                flagged.append(token)
    return flagged


def _inline_refs(node: Any, defs: dict[str, Any]) -> Any:
    """Recursively replace {"$ref": "#/$defs/X"} with the definition itself.

    Some models mishandle $ref/$defs in tool schemas for nested list[Model]
    fields (e.g. flattening catalysts: list[Catalyst] into a bare string) --
    inlining avoids that failure mode entirely.
    """
    if isinstance(node, dict):
        if "$ref" in node:
            ref_name = node["$ref"].rsplit("/", 1)[-1]
            return _inline_refs(defs[ref_name], defs)
        return {k: _inline_refs(v, defs) for k, v in node.items() if k != "$defs"}
    if isinstance(node, list):
        return [_inline_refs(item, defs) for item in node]
    return node


def _report_tool_schema() -> dict[str, Any]:
    """JSON schema for the tool/function call, minus fields the code fills in itself."""
    raw_schema = ResearchReport.model_json_schema()
    defs = raw_schema.pop("$defs", {})
    schema = _inline_refs(raw_schema, defs)

    for field in ("ticker", "generated_at", "data_as_of", "disclaimer"):
        schema.get("properties", {}).pop(field, None)
    if "required" in schema:
        schema["required"] = [
            f for f in schema["required"]
            if f not in ("ticker", "generated_at", "data_as_of", "disclaimer")
        ]
    return schema


def _call_anthropic(ticker: str, context_blob: str, current_date: str) -> dict[str, Any]:
    from anthropic import Anthropic

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY not set. Verify your Claude plan includes Messages "
            "API credits (see support.claude.com / docs.claude.com) and set the "
            "key as an environment variable before calling generate_report()."
        )

    client = Anthropic(api_key=api_key)
    user_message = build_user_message(ticker, context_blob, current_date)
    tool_schema = _report_tool_schema()

    # Identity-linked API keys (created under an org workspace) require this
    # header -- set ANTHROPIC_WORKSPACE_ID in .env if you hit a 400 asking for it.
    extra_headers = {}
    workspace_id = os.environ.get("ANTHROPIC_WORKSPACE_ID")
    if workspace_id:
        extra_headers["anthropic-workspace-id"] = workspace_id

    response = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=8192,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
        tools=[{
            "name": "emit_research_report",
            "description": "Emit the structured research report.",
            "input_schema": tool_schema,
        }],
        tool_choice={"type": "tool", "name": "emit_research_report"},
        extra_headers=extra_headers,
    )

    if response.stop_reason == "max_tokens":
        raise RuntimeError(
            f"Claude's response for {ticker} was truncated (hit max_tokens) before "
            "completing the structured output -- increase max_tokens in _call_anthropic "
            "or shorten the input context."
        )

    tool_use_block = next((b for b in response.content if b.type == "tool_use"), None)
    if tool_use_block is None:
        raise RuntimeError(f"Claude did not return a structured tool-use response for {ticker}.")

    return dict(tool_use_block.input)


def _call_openai(ticker: str, context_blob: str, current_date: str) -> dict[str, Any]:
    from openai import OpenAI

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY not set. Set it as an environment variable (see "
            "platform.openai.com/api-keys) before calling generate_report()."
        )

    client = OpenAI(api_key=api_key)
    user_message = build_user_message(ticker, context_blob, current_date)
    tool_schema = _report_tool_schema()
    tool_schema["additionalProperties"] = False

    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        max_completion_tokens=8192,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        tools=[{
            "type": "function",
            "function": {
                "name": "emit_research_report",
                "description": "Emit the structured research report.",
                "parameters": tool_schema,
            },
        }],
        tool_choice={"type": "function", "function": {"name": "emit_research_report"}},
    )

    choice = response.choices[0]
    if choice.finish_reason == "length":
        raise RuntimeError(
            f"OpenAI's response for {ticker} was truncated (hit max_completion_tokens) "
            "before completing the structured output -- increase the token limit in "
            "_call_openai or shorten the input context."
        )

    tool_calls = choice.message.tool_calls
    if not tool_calls:
        raise RuntimeError(f"OpenAI did not return a structured tool-call response for {ticker}.")

    return json.loads(tool_calls[0].function.arguments)


def _call_llm(ticker: str, context_blob: str, data_as_of: str, current_date: str) -> ResearchReport:
    if LLM_PROVIDER == "openai":
        payload = _call_openai(ticker, context_blob, current_date)
    elif LLM_PROVIDER == "anthropic":
        payload = _call_anthropic(ticker, context_blob, current_date)
    else:
        raise RuntimeError(f"Unknown LLM_PROVIDER '{LLM_PROVIDER}' -- use 'anthropic' or 'openai'.")

    payload["ticker"] = ticker
    payload["generated_at"] = datetime.now(timezone.utc)
    payload["data_as_of"] = data_as_of
    payload["disclaimer"] = FIXED_DISCLAIMER

    try:
        return ResearchReport.model_validate(payload)
    except ValidationError as exc:
        logger.error(f"[{ticker}] {LLM_PROVIDER} returned structurally invalid output: {payload!r}")
        raise RuntimeError(
            f"{LLM_PROVIDER} returned a malformed structured response for {ticker} "
            f"(fields missing or wrong type -- see logs for the raw payload). "
            f"Verify {'ANTHROPIC_MODEL' if LLM_PROVIDER == 'anthropic' else 'OPENAI_MODEL'} "
            f"is a valid, current model id, then retry."
        ) from exc


def generate_report(
    ticker: str,
    market: str = "US",
    dcf_overrides: Optional[dict[str, Any]] = None,
) -> ResearchReport:
    """
    Full Phase 2 pipeline: Phase1 data + RAG context -> LLM -> ResearchReport.

    Fails loudly (Phase1Unavailable) only if Phase 1's API is unreachable.
    Zero RAG documents is not fatal -- the report is still produced using
    only computed numbers, with a note that no recent news was available.
    """
    phase1_data = _fetch_phase1_data(ticker, dcf_overrides)
    data_as_of = phase1_data["fundamentals"].get("as_of_date") or "unknown"

    rag_hits, no_news_available = _gather_rag_context(
        ticker, market, query_text=f"{ticker} recent news earnings outlook risk"
    )

    context_blob = _build_context_blob(ticker, phase1_data, rag_hits, no_news_available)
    current_date = datetime.now(timezone.utc).date().isoformat()

    report = _call_llm(ticker, context_blob, data_as_of, current_date)

    flagged = find_unsupported_numbers(report.thesis, report.counter_thesis, context_blob)
    if flagged:
        logger.warning(f"[{ticker}] Possible hallucinated numbers not found in source context: {flagged}")

    return report