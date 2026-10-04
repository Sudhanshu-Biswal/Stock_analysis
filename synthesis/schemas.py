"""
synthesis/schemas.py — structured LLM output

The LLM fills these in via tool-use / JSON mode, not free text. Any number
appearing in thesis/counter_thesis must trace back to Phase 1 or a
retrieved, sourced document (enforced in synthesis/report.py post-processing).
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Optional

from pydantic import BaseModel


class Catalyst(BaseModel):
    description: str
    source: Optional[str] = None
    published_date: Optional[date] = None


class RiskItem(BaseModel):
    description: str
    category: str  # "business" | "valuation" | "leverage" | "regulatory" | "macro" | "execution"


class ScenarioCase(BaseModel):
    label: str  # "bull" | "base" | "bear"
    narrative: str
    key_assumptions: list[str]


class ResearchReport(BaseModel):
    ticker: str
    generated_at: datetime
    thesis: str
    counter_thesis: str
    catalysts: list[Catalyst]
    risks: list[RiskItem]
    scenarios: list[ScenarioCase]
    sources_used: list[str]  # every source URL/title actually cited
    data_as_of: str          # carried directly from Phase 1's as_of_date, not invented
    disclaimer: str          # fixed text
