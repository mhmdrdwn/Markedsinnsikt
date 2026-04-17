"""Groundedness evaluation for AI-generated insights.

Checks whether AI output is anchored to actual data rather than hallucinated.
Returns a numeric score (0–100) and per-check details for the UI.
"""

from __future__ import annotations

import re

import pandas as pd


def eval_groundedness(insights: dict, df: pd.DataFrame) -> dict:
    """Score how grounded the AI insights are in the actual campaign data.

    Parameters
    ----------
    insights:
        The structured dict returned by ``generate_insights``.
    df:
        The (filtered) dataframe the insights were generated from.

    Returns
    -------
    dict with keys:
        score          — int 0–100
        label          — str "Excellent" | "Good" | "Fair" | "Poor"
        checks         — dict[str, bool] per-check results
        channels_found — list[str] channel names referenced in insights text
    """
    channels  = set(df["channel"].str.lower().unique())
    campaigns = set(df["campaign"].str.lower().unique())

    # Collect all free-text from insights
    parts: list[str] = []
    if insights.get("executive_decision"):
        parts.append(insights["executive_decision"])
    if insights.get("summary"):
        parts.append(insights["summary"])
    for item in insights.get("insights", []):
        parts.append(item.get("title", "") + " " + item.get("detail", ""))
    for rec in insights.get("recommendations", []):
        parts.append(rec.get("action", "") + " " + rec.get("expected_impact", ""))
    for anom in insights.get("anomalies", []):
        parts.append(anom.get("issue", ""))

    full_text = " ".join(parts).lower()

    checks: dict[str, bool] = {}

    # 1. References at least one real channel name
    channels_found = [ch for ch in channels if ch in full_text]
    checks["mentions_real_channel"] = len(channels_found) > 0

    # 2. Contains at least one concrete number (NOK amount, ROAS, %, count)
    checks["contains_numbers"] = bool(
        re.search(r"\b\d[\d\s]*[.,]?\d*\s*(x|%|nok|kr|stk)\b", full_text)
        or re.search(r"\b\d{2,}\b", full_text)  # any 2+ digit number
    )

    # 3. Has at least one actionable recommendation
    checks["has_recommendations"] = len(insights.get("recommendations", [])) >= 1

    # 4. Executive decision is concise (≤ 35 words)
    exec_words = len(insights.get("executive_decision", "").split())
    checks["exec_decision_concise"] = 1 <= exec_words <= 35

    # 5. Required JSON keys present (structural completeness)
    required_keys = {"executive_decision", "summary", "insights", "recommendations", "anomalies"}
    checks["complete_structure"] = required_keys.issubset(insights.keys())

    # Score: each check worth 20 points
    score = sum(20 for v in checks.values() if v)

    label = (
        "Utmerket" if score >= 80
        else "God" if score >= 60
        else "Akseptabel" if score >= 40
        else "Svak"
    )

    return {
        "score": score,
        "label": label,
        "checks": checks,
        "channels_found": channels_found,
    }
