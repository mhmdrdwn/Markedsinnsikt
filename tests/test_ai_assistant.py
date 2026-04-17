"""Tests for ai package — context builder, tools, evals, and anomaly detection."""

import json
import sys
import os

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data import get_dataset
from ai import build_context, detect_anomalies
from ai.tools import ToolExecutor, TOOL_DEFINITIONS
from ai.evals import eval_groundedness
from ai.insights import _build_rag_context


@pytest.fixture(scope="module")
def df():
    return get_dataset()


# ---------------------------------------------------------------------------
# build_context
# ---------------------------------------------------------------------------

def test_build_context_returns_string(df):
    result = build_context(df)
    assert isinstance(result, str)
    assert len(result) > 0


def test_build_context_contains_key_sections(df):
    result = build_context(df)
    assert "AGGREGATE SUMMARY" in result
    assert "PERFORMANCE BY CHANNEL" in result
    assert "PERFORMANCE BY CAMPAIGN" in result


def test_build_context_client_filter(df):
    client = df["client"].iloc[0]
    result = build_context(df, client_filter=client)
    assert "CROSS-CLIENT BENCHMARKING" in result


def test_build_context_all_filter(df):
    assert build_context(df, client_filter="All") == build_context(df)


# ---------------------------------------------------------------------------
# detect_anomalies
# ---------------------------------------------------------------------------

def test_detect_anomalies_returns_list(df):
    assert isinstance(detect_anomalies(df), list)


def test_detect_anomalies_structure(df):
    for a in detect_anomalies(df):
        assert "client" in a
        assert "campaign" in a
        assert "severity" in a
        assert a["severity"] in ("high", "medium", "low")


def test_detect_anomalies_empty_df():
    empty = pd.DataFrame(columns=["client", "campaign", "week", "spend", "revenue"])
    assert detect_anomalies(empty) == []


# ---------------------------------------------------------------------------
# ToolExecutor
# ---------------------------------------------------------------------------

def test_tool_definitions_schema():
    assert len(TOOL_DEFINITIONS) == 5
    for t in TOOL_DEFINITIONS:
        assert t["type"] == "function"
        assert "name" in t["function"]
        assert "parameters" in t["function"]


def test_tool_get_channel_performance_all(df):
    ex = ToolExecutor(df)
    result = json.loads(ex.execute("get_channel_performance", {"channel": "all"}))
    assert isinstance(result, list)
    assert len(result) > 0
    assert all("roas" in r for r in result)


def test_tool_get_top_channel(df):
    ex = ToolExecutor(df)
    result = json.loads(ex.execute("get_top_channel", {}))
    assert "channel" in result
    assert "roas" in result
    assert result["roas"] > 0


def test_tool_compare_channels(df):
    channels = df["channel"].unique().tolist()
    ex = ToolExecutor(df)
    result = json.loads(ex.execute("compare_channels", {"channel_a": channels[0], "channel_b": channels[1]}))
    assert len(result) == 2


def test_tool_get_weekly_trend(df):
    channel = df["channel"].iloc[0]
    ex = ToolExecutor(df)
    result = json.loads(ex.execute("get_weekly_trend", {"channel": channel, "metric": "roas"}))
    assert "weeks" in result
    assert "values" in result
    assert result["trend"] in ("up", "down", "stable")


def test_tool_unknown_returns_error(df):
    ex = ToolExecutor(df)
    result = json.loads(ex.execute("nonexistent_tool", {}))
    assert "error" in result


def test_tools_used_tracking(df):
    ex = ToolExecutor(df)
    ex.execute("get_top_channel", {})
    ex.execute("get_top_channel", {})  # duplicate — should appear once
    ex.execute("get_channel_performance", {"channel": "all"})
    assert ex.tools_used == ["get_top_channel", "get_channel_performance"]


# ---------------------------------------------------------------------------
# eval_groundedness
# ---------------------------------------------------------------------------

def test_eval_groundedness_perfect_score(df):
    channel = df["channel"].iloc[0]
    insights = {
        "executive_decision": f"Øk {channel} budsjett med 20% — ROAS er 4.2x.",
        "summary": f"{channel} leverer ROAS 4.2x mot snitt 2.1x. NOK 120 000 i inntekt siste uke.",
        "insights": [{"title": "Best kanal", "detail": f"{channel}: ROAS 4.2x, NOK 80 000."}],
        "anomalies": [],
        "recommendations": [{"action": "Øk budsjett", "target": channel,
                              "expected_impact": "estimert +15%", "priority": "high"}],
    }
    result = eval_groundedness(insights, df)
    assert result["score"] == 100
    assert result["label"] == "Utmerket"
    assert len(result["channels_found"]) > 0


def test_eval_groundedness_empty_insights(df):
    result = eval_groundedness({}, df)
    assert result["score"] == 0
    assert result["label"] == "Svak"


def test_eval_groundedness_returns_required_keys(df):
    result = eval_groundedness({"executive_decision": "Test", "summary": "Test",
                                 "insights": [], "recommendations": [], "anomalies": []}, df)
    assert "score" in result
    assert "label" in result
    assert "checks" in result
    assert "channels_found" in result


# ---------------------------------------------------------------------------
# RAG-lite context
# ---------------------------------------------------------------------------

def test_rag_context_shorter_than_full(df):
    full = build_context(df)
    rag  = _build_rag_context(df)
    assert len(rag) < len(full)


def test_rag_context_channel_question_adds_trend(df):
    channel = df["channel"].iloc[0]
    ctx = _build_rag_context(df, question=f"Hva er trenden for {channel}?")
    assert "WEEKLY TREND" in ctx.upper()


def test_rag_context_budget_question_adds_signal(df):
    ctx = _build_rag_context(df, question="Hvordan bør vi fordele budsjettet?")
    assert "BUDGET SIGNAL" in ctx.upper()
