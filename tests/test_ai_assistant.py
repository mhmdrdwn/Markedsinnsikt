"""Tests for ai_assistant.py — context builder and anomaly detection."""

import pandas as pd
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data import generate_dataset
from ai_assistant import build_context, detect_anomalies


@pytest.fixture(scope="module")
def df():
    return generate_dataset()


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
    result_all      = build_context(df, client_filter="All")
    result_no_filter = build_context(df)
    assert result_all == result_no_filter


def test_detect_anomalies_returns_list(df):
    result = detect_anomalies(df)
    assert isinstance(result, list)


def test_detect_anomalies_structure(df):
    result = detect_anomalies(df)
    for a in result:
        assert "client" in a
        assert "campaign" in a
        assert "severity" in a
        assert a["severity"] in ("high", "medium", "low")


def test_detect_anomalies_empty_df():
    empty = pd.DataFrame(columns=["client", "campaign", "week", "spend", "revenue"])
    result = detect_anomalies(empty)
    assert result == []
