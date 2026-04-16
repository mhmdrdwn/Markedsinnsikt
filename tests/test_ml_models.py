"""Tests for ml package — forecasting, backtesting, anomaly detection."""

import pandas as pd
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data import generate_dataset
from ml import (
    backtest_models,
    detect_anomalies_zscore,
    detect_anomalies_isolation_forest,
    predict_xgboost_with_intervals,
    suggest_budget_reallocation,
    compute_business_impact,
)


@pytest.fixture(scope="module")
def df():
    return generate_dataset()


def test_backtest_returns_list(df):
    result = backtest_models(df)
    assert isinstance(result, list)


def test_backtest_has_required_keys(df):
    result = backtest_models(df)
    if result:
        row = result[0]
        for key in ("channel", "lr_mae", "xgb_mae", "lr_rmse", "xgb_rmse", "improvement_pct"):
            assert key in row, f"Missing key: {key}"


def test_backtest_mae_non_negative(df):
    result = backtest_models(df)
    for row in result:
        assert row["lr_mae"] >= 0
        assert row["xgb_mae"] >= 0


def test_zscore_anomalies_returns_list(df):
    result = detect_anomalies_zscore(df)
    assert isinstance(result, list)


def test_isolation_forest_returns_list(df):
    result = detect_anomalies_isolation_forest(df)
    assert isinstance(result, list)


def test_xgboost_forecast_structure(df):
    result = predict_xgboost_with_intervals(df)
    assert isinstance(result, list)
    if result:
        p = result[0]
        for key in ("channel", "next_week", "predicted_roas", "lower_90", "upper_90", "history"):
            assert key in p, f"Missing key: {key}"


def test_xgboost_intervals_ordered(df):
    result = predict_xgboost_with_intervals(df)
    for p in result:
        assert p["lower_90"] <= p["predicted_roas"] <= p["upper_90"], (
            f"{p['channel']}: interval not ordered correctly"
        )


def test_compute_business_impact(df):
    bt = backtest_models(df)
    avg_spend = float(df.groupby("week")["spend"].sum().mean())
    impacts = compute_business_impact(bt, avg_spend)
    assert isinstance(impacts, list)
    if impacts:
        for imp in impacts:
            assert 0 <= imp["decision_accuracy_proxy"] <= 100
            assert imp["estimated_weekly_cost_of_error"] >= 0
