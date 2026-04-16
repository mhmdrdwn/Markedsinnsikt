"""ml package — re-exports all public ML functions."""

from ml.features import _zscore, _lag_features
from ml.models import (
    predict_next_week,
    predict_xgboost_with_intervals,
    suggest_budget_reallocation,
)
from ml.backtesting import backtest_models, compute_business_impact
from ml.anomaly import detect_anomalies_zscore, detect_anomalies_isolation_forest

__all__ = [
    "_zscore",
    "_lag_features",
    "predict_next_week",
    "predict_xgboost_with_intervals",
    "suggest_budget_reallocation",
    "backtest_models",
    "compute_business_impact",
    "detect_anomalies_zscore",
    "detect_anomalies_isolation_forest",
]
