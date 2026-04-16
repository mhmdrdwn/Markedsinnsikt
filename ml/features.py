"""Shared feature engineering helpers used by ML models."""

from __future__ import annotations

import numpy as np


def _zscore(values: np.ndarray) -> np.ndarray:
    std = values.std()
    if std == 0:
        return np.zeros_like(values, dtype=float)
    return (values - values.mean()) / std


def _lag_features(values: np.ndarray, n_lags: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """
    Build lag-feature matrix and target vector for time-series models.

    Features per row: [lag_1, lag_2, ..., rolling_mean_3, time_index]
    """
    X_rows, y_vals = [], []
    for i in range(n_lags, len(values)):
        row = list(values[i - n_lags: i])
        row.append(float(np.mean(values[max(0, i - 3): i])))   # 3-week rolling mean
        row.append(float(i))                                     # time trend
        X_rows.append(row)
        y_vals.append(values[i])
    return np.array(X_rows, dtype=float), np.array(y_vals, dtype=float)
