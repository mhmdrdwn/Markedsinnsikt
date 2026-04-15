"""ML models for Markedsinnsikt AI.

Capabilities:
  1. predict_next_week            — linear regression baseline (spend & ROAS)
  2. detect_anomalies_zscore      — z-score anomaly detection on ROAS
  3. suggest_budget_reallocation  — ROI-based budget shift recommendations
  4. predict_xgboost_with_intervals — XGBoost forecasting + 90% prediction interval
  5. backtest_models              — walk-forward validation: Linear vs XGBoost
  6. detect_anomalies_isolation_forest — multi-dimensional Isolation Forest
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# 1. Linear-regression baseline (spend & ROAS)
# ---------------------------------------------------------------------------

def predict_next_week(df: pd.DataFrame) -> list[dict]:
    """
    For each channel, fit linear regression on weekly spend and ROAS,
    then predict the next week value.
    """
    results = []

    for ch, grp in df.groupby("channel"):
        weekly = (
            grp.groupby("week")
            .agg(spend=("spend", "sum"), revenue=("revenue", "sum"))
            .sort_index()
        )
        weekly["roas"] = weekly["revenue"] / weekly["spend"].replace(0, float("nan"))

        if len(weekly) < 3:
            continue

        weeks = weekly.index.values.astype(float)
        next_week = float(weeks[-1] + 1)
        X = weeks.reshape(-1, 1)

        spend_vals = weekly["spend"].values
        m_spend = LinearRegression().fit(X, spend_vals)
        pred_spend = float(m_spend.predict([[next_week]])[0])
        mae_spend  = float(np.mean(np.abs(m_spend.predict(X) - spend_vals)))

        roas_clean = weekly["roas"].dropna()
        mae_roas: float | None = None
        if len(roas_clean) >= 3:
            X_r = roas_clean.index.values.astype(float).reshape(-1, 1)
            m_roas = LinearRegression().fit(X_r, roas_clean.values)
            pred_roas = float(m_roas.predict([[next_week]])[0])
            mae_roas  = float(np.mean(np.abs(m_roas.predict(X_r) - roas_clean.values)))
        else:
            pred_roas = float(roas_clean.mean()) if len(roas_clean) > 0 else 0.0

        history = [
            {
                "week": int(w),
                "actual_spend": round(float(s), 0),
                "actual_roas": round(float(r), 2) if pd.notna(r) else None,
            }
            for w, s, r in zip(weekly.index, weekly["spend"], weekly["roas"])
        ]

        results.append({
            "channel": ch,
            "next_week": int(next_week),
            "predicted_spend": round(max(pred_spend, 0), 0),
            "predicted_roas":  round(max(pred_roas,  0), 2),
            "mae_spend": round(mae_spend, 0),
            "mae_roas":  round(mae_roas, 2) if mae_roas is not None else None,
            "history": history,
        })

    return results


# ---------------------------------------------------------------------------
# 2. Z-score anomaly detection
# ---------------------------------------------------------------------------

def detect_anomalies_zscore(
    df: pd.DataFrame,
    threshold: float = 2.0,
) -> list[dict]:
    """Flag weeks where ROAS deviates ≥ threshold std-devs from campaign mean."""
    anomalies: list[dict] = []

    for (client, campaign), grp in df.groupby(["client", "campaign"]):
        weekly = (
            grp.groupby("week")
            .agg(spend=("spend", "sum"), revenue=("revenue", "sum"))
            .sort_index()
        )
        weekly["roas"] = weekly["revenue"] / weekly["spend"].replace(0, float("nan"))
        roas = weekly["roas"].dropna()

        if len(roas) < 4:
            continue

        z = _zscore(roas.values)
        for i, (week, score) in enumerate(zip(roas.index, z)):
            if abs(score) >= threshold:
                actual = float(roas.iloc[i])
                mean_r = float(roas.mean())
                direction = "unormalt høy" if actual > mean_r else "unormalt lav"
                anomalies.append({
                    "client": client,
                    "campaign": campaign,
                    "week": int(week),
                    "roas": round(actual, 2),
                    "mean_roas": round(mean_r, 2),
                    "z_score": round(float(abs(score)), 2),
                    "direction": direction,
                    "severity": "high" if abs(score) >= 3.0 else "medium",
                    "detail": (
                        f"ROAS {actual:.2f}x i uke {int(week)} er {direction} "
                        f"(snitt {mean_r:.2f}x, z={abs(score):.2f})"
                    ),
                    "method": "Z-score",
                })

    return sorted(anomalies, key=lambda x: x["z_score"], reverse=True)


# ---------------------------------------------------------------------------
# 3. Budget reallocation recommendations
# ---------------------------------------------------------------------------

def suggest_budget_reallocation(
    df: pd.DataFrame,
    realloc_pct: float = 0.20,
) -> list[dict]:
    """Move `realloc_pct` of budget from lower-ROAS channels to the best one."""
    ch = (
        df.groupby("channel")
        .agg(spend=("spend", "sum"), revenue=("revenue", "sum"),
             conversions=("conversions", "sum"))
        .reset_index()
    )
    ch["roas"] = ch["revenue"] / ch["spend"].replace(0, float("nan"))
    ch = ch.dropna(subset=["roas"]).sort_values("roas", ascending=False)

    if len(ch) < 2:
        return []

    best = ch.iloc[0]
    recommendations = []

    for _, row in ch.iloc[1:].iterrows():
        transfer    = row["spend"] * realloc_pct
        current_rev  = transfer * row["roas"]
        expected_rev = transfer * best["roas"]
        gain         = expected_rev - current_rev
        gain_pct     = gain / current_rev * 100 if current_rev > 0 else 0.0

        recommendations.append({
            "from_channel": row["channel"],
            "to_channel":   best["channel"],
            "from_roas":    round(float(row["roas"]), 2),
            "to_roas":      round(float(best["roas"]), 2),
            "transfer_amount":      round(float(transfer), 0),
            "expected_revenue_gain": round(float(gain), 0),
            "gain_pct": round(float(gain_pct), 1),
            "summary": (
                f"Flytt 20% av {row['channel']}-budsjettet (NOK {transfer:,.0f}) "
                f"til {best['channel']} → forventet inntektsgevinst: "
                f"NOK {gain:,.0f} (+{gain_pct:.0f}%)"
            ),
        })

    return recommendations


# ---------------------------------------------------------------------------
# 4. XGBoost forecasting with 90% prediction intervals
# ---------------------------------------------------------------------------

def predict_xgboost_with_intervals(
    df: pd.DataFrame,
    n_lags: int = 2,
    ci: float = 0.90,
) -> list[dict]:
    """
    For each channel, train XGBoost on lag features of weekly ROAS and
    predict next week. Returns point forecast + 90% prediction interval
    derived from in-sample residuals.
    """
    from xgboost import XGBRegressor

    z_val = 1.645  # 90% normal CI
    results = []

    for ch, grp in df.groupby("channel"):
        weekly = (
            grp.groupby("week")
            .agg(spend=("spend", "sum"), revenue=("revenue", "sum"))
            .sort_index()
        )
        weekly["roas"] = weekly["revenue"] / weekly["spend"].replace(0, float("nan"))
        roas_filled = weekly["roas"].fillna(weekly["roas"].mean())

        if len(roas_filled) < n_lags + 2:
            continue

        vals = roas_filled.values.astype(float)
        X, y = _lag_features(vals, n_lags)

        if len(X) < 3:
            continue

        model = XGBRegressor(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            verbosity=0, random_state=42,
        )
        model.fit(X, y)

        # Predict next week
        next_feat = (
            list(vals[-n_lags:])
            + [float(np.mean(vals[-3:])), float(len(vals))]
        )
        pred = float(model.predict(np.array([next_feat], dtype=float))[0])

        # 90% PI from residual std
        residuals = y - model.predict(X)
        std_r = float(np.std(residuals))
        lower = max(pred - z_val * std_r, 0.0)
        upper = pred + z_val * std_r
        mae   = float(np.mean(np.abs(residuals)))

        # Feature importance
        feat_names = [f"lag_{i+1}" for i in range(n_lags)] + ["rolling_mean", "trend"]
        importance = dict(zip(feat_names, model.feature_importances_.tolist()))

        history = [
            {"week": int(w), "roas": round(float(r), 2) if pd.notna(r) else None}
            for w, r in zip(weekly.index, weekly["roas"])
        ]

        results.append({
            "channel":        ch,
            "next_week":      int(weekly.index[-1]) + 1,
            "predicted_roas": round(max(pred, 0), 2),
            "lower_90":       round(lower, 2),
            "upper_90":       round(upper, 2),
            "mae":            round(mae, 3),
            "feature_importance": importance,
            "history":        history,
        })

    return results


# ---------------------------------------------------------------------------
# 5. Backtesting: walk-forward validation (Linear vs XGBoost)
# ---------------------------------------------------------------------------

def backtest_models(
    df: pd.DataFrame,
    n_lags: int = 2,
) -> list[dict]:
    """
    Walk-forward validation on weekly ROAS per channel.

    For each position k (from n_lags+1 to T-1):
      - Train both LinearRegression and XGBoost on weeks 1..k
      - Predict week k+1
      - Record actual vs predicted

    Returns MAE, RMSE per channel plus per-step data for charting.
    """
    from xgboost import XGBRegressor

    results = []

    for ch, grp in df.groupby("channel"):
        weekly = (
            grp.groupby("week")
            .agg(spend=("spend", "sum"), revenue=("revenue", "sum"))
            .sort_index()
        )
        weekly["roas"] = weekly["revenue"] / weekly["spend"].replace(0, float("nan"))
        roas_filled = weekly["roas"].fillna(weekly["roas"].mean())

        if len(roas_filled) < n_lags + 3:
            continue

        vals  = roas_filled.values.astype(float)
        weeks = weekly.index.values.astype(int)

        actuals, lr_preds, xgb_preds, pred_weeks = [], [], [], []
        min_train = n_lags + 1

        for i in range(min_train, len(vals)):
            train = vals[:i]
            X_tr, y_tr = _lag_features(train, n_lags)
            if len(X_tr) < 2:
                continue

            next_feat = (
                list(train[-n_lags:])
                + [float(np.mean(train[-3:])), float(i)]
            )
            next_feat_arr = np.array([next_feat], dtype=float)

            # Linear Regression
            lr = LinearRegression().fit(X_tr, y_tr)
            lr_p = float(lr.predict(next_feat_arr)[0])

            # XGBoost
            xgb = XGBRegressor(
                n_estimators=100, max_depth=3, learning_rate=0.1,
                verbosity=0, random_state=42,
            )
            xgb.fit(X_tr, y_tr)
            xgb_p = float(xgb.predict(next_feat_arr)[0])

            actuals.append(vals[i])
            lr_preds.append(lr_p)
            xgb_preds.append(xgb_p)
            pred_weeks.append(int(weeks[i]))

        if not actuals:
            continue

        act  = np.array(actuals)
        lr_a = np.array(lr_preds)
        xgb_a = np.array(xgb_preds)

        lr_mae    = float(np.mean(np.abs(act - lr_a)))
        xgb_mae   = float(np.mean(np.abs(act - xgb_a)))
        lr_rmse   = float(np.sqrt(np.mean((act - lr_a) ** 2)))
        xgb_rmse  = float(np.sqrt(np.mean((act - xgb_a) ** 2)))
        winner    = "XGBoost" if xgb_mae < lr_mae else "Lineær regresjon"
        imp_pct   = (lr_mae - xgb_mae) / lr_mae * 100 if lr_mae > 0 else 0.0

        results.append({
            "channel":      ch,
            "lr_mae":       round(lr_mae,   3),
            "xgb_mae":      round(xgb_mae,  3),
            "lr_rmse":      round(lr_rmse,  3),
            "xgb_rmse":     round(xgb_rmse, 3),
            "winner":       winner,
            "improvement_pct": round(imp_pct, 1),
            "steps":        len(actuals),
            "backtest_data": [
                {
                    "week":    w,
                    "actual":  round(float(a), 2),
                    "lr":      round(float(l), 2),
                    "xgb":     round(float(x), 2),
                }
                for w, a, l, x in zip(pred_weeks, actuals, lr_preds, xgb_preds)
            ],
        })

    return results


# ---------------------------------------------------------------------------
# 6. Isolation Forest anomaly detection
# ---------------------------------------------------------------------------

def detect_anomalies_isolation_forest(
    df: pd.DataFrame,
    contamination: float = 0.10,
) -> list[dict]:
    """
    Multi-dimensional anomaly detection using Isolation Forest on
    weekly (spend, ROAS, CTR) per campaign.

    More robust than z-score: captures outliers that are unusual only
    in combination (e.g. high spend + low ROAS simultaneously).
    """
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler

    anomalies: list[dict] = []

    for (client, campaign), grp in df.groupby(["client", "campaign"]):
        weekly = (
            grp.groupby("week")
            .agg(
                spend=("spend", "sum"),
                revenue=("revenue", "sum"),
                clicks=("clicks", "sum"),
                impressions=("impressions", "sum"),
            )
            .sort_index()
        )
        weekly["roas"] = weekly["revenue"] / weekly["spend"].replace(0, float("nan"))
        weekly["ctr"]  = weekly["clicks"]  / weekly["impressions"].replace(0, float("nan")) * 100
        weekly = weekly.fillna(weekly.mean(numeric_only=True))

        if len(weekly) < 4:
            continue

        X = weekly[["spend", "roas", "ctr"]].values.astype(float)
        X_scaled = StandardScaler().fit_transform(X)

        iso = IsolationForest(
            n_estimators=100, contamination=contamination,
            random_state=42,
        )
        labels = iso.fit_predict(X_scaled)    # -1 = anomaly
        scores = iso.decision_function(X_scaled)  # lower = more anomalous

        for i, (week, label, score) in enumerate(zip(weekly.index, labels, scores)):
            if label == -1:
                row = weekly.iloc[i]
                anomalies.append({
                    "client":   client,
                    "campaign": campaign,
                    "week":     int(week),
                    "roas":     round(float(row["roas"]), 2),
                    "spend":    round(float(row["spend"]), 0),
                    "ctr":      round(float(row["ctr"]), 2),
                    "anomaly_score": round(float(-score), 3),
                    "severity": "high" if score < -0.15 else "medium",
                    "detail": (
                        f"Uke {int(week)}: unormalt mønster oppdaget "
                        f"(spend NOK {row['spend']:,.0f}, "
                        f"ROAS {row['roas']:.2f}x, CTR {row['ctr']:.2f}%)"
                    ),
                    "method": "Isolation Forest",
                })

    return sorted(anomalies, key=lambda x: x["anomaly_score"], reverse=True)
