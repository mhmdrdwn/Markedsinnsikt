"""Core ML models: linear regression baseline, XGBoost forecasting, budget reallocation."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from ml.features import _lag_features


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
    from datetime import timedelta

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

        # Build week → date mapping from raw data
        week_date_map = grp.groupby("week")["week_date"].first()

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
        next_feat_arr = np.array([next_feat], dtype=float)
        pred = float(model.predict(next_feat_arr)[0])

        # 90% PI from residual std
        residuals = y - model.predict(X)
        std_r = float(np.std(residuals))
        lower = max(pred - z_val * std_r, 0.0)
        upper = pred + z_val * std_r
        mae   = float(np.mean(np.abs(residuals)))

        feat_names = [f"lag_{i+1}" for i in range(n_lags)] + ["rolling_mean", "trend"]

        # SHAP — TreeExplainer is fast (<50ms) for our model/data size
        import shap as _shap
        explainer   = _shap.TreeExplainer(model)
        shap_matrix = explainer.shap_values(X)            # (n_samples, n_features)
        shap_next   = explainer.shap_values(next_feat_arr)[0]  # (n_features,)

        shap_global = dict(zip(feat_names, np.abs(shap_matrix).mean(axis=0).tolist()))
        shap_local  = dict(zip(feat_names, shap_next.tolist()))
        base_value  = float(explainer.expected_value)

        # Compute next week's calendar date (+7 days from last known date)
        last_date = pd.to_datetime(week_date_map.iloc[-1])
        next_date = (last_date + timedelta(weeks=1)).strftime("%Y-%m-%d")

        history = [
            {
                "week":      int(w),
                "week_date": week_date_map.get(w, ""),
                "roas":      round(float(r), 2) if pd.notna(r) else None,
            }
            for w, r in zip(weekly.index, weekly["roas"])
        ]

        results.append({
            "channel":        ch,
            "next_week":      int(weekly.index[-1]) + 1,
            "next_date":      next_date,
            "predicted_roas": round(max(pred, 0), 2),
            "lower_90":       round(lower, 2),
            "upper_90":       round(upper, 2),
            "mae":            round(mae, 3),
            "feature_importance": shap_global,
            "shap_global":    shap_global,
            "shap_local":     shap_local,
            "base_value":     round(base_value, 3),
            "history":        history,
        })

    return results


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
                f"til {best['channel']} -> forventet inntektsgevinst: "
                f"NOK {gain:,.0f} (+{gain_pct:.0f}%)"
            ),
        })

    return recommendations
