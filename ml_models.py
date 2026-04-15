"""ML models for Markedsinnsikt AI.

Three capabilities:
  1. predict_next_week   — linear regression on weekly spend & ROAS per channel
  2. detect_anomalies_zscore — statistical anomaly detection (z-score on ROAS)
  3. suggest_budget_reallocation — ROI-based budget shift recommendations
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


# ---------------------------------------------------------------------------
# 1. Spend & ROAS prediction
# ---------------------------------------------------------------------------

def predict_next_week(df: pd.DataFrame) -> list[dict]:
    """
    For each channel in df, fit linear regression on weekly spend and ROAS,
    then predict the next week value.

    Returns a list of dicts — one per channel — containing:
      channel, next_week, predicted_spend, predicted_roas,
      mae_spend, mae_roas, history (list of {week, actual_spend, actual_roas})
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

        # --- Spend ---
        spend_vals = weekly["spend"].values
        m_spend = LinearRegression().fit(X, spend_vals)
        pred_spend = float(m_spend.predict([[next_week]])[0])
        mae_spend = float(np.mean(np.abs(m_spend.predict(X) - spend_vals)))

        # --- ROAS ---
        roas_clean = weekly["roas"].dropna()
        mae_roas: float | None = None
        if len(roas_clean) >= 3:
            X_r = roas_clean.index.values.astype(float).reshape(-1, 1)
            m_roas = LinearRegression().fit(X_r, roas_clean.values)
            pred_roas = float(m_roas.predict([[next_week]])[0])
            mae_roas = float(np.mean(np.abs(m_roas.predict(X_r) - roas_clean.values)))
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
            "predicted_roas": round(max(pred_roas, 0), 2),
            "mae_spend": round(mae_spend, 0),
            "mae_roas": round(mae_roas, 2) if mae_roas is not None else None,
            "history": history,
        })

    return results


# ---------------------------------------------------------------------------
# 2. Statistical anomaly detection (z-score)
# ---------------------------------------------------------------------------

def _zscore(values: np.ndarray) -> np.ndarray:
    std = values.std()
    if std == 0:
        return np.zeros_like(values, dtype=float)
    return (values - values.mean()) / std


def detect_anomalies_zscore(
    df: pd.DataFrame,
    threshold: float = 2.0,
) -> list[dict]:
    """
    Flag weeks where a campaign's ROAS deviates more than `threshold` standard
    deviations from its own historical mean.

    More statistically rigorous than fixed-percentage thresholds — adapts to each
    campaign's own volatility.
    """
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
                        f"for denne kampanjen (snitt {mean_r:.2f}x, z={abs(score):.2f})"
                    ),
                })

    return sorted(anomalies, key=lambda x: x["z_score"], reverse=True)


# ---------------------------------------------------------------------------
# 3. Budget reallocation recommendations
# ---------------------------------------------------------------------------

def suggest_budget_reallocation(
    df: pd.DataFrame,
    realloc_pct: float = 0.20,
) -> list[dict]:
    """
    Identify the highest-ROAS channel and suggest moving `realloc_pct` of budget
    from each lower-performing channel to it. Calculates expected revenue gain.
    """
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
        transfer = row["spend"] * realloc_pct
        current_rev  = transfer * row["roas"]
        expected_rev = transfer * best["roas"]
        gain = expected_rev - current_rev
        gain_pct = gain / current_rev * 100 if current_rev > 0 else 0.0

        recommendations.append({
            "from_channel": row["channel"],
            "to_channel": best["channel"],
            "from_roas": round(float(row["roas"]), 2),
            "to_roas": round(float(best["roas"]), 2),
            "transfer_amount": round(float(transfer), 0),
            "expected_revenue_gain": round(float(gain), 0),
            "gain_pct": round(float(gain_pct), 1),
            "summary": (
                f"Flytt 20% av {row['channel']}-budsjettet (NOK {transfer:,.0f}) "
                f"til {best['channel']} → forventet inntektsgevinst: "
                f"NOK {gain:,.0f} (+{gain_pct:.0f}%)"
            ),
        })

    return recommendations
