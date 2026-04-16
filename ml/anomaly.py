"""Anomaly detection models: Z-score and Isolation Forest."""

from __future__ import annotations

import pandas as pd

from ml.features import _zscore


def detect_anomalies_zscore(
    df: pd.DataFrame,
    threshold: float = 2.0,
) -> list[dict]:
    """Flag weeks where ROAS deviates >= threshold std-devs from campaign mean."""
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
