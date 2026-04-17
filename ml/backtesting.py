"""Walk-forward backtesting for LinearRegression and XGBoost ROAS models."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ml.features import _lag_features


def backtest_models(
    df: pd.DataFrame,
    n_lags: int = 2,
    window: int = 26,
    max_steps: int = 52,
) -> list[dict]:
    """
    Walk-forward validation on weekly ROAS per channel.

    window    : rolling training window in weeks (default 26 = 6 months).
                Keeps each XGBoost fit small regardless of history length.
    max_steps : cap the number of backtest steps to the most recent N weeks
                (default 52 = 1 year). Prevents O(n²) blowup on long series.

    For each step k, trains both LinearRegression and XGBoost, predicts k+1,
    and records actual vs predicted. Also computes error analysis:
      - bias (mean signed error)
      - direction accuracy (% of weeks where trend direction is correct)
      - worst 3 prediction cases
      - failure mode classification
    """
    from xgboost import XGBRegressor
    from sklearn.linear_model import LinearRegression

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

        # Limit steps to the most recent max_steps weeks
        all_steps = list(range(min_train, len(vals)))
        steps = all_steps[-max_steps:] if len(all_steps) > max_steps else all_steps

        for i in steps:
            start = max(0, i - window)
            train = vals[start:i]
            X_tr, y_tr = _lag_features(train, n_lags)
            if len(X_tr) < 2:
                continue

            next_feat = (
                list(train[-n_lags:])
                + [float(np.mean(train[-3:])), float(i)]
            )
            next_feat_arr = np.array([next_feat], dtype=float)

            lr = LinearRegression().fit(X_tr, y_tr)
            lr_p = float(lr.predict(next_feat_arr)[0])

            xgb = XGBRegressor(
                n_estimators=30, max_depth=2, learning_rate=0.15,
                subsample=0.9, verbosity=0, random_state=42,
            )
            xgb.fit(X_tr, y_tr)
            xgb_p = float(xgb.predict(next_feat_arr)[0])

            actuals.append(vals[i])
            lr_preds.append(lr_p)
            xgb_preds.append(xgb_p)
            pred_weeks.append(int(weeks[i]))

        if not actuals:
            continue

        act   = np.array(actuals)
        lr_a  = np.array(lr_preds)
        xgb_a = np.array(xgb_preds)

        xgb_errors = act - xgb_a   # positive = underpredicted, negative = overpredicted
        lr_errors  = act - lr_a

        lr_mae   = float(np.mean(np.abs(lr_errors)))
        xgb_mae  = float(np.mean(np.abs(xgb_errors)))
        lr_rmse  = float(np.sqrt(np.mean(lr_errors ** 2)))
        xgb_rmse = float(np.sqrt(np.mean(xgb_errors ** 2)))
        winner   = "XGBoost" if xgb_mae < lr_mae else "Lineær regresjon"
        imp_pct  = (lr_mae - xgb_mae) / lr_mae * 100 if lr_mae > 0 else 0.0

        # Bias: mean signed error (positive = model underpredicts, negative = overpredicts)
        xgb_bias = float(np.mean(xgb_errors))

        # Direction accuracy: % of consecutive steps where trend direction is correct
        if len(act) >= 2:
            actual_dirs = np.sign(np.diff(act))
            pred_dirs   = np.sign(np.diff(xgb_a))
            direction_acc = float(np.mean(actual_dirs == pred_dirs)) * 100
        else:
            direction_acc = None

        # Worst 3 prediction cases by absolute error
        abs_err = np.abs(xgb_errors)
        worst_idx = np.argsort(abs_err)[-3:][::-1]
        worst_cases = [
            {
                "week":      int(pred_weeks[i]),
                "actual":    round(float(act[i]), 2),
                "predicted": round(float(xgb_a[i]), 2),
                "error":     round(float(xgb_errors[i]), 2),
                "abs_error": round(float(abs_err[i]), 2),
            }
            for i in worst_idx
        ]

        # Failure mode classification
        if abs(xgb_bias) > 0.15:
            failure_mode = (
                f"Systematisk {'undervurdering' if xgb_bias > 0 else 'overvurdering'} "
                f"({xgb_bias:+.2f}x gjennomsnittlig avvik)"
            )
        elif direction_acc is not None and direction_acc < 55:
            failure_mode = (
                f"Svak trendretning ({direction_acc:.0f}% korrekt) — "
                "modellen treffer dårlig på opp/ned-bevegelser"
            )
        elif xgb_mae > 0.5:
            failure_mode = (
                f"Høy absolutt feil (MAE {xgb_mae:.3f}x) — "
                "for lite treningsdata eller høy volatilitet"
            )
        else:
            failure_mode = "Ingen systematisk svikt oppdaget"

        results.append({
            "channel":          ch,
            "lr_mae":           round(lr_mae,   3),
            "xgb_mae":          round(xgb_mae,  3),
            "lr_rmse":          round(lr_rmse,  3),
            "xgb_rmse":         round(xgb_rmse, 3),
            "winner":           winner,
            "improvement_pct":  round(imp_pct, 1),
            "steps":            len(actuals),
            "avg_actual_roas":  round(float(np.mean(act)), 2),
            "xgb_bias":         round(xgb_bias, 3),
            "direction_accuracy": round(direction_acc, 1) if direction_acc is not None else None,
            "worst_cases":      worst_cases,
            "failure_mode":     failure_mode,
            "validation_type":  f"rolling-{window}w / last-{max_steps}steps",
            "backtest_data": [
                {
                    "week":      w,
                    "actual":    round(float(a), 2),
                    "lr":        round(float(l), 2),
                    "xgb":       round(float(x), 2),
                    "xgb_error": round(float(e), 2),
                }
                for w, a, l, x, e in zip(
                    pred_weeks, actuals, lr_preds, xgb_preds, xgb_errors.tolist()
                )
            ],
        })

    return results


def compute_business_impact(
    backtest_results: list[dict],
    avg_weekly_spend: float,
) -> list[dict]:
    """
    Translate ROAS prediction error into business terms.

    A ROAS prediction error of `mae` applied to `avg_weekly_spend` gives an
    estimate of how much revenue could be mis-allocated per week.

    decision_accuracy_proxy = 1 - (mae / avg_actual_roas)
    Represents how close the model is to a perfect decision (100% = no error).
    """
    impacts = []
    for r in backtest_results:
        mae      = r["xgb_mae"]
        avg_roas = r.get("avg_actual_roas", 1.0) or 1.0
        estimated_cost   = mae * avg_weekly_spend
        decision_accuracy = max(0.0, min(1.0, 1.0 - (mae / avg_roas))) * 100
        impacts.append({
            "channel":                        r["channel"],
            "mae":                            round(mae, 3),
            "avg_actual_roas":                round(avg_roas, 2),
            "estimated_weekly_cost_of_error": round(estimated_cost, 0),
            "decision_accuracy_proxy":        round(decision_accuracy, 1),
        })
    return impacts
