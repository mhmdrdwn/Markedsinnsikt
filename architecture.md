# System Architecture — Markedsinnsikt AI

```
Data ingestion
   ↓
   data.py — Synthetic campaign dataset generator
   (client, campaign, channel, week, spend, revenue,
    clicks, impressions, conversions, ROAS, CTR, goal, audience)
   ↓
Feature engineering
   ↓
   ml_models._lag_features()
   Lag features: [lag_1, lag_2, rolling_mean_3, time_index]
   per channel, computed on weekly ROAS time series
   ↓
ML models (XGBoost / baseline)
   ↓
   ml_models.predict_xgboost_with_intervals()
     XGBRegressor(n_estimators=200, max_depth=3, lr=0.05)
     → point forecast + 90% prediction interval (from residual std)

   ml_models.predict_next_week()
     LinearRegression baseline on weekly spend & ROAS
   ↓
Backtesting & evaluation
   ↓
   ml_models.backtest_models()
     Walk-forward validation (expanding or rolling window)
     Evaluates both LinearRegression and XGBoost per fold
     Metrics: MAE, RMSE, bias, direction accuracy
     Error analysis: worst cases, failure mode classification

   ml_models.compute_business_impact()
     Translates MAE → estimated weekly revenue cost of error
     Decision accuracy proxy = 1 − (MAE / avg ROAS)
   ↓
Anomaly detection
   ↓
   ml_models.detect_anomalies_zscore()
     Statistical: flags ROAS deviations ≥ 2σ from campaign mean

   ml_models.detect_anomalies_isolation_forest()
     Multi-dimensional: IsolationForest on (spend, ROAS, CTR)
     Captures outliers unusual only in combination

   ai_assistant.detect_anomalies()
     Rule-based: ROAS drops ≥ 25% and spend spikes ≥ 50% WoW
   ↓
Insight generator (LLM)
   ↓
   ai_assistant.build_context()
     Aggregates: channel/campaign breakdown, goal-specific KPIs,
     audience segments, WoW trends, predictive signals,
     cross-client benchmarks, anomalies → structured text context

   ai_assistant.generate_insights()
     Provider chain: Groq → Gemini → Mistral (automatic fallback)
     Returns structured JSON: summary, insights, anomalies, recommendations

   ai_assistant.answer_question()
     Multi-turn chat with full campaign context
   ↓
Dashboard (Dash UI)
   ↓
   dashboard.py — Single-process Dash app (no HTTP layer)
   Tab 1 — Analyse:   KPI cards + 4 performance charts
   Tab 2 — AI Innsikt: AI-generated analysis (auto on tab open / filter change)
   Tab 3 — ML-analyse: XGBoost forecast, backtesting, error analysis,
                        business impact, anomaly detection, budget reallocation
   Floating chat widget: multi-turn AI assistant
   Notification bell:    real-time anomaly alerts
```

## Key design decisions

| Decision | Rationale |
|---|---|
| Single Dash process (no FastAPI) | Eliminates network overhead; direct function calls |
| Walk-forward backtest | Respects temporal order; no data leakage |
| Rolling window option | Adapts to non-stationary marketing data |
| Groq → Gemini → Mistral fallback | Resilience against rate limits / provider outages |
| Isolation Forest over pure z-score | Catches multivariate anomalies invisible to univariate methods |
| 90% PI from residual std | Simple, interpretable uncertainty estimate without quantile regression |
| Filter-state store (dcc.Store) | Prevents redundant AI/ML re-runs on tab switches without filter changes |
