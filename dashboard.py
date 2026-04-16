"""Markedsinnsikt AI — Dash dashboard (direct function calls, no HTTP layer)."""

import os
import re
import pandas as pd
from dotenv import load_dotenv
load_dotenv()

import dash
from dash import dcc, html, Input, Output, State, ctx
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go

from data import generate_dataset
from ai_assistant import (
    build_context, generate_insights, answer_question, detect_anomalies
)
from ml_models import (
    detect_anomalies_zscore,
    suggest_budget_reallocation,
    predict_xgboost_with_intervals,
    backtest_models,
    detect_anomalies_isolation_forest,
    compute_business_impact,
)

DEFAULT_MODEL = "llama-3.3-70b-versatile"


def fmt_nok(value: float) -> str:
    """Norwegian number format: NOK 1 234 (space as thousands separator)."""
    return f"NOK {value:,.0f}".replace(",", "\u202f")


def ai_error_alert(e: Exception) -> dbc.Alert:
    """User-friendly Norwegian alert for AI provider errors."""
    msg = str(e)
    if "429" in msg or "rate_limit" in msg.lower():
        wait = re.search(r"Please try again in ([\dmhs.]+)", msg)
        wait_str = f" Prøv igjen om ca. {wait.group(1)}." if wait else ""
        return dbc.Alert(
            [
                html.Strong("Daglig tokengrense nådd. "),
                f"Alle AI-leverandører er midlertidig utilgjengelige.{wait_str}",
                html.Br(),
                html.Small(
                    "Tips: filtrer til én kunde eller kanal for å bruke færre tokens.",
                    className="text-muted",
                ),
            ],
            color="warning",
            className="mt-2",
        )
    return dbc.Alert(f"Feil: {e}", color="danger", className="mt-2")

CHANNEL_COLORS = {
    "Google Ads": "#4285F4",   # Google blue
    "Meta Ads":   "#F97316",   # orange
    "TikTok Ads": "#10B981",   # green
}

# ---------------------------------------------------------------------------
# Data layer (direct calls — no HTTP)
# ---------------------------------------------------------------------------

_df: pd.DataFrame = generate_dataset()  # pre-warm at startup — avoids cold-start race on Render

def get_df() -> pd.DataFrame:
    return _df


def apply_filters(client="All", campaign="All", channel="All") -> pd.DataFrame:
    df = get_df()
    mask = pd.Series(True, index=df.index)
    if client and client != "All":
        mask &= df["client"] == client
    if campaign and campaign != "All":
        mask &= df["campaign"] == campaign
    if channel and channel != "All":
        mask &= df["channel"] == channel
    return df[mask]


def get_filters_data(client="All") -> dict:
    df = get_df()
    filtered = df if client == "All" else df[df["client"] == client]
    return {
        "clients":   ["All"] + sorted(df["client"].unique().tolist()),
        "campaigns": ["All"] + sorted(filtered["campaign"].unique().tolist()),
        "channels":  ["All"] + sorted(df["channel"].unique().tolist()),
    }


def get_kpis_data(client, campaign, channel) -> dict:
    df = apply_filters(client, campaign, channel)
    total_spend       = df["spend"].sum()
    total_revenue     = df["revenue"].sum()
    total_conversions = int(df["conversions"].sum())
    total_clicks      = df["clicks"].sum()
    total_impressions = df["impressions"].sum()
    avg_roas = total_revenue / total_spend if total_spend > 0 else 0
    avg_ctr  = total_clicks / total_impressions * 100 if total_impressions > 0 else 0
    return {
        "total_spend": round(total_spend, 2),
        "total_revenue": round(total_revenue, 2),
        "total_conversions": total_conversions,
        "avg_roas": round(avg_roas, 2),
        "avg_ctr": round(avg_ctr, 2),
        "row_count": len(df),
    }


def get_analytics_summary(client, campaign, channel) -> dict:
    df = apply_filters(client, campaign, channel)
    weekly = (
        df.groupby("week")
        .agg(spend=("spend", "sum"), revenue=("revenue", "sum"),
             conversions=("conversions", "sum"), clicks=("clicks", "sum"),
             impressions=("impressions", "sum"))
        .sort_index()
    )
    weekly["roas"] = weekly["revenue"] / weekly["spend"].replace(0, float("nan"))
    weekly["ctr"]  = weekly["clicks"]  / weekly["impressions"].replace(0, float("nan")) * 100

    def wow(series):
        clean = series.dropna()
        if len(clean) < 2 or clean.iloc[-2] == 0:
            return None
        return round((clean.iloc[-1] - clean.iloc[-2]) / abs(clean.iloc[-2]) * 100, 1)

    return {
        "trends": {
            "spend_wow":       wow(weekly["spend"]),
            "revenue_wow":     wow(weekly["revenue"]),
            "conversions_wow": wow(weekly["conversions"]),
            "roas_wow":        wow(weekly["roas"]),
            "ctr_wow":         wow(weekly["ctr"]),
        },
        "anomalies": detect_anomalies(df),
    }


def compute_portfolio_health(client, campaign, channel) -> dict:
    """
    Score 0–100 combining three signals:
      - ROAS health   (40 pts): avg ROAS vs portfolio benchmark
      - Trend health  (30 pts): share of channels with positive WoW ROAS
      - Anomaly load  (30 pts): penalises detected anomalies relative to campaign count
    """
    df = apply_filters(client, campaign, channel)
    full_df = get_df()

    if df.empty:
        return {"score": 0, "label": "Ingen data", "color": "#94a3b8"}

    # ROAS health (40 pts)
    avg_roas       = df["revenue"].sum() / df["spend"].sum() if df["spend"].sum() > 0 else 0
    portfolio_roas = full_df["revenue"].sum() / full_df["spend"].sum() if full_df["spend"].sum() > 0 else avg_roas
    roas_ratio     = min(avg_roas / portfolio_roas, 1.5) / 1.5  # cap at 1.5× benchmark
    roas_pts       = round(roas_ratio * 40)

    # Trend health (30 pts)
    positive = 0
    total_ch = 0
    for _, grp in df.groupby("channel"):
        weekly = grp.groupby("week").agg(spend=("spend","sum"), revenue=("revenue","sum")).sort_index()
        if len(weekly) < 2:
            continue
        weekly["roas"] = weekly["revenue"] / weekly["spend"].replace(0, float("nan"))
        roas_clean = weekly["roas"].dropna()
        if len(roas_clean) >= 2:
            total_ch += 1
            if roas_clean.iloc[-1] >= roas_clean.iloc[-2]:
                positive += 1
    trend_pts = round((positive / total_ch * 30) if total_ch > 0 else 15)

    # Anomaly load (30 pts)
    n_anomalies  = len(detect_anomalies(df))
    n_campaigns  = df["campaign"].nunique()
    anomaly_rate = n_anomalies / max(n_campaigns, 1)
    anomaly_pts  = round(max(0, 30 - anomaly_rate * 15))

    score = roas_pts + trend_pts + anomaly_pts

    if score >= 75:
        label, color = "God", "#10b981"
    elif score >= 50:
        label, color = "Moderat", "#f59e0b"
    else:
        label, color = "Svak", "#ef4444"

    return {"score": score, "label": label, "color": color}


def get_chart_roas(client, campaign, channel) -> list:
    df = apply_filters(client, campaign, channel)
    agg = df.groupby("channel")[["revenue", "spend"]].sum().reset_index()
    agg["roas"] = (agg["revenue"] / agg["spend"].replace(0, float("nan"))).round(2).fillna(0)
    return agg[["channel", "roas"]].to_dict(orient="records")


def get_chart_conv(client, campaign, channel) -> list:
    df = apply_filters(client, campaign, channel)
    return (
        df.groupby("campaign")["conversions"]
        .sum().reset_index().sort_values("conversions")
        .to_dict(orient="records")
    )


def get_chart_spend(client, campaign, channel) -> list:
    df = apply_filters(client, campaign, channel)
    return df.groupby("channel")["spend"].sum().reset_index().to_dict(orient="records")


def get_chart_weekly(client, campaign, channel) -> list:
    df = apply_filters(client, campaign, channel)
    return (
        df.groupby("week_date")["spend"]
        .sum()
        .reset_index()
        .sort_values("week_date")
        .to_dict(orient="records")
    )


def compute_chart_insights(client, campaign, channel) -> dict:
    """Rule-based one-liner interpretations for each chart."""
    df = apply_filters(client, campaign, channel)
    if df.empty:
        return {"roas": "", "conv": "", "spend": "", "weekly": ""}

    # ROAS by channel
    ch = df.groupby("channel").agg(spend=("spend", "sum"), revenue=("revenue", "sum"))
    ch["roas"] = ch["revenue"] / ch["spend"].replace(0, float("nan"))
    if len(ch) >= 2:
        best = ch["roas"].idxmax()
        worst = ch["roas"].idxmin()
        gap = (ch.loc[best, "roas"] - ch.loc[worst, "roas"]) / ch.loc[worst, "roas"] * 100
        roas_txt = (
            f"{best} leverer høyest ROAS ({ch.loc[best,'roas']:.1f}x), "
            f"{gap:.0f}% over {worst} ({ch.loc[worst,'roas']:.1f}x). "
            "Vurder å flytte budsjett hit."
        )
    else:
        roas_txt = ""

    # Conversions by campaign
    cp = df.groupby("campaign")["conversions"].sum()
    if len(cp) >= 2:
        best_cp = cp.idxmax()
        worst_cp = cp.idxmin()
        conv_txt = (
            f"«{best_cp}» topper med {int(cp[best_cp]):,} konverteringer. "
            f"«{worst_cp}» henger etter ({int(cp[worst_cp]):,}) — "
            "vurder å justere budsjett eller kreativ."
        )
    elif len(cp) == 1:
        best_cp = cp.idxmax()
        conv_txt = f"«{best_cp}»: {int(cp[best_cp]):,} konverteringer."
    else:
        conv_txt = ""

    # Spend by channel
    sp = df.groupby("channel")["spend"].sum()
    if not sp.empty:
        top = sp.idxmax()
        share = sp[top] / sp.sum() * 100
        spend_txt = (
            f"{top} mottar {share:.0f}% av totalt budsjett "
            f"(NOK {sp[top]:,.0f} av NOK {sp.sum():,.0f})."
        )
    else:
        spend_txt = ""

    # Weekly spend trend
    wk = df.groupby("week")["spend"].sum().sort_index()
    if len(wk) >= 2:
        wow = (wk.iloc[-1] - wk.iloc[-2]) / wk.iloc[-2] * 100
        direction = "økt" if wow >= 0 else "sunket"
        weekly_txt = (
            f"Forbruket har {direction} {abs(wow):.1f}% siste uke "
            f"(NOK {wk.iloc[-2]:,.0f} → NOK {wk.iloc[-1]:,.0f})."
        )
    else:
        weekly_txt = ""

    return {"roas": roas_txt, "conv": conv_txt, "spend": spend_txt, "weekly": weekly_txt}


# ---------------------------------------------------------------------------
# Initial data for dropdowns
# ---------------------------------------------------------------------------

# Filters are loaded via callback on page load (not at startup)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.FLATLY,
        "https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap",
    ],
    title="Markedsinnsikt AI",
    suppress_callback_exceptions=True,
)
server = app.server

# Consistent Plotly chart theme
CHART_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, system-ui, sans-serif", size=11.5, color="#374151"),
    margin=dict(t=46, b=24, l=8, r=8),
    xaxis=dict(showgrid=True, gridcolor="#f1f5f9", gridwidth=1, zeroline=False),
    yaxis=dict(showgrid=True, gridcolor="#f1f5f9", gridwidth=1, zeroline=False),
    legend=dict(font=dict(size=11)),
)


# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------

def kpi_card(label: str, value: str, trend: float | None = None, card_class: str = "kpi-card-blue") -> dbc.Col:
    if trend is not None:
        arrow    = "▲" if trend >= 0 else "▼"
        color    = "text-success" if trend >= 0 else "text-danger"
        trend_el = html.Small(f"{arrow} {abs(trend):.1f}% u/u", className=f"{color} fw-semibold")
    else:
        trend_el = html.Span()

    return dbc.Col(
        dbc.Card(
            dbc.CardBody([
                html.P(label, className="text-muted mb-1 small fw-semibold text-uppercase",
                       style={"letterSpacing": "0.06em", "fontSize": "0.7rem"}),
                html.H4(value, className="mb-1 fw-bold", style={"fontSize": "1.3rem"}),
                trend_el,
            ]),
            className="h-100",
        ),
        xs=6, md=True,
        className=f"kpi-card {card_class}",
    )


def section_header(icon: str, title: str) -> html.Div:
    return html.Div(html.Span(f"{icon}  {title}"), className="section-label")


def render_bubble(msg: dict) -> html.Div:
    is_user = msg["role"] == "user"
    return html.Div(
        html.Div(
            dcc.Markdown(msg["content"], style={"margin": 0}),
            style={
                "background": "#2c3e50" if is_user else "#e9ecef",
                "color": "white" if is_user else "#212529",
                "padding": "0.6rem 0.9rem",
                "borderRadius": "16px 16px 4px 16px" if is_user else "16px 16px 16px 4px",
                "maxWidth": "80%",
                "fontSize": "0.88rem",
                "lineHeight": "1.4",
            },
        ),
        style={
            "display": "flex",
            "justifyContent": "flex-end" if is_user else "flex-start",
            "marginBottom": "0.6rem",
        },
    )


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

sidebar = dbc.Col(
    [
        # Brand block
        html.Div(
            [
                html.Div("📊", style={"fontSize": "1.6rem", "lineHeight": "1", "marginBottom": "0.3rem"}),
                html.Div("Markedsinnsikt AI",
                         style={"fontWeight": "700", "fontSize": "0.92rem",
                                "color": "white", "letterSpacing": "-0.01em"}),
                html.Div("KI-drevet markedsanalyse",
                         style={"fontSize": "0.7rem", "color": "rgba(255,255,255,0.5)",
                                "marginTop": "0.15rem"}),
            ],
            className="sidebar-brand",
        ),

        # Filters label
        html.Div(html.Span("Filtre"), className="section-label"),

        html.Div([
            dbc.Label("Kunde", className="fw-semibold text-muted small mb-1"),
            html.Span(id="dot-client", style={"marginLeft": "0.4rem"}),
        ], style={"display": "flex", "alignItems": "center"}),
        dcc.Dropdown(id="dd-client", options=[], value="All", clearable=False, className="mb-3"),

        html.Div([
            dbc.Label("Kampanje", className="fw-semibold text-muted small mb-1"),
            html.Span(id="dot-campaign", style={"marginLeft": "0.4rem"}),
        ], style={"display": "flex", "alignItems": "center"}),
        dcc.Dropdown(id="dd-campaign", options=[], value="All", clearable=False, className="mb-3"),

        html.Div([
            dbc.Label("Kanal", className="fw-semibold text-muted small mb-1"),
            html.Span(id="dot-channel", style={"marginLeft": "0.4rem"}),
        ], style={"display": "flex", "alignItems": "center"}),
        dcc.Dropdown(id="dd-channel", options=[], value="All", clearable=False, className="mb-3"),

        dcc.Interval(id="init-trigger", interval=300, max_intervals=1),

        html.Hr(style={"borderColor": "#e2e8f0", "marginTop": "0.5rem"}),
        html.Div(id="row-count", className="text-muted small text-center"),

        html.Hr(style={"borderColor": "#e2e8f0", "marginTop": "0.75rem"}),
        html.Div(html.Span("Last ned"), className="section-label"),
        dbc.Button(
            "⬇ CSV",
            id="btn-download-csv",
            color="outline-secondary",
            size="sm",
            className="w-100",
        ),
        dcc.Download(id="download-csv"),
        dcc.Download(id="download-pdf"),
    ],
    width=2,
    style={
        "position": "sticky",
        "top": 0,
        "height": "100vh",
        "overflowY": "auto",
        "background": "#f8fafc",
        "padding": "1.25rem",
        "borderRight": "1px solid #e2e8f0",
    },
)

# ---------------------------------------------------------------------------
# Chat widget (floating)
# ---------------------------------------------------------------------------

chat_widget = html.Div(
    [
        # Panel
        html.Div(
            [
                # Header
                html.Div(
                    [
                        html.Span("💬 AI-assistent", style={"fontWeight": "600", "fontSize": "0.95rem"}),
                        html.Div(
                            [
                                dbc.Button(
                                    "Tøm",
                                    id="chat-clear-btn",
                                    size="sm",
                                    color="link",
                                    style={"color": "rgba(255,255,255,0.75)", "fontSize": "0.78rem", "padding": "0 0.5rem"},
                                ),
                                dbc.Button(
                                    "✕",
                                    id="chat-close-btn",
                                    size="sm",
                                    color="link",
                                    style={"color": "white", "fontSize": "1rem", "padding": "0 0.25rem"},
                                ),
                            ],
                            style={"display": "flex", "alignItems": "center"},
                        ),
                    ],
                    style={
                        "display": "flex",
                        "justifyContent": "space-between",
                        "alignItems": "center",
                        "background": "#2c3e50",
                        "color": "white",
                        "padding": "0.75rem 1rem",
                        "borderRadius": "14px 14px 0 0",
                    },
                ),

                # Messages
                html.Div(
                    id="chat-messages",
                    style={
                        "height": "380px",
                        "overflowY": "auto",
                        "padding": "1rem",
                        "background": "#ffffff",
                        "display": "flex",
                        "flexDirection": "column",
                    },
                ),

                # Loading indicator
                dcc.Loading(html.Div(id="chat-loading"), type="dot"),

                # Example questions
                html.Div(
                    [
                        html.Small("Prøv:", className="text-muted me-2 fw-semibold"),
                        dbc.Button("Beste kanal?", id="q-btn-1", size="sm",
                                   color="outline-secondary", className="me-1 mb-1"),
                        dbc.Button("Hva bør optimaliseres?", id="q-btn-2", size="sm",
                                   color="outline-secondary", className="me-1 mb-1"),
                        dbc.Button("Hvorfor lave konverteringer?", id="q-btn-3", size="sm",
                                   color="outline-secondary", className="mb-1"),
                    ],
                    style={
                        "padding": "0.5rem 0.75rem 0.25rem",
                        "background": "#f8f9fa",
                        "borderTop": "1px solid #dee2e6",
                        "flexWrap": "wrap",
                        "display": "flex",
                        "alignItems": "center",
                    },
                ),

                # Input row
                html.Div(
                    [
                        dcc.Input(
                            id="chat-input",
                            type="text",
                            placeholder="Skriv en melding...",
                            debounce=False,
                            n_submit=0,
                            style={
                                "flex": 1,
                                "border": "1px solid #dee2e6",
                                "borderRadius": "8px",
                                "padding": "0.5rem 0.75rem",
                                "fontSize": "0.88rem",
                                "outline": "none",
                            },
                        ),
                        dbc.Button(
                            "Send",
                            id="chat-send-btn",
                            color="primary",
                            size="sm",
                            style={"borderRadius": "8px", "flexShrink": 0},
                        ),
                    ],
                    style={
                        "display": "flex",
                        "gap": "0.5rem",
                        "padding": "0.75rem",
                        "borderTop": "1px solid #dee2e6",
                        "background": "#f8f9fa",
                        "borderRadius": "0 0 14px 14px",
                    },
                ),
            ],
            id="chat-panel",
            style={
                "display": "none",
                "position": "fixed",
                "bottom": "5.5rem",
                "right": "2rem",
                "width": "380px",
                "borderRadius": "14px",
                "boxShadow": "0 8px 32px rgba(0,0,0,0.18)",
                "zIndex": 1050,
                "overflow": "hidden",
            },
        ),

        # Toggle button
        dbc.Button(
            "💬 AI-assistent",
            id="chat-toggle-btn",
            color="primary",
            style={
                "position": "fixed",
                "bottom": "2rem",
                "right": "2rem",
                "zIndex": 1051,
                "borderRadius": "50px",
                "padding": "0.6rem 1.4rem",
                "boxShadow": "0 4px 16px rgba(0,0,0,0.2)",
                "fontWeight": "600",
            },
        ),

        # Persistent chat history (localStorage)
        dcc.Store(id="chat-store", storage_type="local"),
    ]
)

# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------

def chart_insight_text(div_id: str) -> html.Div:
    return html.Div(id=div_id, className="chart-insight")


main = dbc.Col(
    [
        # ── Page header banner ────────────────────────────────
        html.Div(
            [
                html.H2("Markedsinnsikt AI"),
                html.P(
                    "Et beslutningsstøtteverktøy som omgjør markedsdata til handlingsrettede "
                    "anbefalinger på tvers av kampanjer, kanaler og kunder."
                ),
            ],
            className="page-header",
        ),

        dbc.Tabs(
            [
                # ── Tab 1: Analyse ────────────────────────────
                dbc.Tab(
                    [
                        html.Div(
                            dbc.Button(
                                "⬇ Last ned rapport",
                                id="btn-download-pdf-analyse",
                                color="outline-primary",
                                size="sm",
                                className="float-end mt-3",
                            ),
                        ),
                        section_header("📈", "Oversikt"),
                        html.Div(id="portfolio-health-bar", className="mb-3"),
                        dbc.Row(id="kpi-row", className="mb-4 g-3"),

                        section_header("📊", "Ytelse"),
                        dbc.Row([
                            dbc.Col([
                                html.Div(dcc.Graph(id="chart-roas"), className="chart-wrapper"),
                                chart_insight_text("insight-roas"),
                            ], md=6),
                            dbc.Col([
                                html.Div(dcc.Graph(id="chart-conv"), className="chart-wrapper"),
                                chart_insight_text("insight-conv"),
                            ], md=6),
                        ], className="mb-1"),
                        dbc.Row([
                            dbc.Col([
                                html.Div(dcc.Graph(id="chart-spend-pie"), className="chart-wrapper"),
                                chart_insight_text("insight-spend"),
                            ], md=6),
                            dbc.Col([
                                html.Div(dcc.Graph(id="chart-weekly"), className="chart-wrapper"),
                                chart_insight_text("insight-weekly"),
                            ], md=6),
                        ], className="mb-3"),
                    ],
                    label="📊 Analyse",
                    tab_id="tab-analyse",
                    className="pt-4",
                ),

                # ── Tab 2: ML-analyse ─────────────────────────
                dbc.Tab(
                    [
                        html.Div(
                            [
                                html.P(
                                    "XGBoost-tidsserieprediksjoner med 90% konfidensintervaller · "
                                    "Walk-forward backtesting · Isolation Forest + z-score avviksdeteksjon.",
                                    className="text-muted small mb-0",
                                    style={"flex": 1},
                                ),
                                dbc.Button(
                                    "⬇ Last ned rapport",
                                    id="btn-download-pdf-ml",
                                    color="outline-primary",
                                    size="sm",
                                    style={"whiteSpace": "nowrap"},
                                ),
                            ],
                            className="d-flex align-items-center gap-3 mt-4 mb-3",
                        ),
                        dcc.Loading(html.Div(id="ml-out"), type="circle"),
                    ],
                    label="🔮 ML-analyse",
                    tab_id="tab-ml",
                    className="pt-2",
                ),

                # ── Tab 3: AI Innsikt ─────────────────────────
                dbc.Tab(
                    [
                        html.Div(
                            dbc.Button(
                                "⬇ Last ned rapport",
                                id="btn-download-pdf-ai",
                                color="outline-primary",
                                size="sm",
                            ),
                            className="d-flex justify-content-end mt-3 mb-2",
                        ),
                        dcc.Loading(
                            html.Div(id="live-insights-panel"),
                            type="circle",
                        ),
                    ],
                    label="🧠 AI Innsikt",
                    tab_id="tab-innsikt",
                    className="pt-2",
                ),
            ],
            id="main-tabs",
            active_tab="tab-analyse",
            className="mb-2",
        ),

        html.Hr(style={"marginTop": "3rem", "borderColor": "#e2e8f0"}),
        html.P(
            "Dataene er syntetiske og kun for demoformål.",
            className="text-muted text-center small mb-5",
        ),
    ],
    width=10,
    className="p-4",
)

# ---------------------------------------------------------------------------
# Notification bell (anomalies)
# ---------------------------------------------------------------------------

anomaly_notification = html.Div(
    [
        # Bell button
        html.Div(
            dbc.Button(
                [html.Span("🔔 "), html.Span("0", id="notif-badge")],
                id="notif-btn",
                color="danger",
                size="sm",
                style={
                    "borderRadius": "50px",
                    "fontWeight": "600",
                    "boxShadow": "0 2px 8px rgba(0,0,0,0.2)",
                },
            ),
            id="notif-wrapper",
            style={
                "display": "none",
                "position": "fixed",
                "top": "1rem",
                "right": "1rem",
                "zIndex": 1060,
            },
        ),

        # Modal
        dbc.Modal(
            [
                dbc.ModalHeader(dbc.ModalTitle("⚠️ Oppdagede avvik")),
                dbc.ModalBody(html.Div(id="notif-modal-body")),
                dbc.ModalFooter(
                    dbc.Button("Lukk", id="notif-modal-close", color="secondary", size="sm")
                ),
            ],
            id="notif-modal",
            is_open=False,
            scrollable=True,
        ),

        # Store anomaly data
        dcc.Store(id="anomaly-store"),
        # Track last-run filters per tab to avoid redundant re-runs
        dcc.Store(id="insights-last-filters", data=None),
        dcc.Store(id="ml-last-filters", data=None),
        # Cache ML results so AI tab can read without recomputing
        dcc.Store(id="ml-results-store", data=None),
    ]
)

app.layout = dbc.Container(
    [
        dbc.Row([sidebar, main], className="g-0"),
        chat_widget,
        anomaly_notification,
    ],
    fluid=True,
)

# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

@app.callback(
    Output("dd-client", "options"),
    Output("dd-client", "value"),
    Output("dd-channel", "options"),
    Output("dd-channel", "value"),
    Input("init-trigger", "n_intervals"),
)
def load_filters(_):
    data = get_filters_data()
    client_opts  = [{"label": c, "value": c} for c in data["clients"]]
    channel_opts = [{"label": c, "value": c} for c in data["channels"]]
    return client_opts, "All", channel_opts, "All"


@app.callback(
    Output("dd-campaign", "options"),
    Output("dd-campaign", "value"),
    Input("dd-client", "value"),
)
def update_campaign_options(client):
    data = get_filters_data(client)
    options = [{"label": c, "value": c} for c in data["campaigns"]]
    return options, "All"


@app.callback(
    Output("portfolio-health-bar", "children"),
    Output("kpi-row", "children"),
    Output("row-count", "children"),
    Output("anomaly-store", "data"),
    Output("notif-badge", "children"),
    Output("notif-wrapper", "style"),
    Input("dd-client", "value"),
    Input("dd-campaign", "value"),
    Input("dd-channel", "value"),
)
def update_kpis(client, campaign, channel):
    data      = get_kpis_data(client, campaign, channel)
    summary   = get_analytics_summary(client, campaign, channel)
    trends    = summary.get("trends", {})
    anomalies = summary.get("anomalies", [])
    health    = compute_portfolio_health(client, campaign, channel)

    health_bar = html.Div(
        [
            html.Div(
                [
                    html.Span("Porteføljehelse", style={
                        "fontSize": "0.7rem", "fontWeight": "700", "textTransform": "uppercase",
                        "letterSpacing": "0.08em", "color": "#64748b", "marginRight": "0.75rem",
                    }),
                    html.Span(f"{health['score']}/100", style={
                        "fontSize": "1.1rem", "fontWeight": "700", "color": health["color"],
                        "marginRight": "0.4rem",
                    }),
                    html.Span(health["label"], style={
                        "fontSize": "0.78rem", "fontWeight": "600", "color": health["color"],
                    }),
                ],
                style={"display": "flex", "alignItems": "center", "marginBottom": "0.4rem"},
            ),
            html.Div(
                html.Div(style={
                    "width": f"{health['score']}%",
                    "height": "6px",
                    "borderRadius": "3px",
                    "background": health["color"],
                    "transition": "width 0.4s ease",
                }),
                style={
                    "background": "#e2e8f0", "borderRadius": "3px",
                    "height": "6px", "width": "100%",
                },
            ),
        ],
        style={"padding": "0.75rem 1rem", "background": "#f8fafc",
               "borderRadius": "8px", "border": "1px solid #e2e8f0"},
    )

    cards = [
        kpi_card("Totalt forbruk", fmt_nok(data['total_spend']),        trends.get("spend_wow"),       "kpi-card-amber"),
        kpi_card("Total inntekt",  fmt_nok(data['total_revenue']),      trends.get("revenue_wow"),     "kpi-card-green"),
        kpi_card("Konverteringer", f"{data['total_conversions']:,}",    trends.get("conversions_wow"), "kpi-card-blue"),
        kpi_card("Gj.snitt ROAS",  f"{data['avg_roas']:.2f}x",         trends.get("roas_wow"),        "kpi-card-purple"),
        kpi_card("Gj.snitt CTR",   f"{data['avg_ctr']:.2f}%",          trends.get("ctr_wow"),         "kpi-card-teal"),
    ]

    bell_style_base = {
        "position": "fixed", "top": "1rem", "right": "1rem", "zIndex": 1060,
    }
    bell_style = {**bell_style_base, "display": "block" if anomalies else "none"}

    return (
        health_bar,
        cards,
        f"{data['row_count']} rader i gjeldende visning",
        anomalies,
        str(len(anomalies)),
        bell_style,
    )


@app.callback(
    Output("notif-modal-body", "children"),
    Input("anomaly-store", "data"),
)
def render_anomaly_modal_body(anomalies):
    if not anomalies:
        return html.P("Ingen avvik oppdaget.", className="text-muted")
    severity_color = {"high": "danger", "medium": "warning"}
    return [
        dbc.Alert(
            [
                html.Strong(f"[{a['severity'].upper()}] [{a['client']}] {a['campaign']}: "),
                a["detail"],
            ],
            color=severity_color.get(a["severity"], "warning"),
            className="py-2 mb-2",
        )
        for a in anomalies
    ]


@app.callback(
    Output("notif-modal", "is_open"),
    Input("notif-btn", "n_clicks"),
    Input("notif-modal-close", "n_clicks"),
    State("notif-modal", "is_open"),
    prevent_initial_call=True,
)
def toggle_notif_modal(_, _close, is_open):
    return not is_open


def _empty_fig(msg: str = "Ingen data for dette utvalget") -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=msg, xref="paper", yref="paper",
                       x=0.5, y=0.5, showarrow=False,
                       font=dict(size=13, color="#94a3b8"))
    fig.update_layout(**{k: v for k, v in CHART_LAYOUT.items() if k not in ("xaxis", "yaxis")},
                      xaxis=dict(visible=False), yaxis=dict(visible=False))
    return fig


@app.callback(
    Output("chart-roas", "figure"),
    Output("chart-conv", "figure"),
    Output("chart-spend-pie", "figure"),
    Output("chart-weekly", "figure"),
    Input("dd-client", "value"),
    Input("dd-campaign", "value"),
    Input("dd-channel", "value"),
)
def update_charts(client, campaign, channel):
    roas_data   = get_chart_roas(client, campaign, channel)
    conv_data   = get_chart_conv(client, campaign, channel)
    spend_data  = get_chart_spend(client, campaign, channel)
    weekly_data = get_chart_weekly(client, campaign, channel)

    # ── ROAS per kanal ───────────────────────────────────────────────────
    if not roas_data:
        fig_roas = _empty_fig()
    else:
        avg_roas = sum(r["roas"] for r in roas_data) / len(roas_data)
        fig_roas = px.bar(
            roas_data, x="channel", y="roas",
            color="channel", color_discrete_map=CHANNEL_COLORS,
            text_auto=".2f", title="ROAS per kanal",
            labels={"roas": "ROAS (x)", "channel": ""},
        )
        fig_roas.update_traces(
            textposition="outside", marker_line_width=0,
            hovertemplate="<b>%{x}</b><br>ROAS: %{y:.2f}x<extra></extra>",
        )
        fig_roas.add_hline(y=avg_roas, line_dash="dot", line_color="#94a3b8",
                           annotation_text=f"Snitt {avg_roas:.2f}x",
                           annotation_position="top right",
                           annotation_font=dict(size=10, color="#94a3b8"))
        fig_roas.update_layout(**CHART_LAYOUT, showlegend=False)

    # ── Konverteringer per kampanje ──────────────────────────────────────
    if not conv_data:
        fig_conv = _empty_fig()
    else:
        fig_conv = px.bar(
            conv_data, x="conversions", y="campaign",
            orientation="h", text_auto=True,
            title="Konverteringer per kampanje",
            labels={"conversions": "Konverteringer", "campaign": ""},
            color="conversions", color_continuous_scale="Blues",
        )
        fig_conv.update_traces(
            hovertemplate="<b>%{y}</b><br>Konverteringer: %{x:,}<extra></extra>",
        )
        fig_conv.update_layout(**CHART_LAYOUT, coloraxis_showscale=False)

    # ── Forbruk per kanal ────────────────────────────────────────────────
    if not spend_data:
        fig_spend = _empty_fig()
    else:
        fig_spend = px.pie(
            spend_data, names="channel", values="spend",
            color="channel", color_discrete_map=CHANNEL_COLORS,
            title="Forbruk per kanal", hole=0.38,
        )
        fig_spend.update_traces(
            textposition="inside", textinfo="percent+label",
            marker=dict(line=dict(color="white", width=2)),
            hovertemplate="<b>%{label}</b><br>Forbruk: NOK %{value:,.0f}<br>Andel: %{percent}<extra></extra>",
        )
        fig_spend.update_layout(**{k: v for k, v in CHART_LAYOUT.items()
                                   if k not in ("xaxis", "yaxis")})

    # ── Ukentlig forbrukstrend ───────────────────────────────────────────
    if not weekly_data:
        fig_weekly = _empty_fig()
    else:
        fig_weekly = px.line(
            weekly_data, x="week_date", y="spend",
            markers=True, title="Ukentlig forbrukstrend",
            labels={"spend": "Forbruk (NOK)", "week_date": "Dato"},
            color_discrete_sequence=["#3b82f6"],
        )
        fig_weekly.update_traces(
            line=dict(width=2.5), marker=dict(size=7),
            hovertemplate="%{x|%d %b %Y}<br>Forbruk: NOK %{y:,.0f}<extra></extra>",
        )
        fig_weekly.update_layout(**CHART_LAYOUT)

    return fig_roas, fig_conv, fig_spend, fig_weekly


def render_insights(data: dict, ml_recs: list | None = None, impacts: list | None = None) -> html.Div:
    unified = _render_unified_action_plan(data.get("recommendations", []), ml_recs or [], impacts or [])

    exec_decision = data.get("executive_decision", "")
    exec_block = html.Div(
        [
            html.Div(
                "Ukens beslutning",
                style={
                    "fontSize": "0.65rem", "fontWeight": "700", "letterSpacing": "0.1em",
                    "textTransform": "uppercase", "color": "rgba(255,255,255,0.7)",
                    "marginBottom": "0.4rem",
                },
            ),
            html.Div(
                exec_decision,
                style={
                    "fontSize": "1.15rem", "fontWeight": "700", "lineHeight": "1.4",
                    "color": "white",
                },
            ),
        ],
        style={
            "background": "linear-gradient(135deg, #1e3a5f 0%, #2c3e50 100%)",
            "borderRadius": "12px",
            "padding": "1.25rem 1.5rem",
            "marginBottom": "1.25rem",
            "borderLeft": "4px solid #3b82f6",
            "boxShadow": "0 4px 16px rgba(0,0,0,0.12)",
        },
    ) if exec_decision else html.Div()

    return html.Div([
        exec_block,

        dbc.Alert(data.get("summary", ""), color="info", className="mb-3"),

        html.H6("Nøkkelinnsikt", className="fw-bold"),
        html.Ul([
            html.Li([html.Strong(i.get("title", "") + ": "), i.get("detail", "")])
            for i in data.get("insights", [])
        ], className="mb-3"),

        unified,
    ])


def _render_unified_action_plan(ai_recs: list, ml_recs: list, impacts: list) -> html.Div:
    """Merge AI recommendations + ML budget signals into a single ranked action plan."""

    rows = []

    # AI recommendations → rank by priority
    priority_rank = {"high": 0, "medium": 1, "low": 2}
    for r in sorted(ai_recs, key=lambda x: priority_rank.get(x.get("priority", "low"), 2)):
        pri = r.get("priority", "low")
        rows.append({
            "rank":   priority_rank.get(pri, 2),
            "source": "AI",
            "badge":  pri.upper(),
            "color":  {"high": "danger", "medium": "warning", "low": "success"}.get(pri, "secondary"),
            "action": r.get("action", ""),
            "detail": r.get("expected_impact", r.get("target", "")),
        })

    # ML budget reallocation → always "medium" priority
    for r in ml_recs:
        rows.append({
            "rank":   1,
            "source": "ML",
            "badge":  "BUDSJETT",
            "color":  "info",
            "action": f"Flytt budsjett: {r['from_channel']} → {r['to_channel']}",
            "detail": f"ROAS: {r['from_roas']:.1f}x → {r['to_roas']:.1f}x",
        })

    # ML business impact → flag lowest-accuracy channel as high risk
    if impacts:
        worst = min(impacts, key=lambda x: x["decision_accuracy_proxy"])
        if worst["decision_accuracy_proxy"] < 75:
            rows.append({
                "rank":   0,
                "source": "ML",
                "badge":  "RISIKO",
                "color":  "danger",
                "action": f"Forbedre prediksjonsnøyaktighet for {worst['channel']}",
                "detail": (
                    f"~{round(worst['decision_accuracy_proxy'] / 5) * 5:.0f}% nøyaktighet · "
                    f"~{fmt_nok(round(worst['estimated_weekly_cost_of_error'] / 500) * 500)}/uke risiko"
                ),
            })

    if not rows:
        return html.Div()

    rows.sort(key=lambda x: x["rank"])

    table_rows = [
        html.Tr([
            html.Td(html.Span(r["badge"], className=f"badge bg-{r['color']}")),
            html.Td(html.Span(r["source"], className="badge bg-secondary")),
            html.Td(html.Strong(r["action"])),
            html.Td(html.Small(r["detail"], className="text-muted")),
        ])
        for r in rows
    ]

    return html.Div([
        html.H6("Samlet handlingsplan", className="fw-bold mt-2 mb-1"),
        html.P(
            "Prioriterte tiltak basert på AI-analyse og ML-modellsignaler.",
            className="text-muted small mb-2",
        ),
        dbc.Table(
            [
                html.Thead(html.Tr([
                    html.Th("Prioritet", style={"width": "9%"}),
                    html.Th("Kilde",     style={"width": "7%"}),
                    html.Th("Tiltak",    style={"width": "35%"}),
                    html.Th("Detalj"),
                ])),
                html.Tbody(table_rows),
            ],
            bordered=True, hover=True, responsive=True, size="sm",
        ),
    ])




# ---------------------------------------------------------------------------
# Chat callbacks
# ---------------------------------------------------------------------------

@app.callback(
    Output("chat-panel", "style"),
    Input("chat-toggle-btn", "n_clicks"),
    Input("chat-close-btn", "n_clicks"),
    State("chat-panel", "style"),
    prevent_initial_call=True,
)
def toggle_chat_panel(_, _close, current_style):
    if ctx.triggered_id == "chat-close-btn":
        return {**current_style, "display": "none"}
    new_display = "none" if current_style.get("display") == "block" else "block"
    return {**current_style, "display": new_display}


@app.callback(
    Output("chat-store", "data", allow_duplicate=True),
    Output("chat-loading", "children", allow_duplicate=True),
    Input("chat-panel", "style"),
    State("chat-store", "data"),
    State("dd-client", "value"),
    State("dd-campaign", "value"),
    State("dd-channel", "value"),
    prevent_initial_call=True,
)
def auto_greet(panel_style, history, client, campaign, channel):
    if not panel_style or panel_style.get("display") != "block":
        return dash.no_update, None
    if history:
        return dash.no_update, None
    try:
        context = build_context(get_df(), client, campaign, channel)
        greeting = answer_question(
            [{"role": "user", "content": (
                "Gi meg en kort velkomstmelding på 2 setninger basert på dataene du ser nå. "
                "Nevn det viktigste tallet eller trenden, og spør hva jeg ønsker å analysere."
            )}],
            context,
            model=DEFAULT_MODEL,
        )
        return [{"role": "assistant", "content": greeting}], None
    except Exception:
        return dash.no_update, None


@app.callback(
    Output("chat-store", "data"),
    Output("chat-input", "value"),
    Output("chat-loading", "children"),
    Input("chat-send-btn", "n_clicks"),
    Input("chat-input", "n_submit"),
    State("chat-input", "value"),
    State("chat-store", "data"),
    State("dd-client", "value"),
    State("dd-campaign", "value"),
    State("dd-channel", "value"),
    prevent_initial_call=True,
)
def send_message(_, _submit, message, history, client, campaign, channel):
    if not message or not message.strip():
        return history, "", None

    history = history or []
    history.append({"role": "user", "content": message.strip()})

    try:
        context = build_context(get_df(), client, campaign, channel)
        reply = answer_question(history, context, model=DEFAULT_MODEL)
        history.append({"role": "assistant", "content": reply})
    except Exception as e:
        msg = str(e)
        if "429" in msg or "rate_limit" in msg.lower():
            wait = re.search(r"Please try again in ([\dmhs.]+)", msg)
            wait_str = f" Prøv igjen om ca. {wait.group(1)}." if wait else ""
            history.append({"role": "assistant", "content":
                f"Alle AI-leverandører er midlertidig utilgjengelige.{wait_str} "
                "Prøv igjen senere, eller filtrer til én kunde for å bruke færre tokens."})
        else:
            history.append({"role": "assistant", "content": f"Beklager, noe gikk galt: {e}"})

    return history, "", None


@app.callback(
    Output("chat-messages", "children"),
    Input("chat-store", "data"),
)
def render_messages(history):
    if not history:
        return html.P(
            "Ingen meldinger.",
            className="text-muted text-center",
            style={"marginTop": "2rem", "fontSize": "0.88rem"},
        )
    return [render_bubble(msg) for msg in history]


@app.callback(
    Output("chat-store", "data", allow_duplicate=True),
    Input("chat-clear-btn", "n_clicks"),
    prevent_initial_call=True,
)
def clear_chat(_):
    return []


def _sev_color(sev: str) -> str:
    return "danger" if sev == "high" else "warning"


def render_ml_results(
    xgb_results: list,
    backtest: list,
    zscore_anomalies: list,
    if_anomalies: list,
    impacts: list | None = None,
) -> html.Div:

    # ── 1. XGBoost forecast cards ─────────────────────────────────────────
    xgb_cards = [
        dbc.Col(
            dbc.Card([
                dbc.CardHeader(html.Strong(p["channel"])),
                dbc.CardBody([
                    html.P([
                        html.Span("ROAS prediksjon ", className="text-muted small"),
                        html.Span(
                            pd.to_datetime(p.get("next_date", "")).strftime("%d %b")
                            if p.get("next_date") else f"uke {p['next_week']}",
                            className="fw-bold"
                        ),
                        html.Span(f": {p['predicted_roas']:.2f}x",
                                  className="fw-bold text-primary ms-1"),
                    ], className="mb-1"),
                    html.P(
                        f"90% intervall: [{p['lower_90']:.2f}x – {p['upper_90']:.2f}x]",
                        className="text-muted small mb-0",
                    ),
                ]),
            ], className="shadow-sm h-100"),
            md=4, className="mb-3",
        )
        for p in xgb_results
    ]

    # ── 2. XGBoost forecast chart with 90% CI error bars ─────────────────
    fig_xgb = go.Figure()
    for p in xgb_results:
        ch    = p["channel"]
        color = CHANNEL_COLORS.get(ch, "#888")
        dates = [h.get("week_date", str(h["week"])) for h in p["history"]]
        roas  = [h["roas"] for h in p["history"]]

        fig_xgb.add_trace(go.Scatter(
            x=dates, y=roas, name=f"{ch} (faktisk)",
            mode="lines+markers",
            line=dict(color=color, width=2.5),
            marker=dict(size=6),
            hovertemplate="%{x|%d %b}<br>ROAS: %{y:.2f}x<extra></extra>",
        ))
        fig_xgb.add_trace(go.Scatter(
            x=[p.get("next_date", str(p["next_week"]))], y=[p["predicted_roas"]],
            name=f"{ch} prediksjon",
            mode="markers",
            marker=dict(symbol="star", size=18, color=color,
                        line=dict(width=1, color="white")),
            error_y=dict(
                type="data", symmetric=False,
                array=[p["upper_90"] - p["predicted_roas"]],
                arrayminus=[p["predicted_roas"] - p["lower_90"]],
                color=color, thickness=2, width=6,
            ),
            hovertemplate="%{x|%d %b}<br>Prediksjon: %{y:.2f}x<extra></extra>",
        ))

    fig_xgb.update_layout(**CHART_LAYOUT)
    fig_xgb.update_layout(
        title="XGBoost ROAS-prediksjon med 90% usikkerhetsintervall",
        xaxis_title="Dato", yaxis_title="ROAS (x)",
        legend=dict(orientation="h", y=-0.28, font=dict(size=10)),
        margin=dict(t=50, b=90, l=8, r=8),
    )

    # ── 3. SHAP explainability ────────────────────────────────────────────
    FEAT_LABELS = {"lag_1": "Forrige uke (lag 1)", "lag_2": "To uker siden (lag 2)",
                   "rolling_mean": "Rullende snitt (3 uker)", "trend": "Tidsindeks"}

    fi_section: list = []
    if xgb_results:
        # Global SHAP — mean |SHAP| across all channels
        all_channels = [p["channel"] for p in xgb_results]
        rows_global = []
        for p in xgb_results:
            for feat, val in p["shap_global"].items():
                rows_global.append({"feature": FEAT_LABELS.get(feat, feat),
                                     "channel": p["channel"], "shap": round(val, 4)})

        fig_global = px.bar(
            rows_global, x="shap", y="feature", color="channel",
            orientation="h", barmode="group",
            color_discrete_map=CHANNEL_COLORS,
            title="SHAP — gjennomsnittlig absoluttverdi per feature",
            labels={"shap": "Gj.snitt |SHAP|", "feature": ""},
        )
        fig_global.update_layout(**CHART_LAYOUT)
        fig_global.update_layout(
            legend=dict(orientation="h", y=-0.3, font=dict(size=10)),
            margin=dict(t=46, b=80, l=8, r=8),
        )

        # Local SHAP — next-week prediction breakdown for first channel
        p0 = xgb_results[0]
        local_items = sorted(p0["shap_local"].items(), key=lambda x: x[1])
        rows_local = [
            {
                "feature": FEAT_LABELS.get(k, k),
                "shap": round(v, 4),
                "color": "#10b981" if v >= 0 else "#ef4444",
            }
            for k, v in local_items
        ]
        fig_local = go.Figure(go.Bar(
            x=[r["shap"] for r in rows_local],
            y=[r["feature"] for r in rows_local],
            orientation="h",
            marker_color=[r["color"] for r in rows_local],
            text=[f"{r['shap']:+.3f}" for r in rows_local],
            textposition="outside",
        ))
        base = p0["base_value"]
        pred = p0["predicted_roas"]
        fig_local.update_layout(**CHART_LAYOUT)
        fig_local.update_layout(
            title=f"SHAP lokal forklaring — {p0['channel']} neste uke "
                  f"(basis {base:.2f}x → prediksjon {pred:.2f}x)",
            xaxis_title="SHAP-bidrag (x)", yaxis_title="",
            margin=dict(t=50, b=24, l=8, r=60),
        )
        fig_local.add_vline(x=0, line_width=1, line_color="#94a3b8")

        fi_section = [
            html.H6("SHAP-forklarbarhet", className="fw-bold mt-3 mb-1"),
            html.P(
                "Global: hvilke features driver ROAS-prediksjoner på tvers av kanaler. "
                "Lokal: hvordan hver feature bidro til neste ukes prediksjon.",
                className="text-muted small mb-2",
            ),
            dbc.Row([
                dbc.Col(dcc.Graph(figure=fig_global), md=6),
                dbc.Col(dcc.Graph(figure=fig_local),  md=6),
            ]),
            html.Hr(),
        ]

    # ── Tooltip helper ────────────────────────────────────────────────────
    def _th(label: str, tip_id: str, tip_text: str) -> html.Th:
        return html.Th([
            label,
            html.Sup(
                " ⓘ",
                id=tip_id,
                style={"cursor": "help", "color": "#94a3b8", "fontSize": "0.65rem"},
            ),
            dbc.Tooltip(tip_text, target=tip_id, placement="top"),
        ])

    # ── 4. Backtesting table ──────────────────────────────────────────────
    if backtest:
        bt_rows = [
            html.Tr([
                html.Td(r["channel"]),
                html.Td(f"{r['lr_mae']:.3f}"),
                html.Td(f"{r['xgb_mae']:.3f}"),
                html.Td(f"{r['lr_rmse']:.3f}"),
                html.Td(f"{r['xgb_rmse']:.3f}"),
                html.Td(f"{r['improvement_pct']:+.1f}%"),
            ])
            for r in backtest
        ]
        bt_table = dbc.Table(
            [
                html.Thead(html.Tr([
                    html.Th("Kanal"),
                    _th("Lin. MAE",  "tip-lr-mae",   "Mean Absolute Error — gjennomsnittlig absolutt ROAS-avvik for lineær regresjon."),
                    _th("XGB MAE",   "tip-xgb-mae",  "Mean Absolute Error for XGBoost. Lavere er bedre."),
                    _th("Lin. RMSE", "tip-lr-rmse",  "Root Mean Squared Error — straffer store feil tyngre enn MAE."),
                    _th("XGB RMSE",  "tip-xgb-rmse", "Root Mean Squared Error for XGBoost."),
                    _th("XGB forbedring vs lin.", "tip-improvement", "Positiv = XGBoost er bedre enn lineær baseline. Negativ = baseline vinner."),
                ])),
                html.Tbody(bt_rows),
            ],
            bordered=True, hover=True, responsive=True, size="sm",
            className="mb-3",
        )

        # Backtesting chart — one tab per channel
        bt_tabs = dbc.Tabs([
            dbc.Tab(
                dcc.Graph(figure=_make_backtest_fig(r)),
                label=r["channel"],
                tab_id=f"bt-{r['channel'].replace(' ', '-')}",
            )
            for r in backtest
        ], active_tab=f"bt-{backtest[0]['channel'].replace(' ', '-')}")

        backtest_section = [
            html.H6("Backtesting — walk-forward validering", className="fw-bold mt-3 mb-1"),
            html.P(
                "Trener på t=1..k, predikerer uke k+1 for hvert steg. "
                "Sammenligner Lineær regresjon vs XGBoost.",
                className="text-muted small mb-2",
            ),
            bt_table,
            bt_tabs,
        ]
    else:
        backtest_section = [
            html.P("For lite data for backtesting (trenger ≥5 uker per kanal).",
                   className="text-muted small"),
        ]

    # ── 5. Anomaly detection (z-score + Isolation Forest) ────────────────
    def anomaly_card(a: dict) -> dbc.Alert:
        method_badge = html.Span(
            a.get("method", ""),
            className="badge bg-secondary me-2",
        )
        score_badge = (
            html.Span(f"z={a['z_score']:.2f}", className=f"badge bg-{_sev_color(a['severity'])} me-2")
            if "z_score" in a else
            html.Span(f"IF={a['anomaly_score']:.3f}", className=f"badge bg-{_sev_color(a['severity'])} me-2")
        )
        return dbc.Alert(
            [method_badge, score_badge,
             html.Strong(f"[{a['client']}] {a['campaign']}: "),
             a["detail"]],
            color=_sev_color(a["severity"]),
            className="py-2 mb-2",
        )

    all_anomalies = zscore_anomalies[:6] + if_anomalies[:6]
    anomaly_section = [
        html.H6("Avviksdeteksjon: Z-score + Isolation Forest",
                className="fw-bold mt-3 mb-1"),
        html.P(
            f"Z-score flagget {len(zscore_anomalies)} avvik · "
            f"Isolation Forest flagget {len(if_anomalies)} avvik "
            "(multidimensjonalt: spend + ROAS + CTR samtidig).",
            className="text-muted small mb-2",
        ),
    ] + (
        [anomaly_card(a) for a in all_anomalies]
        if all_anomalies else
        [html.P("Ingen avvik oppdaget.", className="text-muted small")]
    )

    # ── 6. Error analysis ─────────────────────────────────────────────────
    error_section: list = []
    if backtest:
        def _failure_badge(r: dict) -> html.Span:
            bias    = r.get("xgb_bias", 0)
            dir_acc = r.get("direction_accuracy")
            mae     = r.get("xgb_mae", 0)
            if abs(bias) > 0.15:
                return html.Span(f"🔴 Bias ({bias:+.2f}x)", className="badge bg-danger")
            elif dir_acc is not None and dir_acc < 55:
                return html.Span(f"🟡 Trend ({dir_acc:.0f}%)", className="badge bg-warning text-dark")
            elif mae > 0.5:
                return html.Span(f"🟡 Høy MAE ({mae:.2f})", className="badge bg-warning text-dark")
            else:
                return html.Span("🟢 OK", className="badge bg-success")

        error_rows = []
        worst_cases_all: list = []
        for r in backtest:
            bias = r.get("xgb_bias", 0)
            dir_acc = r.get("direction_accuracy")
            dir_str = f"{dir_acc:.0f}%" if dir_acc is not None else "–"
            bias_color = "text-danger" if abs(bias) > 0.15 else "text-success"
            error_rows.append(html.Tr([
                html.Td(r["channel"]),
                html.Td(html.Span(f"{bias:+.3f}x", className=bias_color)),
                html.Td(dir_str),
                html.Td(_failure_badge(r)),
            ]))
            for wc in r.get("worst_cases", []):
                worst_cases_all.append({**wc, "channel": r["channel"]})

        worst_cases_all.sort(key=lambda x: x["abs_error"], reverse=True)

        error_section = [
            html.H6("Feilanalyse", className="fw-bold mt-3 mb-1"),
            dbc.Table(
                [
                    html.Thead(html.Tr([
                        html.Th("Kanal"),
                        _th("Bias",         "tip-err-bias",  "Gjennomsnittlig retningsfeil. Positiv = modellen undervurderer. Negativ = overvurderer."),
                        _th("Trendretning", "tip-err-dir",   "Andel uker der modellen korrekt predikerer om ROAS stiger eller faller."),
                        _th("Sviktmodus",   "tip-err-fail",  "Klassifisert feilmønster basert på bias og trendretning."),
                    ])),
                    html.Tbody(error_rows),
                ],
                bordered=True, hover=True, responsive=True, size="sm", className="mb-3",
            ),
            html.H6("Verste prediksjoner (topp 5)", className="fw-bold mb-1"),
            dbc.Table(
                [
                    html.Thead(html.Tr([
                        html.Th("Kanal"), html.Th("Uke"),
                        html.Th("Faktisk"), html.Th("Predikert"), html.Th("Feil"),
                    ])),
                    html.Tbody([
                        html.Tr([
                            html.Td(wc["channel"]),
                            html.Td(str(wc["week"])),
                            html.Td(f"{wc['actual']:.2f}x"),
                            html.Td(f"{wc['predicted']:.2f}x"),
                            html.Td(
                                html.Span(
                                    f"{wc['error']:+.2f}x",
                                    className="text-danger" if abs(wc["error"]) > 0.2 else "text-warning",
                                )
                            ),
                        ])
                        for wc in worst_cases_all[:5]
                    ]),
                ],
                bordered=True, hover=True, responsive=True, size="sm", className="mb-3",
            ),
        ]

    # ── 8. Business impact ────────────────────────────────────────────────
    if impacts:
        impact_cards = [
            dbc.Col(
                dbc.Card([
                    dbc.CardHeader(html.Strong(imp["channel"])),
                    dbc.CardBody([
                        html.P([
                            html.Span([
                                "Beslutningsnøyaktighet ",
                                html.Sup("ⓘ", id=f"tip-da-{imp['channel'].replace(' ','-')}",
                                         style={"cursor": "help", "color": "#94a3b8", "fontSize": "0.65rem"}),
                                dbc.Tooltip(
                                    "1 − (MAE / gj.snitt ROAS). 100% = perfekt prediksjon.",
                                    target=f"tip-da-{imp['channel'].replace(' ','-')}",
                                    placement="top",
                                ),
                            ], className="text-muted small"),
                            html.Span(
                                f"ca. {round(imp['decision_accuracy_proxy'] / 5) * 5:.0f}%",
                                className=(
                                    "fw-bold text-success" if imp["decision_accuracy_proxy"] >= 80
                                    else "fw-bold text-warning" if imp["decision_accuracy_proxy"] >= 60
                                    else "fw-bold text-danger"
                                ),
                            ),
                        ], className="mb-1"),
                        html.P([
                            html.Span([
                                "Ukentlig feilkostnad ",
                                html.Sup("ⓘ", id=f"tip-wc-{imp['channel'].replace(' ','-')}",
                                         style={"cursor": "help", "color": "#94a3b8", "fontSize": "0.65rem"}),
                                dbc.Tooltip(
                                    "Estimat: MAE × gj.snitt ukentlig forbruk. Indikerer størrelsesorden, ikke eksakt tap.",
                                    target=f"tip-wc-{imp['channel'].replace(' ','-')}",
                                    placement="top",
                                ),
                            ], className="text-muted small"),
                            html.Span(
                                f"~NOK {round(imp['estimated_weekly_cost_of_error'] / 500) * 500:,.0f}",
                                className="fw-bold",
                            ),
                        ], className="mb-0"),
                        html.Small(f"Gj.snitt ROAS: {imp['avg_actual_roas']:.1f}x",
                                   className="text-muted"),
                    ]),
                ], className="shadow-sm h-100"),
                md=4, className="mb-3",
            )
            for imp in impacts
        ]
        impact_section: list = [
            html.H6("Forretningsmessig konsekvens av prediksjonfeil", className="fw-bold mt-3 mb-1"),
            html.P(
                "Estimert ukentlig inntektstap ved å handle på feil ROAS-prediksjon. "
                "Beslutningsnøyaktighet = 1 − (MAE / gj.snitt ROAS).",
                className="text-muted small mb-2",
            ),
            dbc.Row(impact_cards),
        ]
    else:
        impact_section = []

    return html.Div([
        section_header("🔮", "XGBoost-prediksjon neste uke"),
        dbc.Row(xgb_cards),
        html.Div(dcc.Graph(figure=fig_xgb), className="chart-wrapper mb-3"),
        *fi_section,

        html.Hr(),
        *backtest_section,

        html.Hr(),
        *error_section,

        html.Hr(),
        *impact_section,

        html.Hr(),
        *anomaly_section,
    ])


def _make_backtest_fig(r: dict) -> go.Figure:
    """Line chart for one channel's walk-forward backtest results."""
    data = r["backtest_data"]
    weeks = [d["week"]   for d in data]
    act   = [d["actual"] for d in data]
    lr    = [d["lr"]     for d in data]
    xgb   = [d["xgb"]   for d in data]

    ch    = r["channel"]
    color = CHANNEL_COLORS.get(ch, "#888")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=weeks, y=act, name="Faktisk",
        mode="lines+markers", line=dict(color=color, width=2.5),
    ))
    fig.add_trace(go.Scatter(
        x=weeks, y=lr, name="Lineær reg.",
        mode="lines+markers",
        line=dict(color="#94a3b8", width=1.5, dash="dot"),
        marker=dict(size=5),
    ))
    fig.add_trace(go.Scatter(
        x=weeks, y=xgb, name="XGBoost",
        mode="lines+markers",
        line=dict(color="#10b981", width=1.5, dash="dash"),
        marker=dict(size=5),
    ))
    fig.update_layout(**CHART_LAYOUT)
    fig.update_layout(
        title=f"Backtesting: {ch} — MAE lin={r['lr_mae']:.3f} · xgb={r['xgb_mae']:.3f}",
        xaxis_title="Uke", yaxis_title="ROAS (x)",
        legend=dict(orientation="h", y=-0.28, font=dict(size=10)),
        margin=dict(t=50, b=80, l=8, r=8),
    )
    return fig


@app.callback(
    Output("ml-out", "children"),
    Output("ml-last-filters", "data"),
    Output("ml-results-store", "data"),
    Input("dd-client", "value"),
    Input("dd-campaign", "value"),
    Input("dd-channel", "value"),
    State("ml-last-filters", "data"),
)
def run_ml_analysis(client, campaign, channel, last_filters):
    current = {"client": client, "campaign": campaign, "channel": channel}
    if current == last_filters:
        return dash.no_update, dash.no_update, dash.no_update
    try:
        df = apply_filters(client, campaign, channel)
        xgb_results  = predict_xgboost_with_intervals(df)
        bt           = backtest_models(df)
        z_anomalies  = detect_anomalies_zscore(df)
        if_anomalies = detect_anomalies_isolation_forest(df)
        recs         = suggest_budget_reallocation(df)
        avg_spend    = float(df.groupby("week")["spend"].sum().mean())
        impacts      = compute_business_impact(bt, avg_spend)
        cached = {"bt": bt, "impacts": impacts, "ml_recs": recs}
        return render_ml_results(xgb_results, bt, z_anomalies, if_anomalies, impacts), current, cached
    except Exception as e:
        return dbc.Alert(f"ML-feil: {e}", color="danger"), current, dash.no_update


@app.callback(
    Output("insight-roas", "children"),
    Output("insight-conv", "children"),
    Output("insight-spend", "children"),
    Output("insight-weekly", "children"),
    Input("dd-client", "value"),
    Input("dd-campaign", "value"),
    Input("dd-channel", "value"),
)
def update_chart_insights(client, campaign, channel):
    hints = compute_chart_insights(client, campaign, channel)
    return hints["roas"], hints["conv"], hints["spend"], hints["weekly"]


_DOT_ACTIVE = "●"
_DOT_STYLE_ON  = {"color": "#3b82f6", "fontSize": "0.55rem", "verticalAlign": "middle"}
_DOT_STYLE_OFF = {"display": "none"}

@app.callback(
    Output("dot-client",   "children"), Output("dot-client",   "style"),
    Output("dot-campaign", "children"), Output("dot-campaign", "style"),
    Output("dot-channel",  "children"), Output("dot-channel",  "style"),
    Input("dd-client",   "value"),
    Input("dd-campaign", "value"),
    Input("dd-channel",  "value"),
)
def update_filter_dots(client, campaign, channel):
    def dot(val):
        active = val and val != "All"
        return (_DOT_ACTIVE, _DOT_STYLE_ON) if active else ("", _DOT_STYLE_OFF)
    c, cs = dot(client)
    p, ps = dot(campaign)
    ch, chs = dot(channel)
    return c, cs, p, ps, ch, chs


@app.callback(
    Output("live-insights-panel", "children"),
    Output("insights-last-filters", "data"),
    Input("dd-client", "value"),
    Input("dd-campaign", "value"),
    Input("dd-channel", "value"),
    State("insights-last-filters", "data"),
    State("ml-results-store", "data"),
)
def update_live_insights(client, campaign, channel, last_filters, ml_cache):
    current = {"client": client, "campaign": campaign, "channel": channel}
    if current == last_filters:
        return dash.no_update, dash.no_update
    try:
        context = build_context(get_df(), client, campaign, channel)
        data = generate_insights(context, model=DEFAULT_MODEL)

        # Use cached ML results if available — avoids recomputing backtest
        if ml_cache:
            impacts = ml_cache.get("impacts", [])
            ml_recs = ml_cache.get("ml_recs", [])
        else:
            try:
                df = apply_filters(client, campaign, channel)
                bt = backtest_models(df)
                avg_spend = float(df.groupby("week")["spend"].sum().mean())
                impacts   = compute_business_impact(bt, avg_spend)
                ml_recs   = suggest_budget_reallocation(df)
            except Exception:
                impacts, ml_recs = [], []

        return render_insights(data, ml_recs, impacts), current
    except Exception as e:
        return ai_error_alert(e), current


@app.callback(
    Output("chat-input", "value", allow_duplicate=True),
    Input("q-btn-1", "n_clicks"),
    Input("q-btn-2", "n_clicks"),
    Input("q-btn-3", "n_clicks"),
    prevent_initial_call=True,
)
def prefill_question(*_):
    questions = {
        "q-btn-1": "Hvilken kanal presterer best?",
        "q-btn-2": "Hva bør jeg optimalisere?",
        "q-btn-3": "Hvorfor er konverteringene lave?",
    }
    return questions.get(ctx.triggered_id, dash.no_update)


@app.callback(
    Output("download-csv", "data"),
    Input("btn-download-csv", "n_clicks"),
    State("dd-client",   "value"),
    State("dd-campaign", "value"),
    State("dd-channel",  "value"),
    prevent_initial_call=True,
)
def download_csv(_, client, campaign, channel):
    df = apply_filters(client, campaign, channel)
    # Human-readable column order for the export
    cols = ["client", "campaign", "channel", "week_date", "spend",
            "revenue", "roas", "ctr", "clicks", "conversions", "impressions",
            "goal", "audience", "ad_text"]
    export = df[[c for c in cols if c in df.columns]].rename(columns={
        "client": "Kunde", "campaign": "Kampanje", "channel": "Kanal",
        "week_date": "Dato", "spend": "Forbruk (NOK)", "revenue": "Inntekt (NOK)",
        "roas": "ROAS", "ctr": "CTR (%)", "clicks": "Klikk",
        "conversions": "Konverteringer", "impressions": "Visninger",
        "goal": "Mål", "audience": "Målgruppe", "ad_text": "Annonsetekst",
    })
    parts = [p for p in [client, campaign, channel] if p and p != "All"]
    filename = "markedsinnsikt_" + ("_".join(parts) if parts else "alle") + ".csv"
    return dcc.send_data_frame(export.to_csv, filename, index=False)


@app.callback(
    Output("download-pdf", "data"),
    Input("btn-download-pdf-analyse", "n_clicks"),
    Input("btn-download-pdf-ml",      "n_clicks"),
    Input("btn-download-pdf-ai",      "n_clicks"),
    State("dd-client",                "value"),
    State("dd-campaign",              "value"),
    State("dd-channel",               "value"),
    State("ml-results-store",         "data"),
    prevent_initial_call=True,
)
def download_pdf_report(_an, _ml, _ai, client, campaign, channel, ml_cache):
    from datetime import datetime as _dt

    # ── 1. AI insights ────────────────────────────────────────────────────
    try:
        context  = build_context(get_df(), client, campaign, channel)
        ai_data  = generate_insights(context, model=DEFAULT_MODEL)
    except Exception as e:
        ai_data = {"summary": f"AI ikke tilgjengelig: {e}", "insights": [],
                   "recommendations": [], "executive_decision": ""}

    # ── 2. ML results (from cache or fresh) ───────────────────────────────
    if ml_cache:
        bt      = ml_cache.get("bt", [])
        impacts = ml_cache.get("impacts", [])
        ml_recs = ml_cache.get("ml_recs", [])
    else:
        try:
            df        = apply_filters(client, campaign, channel)
            bt        = backtest_models(df)
            avg_spend = float(df.groupby("week")["spend"].sum().mean())
            impacts   = compute_business_impact(bt, avg_spend)
            ml_recs   = suggest_budget_reallocation(df)
        except Exception:
            bt, impacts, ml_recs = [], [], []

    # ── 3. KPIs ───────────────────────────────────────────────────────────
    kpis   = get_kpis_data(client, campaign, channel)
    health = compute_portfolio_health(client, campaign, channel)

    # ── 4. Build HTML report ──────────────────────────────────────────────
    generated = _dt.now().strftime("%d.%m.%Y %H:%M")
    scope_parts = [p for p in [client, campaign, channel] if p and p != "All"]
    scope_str   = " · ".join(scope_parts) if scope_parts else "Alle kunder / kampanjer / kanaler"

    def _kpi_row(label, value):
        return f"<tr><td>{label}</td><td><strong>{value}</strong></td></tr>"

    def _insight_rows():
        rows = ""
        for i in ai_data.get("insights", []):
            rows += f"<tr><td><strong>{i.get('title','')}</strong></td><td>{i.get('detail','')}</td></tr>"
        return rows or "<tr><td colspan='2'>Ingen innsikter.</td></tr>"

    def _rec_rows():
        rows = ""
        priority_rank = {"high": 0, "medium": 1, "low": 2}
        recs = sorted(ai_data.get("recommendations", []),
                      key=lambda x: priority_rank.get(x.get("priority", "low"), 2))
        for r in recs:
            pri_no = {"high": "HØY", "medium": "MIDDELS", "low": "LAV"}.get(r.get("priority",""), "–")
            rows += (f"<tr><td><span class='badge-{r.get('priority','low')}'>{pri_no}</span></td>"
                     f"<td>{r.get('action','')}</td>"
                     f"<td>{r.get('expected_impact','')}</td></tr>")
        return rows or "<tr><td colspan='3'>Ingen anbefalinger.</td></tr>"

    def _ml_bt_rows():
        rows = ""
        for r in bt:
            bias = r.get("xgb_bias", 0)
            bias_str = f"{bias:+.3f}x"
            dir_acc = r.get("direction_accuracy")
            dir_str = f"{dir_acc:.0f}%" if dir_acc is not None else "–"
            imp = f"{r.get('improvement_pct', 0):+.1f}%"
            rows += (f"<tr><td>{r['channel']}</td>"
                     f"<td>{r['xgb_mae']:.3f}</td><td>{r['xgb_rmse']:.3f}</td>"
                     f"<td>{bias_str}</td><td>{dir_str}</td><td>{imp}</td></tr>")
        return rows or "<tr><td colspan='6'>Ikke nok data for backtesting.</td></tr>"

    def _ml_rec_rows():
        rows = ""
        for r in ml_recs:
            rows += (f"<tr><td>{r['from_channel']}</td><td>{r['to_channel']}</td>"
                     f"<td>{r['from_roas']:.2f}x</td><td>{r['to_roas']:.2f}x</td>"
                     f"<td>{r['summary']}</td></tr>")
        return rows or "<tr><td colspan='5'>Ingen budsjettanbefalinger.</td></tr>"

    def _impact_rows():
        rows = ""
        for imp in impacts:
            acc = round(imp["decision_accuracy_proxy"] / 5) * 5
            cost = round(imp["estimated_weekly_cost_of_error"] / 500) * 500
            rows += (f"<tr><td>{imp['channel']}</td>"
                     f"<td>ca. {acc:.0f}%</td>"
                     f"<td>~NOK {cost:,.0f}</td>"
                     f"<td>{imp['avg_actual_roas']:.2f}x</td></tr>")
        return rows or "<tr><td colspan='4'>Ingen data.</td></tr>"

    exec_decision = ai_data.get("executive_decision", "")
    exec_block = (
        f"<div class='exec-box'><div class='exec-label'>UKENS BESLUTNING</div>"
        f"<div class='exec-text'>{exec_decision}</div></div>"
        if exec_decision else ""
    )

    html_report = f"""<!DOCTYPE html>
<html lang="no">
<head>
<meta charset="UTF-8">
<title>Markedsinnsikt AI — Rapport {generated}</title>
<style>
  body {{ font-family: 'Segoe UI', Arial, sans-serif; font-size: 13px;
          color: #1e293b; margin: 0; padding: 2rem 2.5rem; background: #fff; }}
  h1   {{ font-size: 1.5rem; color: #0f172a; margin-bottom: 0.2rem; }}
  h2   {{ font-size: 1.05rem; color: #1e3a5f; border-bottom: 2px solid #e2e8f0;
          padding-bottom: 0.3rem; margin-top: 2rem; margin-bottom: 0.75rem; }}
  h3   {{ font-size: 0.9rem; color: #475569; margin: 1rem 0 0.4rem; }}
  .meta {{ color: #64748b; font-size: 0.82rem; margin-bottom: 1.5rem; }}
  .exec-box {{ background: linear-gradient(135deg,#1e3a5f,#2c3e50); color: white;
               border-left: 4px solid #3b82f6; border-radius: 8px;
               padding: 1rem 1.25rem; margin-bottom: 1.25rem; }}
  .exec-label {{ font-size: 0.65rem; letter-spacing: 0.1em; color: rgba(255,255,255,0.65);
                 text-transform: uppercase; margin-bottom: 0.3rem; }}
  .exec-text  {{ font-size: 1.05rem; font-weight: 700; line-height: 1.4; }}
  .summary-box {{ background: #eff6ff; border-left: 3px solid #3b82f6;
                  padding: 0.75rem 1rem; border-radius: 4px; margin-bottom: 1rem;
                  font-size: 0.92rem; }}
  .health-bar-wrap {{ background: #e2e8f0; border-radius: 4px; height: 8px; width: 200px;
                      display: inline-block; vertical-align: middle; margin-left: 0.5rem; }}
  .health-bar {{ height: 8px; border-radius: 4px; background: {health['color']};
                 width: {health['score']}%; }}
  table  {{ width: 100%; border-collapse: collapse; font-size: 0.82rem; margin-bottom: 1rem; }}
  th     {{ background: #f1f5f9; text-align: left; padding: 0.45rem 0.6rem;
            border-bottom: 2px solid #e2e8f0; font-weight: 600; }}
  td     {{ padding: 0.4rem 0.6rem; border-bottom: 1px solid #f1f5f9; vertical-align: top; }}
  tr:hover td {{ background: #f8fafc; }}
  .badge-high   {{ background:#fee2e2; color:#b91c1c; padding:2px 6px; border-radius:4px; font-size:0.75rem; }}
  .badge-medium {{ background:#fef3c7; color:#92400e; padding:2px 6px; border-radius:4px; font-size:0.75rem; }}
  .badge-low    {{ background:#d1fae5; color:#065f46; padding:2px 6px; border-radius:4px; font-size:0.75rem; }}
  .footer {{ margin-top: 3rem; font-size: 0.75rem; color: #94a3b8; border-top: 1px solid #e2e8f0;
             padding-top: 0.75rem; }}
  @media print {{
    body {{ padding: 0; }}
    @page {{ margin: 1.5cm; }}
  }}
</style>
</head>
<body>

<h1>📊 Markedsinnsikt AI — Analyserapport</h1>
<div class="meta">
  Generert: {generated} &nbsp;|&nbsp; Utvalg: <strong>{scope_str}</strong>
</div>

{exec_block}

<div class="summary-box">{ai_data.get('summary', '')}</div>

<h2>Porteføljehelse</h2>
<table>
  <tr>
    <th>Indikator</th><th>Verdi</th>
  </tr>
  {_kpi_row("Totalt forbruk", fmt_nok(kpis["total_spend"]))}
  {_kpi_row("Total inntekt",  fmt_nok(kpis["total_revenue"]))}
  {_kpi_row("Konverteringer", f"{kpis['total_conversions']:,}")}
  {_kpi_row("Gj.snitt ROAS",  f"{kpis['avg_roas']:.2f}x")}
  {_kpi_row("Gj.snitt CTR",   f"{kpis['avg_ctr']:.2f}%")}
  {_kpi_row("Porteføljehelse",
    f"{health['score']}/100 {health['label']} "
    f"<span class='health-bar-wrap'><span class='health-bar'></span></span>")}
</table>

<h2>AI-innsikt</h2>
<table>
  <tr><th style="width:28%">Tema</th><th>Detalj</th></tr>
  {_insight_rows()}
</table>

<h2>Anbefalinger (AI)</h2>
<table>
  <tr><th style="width:10%">Prioritet</th><th style="width:40%">Tiltak</th><th>Forventet effekt</th></tr>
  {_rec_rows()}
</table>

<h2>ML — Budsjettomfordeling</h2>
<table>
  <tr><th>Fra kanal</th><th>Til kanal</th><th>Fra ROAS</th><th>Til ROAS</th><th>Begrunnelse</th></tr>
  {_ml_rec_rows()}
</table>

<h2>ML — Backtesting (walk-forward validering)</h2>
<table>
  <tr><th>Kanal</th><th>XGB MAE</th><th>XGB RMSE</th><th>Bias</th><th>Trendretning</th><th>Forbedring vs lin.</th></tr>
  {_ml_bt_rows()}
</table>

<h2>ML — Forretningsmessig konsekvens av prediksjonfeil</h2>
<table>
  <tr><th>Kanal</th><th>Beslutningsnøyaktighet</th><th>Ukentlig feilkostnad</th><th>Gj.snitt ROAS</th></tr>
  {_impact_rows()}
</table>

<div class="footer">
  Markedsinnsikt AI · Syntetiske data for demoformål ·
  Rapport generert {generated}
</div>

</body>
</html>"""

    parts = [p for p in [client, campaign, channel] if p and p != "All"]
    filename = "rapport_" + ("_".join(parts) if parts else "alle") + f"_{_dt.now().strftime('%Y%m%d')}.html"
    return dcc.send_string(html_report, filename)


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8050))
    app.run(host="0.0.0.0", port=port, debug=False)
