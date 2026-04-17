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

from data import get_dataset
from ai import (
    build_context, generate_insights, answer_question, detect_anomalies,
    generate_insights_with_meta, answer_question_with_tools,
)
from ml import (
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

_df: pd.DataFrame = get_dataset()  # Robyn MMM dataset — pre-warmed at startup

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

def kpi_card(label: str, value: str, trend: float | None = None, card_class: str = "kpi-card-blue", icon: str = "") -> dbc.Col:
    if trend is not None:
        arrow    = "▲" if trend >= 0 else "▼"
        color    = "text-success" if trend >= 0 else "text-danger"
        trend_el = html.Small(f"{arrow} {abs(trend):.1f}% u/u", className=f"{color} fw-semibold",
                              style={"fontSize": "0.75rem"})
    else:
        trend_el = html.Span()

    return dbc.Col(
        dbc.Card(
            dbc.CardBody([
                html.Div([
                    html.P(label, className="kpi-label mb-0"),
                    html.Span(icon, className="kpi-icon") if icon else html.Span(),
                ], className="d-flex justify-content-between align-items-start mb-2"),
                html.Div(value, className="kpi-value mb-1"),
                trend_el,
            ], style={"padding": "1rem 1.1rem"}),
            className="h-100",
        ),
        xs=6, md=True,
        className=f"kpi-card {card_class}",
    )


def section_header(icon: str, title: str) -> html.Div:
    return html.Div(html.Span(f"{icon}  {title}"), className="section-label")


_TOOL_LABEL_MAP = {
    "get_channel_performance": "kanal-data",
    "get_top_channel":         "beste kanal",
    "compare_channels":        "sammenlign kanaler",
    "get_weekly_trend":        "ukentlig trend",
    "get_anomalies":           "avvikssøk",
}


def render_bubble(msg: dict) -> html.Div:
    is_user = msg["role"] == "user"

    # Build meta footer for assistant messages
    meta_items: list = []
    if not is_user:
        tools = msg.get("tools_used") or []
        latency = msg.get("latency_ms")
        tokens  = msg.get("tokens")
        provider = msg.get("provider", "")

        if tools:
            tool_chips = [
                html.Span(
                    _TOOL_LABEL_MAP.get(t, t),
                    style={
                        "background": "rgba(59,130,246,0.18)",
                        "color": "#93c5fd",
                        "borderRadius": "10px",
                        "padding": "1px 7px",
                        "fontSize": "0.62rem",
                        "fontWeight": "600",
                        "letterSpacing": "0.02em",
                    },
                )
                for t in tools
            ]
            meta_items += [
                html.Span("verktøy: ", style={"color": "#94a3b8", "fontSize": "0.62rem"}),
                *tool_chips,
            ]

        if latency:
            if meta_items:
                meta_items.append(html.Span(" · ", style={"color": "#64748b"}))
            label = f"{provider} · " if provider else ""
            if tokens and tokens > 0:
                label += f"{latency} ms · {tokens:,} tok"
            else:
                label += f"{latency} ms"
            meta_items.append(html.Span(label, style={"color": "#64748b", "fontSize": "0.62rem", "fontFamily": "monospace"}))

    bubble_content = [dcc.Markdown(msg["content"], style={"margin": 0})]
    if meta_items:
        bubble_content.append(
            html.Div(meta_items, style={"display": "flex", "gap": "4px", "flexWrap": "wrap", "marginTop": "0.4rem", "alignItems": "center"})
        )

    return html.Div(
        html.Div(
            bubble_content,
            style={
                "background": "#2c3e50" if is_user else "#e9ecef",
                "color": "white" if is_user else "#212529",
                "padding": "0.6rem 0.9rem",
                "borderRadius": "16px 16px 4px 16px" if is_user else "16px 16px 16px 4px",
                "maxWidth": "85%",
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
                html.Div(
                    [
                        html.Span("📊", style={"fontSize": "1.4rem"}),
                        html.Div([
                            html.Div("Markedsinnsikt AI",
                                     style={"fontWeight": "700", "fontSize": "0.88rem",
                                            "color": "white", "letterSpacing": "-0.01em",
                                            "lineHeight": "1.2"}),
                            html.Div("KI-drevet markedsanalyse",
                                     style={"fontSize": "0.67rem", "color": "rgba(255,255,255,0.42)",
                                            "marginTop": "0.1rem"}),
                        ]),
                    ],
                    style={"display": "flex", "alignItems": "center", "gap": "0.65rem"},
                ),
            ],
            className="sidebar-brand",
        ),

        # Filters
        html.Div(html.Span("Filtre"), className="section-label"),

        html.Div([
            dbc.Label("Kunde", className="mb-1"),
            html.Span(id="dot-client", style={"marginLeft": "0.4rem", "fontSize": "0.5rem", "color": "#3b82f6"}),
        ], style={"display": "flex", "alignItems": "center"}),
        dcc.Dropdown(id="dd-client", options=[], value="All", clearable=False, className="mb-3"),

        html.Div([
            dbc.Label("Kampanje", className="mb-1"),
            html.Span(id="dot-campaign", style={"marginLeft": "0.4rem", "fontSize": "0.5rem", "color": "#3b82f6"}),
        ], style={"display": "flex", "alignItems": "center"}),
        dcc.Dropdown(id="dd-campaign", options=[], value="All", clearable=False, className="mb-3"),

        html.Div([
            dbc.Label("Kanal", className="mb-1"),
            html.Span(id="dot-channel", style={"marginLeft": "0.4rem", "fontSize": "0.5rem", "color": "#3b82f6"}),
        ], style={"display": "flex", "alignItems": "center"}),
        dcc.Dropdown(id="dd-channel", options=[], value="All", clearable=False, className="mb-3"),

        dcc.Interval(id="init-trigger", interval=300, max_intervals=1),

        html.Hr(),
        html.Div(id="row-count", className="text-center", style={"fontSize": "0.7rem"}),

        html.Hr(),
        html.Div(html.Span("Last ned"), className="section-label"),
        dbc.Button(
            [html.Span("⬇", style={"marginRight": "0.4rem"}), "CSV-eksport"],
            id="btn-download-csv",
            color="outline-secondary",
            size="sm",
            className="w-100",
        ),
        dcc.Download(id="download-csv"),
        dcc.Download(id="download-pdf"),
    ],
    width=2,
    className="dark-sidebar",
    style={
        "position": "sticky",
        "top": 0,
        "height": "100vh",
        "overflowY": "auto",
        "padding": "1.25rem",
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
                html.Div("📊  Markedsanalyse-plattform", className="page-header-badge"),
                html.H2("Markedsinnsikt AI"),
                html.P(
                    "Omgjør markedsdata til handlingsrettede anbefalinger "
                    "på tvers av kampanjer, kanaler og kunder — drevet av KI og ML."
                ),
            ],
            className="page-header fade-up",
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
            "Datakilde: Meta Robyn MMM-datasett (2015–2019) — reelle ukentlige forbruksmønstre med adstock-effekter.",
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
        # Cache AI results so PDF can read without re-calling the API
        dcc.Store(id="ai-results-store", data=None),
        # Store AI metadata (eval score, observability) for display
        dcc.Store(id="ai-meta-store", data=None),
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
        kpi_card("Totalt forbruk", fmt_nok(data['total_spend']),        trends.get("spend_wow"),       "kpi-card-amber",  "💰"),
        kpi_card("Total inntekt",  fmt_nok(data['total_revenue']),      trends.get("revenue_wow"),     "kpi-card-green",  "📈"),
        kpi_card("Konverteringer", f"{data['total_conversions']:,}",    trends.get("conversions_wow"), "kpi-card-blue",   "🎯"),
        kpi_card("Gj.snitt ROAS",  f"{data['avg_roas']:.2f}x",         trends.get("roas_wow"),        "kpi-card-purple", "⚡"),
        kpi_card("Gj.snitt CTR",   f"{data['avg_ctr']:.2f}%",          trends.get("ctr_wow"),         "kpi-card-teal",   "👆"),
    ]

    bell_style_base = {
        "position": "fixed", "top": "1rem", "right": "1rem", "zIndex": 1060,
    }
    bell_style = {**bell_style_base, "display": "block" if anomalies else "none"}

    return (
        html.Div([_filter_banner(client, campaign, channel), health_bar]),
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


def build_analyse_figs(client, campaign, channel):
    """Build the 4 Analyse-tab figures. Shared by the live callback and PDF export."""
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
    return build_analyse_figs(client, campaign, channel)


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


def _obs_badge(obs: dict | None) -> html.Span:
    """Small observability pill: model | latency | tokens."""
    if not obs:
        return html.Span()
    provider  = obs.get("provider", "")
    model     = obs.get("model", "")
    latency   = obs.get("latency_ms", 0)
    tokens    = obs.get("total_tokens", obs.get("prompt_tokens", 0) + obs.get("completion_tokens", 0))
    model_str = f"{provider} / {model}" if provider else model
    parts = [model_str]
    if latency:
        parts.append(f"{latency} ms")
    if tokens and tokens > 0:
        parts.append(f"{tokens:,} tok")
    return html.Span(
        " · ".join(parts),
        style={
            "fontSize": "0.68rem", "color": "rgba(255,255,255,0.55)",
            "background": "rgba(255,255,255,0.06)", "borderRadius": "20px",
            "padding": "2px 8px", "fontFamily": "monospace",
        },
    )


def _eval_badge(eval_result: dict | None) -> html.Span:
    """Groundedness score pill."""
    if not eval_result:
        return html.Span()
    score = eval_result.get("score", 0)
    label = eval_result.get("label", "")
    color = (
        "#22c55e" if score >= 80
        else "#f59e0b" if score >= 60
        else "#f97316" if score >= 40
        else "#ef4444"
    )
    return html.Span(
        [
            html.Span("●", style={"color": color, "marginRight": "4px"}),
            f"Grunnfesting: {label} ({score}/100)",
        ],
        title=(
            "Grunnfestingspoeng — sjekker om AI-innsikten er forankret i faktiske data.\n"
            + "\n".join(
                f"  {'✓' if v else '✗'} {k}"
                for k, v in eval_result.get("checks", {}).items()
            )
        ),
        style={
            "fontSize": "0.68rem", "color": "rgba(255,255,255,0.7)",
            "background": "rgba(255,255,255,0.06)", "borderRadius": "20px",
            "padding": "2px 8px", "cursor": "help",
        },
    )


def render_insights(
    data: dict,
    ml_recs: list | None = None,
    impacts: list | None = None,
    eval_result: dict | None = None,
    obs: dict | None = None,
) -> html.Div:
    unified = _render_unified_action_plan(data.get("recommendations", []), ml_recs or [], impacts or [])

    exec_decision = data.get("executive_decision", "")

    # Meta bar: eval badge + obs badge
    meta_bar = html.Div(
        [_eval_badge(eval_result), _obs_badge(obs)],
        style={
            "display": "flex", "gap": "0.5rem", "alignItems": "center",
            "marginBottom": "0.75rem", "flexWrap": "wrap",
        },
    ) if (eval_result or obs) else html.Div()

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
        meta_bar,
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
        greeting_msg = [{"role": "user", "content": (
            "Gi meg en kort velkomstmelding på 2 setninger basert på dataene du ser nå. "
            "Nevn det viktigste tallet eller trenden, og spør hva jeg ønsker å analysere."
        )}]
        greeting, obs = answer_question_with_tools(
            greeting_msg, get_df(), client, campaign, channel, model=DEFAULT_MODEL
        )
        return [{
            "role": "assistant", "content": greeting,
            "tools_used": obs.tools_used, "latency_ms": obs.latency_ms,
            "tokens": obs.total_tokens, "provider": obs.provider,
        }], None
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

    # Pass only role+content to the LLM (strip UI-only fields like tools_used, latency_ms)
    llm_messages = [{"role": m["role"], "content": m["content"]} for m in history]

    try:
        reply, obs = answer_question_with_tools(
            llm_messages, get_df(), client, campaign, channel, model=DEFAULT_MODEL
        )
        history.append({
            "role": "assistant",
            "content": reply,
            "tools_used": obs.tools_used,
            "latency_ms": obs.latency_ms,
            "tokens": obs.total_tokens,
            "provider": obs.provider,
        })
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


def build_ml_figs(xgb_results: list):
    """Build XGBoost forecast + SHAP figures. Returns (fig_xgb, fig_global, fig_local).
    fig_global and fig_local are None when xgb_results is empty."""
    FEAT_LABELS = {
        "lag_1": "Forrige uke (lag 1)", "lag_2": "To uker siden (lag 2)",
        "rolling_mean": "Rullende snitt (3 uker)", "trend": "Tidsindeks",
    }

    # XGBoost forecast with 90% CI error bars
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

    if not xgb_results:
        return fig_xgb, None, None

    # Global SHAP — mean |SHAP| across all channels
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

    return fig_xgb, fig_global, fig_local


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

    # ── 2 & 3. XGBoost forecast + SHAP charts ────────────────────────────
    fig_xgb, fig_global, fig_local = build_ml_figs(xgb_results)

    fi_section: list = []
    if xgb_results and fig_global is not None and fig_local is not None:
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


def _filter_banner(client, campaign, channel) -> html.Div:
    """Blue context banner shown at the top of ML and AI tabs when a filter is active."""
    parts = [(client, "Kunde"), (campaign, "Kampanje"), (channel, "Kanal")]
    active = [(val, lbl) for val, lbl in parts if val and val != "All"]
    if not active:
        return html.Div()
    chips = [
        html.Span([
            html.Span(lbl + ": ", style={"opacity": "0.65", "fontSize": "0.72rem", "fontWeight": 600}),
            html.Span(val, style={"fontWeight": 700}),
        ], style={
            "background": "rgba(255,255,255,0.12)", "borderRadius": "20px",
            "padding": "0.25rem 0.75rem", "fontSize": "0.82rem", "color": "white",
        })
        for val, lbl in active
    ]
    return html.Div(
        [html.Span("🔍  Analyserer:", style={"color": "rgba(255,255,255,0.6)", "fontSize": "0.78rem", "marginRight": "0.6rem"})] + chips,
        style={
            "background": "linear-gradient(135deg, #1e3a5f 0%, #1e40af 100%)",
            "borderRadius": "10px", "padding": "0.65rem 1rem",
            "display": "flex", "alignItems": "center", "gap": "0.5rem",
            "flexWrap": "wrap", "marginBottom": "1rem",
            "boxShadow": "0 2px 8px rgba(30,64,175,0.25)",
        }
    )


@app.callback(
    Output("ml-out", "children"),
    Output("ml-last-filters", "data"),
    Output("ml-results-store", "data"),
    Input("dd-client", "value"),
    Input("dd-campaign", "value"),
    Input("dd-channel", "value"),
    Input("main-tabs", "active_tab"),
    State("ml-last-filters", "data"),
    prevent_initial_call=True,
)
def run_ml_analysis(client, campaign, channel, active_tab, last_filters):
    if active_tab != "tab-ml":
        return dash.no_update, dash.no_update, dash.no_update
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
        cached = {
            "bt": bt, "impacts": impacts, "ml_recs": recs,
            "xgb_results": xgb_results,
            "z_anomalies": z_anomalies,
            "if_anomalies": if_anomalies,
        }
        content = html.Div([
            _filter_banner(client, campaign, channel),
            render_ml_results(xgb_results, bt, z_anomalies, if_anomalies, impacts),
        ])
        return content, current, cached
    except Exception as e:
        return dbc.Alert(f"ML-feil: {e}", color="danger"), current, dash.no_update



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
    Output("ai-results-store", "data"),
    Output("ai-meta-store", "data"),
    Input("dd-client", "value"),
    Input("dd-campaign", "value"),
    Input("dd-channel", "value"),
    Input("main-tabs", "active_tab"),
    State("insights-last-filters", "data"),
    State("ml-results-store", "data"),
    prevent_initial_call=True,
)
def update_live_insights(client, campaign, channel, active_tab, last_filters, ml_cache):
    if active_tab != "tab-innsikt":
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update
    current = {"client": client, "campaign": campaign, "channel": channel}
    if current == last_filters:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update
    try:
        # Use v3.0 API — returns insights + eval + observability
        data, eval_result, obs = generate_insights_with_meta(
            get_df(), client, campaign, channel, model=DEFAULT_MODEL
        )

        # Serialize obs (dataclass → dict) for storage
        obs_dict = {
            "provider": obs.provider, "model": obs.model,
            "prompt_tokens": obs.prompt_tokens, "completion_tokens": obs.completion_tokens,
            "total_tokens": obs.total_tokens, "latency_ms": obs.latency_ms,
            "prompt_version": obs.prompt_version, "tools_used": obs.tools_used,
        }
        meta = {"eval": eval_result, "obs": obs_dict}

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

        content = html.Div([
            _filter_banner(client, campaign, channel),
            render_insights(data, ml_recs, impacts, eval_result=eval_result, obs=obs_dict),
        ])
        return content, current, data, meta
    except Exception as e:
        return ai_error_alert(e), current, dash.no_update, dash.no_update


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
    State("ai-results-store",         "data"),
    prevent_initial_call=True,
)
def download_pdf_report(_an, _ml, _ai, client, campaign, channel, ml_cache, ai_cache):
    from datetime import datetime as _dt
    import plotly.io as _pio

    triggered = ctx.triggered_id

    # ── Only load what the triggered tab actually needs ───────────────────
    kpis   = get_kpis_data(client, campaign, channel)
    health = compute_portfolio_health(client, campaign, channel)

    # ML data: needed by ML tab and AI tab (budget summary)
    need_ml = triggered in ("btn-download-pdf-ml", "btn-download-pdf-ai")
    if need_ml:
        if ml_cache:
            bt           = ml_cache.get("bt", [])
            impacts      = ml_cache.get("impacts", [])
            ml_recs      = ml_cache.get("ml_recs", [])
            xgb_results  = ml_cache.get("xgb_results", [])
            z_anomalies  = ml_cache.get("z_anomalies", [])
            if_anomalies = ml_cache.get("if_anomalies", [])
        else:
            try:
                df           = apply_filters(client, campaign, channel)
                xgb_results  = predict_xgboost_with_intervals(df)
                bt           = backtest_models(df)
                avg_spend    = float(df.groupby("week")["spend"].sum().mean())
                impacts      = compute_business_impact(bt, avg_spend)
                ml_recs      = suggest_budget_reallocation(df)
                z_anomalies  = detect_anomalies_zscore(df)
                if_anomalies = detect_anomalies_isolation_forest(df)
            except Exception:
                bt = impacts = ml_recs = xgb_results = z_anomalies = if_anomalies = []
    else:
        bt = impacts = ml_recs = xgb_results = z_anomalies = if_anomalies = []

    # AI data: only needed by AI tab
    if triggered == "btn-download-pdf-ai":
        if ai_cache:
            ai_data = ai_cache
        else:
            try:
                context = build_context(get_df(), client, campaign, channel)
                ai_data = generate_insights(context, model=DEFAULT_MODEL)
            except Exception as e:
                ai_data = {"summary": f"AI ikke tilgjengelig: {e}", "insights": [],
                           "recommendations": [], "executive_decision": ""}
    else:
        ai_data = {}

    # ── HTML report helpers ───────────────────────────────────────────────
    generated   = _dt.now().strftime("%d.%m.%Y %H:%M")
    scope_parts = [p for p in [client, campaign, channel] if p and p != "All"]
    scope_str   = " | ".join(scope_parts) if scope_parts else "Alle kunder / kampanjer / kanaler"

    def _fig_html(fig) -> str:
        return _pio.to_html(fig, include_plotlyjs=False, full_html=False,
                            config={"displayModeBar": False})

    _BASE_CSS = """
    <style>
      body { font-family: 'Inter', system-ui, sans-serif; background:#f1f5f9; margin:0; padding:0; color:#0f172a; }
      .wrap { max-width:960px; margin:0 auto; padding:2rem 1.5rem; }
      .cover { background:linear-gradient(135deg,#0c1628 0%,#1a3a7a 100%);
               border-radius:14px; padding:2rem 2.5rem; color:white; margin-bottom:2rem; }
      .cover h1 { margin:0 0 .25rem; font-size:1.8rem; font-weight:800; }
      .cover p  { margin:0; color:rgba(255,255,255,.55); font-size:.9rem; }
      h2 { font-size:1rem; font-weight:700; text-transform:uppercase;
           letter-spacing:.08em; color:#1e3a8a; border-bottom:2px solid #e2e8f0;
           padding-bottom:.4rem; margin:1.75rem 0 .9rem; }
      table { width:100%; border-collapse:collapse; font-size:.85rem; margin-bottom:1.25rem; }
      th { background:#f1f5f9; text-align:left; padding:.45rem .7rem;
           font-weight:600; border-bottom:2px solid #e2e8f0; }
      td { padding:.4rem .7rem; border-bottom:1px solid #e8edf3; }
      tr:nth-child(even) td { background:#f8fafc; }
      .kpi-grid { display:grid; grid-template-columns:repeat(3,1fr); gap:1rem; margin-bottom:1.5rem; }
      .kpi { background:white; border-radius:10px; border-left:4px solid #3b82f6;
             box-shadow:0 1px 4px rgba(0,0,0,.06); padding:.85rem 1rem; }
      .kpi-label { font-size:.67rem; font-weight:700; text-transform:uppercase;
                   letter-spacing:.08em; color:#94a3b8; margin-bottom:.3rem; }
      .kpi-value { font-size:1.4rem; font-weight:800; color:#0f172a; }
      .chart-box { background:white; border-radius:12px; border:1px solid #e8edf3;
                   box-shadow:0 1px 4px rgba(0,0,0,.04); padding:.5rem .5rem 0;
                   margin-bottom:1.25rem; }
      .insight { font-size:.8rem; color:#64748b; padding:.35rem .75rem;
                 border-left:3px solid #3b82f6; background:#f8fafc;
                 border-radius:0 6px 6px 0; margin:.25rem .25rem 1rem; }
      .badge { display:inline-block; background:rgba(59,130,246,.15);
               border:1px solid rgba(59,130,246,.3); color:#1d4ed8;
               font-size:.68rem; font-weight:700; text-transform:uppercase;
               letter-spacing:.08em; padding:.2rem .65rem; border-radius:20px;
               margin-bottom:.65rem; }
      .exec { background:linear-gradient(135deg,#1e3a5f,#2c3e50); color:white;
              border-radius:10px; padding:1rem 1.25rem; margin-bottom:1.25rem;
              border-left:4px solid #3b82f6; }
      .exec-label { font-size:.65rem; font-weight:700; text-transform:uppercase;
                    letter-spacing:.1em; color:rgba(255,255,255,.6); margin-bottom:.4rem; }
      .footer { text-align:center; font-size:.75rem; color:#94a3b8; margin-top:2.5rem;
                padding-top:1rem; border-top:1px solid #e2e8f0; }
      @media print { body { background:white; } .wrap { padding:0; } }
    </style>"""

    def _h(tag, text, **kw):
        attrs = " ".join(f'{k}="{v}"' for k, v in kw.items())
        return f"<{tag} {attrs}>{text}</{tag}>" if attrs else f"<{tag}>{text}</{tag}>"

    def _fmt_nok(value: float) -> str:
        return f"NOK {value:,.0f}".replace(",", "\u202f")

    def _table(headers: list, rows: list) -> str:
        ths = "".join(f"<th>{h}</th>" for h in headers)
        trs = "".join(
            "<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>"
            for row in rows
        )
        return f"<table><thead><tr>{ths}</tr></thead><tbody>{trs}</tbody></table>"

    def _cover(subtitle: str) -> str:
        return (
            f'<div class="cover">'
            f'<div class="badge">Markedsinnsikt AI</div>'
            f'<h1>{subtitle}</h1>'
            f'<p>{generated} &nbsp;|&nbsp; {scope_str}</p>'
            f'</div>'
        )

    def _kpi_grid(items: list) -> str:
        cards = "".join(
            f'<div class="kpi"><div class="kpi-label">{lbl}</div>'
            f'<div class="kpi-value">{val}</div></div>'
            for lbl, val in items
        )
        return f'<div class="kpi-grid">{cards}</div>'

    def _wrap(body: str) -> str:
        return (
            "<!DOCTYPE html><html lang='no'><head>"
            "<meta charset='utf-8'>"
            "<script src='https://cdn.plot.ly/plotly-2.35.2.min.js'></script>"
            f"{_BASE_CSS}"
            "</head><body>"
            f'<div class="wrap">{body}'
            f'<div class="footer">Generert: {generated} &nbsp;|&nbsp; {scope_str} &nbsp;|&nbsp; Robyn MMM-data</div>'
            "</div></body></html>"
        )

    parts    = [p for p in [client, campaign, channel] if p and p != "All"]
    tag      = "_".join(parts) if parts else "alle"
    date_tag = _dt.now().strftime("%Y%m%d")

    # ══════════════════════════════════════════════════════════════════════
    # ANALYSE TAB — KPIs + chart insights + portfolio health
    # ══════════════════════════════════════════════════════════════════════
    # ══════════════════════════════════════════════════════════════════════
    # ANALYSE TAB
    # ══════════════════════════════════════════════════════════════════════
    if triggered == "btn-download-pdf-analyse":
        fig_roas, fig_conv, fig_spend, fig_weekly = build_analyse_figs(client, campaign, channel)
        hints = compute_chart_insights(client, campaign, channel)

        charts_html = "".join(
            f'<h2>{title}</h2><div class="chart-box">{_fig_html(fig)}</div>'
            f'<div class="insight">{hints.get(key, "")}</div>'
            for title, fig, key in [
                ("ROAS per kanal",           fig_roas,    "roas"),
                ("Konverteringer per kampanje", fig_conv,  "conv"),
                ("Forbruk per kanal",         fig_spend,  "spend"),
                ("Ukentlig forbrukstrend",    fig_weekly, "weekly"),
            ]
        )

        body = (
            _cover("Oversiktsrapport — Analyse")
            + _kpi_grid([
                ("Totalt forbruk",  _fmt_nok(kpis["total_spend"])),
                ("Total inntekt",   _fmt_nok(kpis["total_revenue"])),
                ("Konverteringer",  f"{kpis['total_conversions']:,}"),
                ("Gj.snitt ROAS",   f"{kpis['avg_roas']:.2f}x"),
                ("Gj.snitt CTR",    f"{kpis['avg_ctr']:.2f}%"),
                ("Portefolgehelse", f"{health['score']}/100 — {health['label']}"),
            ])
            + charts_html
        )
        filename = f"rapport_oversikt_{tag}_{date_tag}.html"

    # ══════════════════════════════════════════════════════════════════════
    # ML TAB
    # ══════════════════════════════════════════════════════════════════════
    elif triggered == "btn-download-pdf-ml":
        fig_xgb, fig_global, fig_local = build_ml_figs(xgb_results)

        xgb_rows = [
            [p["channel"], f"{p['predicted_roas']:.2f}x",
             f"[{p['lower_90']:.2f} – {p['upper_90']:.2f}]x",
             p.get("next_date", f"Uke {p['next_week']}")]
            for p in xgb_results
        ] or [["Ingen prediksjoner.", "", "", ""]]

        bt_rows = [
            [r["channel"], f"{r['xgb_mae']:.3f}", f"{r['xgb_rmse']:.3f}",
             f"{r.get('xgb_bias',0):+.3f}x",
             f"{r['direction_accuracy']:.0f}%" if r.get("direction_accuracy") is not None else "–",
             f"{r.get('improvement_pct',0):+.1f}%"]
            for r in bt
        ] or [["Ikke nok data.", "", "", "", "", ""]]

        impact_rows = [
            [imp["channel"],
             f"ca. {round(imp['decision_accuracy_proxy']/5)*5:.0f}%",
             _fmt_nok(round(imp["estimated_weekly_cost_of_error"]/500)*500),
             f"{imp['avg_actual_roas']:.2f}x"]
            for imp in impacts
        ] or [["Ingen data.", "", "", ""]]

        all_anomalies = (z_anomalies or [])[:5] + (if_anomalies or [])[:5]
        anom_rows = [
            [a.get("severity","").upper(),
             f"{a.get('client','')} / {a.get('campaign','')}",
             a.get("detail","")]
            for a in all_anomalies
        ] or [["Ingen avvik oppdaget.", "", ""]]

        ml_rec_rows = [
            [r["from_channel"], r["to_channel"],
             f"{r['from_roas']:.2f}x", f"{r['to_roas']:.2f}x", r.get("summary","")]
            for r in (ml_recs or [])
        ] or [["Ingen anbefalinger.", "", "", "", ""]]

        charts_html = (
            f'<div class="chart-box">{_fig_html(fig_xgb)}</div>'
            + (f'<div class="chart-box">{_fig_html(fig_global)}</div>'
               f'<div class="chart-box">{_fig_html(fig_local)}</div>'
               if fig_global else "")
        )

        body = (
            _cover("ML-analyserapport")
            + "<h2>XGBoost ROAS-prediksjon neste uke</h2>"
            + _table(["Kanal","Prediksjon","90% intervall","Dato"], xgb_rows)
            + charts_html
            + "<h2>Backtesting — walk-forward validering</h2>"
            + _table(["Kanal","XGB MAE","XGB RMSE","Bias","Trendretning","Forbedring"], bt_rows)
            + "<h2>Forretningsmessig konsekvens</h2>"
            + _table(["Kanal","Beslutningsnøyaktighet","Ukentlig feilkostnad","Gj.snitt ROAS"], impact_rows)
            + "<h2>Avviksdeteksjon</h2>"
            + _table(["Alvorlighet","Kunde / Kampanje","Detalj"], anom_rows)
            + "<h2>ML — Budsjettomfordeling</h2>"
            + _table(["Fra kanal","Til kanal","Fra ROAS","Til ROAS","Begrunnelse"], ml_rec_rows)
        )
        filename = f"rapport_ml_{tag}_{date_tag}.html"

    # ══════════════════════════════════════════════════════════════════════
    # AI TAB
    # ══════════════════════════════════════════════════════════════════════
    else:
        exec_decision = ai_data.get("executive_decision", "")
        exec_block = (
            f'<div class="exec"><div class="exec-label">Ukens beslutning</div>'
            f'<strong>{exec_decision}</strong></div>'
            if exec_decision else ""
        )

        summary_block = (
            f'<p style="background:#eff6ff;padding:.9rem 1rem;border-radius:8px;'
            f'font-size:.88rem;line-height:1.6">{ai_data.get("summary","")}</p>'
        )

        insight_rows = [
            [ins.get("title",""), ins.get("detail","")]
            for ins in ai_data.get("insights", [])
        ] or [["Ingen innsikter.", ""]]

        pri_label = {"high":"🔴 HØY","medium":"🟡 MIDDELS","low":"🟢 LAV"}
        recs = sorted(ai_data.get("recommendations",[]),
                      key=lambda x: {"high":0,"medium":1,"low":2}.get(x.get("priority","low"),2))
        rec_rows = [
            [pri_label.get(r.get("priority","low"),"-"), r.get("action",""), r.get("expected_impact","")]
            for r in recs
        ] or [["Ingen anbefalinger.", "", ""]]

        ml_rec_rows = [
            [r["from_channel"], r["to_channel"],
             f"{r['from_roas']:.2f}x", f"{r['to_roas']:.2f}x"]
            for r in (ml_recs or [])
        ] or [["Ingen anbefalinger.", "", "", ""]]

        body = (
            _cover("AI Innsiktsrapport")
            + exec_block
            + summary_block
            + _kpi_grid([
                ("Totalt forbruk",  _fmt_nok(kpis["total_spend"])),
                ("Total inntekt",   _fmt_nok(kpis["total_revenue"])),
                ("Gj.snitt ROAS",   f"{kpis['avg_roas']:.2f}x"),
                ("Gj.snitt CTR",    f"{kpis['avg_ctr']:.2f}%"),
                ("Portefolgehelse", f"{health['score']}/100 — {health['label']}"),
                ("Datarader",       f"{kpis['row_count']:,}"),
            ])
            + "<h2>AI-innsikt</h2>"
            + _table(["Tema","Detalj"], insight_rows)
            + "<h2>Anbefalinger (AI)</h2>"
            + _table(["Prioritet","Tiltak","Forventet effekt"], rec_rows)
            + "<h2>ML — Budsjettomfordeling</h2>"
            + _table(["Fra kanal","Til kanal","Fra ROAS","Til ROAS"], ml_rec_rows)
        )
        filename = f"rapport_ai_{tag}_{date_tag}.html"

    # ── Output ────────────────────────────────────────────────────────────
    return dcc.send_string(_wrap(body), filename)


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8050))
    app.run(host="0.0.0.0", port=port, debug=False)
