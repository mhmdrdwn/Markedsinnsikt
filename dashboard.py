"""Markedsinnsikt AI — Dash dashboard (direct function calls, no HTTP layer)."""

import os
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
from ml_models import predict_next_week, detect_anomalies_zscore, suggest_budget_reallocation

DEFAULT_MODEL = "llama-3.3-70b-versatile"


def groq_error_alert(e: Exception) -> dbc.Alert:
    """Return a user-friendly Norwegian alert for Groq API errors."""
    msg = str(e)
    if "429" in msg or "rate_limit" in msg.lower():
        # Try to extract the suggested wait time from the error message
        import re
        wait = re.search(r"Please try again in ([\dmhs.]+)", msg)
        wait_str = f" Prøv igjen om ca. {wait.group(1)}." if wait else ""
        return dbc.Alert(
            [
                html.Strong("Groq daglig grense nådd. "),
                f"Gratis-nivået har et daglig tokengrense.{wait_str}",
                html.Br(),
                html.Small(
                    "Tips: reduser filteret til én kunde/kanal for å bruke færre tokens.",
                    className="text-muted",
                ),
            ],
            color="warning",
            className="mt-2",
        )
    return dbc.Alert(f"Feil: {e}", color="danger", className="mt-2")

CHANNEL_COLORS = {
    "Google Ads": "#4285F4",
    "Meta Ads": "#1877F2",
    "TikTok Ads": "#69C9D0",
}

# ---------------------------------------------------------------------------
# Data layer (direct calls — no HTTP)
# ---------------------------------------------------------------------------

_df: pd.DataFrame | None = None

def get_df() -> pd.DataFrame:
    global _df
    if _df is None:
        _df = generate_dataset()
    return _df


def apply_filters(client="All", campaign="All", channel="All") -> pd.DataFrame:
    df = get_df().copy()
    if client and client != "All":
        df = df[df["client"] == client]
    if campaign and campaign != "All":
        df = df[df["campaign"] == campaign]
    if channel and channel != "All":
        df = df[df["channel"] == channel]
    return df


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


def get_chart_roas(client, campaign, channel) -> list:
    df = apply_filters(client, campaign, channel)
    grouped = (
        df.groupby("channel")
        .apply(lambda x: round(x["revenue"].sum() / x["spend"].sum(), 2) if x["spend"].sum() > 0 else 0)
        .reset_index(name="roas")
    )
    return grouped.to_dict(orient="records")


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
    return df.groupby("week")["spend"].sum().reset_index().to_dict(orient="records")


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

        dbc.Label("Kunde", className="fw-semibold text-muted small mb-1"),
        dcc.Dropdown(id="dd-client", options=[], value="All", clearable=False, className="mb-3"),

        dbc.Label("Kampanje", className="fw-semibold text-muted small mb-1"),
        dcc.Dropdown(id="dd-campaign", options=[], value="All", clearable=False, className="mb-3"),

        dbc.Label("Kanal", className="fw-semibold text-muted small mb-1"),
        dcc.Dropdown(id="dd-channel", options=[], value="All", clearable=False, className="mb-3"),

        dcc.Interval(id="init-trigger", interval=300, max_intervals=1),

        html.Hr(style={"borderColor": "#e2e8f0", "marginTop": "0.5rem"}),
        html.Div(id="row-count", className="text-muted small text-center"),
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
                html.Div(
                    [
                        html.Span("● Live",
                                  style={"background": "rgba(16,185,129,0.18)",
                                         "color": "#6ee7b7",
                                         "padding": "0.2rem 0.65rem",
                                         "borderRadius": "20px",
                                         "fontSize": "0.72rem",
                                         "fontWeight": "600",
                                         "marginRight": "0.5rem"}),
                        html.Span("Groq · llama-3.3-70b",
                                  style={"background": "rgba(255,255,255,0.08)",
                                         "color": "rgba(255,255,255,0.5)",
                                         "padding": "0.2rem 0.65rem",
                                         "borderRadius": "20px",
                                         "fontSize": "0.72rem"}),
                    ],
                    style={"marginTop": "0.8rem"},
                ),
            ],
            className="page-header",
        ),

        # ── Oversikt ──────────────────────────────────────────
        section_header("📈", "Oversikt"),
        dbc.Row(id="kpi-row", className="mb-4 g-3"),

        # ── Ytelse ────────────────────────────────────────────
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

        # ── KI-analyser ───────────────────────────────────────
        section_header("🤖", "KI-analyser"),
        html.P(
            "Automatisk analyse av kampanjeytelse, avvik og prioriterte anbefalinger — drevet av Groq.",
            className="text-muted small mb-3",
        ),
        dbc.Button("✦ Generer innsikt", id="btn-insights", color="primary", className="mb-3"),
        dcc.Loading(html.Div(id="insights-out"), type="circle"),

        html.Br(),

        # ── Prediksjoner & ML ─────────────────────────────────
        section_header("🔮", "Prediksjoner & ML"),
        html.P(
            "Lineær regresjon for neste ukes forbruk og ROAS · "
            "Statistisk avviksdeteksjon (z-score) · "
            "Budsjettoptimalisering basert på kanalytelse.",
            className="text-muted small mb-3",
        ),
        dbc.Button("✦ Kjør ML-analyse", id="btn-ml", color="success", className="mb-3"),
        dcc.Loading(html.Div(id="ml-out"), type="circle"),

        html.Hr(style={"marginTop": "3rem", "borderColor": "#e2e8f0"}),
        html.P(
            "Markedsinnsikt AI — Bygget med Dash & Groq · Dataene er syntetiske og kun for demoformål.",
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

    cards = [
        kpi_card("Totalt forbruk", f"NOK {data['total_spend']:,.0f}",  trends.get("spend_wow"),       "kpi-card-amber"),
        kpi_card("Total inntekt",  f"NOK {data['total_revenue']:,.0f}", trends.get("revenue_wow"),     "kpi-card-green"),
        kpi_card("Konverteringer", f"{data['total_conversions']:,}",    trends.get("conversions_wow"), "kpi-card-blue"),
        kpi_card("Gj.snitt ROAS",  f"{data['avg_roas']:.2f}x",         trends.get("roas_wow"),        "kpi-card-purple"),
        kpi_card("Gj.snitt CTR",   f"{data['avg_ctr']:.2f}%",          trends.get("ctr_wow"),         "kpi-card-teal"),
    ]

    bell_style_base = {
        "position": "fixed", "top": "1rem", "right": "1rem", "zIndex": 1060,
    }
    bell_style = {**bell_style_base, "display": "block" if anomalies else "none"}

    return (
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

    fig_roas = px.bar(
        roas_data,
        x="channel", y="roas",
        color="channel",
        color_discrete_map=CHANNEL_COLORS,
        text_auto=".2f",
        title="ROAS per kanal",
        labels={"roas": "ROAS (x)", "channel": ""},
    )
    fig_roas.update_traces(textposition="outside", marker_line_width=0)
    fig_roas.update_layout(**CHART_LAYOUT, showlegend=False)

    fig_conv = px.bar(
        conv_data,
        x="conversions", y="campaign",
        orientation="h",
        text_auto=True,
        title="Konverteringer per kampanje",
        labels={"conversions": "Konverteringer", "campaign": ""},
        color="conversions",
        color_continuous_scale="Blues",
    )
    fig_conv.update_layout(**CHART_LAYOUT, coloraxis_showscale=False)

    fig_spend = px.pie(
        spend_data,
        names="channel", values="spend",
        color="channel",
        color_discrete_map=CHANNEL_COLORS,
        title="Forbruk per kanal",
        hole=0.38,
    )
    fig_spend.update_traces(textposition="inside", textinfo="percent+label",
                            marker=dict(line=dict(color="white", width=2)))
    fig_spend.update_layout(**{k: v for k, v in CHART_LAYOUT.items()
                               if k not in ("xaxis", "yaxis")})

    fig_weekly = px.line(
        weekly_data,
        x="week", y="spend",
        markers=True,
        title="Ukentlig forbrukstrend",
        labels={"spend": "Forbruk (NOK)", "week": "Uke"},
        color_discrete_sequence=["#3b82f6"],
    )
    fig_weekly.update_traces(line=dict(width=2.5), marker=dict(size=7))
    fig_weekly.update_layout(**CHART_LAYOUT)

    return fig_roas, fig_conv, fig_spend, fig_weekly


def render_insights(data: dict) -> html.Div:
    priority_color = {"high": "danger", "medium": "warning", "low": "success"}

    anomaly_cards = []
    for a in data.get("anomalies", []):
        color = "danger" if a.get("severity") == "high" else "warning"
        anomaly_cards.append(
            dbc.Alert(
                [html.Strong(f"[{a.get('severity','').upper()}] {a.get('campaign','')}: "), a.get("issue", "")],
                color=color,
                className="py-2 mb-2",
            )
        )

    rec_cards = [
        dbc.Col(
            dbc.Card([
                dbc.CardHeader(
                    html.Span(
                        r.get("priority", "").upper(),
                        className=f"badge bg-{priority_color.get(r.get('priority',''), 'secondary')}",
                    )
                ),
                dbc.CardBody([
                    html.P(html.Strong(r.get("action", "")), className="mb-1"),
                    html.Small(f"Mål: {r.get('target', '')}", className="text-muted d-block"),
                    html.Small(f"Forventet effekt: {r.get('expected_impact', '')}", className="text-muted"),
                ]),
            ], className="h-100 shadow-sm"),
            md=4, className="mb-3",
        )
        for r in data.get("recommendations", [])
    ]

    return html.Div([
        dbc.Alert(data.get("summary", ""), color="info", className="mb-3"),

        html.H6("Nøkkelinnsikt", className="fw-bold"),
        html.Ul([
            html.Li([html.Strong(i.get("title", "") + ": "), i.get("detail", "")])
            for i in data.get("insights", [])
        ], className="mb-3"),

        *(
            [html.H6("Avvik oppdaget", className="fw-bold text-danger"), *anomaly_cards]
            if anomaly_cards else []
        ),

        html.H6("Anbefalinger", className="fw-bold mt-2"),
        dbc.Row(rec_cards),
    ])


@app.callback(
    Output("insights-out", "children"),
    Input("btn-insights", "n_clicks"),
    State("dd-client", "value"),
    State("dd-campaign", "value"),
    State("dd-channel", "value"),
    prevent_initial_call=True,
)
def generate_insights_cb(_, client, campaign, channel):
    try:
        context = build_context(get_df(), client, campaign, channel)
        data = generate_insights(context, model=DEFAULT_MODEL)
        return render_insights(data)
    except Exception as e:
        return groq_error_alert(e)


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
            import re
            wait = re.search(r"Please try again in ([\dmhs.]+)", msg)
            wait_str = f" Prøv igjen om ca. {wait.group(1)}." if wait else ""
            history.append({"role": "assistant", "content":
                f"Groq daglig tokengrense er nådd.{wait_str} Prøv igjen senere, "
                "eller filtrer til én kunde for å bruke færre tokens."})
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
            "Ingen meldinger ennå. Still et spørsmål!",
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


def render_ml_results(predictions: list, anomalies: list, recommendations: list) -> html.Div:
    severity_color = {"high": "danger", "medium": "warning"}

    # --- Prediction cards ---
    pred_cards = []
    for p in predictions:
        mae_roas_str = f"±{p['mae_roas']:.2f}x" if p["mae_roas"] is not None else "–"
        pred_cards.append(
            dbc.Col(
                dbc.Card([
                    dbc.CardHeader(html.Strong(p["channel"])),
                    dbc.CardBody([
                        html.P([
                            html.Span("Forbruk uke ", className="text-muted small"),
                            html.Span(str(p["next_week"]), className="fw-bold"),
                            html.Span(f":  NOK {p['predicted_spend']:,.0f}", className="fw-bold"),
                        ], className="mb-1"),
                        html.P([
                            html.Span("ROAS: ", className="text-muted small"),
                            html.Span(f"{p['predicted_roas']:.2f}x", className="fw-bold text-primary"),
                        ], className="mb-1"),
                        html.Small(
                            f"MAE forbruk ±NOK {p['mae_spend']:,.0f}  |  MAE ROAS {mae_roas_str}",
                            className="text-muted",
                        ),
                    ]),
                ], className="shadow-sm h-100"),
                md=4, className="mb-3",
            )
        )

    # --- Prediction chart (actual ROAS + predicted point) ---
    fig_pred = go.Figure()
    for p in predictions:
        ch = p["channel"]
        color = CHANNEL_COLORS.get(ch, "#888")
        history = p["history"]
        weeks_h = [h["week"] for h in history]
        roas_h  = [h["actual_roas"] for h in history]

        fig_pred.add_trace(go.Scatter(
            x=weeks_h, y=roas_h,
            name=f"{ch} (faktisk)",
            mode="lines+markers",
            line=dict(color=color, width=2),
            marker=dict(size=6),
        ))
        fig_pred.add_trace(go.Scatter(
            x=[p["next_week"]], y=[p["predicted_roas"]],
            name=f"{ch} (prediksjon uke {p['next_week']})",
            mode="markers",
            marker=dict(symbol="star", size=16, color=color,
                        line=dict(width=1, color="white")),
            showlegend=True,
        ))

    fig_pred.update_layout(**CHART_LAYOUT)
    fig_pred.update_layout(
        title="ROAS: faktisk historikk + prediksjon neste uke (★)",
        xaxis_title="Uke",
        yaxis_title="ROAS (x)",
        legend=dict(orientation="h", y=-0.28, font=dict(size=10)),
        margin=dict(t=50, b=90, l=8, r=8),
    )

    # --- Z-score anomalies ---
    if anomalies:
        anomaly_items = [
            dbc.Alert(
                [
                    html.Span(
                        f"z={a['z_score']:.2f}  ",
                        className=f"badge bg-{severity_color.get(a['severity'], 'secondary')} me-2",
                    ),
                    html.Strong(f"[{a['client']}] {a['campaign']}: "),
                    a["detail"],
                ],
                color=severity_color.get(a["severity"], "warning"),
                className="py-2 mb-2",
            )
            for a in anomalies[:8]          # cap at 8 for readability
        ]
        anomaly_section = [
            html.H6("🔍 Statistisk avviksdeteksjon (z-score ≥ 2.0)", className="fw-bold mt-2 mb-2"),
            html.P(
                "Uker der ROAS avviker ≥2 standardavvik fra kampanjens egen historikk.",
                className="text-muted small mb-2",
            ),
            *anomaly_items,
        ]
    else:
        anomaly_section = [
            html.P("Ingen statistiske avvik oppdaget (z-score < 2.0 for alle kampanjer).",
                   className="text-muted small"),
        ]

    # --- Budget reallocation ---
    if recommendations:
        rec_items = [
            dbc.Alert(
                [
                    html.Strong(f"{r['from_channel']} → {r['to_channel']}:  "),
                    r["summary"],
                    html.Br(),
                    html.Small(
                        f"ROAS sammenligning: {r['from_roas']:.2f}x vs {r['to_roas']:.2f}x",
                        className="text-muted",
                    ),
                ],
                color="success",
                className="py-2 mb-2",
            )
            for r in recommendations
        ]
        budget_section = [
            html.H6("💡 Budsjettoptimalisering", className="fw-bold mt-2 mb-2"),
            html.P(
                "Estimert gevinst ved å flytte 20% av budsjett fra lavere- til høyere-ytende kanal.",
                className="text-muted small mb-2",
            ),
            *rec_items,
        ]
    else:
        budget_section = [
            html.P("Ikke nok kanaldata for budsjettanbefalinger.", className="text-muted small"),
        ]

    return html.Div([
        # Prediction cards
        html.H6("📈 Forbruk & ROAS-prediksjon neste uke", className="fw-bold mb-2"),
        dbc.Row(pred_cards),

        # Prediction chart
        dcc.Graph(figure=fig_pred, className="mb-3"),

        html.Hr(),
        *anomaly_section,

        html.Hr(),
        *budget_section,
    ])


@app.callback(
    Output("ml-out", "children"),
    Input("btn-ml", "n_clicks"),
    State("dd-client", "value"),
    State("dd-campaign", "value"),
    State("dd-channel", "value"),
    prevent_initial_call=True,
)
def run_ml_analysis(_, client, campaign, channel):
    try:
        df = apply_filters(client, campaign, channel)
        predictions    = predict_next_week(df)
        anomalies      = detect_anomalies_zscore(df)
        recommendations = suggest_budget_reallocation(df)
        return render_ml_results(predictions, anomalies, recommendations)
    except Exception as e:
        return dbc.Alert(f"ML-feil: {e}", color="danger")


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


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8050))
    app.run(host="0.0.0.0", port=port, debug=False)
