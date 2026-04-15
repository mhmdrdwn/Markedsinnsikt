"""Markedsinnsikt AI — Dash dashboard (direct function calls, no HTTP layer)."""

import os
import pandas as pd
from dotenv import load_dotenv
load_dotenv()

import dash
from dash import dcc, html, Input, Output, State, ctx
import dash_bootstrap_components as dbc
import plotly.express as px

from data import generate_dataset
from ai_assistant import (
    build_context, generate_insights, answer_question, detect_anomalies
)

DEFAULT_MODEL = "llama-3.3-70b-versatile"

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


# ---------------------------------------------------------------------------
# Initial data for dropdowns
# ---------------------------------------------------------------------------

# Filters are loaded via callback on page load (not at startup)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY],
    title="Markedsinnsikt AI",
)
server = app.server


# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------

def kpi_card(label: str, value: str, trend: float | None = None) -> dbc.Col:
    if trend is not None:
        arrow  = "▲" if trend >= 0 else "▼"
        color  = "text-success" if trend >= 0 else "text-danger"
        trend_el = html.Small(f"{arrow} {abs(trend):.1f}% WoW", className=f"{color} fw-semibold")
    else:
        trend_el = html.Span()

    return dbc.Col(
        dbc.Card(
            dbc.CardBody([
                html.P(label, className="text-muted mb-1 small fw-semibold"),
                html.H4(value, className="mb-0 fw-bold"),
                trend_el,
            ]),
            className="shadow-sm h-100",
        ),
        xs=6, md=True,
    )


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
        html.H5("📊 Markedsinnsikt AI", className="fw-bold mb-0"),
        html.Small("KI-drevet markedsanalyse", className="text-muted d-block mb-3"),
        html.Hr(),

        dbc.Label("Kunde", className="fw-semibold"),
        dcc.Dropdown(id="dd-client", options=[], value="All", clearable=False, className="mb-3"),

        dbc.Label("Kampanje", className="fw-semibold"),
        dcc.Dropdown(id="dd-campaign", options=[], value="All", clearable=False, className="mb-3"),

        dbc.Label("Kanal", className="fw-semibold"),
        dcc.Dropdown(id="dd-channel", options=[], value="All", clearable=False, className="mb-3"),

        dcc.Interval(id="init-trigger", interval=300, max_intervals=1),

        html.Hr(),
        html.Small(id="row-count", className="text-muted"),
    ],
    width=2,
    style={
        "position": "sticky",
        "top": 0,
        "height": "100vh",
        "overflowY": "auto",
        "background": "#f8f9fa",
        "padding": "1.5rem",
        "borderRight": "1px solid #dee2e6",
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

main = dbc.Col(
    [
        html.H2("📊 Markedsinnsikt AI", className="mt-3 mb-0"),
        html.P(
            "KI-drevet markedsanalyseverktøy for kampanjer på tvers av kunder og kanaler.",
            className="text-muted",
        ),
        html.Hr(),

        # KPI row
        dbc.Row(id="kpi-row", className="mb-4 g-3"),
        html.Hr(),

        # Charts — row 1
        dbc.Row([
            dbc.Col(dcc.Graph(id="chart-roas"), md=6),
            dbc.Col(dcc.Graph(id="chart-conv"), md=6),
        ], className="mb-4"),

        # Charts — row 2
        dbc.Row([
            dbc.Col(dcc.Graph(id="chart-spend-pie"), md=6),
            dbc.Col(dcc.Graph(id="chart-weekly"), md=6),
        ], className="mb-4"),

        html.Hr(),

        # AI Insights
        html.H4("🤖 KI-drevne ytelsesanalyser"),
        dbc.Button("Generer innsikt", id="btn-insights", color="primary", className="mb-3"),
        dcc.Loading(html.Div(id="insights-out"), type="circle"),

        html.Hr(),
        html.Small(
            "Markedsinnsikt AI — Bygget med Dash & Groq · Dataene er syntetiske og kun for demoformål.",
            className="text-muted d-block mb-5",
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
        kpi_card("Totalt forbruk", f"NOK {data['total_spend']:,.0f}",  trends.get("spend_wow")),
        kpi_card("Total inntekt",  f"NOK {data['total_revenue']:,.0f}", trends.get("revenue_wow")),
        kpi_card("Konverteringer", f"{data['total_conversions']:,}",    trends.get("conversions_wow")),
        kpi_card("Gj.snitt ROAS",  f"{data['avg_roas']:.2f}x",         trends.get("roas_wow")),
        kpi_card("Gj.snitt CTR",   f"{data['avg_ctr']:.2f}%",          trends.get("ctr_wow")),
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
    fig_roas.update_traces(textposition="outside")
    fig_roas.update_layout(showlegend=False)

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
    fig_conv.update_layout(coloraxis_showscale=False)

    fig_spend = px.pie(
        spend_data,
        names="channel", values="spend",
        color="channel",
        color_discrete_map=CHANNEL_COLORS,
        title="Forbruk per kanal",
    )
    fig_spend.update_traces(textposition="inside", textinfo="percent+label")

    fig_weekly = px.line(
        weekly_data,
        x="week", y="spend",
        markers=True,
        title="Ukentlig forbrukstrend",
        labels={"spend": "Forbruk (NOK)", "week": "Uke"},
    )

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
        return dbc.Alert(f"Feil: {e}", color="danger")


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


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8050))
    app.run(host="0.0.0.0", port=port, debug=False)
