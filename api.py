"""Markedsinnsikt AI — FastAPI backend."""

from __future__ import annotations

from typing import Optional

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import pandas as pd

from data import generate_dataset
from ai_assistant import build_context, generate_insights, answer_question, detect_anomalies
app = FastAPI(title="Markedsinnsikt AI API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_df: Optional[pd.DataFrame] = None


def get_df() -> pd.DataFrame:
    global _df
    if _df is None:
        _df = generate_dataset()
    return _df


def apply_filters(
    df: pd.DataFrame,
    client: str = "All",
    campaign: str = "All",
    channel: str = "All",
) -> pd.DataFrame:
    if client and client != "All":
        df = df[df["client"] == client]
    if campaign and campaign != "All":
        df = df[df["campaign"] == campaign]
    if channel and channel != "All":
        df = df[df["channel"] == channel]
    return df


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/")
def root():
    return {"status": "ok", "service": "Markedsinnsikt AI API", "docs": "/docs"}


# ---------------------------------------------------------------------------
# Filter options
# ---------------------------------------------------------------------------

@app.get("/filters")
def get_filters(client: str = "All"):
    df = get_df()
    filtered = df if client == "All" else df[df["client"] == client]
    return {
        "clients": ["All"] + sorted(df["client"].unique().tolist()),
        "campaigns": ["All"] + sorted(filtered["campaign"].unique().tolist()),
        "channels": ["All"] + sorted(df["channel"].unique().tolist()),
    }


# ---------------------------------------------------------------------------
# KPIs
# ---------------------------------------------------------------------------

@app.get("/kpis")
def get_kpis(client: str = "All", campaign: str = "All", channel: str = "All"):
    df = apply_filters(get_df(), client, campaign, channel)
    total_spend = df["spend"].sum()
    total_revenue = df["revenue"].sum()
    total_conversions = int(df["conversions"].sum())
    total_clicks = df["clicks"].sum()
    total_impressions = df["impressions"].sum()
    avg_roas = total_revenue / total_spend if total_spend > 0 else 0
    avg_ctr = total_clicks / total_impressions * 100 if total_impressions > 0 else 0
    return {
        "total_spend": round(total_spend, 2),
        "total_revenue": round(total_revenue, 2),
        "total_conversions": total_conversions,
        "avg_roas": round(avg_roas, 2),
        "avg_ctr": round(avg_ctr, 2),
        "row_count": len(df),
    }


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

@app.get("/charts/roas-by-channel")
def roas_by_channel(client: str = "All", campaign: str = "All", channel: str = "All"):
    df = apply_filters(get_df(), client, campaign, channel)
    grouped = (
        df.groupby("channel")
        .apply(lambda x: round(x["revenue"].sum() / x["spend"].sum(), 2) if x["spend"].sum() > 0 else 0)
        .reset_index(name="roas")
    )
    return grouped.to_dict(orient="records")


@app.get("/charts/conversions-by-campaign")
def conversions_by_campaign(client: str = "All", campaign: str = "All", channel: str = "All"):
    df = apply_filters(get_df(), client, campaign, channel)
    grouped = (
        df.groupby("campaign")["conversions"]
        .sum()
        .reset_index()
        .sort_values("conversions")
    )
    return grouped.to_dict(orient="records")


@app.get("/charts/spend-by-channel")
def spend_by_channel(client: str = "All", campaign: str = "All", channel: str = "All"):
    df = apply_filters(get_df(), client, campaign, channel)
    grouped = df.groupby("channel")["spend"].sum().reset_index()
    return grouped.to_dict(orient="records")


@app.get("/charts/weekly-spend")
def weekly_spend_chart(client: str = "All", campaign: str = "All", channel: str = "All"):
    df = apply_filters(get_df(), client, campaign, channel)
    grouped = df.groupby("week")["spend"].sum().reset_index()
    return grouped.to_dict(orient="records")


# ---------------------------------------------------------------------------
# Analytics summary (trends + anomalies — no LLM)
# ---------------------------------------------------------------------------

@app.get("/analytics/summary")
def analytics_summary(client: str = "All", campaign: str = "All", channel: str = "All"):
    df = apply_filters(get_df(), client, campaign, channel)

    weekly = (
        df.groupby("week")
        .agg(
            spend=("spend", "sum"),
            revenue=("revenue", "sum"),
            conversions=("conversions", "sum"),
            clicks=("clicks", "sum"),
            impressions=("impressions", "sum"),
        )
        .sort_index()
    )
    weekly["roas"] = weekly["revenue"] / weekly["spend"].replace(0, float("nan"))
    weekly["ctr"]  = weekly["clicks"]  / weekly["impressions"].replace(0, float("nan")) * 100

    def wow(series: pd.Series):
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


# ---------------------------------------------------------------------------
# AI endpoints
# ---------------------------------------------------------------------------

class InsightRequest(BaseModel):
    client: str = "All"
    campaign: str = "All"
    channel: str = "All"
    model: str = "llama-3.3-70b-versatile"


class QuestionRequest(BaseModel):
    messages: list[dict]   # full conversation history [{"role": ..., "content": ...}]
    client: str = "All"
    campaign: str = "All"
    channel: str = "All"
    model: str = "llama-3.3-70b-versatile"


@app.post("/ai/insights")
def ai_insights(req: InsightRequest):
    context = build_context(get_df(), req.client, req.campaign, req.channel)
    return generate_insights(context, model=req.model)  # returns structured dict


@app.post("/ai/ask")
def ai_ask(req: QuestionRequest):
    context = build_context(get_df(), req.client, req.campaign, req.channel)
    return {"answer": answer_question(req.messages, context, model=req.model)}
