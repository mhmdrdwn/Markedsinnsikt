"""Tool definitions and executor for LLM function calling.

The LLM can call these tools during a chat turn to retrieve specific
data slices instead of relying on a pre-built context string.
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd


# ---------------------------------------------------------------------------
# Tool schemas (OpenAI / Groq compatible)
# ---------------------------------------------------------------------------

TOOL_DEFINITIONS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "get_channel_performance",
            "description": (
                "Get spend, revenue, ROAS, and conversions aggregated by channel. "
                "Use channel='all' to see all channels at once."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "channel": {
                        "type": "string",
                        "description": (
                            "Channel name e.g. 'Google Ads', 'Meta Ads', 'TikTok Ads', "
                            "or 'all' for the full breakdown"
                        ),
                    }
                },
                "required": ["channel"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_top_channel",
            "description": "Return the single best-performing channel by ROAS.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_channels",
            "description": "Compare two channels side-by-side on ROAS, spend, revenue, and conversions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "channel_a": {"type": "string", "description": "First channel name"},
                    "channel_b": {"type": "string", "description": "Second channel name"},
                },
                "required": ["channel_a", "channel_b"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weekly_trend",
            "description": "Return week-by-week values for a given metric and channel.",
            "parameters": {
                "type": "object",
                "properties": {
                    "channel": {"type": "string", "description": "Channel name"},
                    "metric": {
                        "type": "string",
                        "enum": ["roas", "spend", "revenue", "conversions"],
                        "description": "The metric to retrieve",
                    },
                },
                "required": ["channel", "metric"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_anomalies",
            "description": (
                "Return detected anomalies: significant ROAS drops or spend spikes "
                "across campaigns."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]


# ---------------------------------------------------------------------------
# Tool executor
# ---------------------------------------------------------------------------

class ToolExecutor:
    """Executes tool calls and records which tools were invoked."""

    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df
        self._calls: list[str] = []

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def execute(self, name: str, args: dict[str, Any]) -> str:
        """Dispatch a tool call by name and return a JSON string result."""
        self._calls.append(name)
        handler = getattr(self, f"_tool_{name}", None)
        if handler is None:
            return json.dumps({"error": f"Unknown tool: {name}"})
        try:
            return handler(**args)
        except Exception as exc:
            return json.dumps({"error": str(exc)})

    @property
    def tools_used(self) -> list[str]:
        """Unique tool names in call order."""
        seen: dict[str, None] = {}
        for t in self._calls:
            seen[t] = None
        return list(seen)

    # ------------------------------------------------------------------
    # Tool implementations
    # ------------------------------------------------------------------

    def _tool_get_channel_performance(self, channel: str) -> str:
        df = self._df
        if channel.lower() != "all":
            df = df[df["channel"].str.lower() == channel.lower()]
        if df.empty:
            return json.dumps({"error": f"No data for channel: {channel}"})
        result = (
            df.groupby("channel")
            .agg(
                spend=("spend", "sum"),
                revenue=("revenue", "sum"),
                conversions=("conversions", "sum"),
            )
            .reset_index()
        )
        result["roas"] = (result["revenue"] / result["spend"].replace(0, float("nan"))).round(2)
        result["spend"] = result["spend"].round(0)
        result["revenue"] = result["revenue"].round(0)
        return result.to_json(orient="records", force_ascii=False)

    def _tool_get_top_channel(self) -> str:
        ch = (
            self._df.groupby("channel")
            .agg(spend=("spend", "sum"), revenue=("revenue", "sum"))
            .reset_index()
        )
        ch["roas"] = ch["revenue"] / ch["spend"].replace(0, float("nan"))
        ch = ch.dropna(subset=["roas"])
        if ch.empty:
            return json.dumps({"error": "No channel data available"})
        best = ch.loc[ch["roas"].idxmax()]
        return json.dumps({
            "channel": str(best["channel"]),
            "roas": round(float(best["roas"]), 2),
            "spend": round(float(best["spend"]), 0),
            "revenue": round(float(best["revenue"]), 0),
        })

    def _tool_compare_channels(self, channel_a: str, channel_b: str) -> str:
        results = []
        for ch in [channel_a, channel_b]:
            sub = self._df[self._df["channel"].str.lower() == ch.lower()]
            if sub.empty:
                results.append({"channel": ch, "error": "not found"})
                continue
            spend = float(sub["spend"].sum())
            revenue = float(sub["revenue"].sum())
            results.append({
                "channel": ch,
                "spend": round(spend, 0),
                "revenue": round(revenue, 0),
                "roas": round(revenue / spend, 2) if spend > 0 else 0,
                "conversions": int(sub["conversions"].sum()),
            })
        return json.dumps(results, ensure_ascii=False)

    def _tool_get_weekly_trend(self, channel: str, metric: str) -> str:
        sub = self._df[self._df["channel"].str.lower() == channel.lower()]
        if sub.empty:
            return json.dumps({"error": f"No data for channel: {channel}"})
        weekly = (
            sub.groupby("week")
            .agg(
                spend=("spend", "sum"),
                revenue=("revenue", "sum"),
                conversions=("conversions", "sum"),
            )
            .sort_index()
        )
        weekly["roas"] = weekly["revenue"] / weekly["spend"].replace(0, float("nan"))
        if metric not in weekly.columns:
            return json.dumps({"error": f"Unknown metric: {metric}"})
        vals = weekly[metric].tolist()
        weeks = [int(w) for w in weekly.index.tolist()]
        cleaned = [round(float(v), 2) if v == v else None for v in vals]
        trend = "stable"
        if len(cleaned) >= 2 and cleaned[-1] is not None and cleaned[-2] is not None:
            trend = "up" if cleaned[-1] > cleaned[-2] else "down"
        return json.dumps({
            "channel": channel,
            "metric": metric,
            "weeks": weeks,
            "values": cleaned,
            "trend": trend,
        })

    def _tool_get_anomalies(self) -> str:
        # Import locally to avoid circular imports
        from ai.insights import detect_anomalies
        anomalies = detect_anomalies(self._df)
        return json.dumps(anomalies[:5], ensure_ascii=False)
