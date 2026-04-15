"""AI assistant layer — builds context from the dataframe and calls Groq."""

from __future__ import annotations

import json
import os
import textwrap
from typing import Optional

import pandas as pd
from groq import Groq

_client: Optional[Groq] = None


def get_groq_client() -> Groq:
    global _client
    if _client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY environment variable is not set. "
                "Add it to your shell or create a .env file."
            )
        _client = Groq(api_key=api_key)
    return _client


# ---------------------------------------------------------------------------
# Trend analysis
# ---------------------------------------------------------------------------

def _pct_change_last_two(series: pd.Series) -> float:
    clean = series.dropna()
    if len(clean) < 2 or clean.iloc[-2] == 0:
        return 0.0
    return (clean.iloc[-1] - clean.iloc[-2]) / abs(clean.iloc[-2]) * 100


def compute_trends(df: pd.DataFrame) -> str:
    lines = ["=== WEEK-OVER-WEEK TRENDS (last 2 weeks) ==="]

    for channel, grp in df.groupby("channel"):
        weekly = (
            grp.groupby("week")
            .agg(spend=("spend", "sum"), revenue=("revenue", "sum"), conversions=("conversions", "sum"))
            .sort_index()
        )
        if len(weekly) < 2:
            continue
        weekly["roas"] = weekly["revenue"] / weekly["spend"].replace(0, float("nan"))
        lines.append(
            f"  {channel}: ROAS {_pct_change_last_two(weekly['roas']):+.1f}% WoW | "
            f"Spend {_pct_change_last_two(weekly['spend']):+.1f}% WoW | "
            f"Conversions {_pct_change_last_two(weekly['conversions']):+.1f}% WoW"
        )

    overall = df.groupby("week").agg(spend=("spend", "sum"), revenue=("revenue", "sum")).sort_index()
    if len(overall) >= 2:
        overall["roas"] = overall["revenue"] / overall["spend"].replace(0, float("nan"))
        lines.append(
            f"  OVERALL: ROAS {_pct_change_last_two(overall['roas']):+.1f}% WoW | "
            f"Spend {_pct_change_last_two(overall['spend']):+.1f}% WoW"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Anomaly detection
# ---------------------------------------------------------------------------

def detect_anomalies(
    df: pd.DataFrame,
    roas_drop_pct: float = -25.0,
    spend_spike_pct: float = 50.0,
) -> list[dict]:
    anomalies: list[dict] = []

    for (client, campaign), grp in df.groupby(["client", "campaign"]):
        weekly = (
            grp.groupby("week")
            .agg(spend=("spend", "sum"), revenue=("revenue", "sum"))
            .sort_index()
        )
        if len(weekly) < 2:
            continue

        weekly["roas"] = weekly["revenue"] / weekly["spend"].replace(0, float("nan"))
        last_roas, prev_roas = weekly["roas"].iloc[-1], weekly["roas"].iloc[-2]

        if pd.notna(last_roas) and pd.notna(prev_roas) and prev_roas != 0:
            roas_change = (last_roas - prev_roas) / abs(prev_roas) * 100
            if roas_change <= roas_drop_pct:
                anomalies.append({
                    "type": "roas_drop", "client": client, "campaign": campaign,
                    "detail": (
                        f"ROAS dropped {roas_change:.1f}% "
                        f"(week {int(weekly.index[-2])} → {int(weekly.index[-1])}): "
                        f"{prev_roas:.2f}x → {last_roas:.2f}x"
                    ),
                    "severity": "high" if roas_change <= -40 else "medium",
                })

        prior_avg = weekly["spend"].iloc[:-1].mean()
        last_spend = weekly["spend"].iloc[-1]
        if prior_avg > 0 and (last_spend - prior_avg) / prior_avg * 100 >= spend_spike_pct:
            anomalies.append({
                "type": "spend_spike", "client": client, "campaign": campaign,
                "detail": (
                    f"Spend spiked {(last_spend - prior_avg) / prior_avg * 100:.1f}% above prior average "
                    f"(NOK {prior_avg:,.0f} → NOK {last_spend:,.0f})"
                ),
                "severity": "medium",
            })

    return anomalies


# ---------------------------------------------------------------------------
# Goal-aware analysis
# ---------------------------------------------------------------------------

def compute_goal_context(df: pd.DataFrame) -> str:
    """Evaluate performance by campaign goal using goal-appropriate KPIs."""
    lines = ["=== GOAL-SPECIFIC PERFORMANCE ==="]

    for goal, grp in df.groupby("goal"):
        spend       = grp["spend"].sum()
        impressions = grp["impressions"].sum()
        conversions = grp["conversions"].sum()
        revenue     = grp["revenue"].sum()
        roas        = revenue / spend if spend > 0 else 0
        cpm         = spend / impressions * 1000 if impressions > 0 else 0
        cpl         = spend / conversions if conversions > 0 else 0

        if goal == "Brand Awareness":
            lines.append(
                f"  {goal}: CPM NOK {cpm:.0f} | Impressions {impressions:,} | "
                f"Reach-spend NOK {spend:,.0f} — judge on CPM/reach, NOT ROAS"
            )
        elif goal == "Lead Generation":
            lines.append(
                f"  {goal}: CPL NOK {cpl:.0f} | Leads {int(conversions):,} | "
                f"Spend NOK {spend:,.0f} — judge on Cost Per Lead"
            )
        elif goal == "Direct Sales":
            lines.append(
                f"  {goal}: ROAS {roas:.2f}x | Revenue NOK {revenue:,.0f} | "
                f"Conversions {int(conversions):,} — judge on ROAS and revenue"
            )
        elif goal == "App Installs":
            lines.append(
                f"  {goal}: CPI NOK {cpl:.0f} | Installs {int(conversions):,} | "
                f"Spend NOK {spend:,.0f} — judge on Cost Per Install"
            )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Audience segment analysis
# ---------------------------------------------------------------------------

def compute_audience_context(df: pd.DataFrame) -> str:
    """Show performance by audience segment, sorted by ROAS."""
    lines = ["=== AUDIENCE SEGMENT PERFORMANCE ==="]

    aud = (
        df.groupby("audience")
        .agg(
            spend=("spend", "sum"),
            revenue=("revenue", "sum"),
            conversions=("conversions", "sum"),
            impressions=("impressions", "sum"),
        )
        .reset_index()
    )
    aud["roas"] = (aud["revenue"] / aud["spend"].replace(0, float("nan"))).round(2)
    aud["cpm"]  = (aud["spend"] / aud["impressions"].replace(0, float("nan")) * 1000).round(0)
    aud = aud.sort_values("roas", ascending=False)

    for _, row in aud.iterrows():
        lines.append(
            f"  {row['audience']}: ROAS {row['roas']:.2f}x | "
            f"Conversions {int(row['conversions']):,} | CPM NOK {row['cpm']:.0f}"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Predictive extrapolation
# ---------------------------------------------------------------------------

def compute_predictions(df: pd.DataFrame) -> str:
    """Detect multi-week declining trends and accelerating spend trajectories."""
    lines = ["=== TREND PREDICTIONS ==="]
    findings = []

    for (client, campaign), grp in df.groupby(["client", "campaign"]):
        weekly = (
            grp.groupby("week")
            .agg(spend=("spend", "sum"), revenue=("revenue", "sum"))
            .sort_index()
        )
        if len(weekly) < 3:
            continue

        weekly["roas"] = weekly["revenue"] / weekly["spend"].replace(0, float("nan"))
        roas_vals = weekly["roas"].dropna().tolist()

        # 3+ consecutive weeks of declining ROAS
        if len(roas_vals) >= 3 and roas_vals[-3] > roas_vals[-2] > roas_vals[-1]:
            total_drop = (roas_vals[-1] - roas_vals[-3]) / abs(roas_vals[-3]) * 100
            findings.append(
                f"  [{client}] {campaign}: ROAS declining 3+ weeks in a row "
                f"({roas_vals[-3]:.2f}x → {roas_vals[-2]:.2f}x → {roas_vals[-1]:.2f}x, "
                f"{total_drop:.1f}% total) — immediate review recommended"
            )

        # Accelerating spend (recent 2 weeks avg > 30% above prior average)
        spend_vals = weekly["spend"].tolist()
        recent_avg = sum(spend_vals[-2:]) / 2
        prior_avg  = sum(spend_vals[:-2]) / max(len(spend_vals) - 2, 1)
        if prior_avg > 0 and (recent_avg - prior_avg) / prior_avg * 100 >= 30:
            findings.append(
                f"  [{client}] {campaign}: Spend accelerating "
                f"(NOK {prior_avg:,.0f}/week prior avg → NOK {recent_avg:,.0f}/week recently, "
                f"+{(recent_avg - prior_avg) / prior_avg * 100:.0f}%) — monitor budget pacing"
            )

    lines += findings if findings else ["  No significant trend predictions at this time."]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Cross-client benchmarking
# ---------------------------------------------------------------------------

def compute_benchmark_context(df: pd.DataFrame, client_filter: str) -> str:
    """Compare selected client against portfolio averages."""
    portfolio_spend  = df["spend"].sum()
    portfolio_rev    = df["revenue"].sum()
    portfolio_clicks = df["clicks"].sum()
    portfolio_impr   = df["impressions"].sum()
    portfolio_conv   = df["conversions"].sum()

    p_roas = portfolio_rev  / portfolio_spend  if portfolio_spend > 0 else 0
    p_ctr  = portfolio_clicks / portfolio_impr * 100 if portfolio_impr > 0 else 0
    p_cpl  = portfolio_spend  / portfolio_conv if portfolio_conv > 0 else 0

    client_df   = df[df["client"] == client_filter]
    c_spend     = client_df["spend"].sum()
    c_rev       = client_df["revenue"].sum()
    c_clicks    = client_df["clicks"].sum()
    c_impr      = client_df["impressions"].sum()
    c_conv      = client_df["conversions"].sum()

    c_roas = c_rev    / c_spend  if c_spend > 0 else 0
    c_ctr  = c_clicks / c_impr * 100 if c_impr > 0 else 0
    c_cpl  = c_spend  / c_conv if c_conv > 0 else 0

    roas_vs = (c_roas - p_roas) / p_roas * 100 if p_roas > 0 else 0
    ctr_vs  = (c_ctr  - p_ctr)  / p_ctr  * 100 if p_ctr  > 0 else 0
    cpl_vs  = (c_cpl  - p_cpl)  / p_cpl  * 100 if p_cpl  > 0 else 0

    return "\n".join([
        f"=== CROSS-CLIENT BENCHMARKING ({client_filter} vs. portfolio) ===",
        f"  ROAS : {client_filter} {c_roas:.2f}x vs portfolio avg {p_roas:.2f}x ({roas_vs:+.1f}%)",
        f"  CTR  : {client_filter} {c_ctr:.2f}%  vs portfolio avg {p_ctr:.2f}%  ({ctr_vs:+.1f}%)",
        f"  CPL  : {client_filter} NOK {c_cpl:.0f}  vs portfolio avg NOK {p_cpl:.0f}  ({cpl_vs:+.1f}%)",
    ])


# ---------------------------------------------------------------------------
# Context builder
# ---------------------------------------------------------------------------

def build_context(
    df: pd.DataFrame,
    client_filter: Optional[str] = None,
    campaign_filter: Optional[str] = None,
    channel_filter: Optional[str] = None,
) -> str:
    filtered = df.copy()

    if client_filter and client_filter != "All":
        filtered = filtered[filtered["client"] == client_filter]
    if campaign_filter and campaign_filter != "All":
        filtered = filtered[filtered["campaign"] == campaign_filter]
    if channel_filter and channel_filter != "All":
        filtered = filtered[filtered["channel"] == channel_filter]

    lines = []

    # Aggregate summary
    total_spend       = filtered["spend"].sum()
    total_revenue     = filtered["revenue"].sum()
    total_conversions = filtered["conversions"].sum()
    total_clicks      = filtered["clicks"].sum()
    total_impressions = filtered["impressions"].sum()
    avg_roas = (total_revenue / total_spend) if total_spend > 0 else 0
    avg_ctr  = (total_clicks / total_impressions * 100) if total_impressions > 0 else 0

    lines += [
        "=== AGGREGATE SUMMARY ===",
        f"Rows included     : {len(filtered)}",
        f"Total Spend       : NOK {total_spend:,.0f}",
        f"Total Revenue     : NOK {total_revenue:,.0f}",
        f"Total Conversions : {total_conversions:,}",
        f"Avg ROAS          : {avg_roas:.2f}x",
        f"Avg CTR           : {avg_ctr:.2f}%",
        "",
    ]

    # Channel breakdown
    lines.append("=== PERFORMANCE BY CHANNEL ===")
    ch_group = (
        filtered.groupby("channel")
        .agg(spend=("spend", "sum"), revenue=("revenue", "sum"), conversions=("conversions", "sum"))
        .reset_index()
    )
    ch_group["roas"] = (ch_group["revenue"] / ch_group["spend"]).round(2)
    for _, row in ch_group.iterrows():
        lines.append(
            f"  {row['channel']}: spend NOK {row['spend']:,.0f} | "
            f"revenue NOK {row['revenue']:,.0f} | conversions {int(row['conversions'])} | ROAS {row['roas']}x"
        )
    lines.append("")

    # Campaign breakdown (top 10 by spend to keep context size bounded)
    lines.append("=== PERFORMANCE BY CAMPAIGN (top 10 by spend) ===")
    cp_group = (
        filtered.groupby(["client", "campaign"])
        .agg(spend=("spend", "sum"), revenue=("revenue", "sum"), conversions=("conversions", "sum"))
        .reset_index()
    )
    cp_group["roas"] = (cp_group["revenue"] / cp_group["spend"]).round(2)
    cp_group = cp_group.nlargest(10, "spend")
    for _, row in cp_group.iterrows():
        lines.append(
            f"  [{row['client']}] {row['campaign']}: spend NOK {row['spend']:,.0f} | "
            f"revenue NOK {row['revenue']:,.0f} | conversions {int(row['conversions'])} | ROAS {row['roas']}x"
        )
    lines.append("")

    # Goal-aware analysis
    lines.append(compute_goal_context(filtered))
    lines.append("")

    # Audience segment analysis
    lines.append(compute_audience_context(filtered))
    lines.append("")

    # Trends
    lines.append(compute_trends(filtered))
    lines.append("")

    # Predictive extrapolation
    lines.append(compute_predictions(filtered))
    lines.append("")

    # Cross-client benchmarking (only when a specific client is selected)
    if client_filter and client_filter != "All":
        lines.append(compute_benchmark_context(df, client_filter))
        lines.append("")

    # Anomalies (cap at 5 to avoid bloat)
    anomalies = detect_anomalies(filtered)
    if anomalies:
        lines.append("=== DETECTED ANOMALIES ===")
        for a in anomalies[:5]:
            lines.append(f"  [{a['severity'].upper()}] [{a['client']}] {a['campaign']}: {a['detail']}")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = textwrap.dedent("""\
    Du er en erfaren markedsanalytiker spesialisert på performance marketing \
    på tvers av Google Ads, Meta Ads og TikTok Ads. Du jobber med skandinaviske \
    merkevarer og forstår norsk markedskontekst. Vær konsis, datadrevet og \
    handlingsorientert. Valuta er NOK. Svar alltid på norsk (bokmål).

    Evaluer alltid kampanjer basert på deres mål:
    - Brand Awareness: vurder CPM og rekkevidde — IKKE ROAS
    - Lead Generation: vurder kostnad per lead (CPL)
    - Direct Sales: vurder ROAS og inntekt
    - App Installs: vurder kostnad per installasjon (CPI)\
""")

INSIGHT_PROMPT = textwrap.dedent("""\
    Analyser følgende kampanjedata — inkludert mål-spesifikke KPIer, målgruppesegmenter, \
    uke-over-uke trender, prediksjoner og avvik — og returner et JSON-objekt med NØYAKTIG denne strukturen:

    {{
      "summary": "2-3 setninger oppsummering av samlet ytelse",
      "insights": [
        {{"title": "kort tittel", "detail": "1-2 setninger med spesifikke tall"}}
      ],
      "anomalies": [
        {{"campaign": "kampanjenavn", "issue": "hva som er galt", "severity": "high|medium|low"}}
      ],
      "recommendations": [
        {{
          "action": "konkret handling",
          "target": "kanal eller kampanjenavn",
          "expected_impact": "forventet resultat med estimerte tall om mulig",
          "priority": "high|medium|low"
        }}
      ]
    }}

    Regler:
    - insights: 3-5 punkter, inkluder målgruppe- og målinnsikter
    - anomalies: tom liste [] hvis ingen oppdaget
    - recommendations: 3-5 konkrete, prioriterte tiltak
    - Returner KUN gyldig JSON — ingen markdown, ingen forklaring

    Data:
    {context}
""")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_json(text: str) -> dict:
    """Parse JSON, raising a clear error if the response was truncated."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to salvage a truncated response by closing open braces/brackets
        open_braces   = text.count("{") - text.count("}")
        open_brackets = text.count("[") - text.count("]")
        patched = text + ("]" * open_brackets) + ("}" * open_braces)
        try:
            return json.loads(patched)
        except json.JSONDecodeError:
            raise ValueError(
                "AI-svaret ble avkortet (for mye data). "
                "Prøv å filtrere til én kunde eller kanal og prøv igjen."
            )


# ---------------------------------------------------------------------------
# Provider implementations
# ---------------------------------------------------------------------------

def _groq_insights(context: str, model: str) -> dict:
    client = get_groq_client()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": INSIGHT_PROMPT.format(context=context)},
        ],
        response_format={"type": "json_object"},
        temperature=0.3,
        max_tokens=2000,
    )
    return _safe_json(response.choices[0].message.content)


def _groq_answer(messages: list[dict], context: str, model: str) -> str:
    client = get_groq_client()
    system = (
        SYSTEM_PROMPT
        + "\n\nDu har tilgang til følgende kampanjedata. Svar basert på disse dataene. "
        "Vær spesifikk, bruk tall, og gi handlingsrettede råd. "
        "Hold svar under 200 ord med mindre mer er etterspurt.\n\n"
        + context
    )
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system}, *messages],
        temperature=0.3,
        max_tokens=700,
    )
    return response.choices[0].message.content.strip()


def _gemini_insights(context: str) -> dict:
    from google import genai as google_genai
    from google.genai import types as genai_types
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not set")
    model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    client = google_genai.Client(api_key=api_key)
    prompt = f"{SYSTEM_PROMPT}\n\n{INSIGHT_PROMPT.format(context=context)}"
    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config=genai_types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.3,
            max_output_tokens=1500,
        ),
    )
    return _safe_json(response.text)


def _gemini_answer(messages: list[dict], context: str) -> str:
    from google import genai as google_genai
    from google.genai import types as genai_types
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not set")
    model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    client = google_genai.Client(api_key=api_key)
    system = (
        SYSTEM_PROMPT
        + "\n\nDu har tilgang til følgende kampanjedata. Svar basert på disse dataene. "
        "Vær spesifikk, bruk tall, og gi handlingsrettede råd. "
        "Hold svar under 200 ord med mindre mer er etterspurt.\n\n"
        + context
    )
    # Build Gemini-format history
    history = []
    for m in messages[:-1]:
        history.append(
            genai_types.Content(
                role="user" if m["role"] == "user" else "model",
                parts=[genai_types.Part(text=m["content"])],
            )
        )
    history.append(
        genai_types.Content(
            role="user",
            parts=[genai_types.Part(text=messages[-1]["content"])],
        )
    )
    response = client.models.generate_content(
        model=model_name,
        contents=history,
        config=genai_types.GenerateContentConfig(
            system_instruction=system,
            temperature=0.3,
            max_output_tokens=700,
        ),
    )
    return response.text.strip()


def _mistral_insights(context: str) -> dict:
    from mistralai.client import Mistral
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise ValueError("MISTRAL_API_KEY not set")
    client = Mistral(api_key=api_key)
    response = client.chat.complete(
        model="mistral-small-latest",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": INSIGHT_PROMPT.format(context=context)},
        ],
        response_format={"type": "json_object"},
        temperature=0.3,
        max_tokens=2000,
    )
    return _safe_json(response.choices[0].message.content)


def _mistral_answer(messages: list[dict], context: str) -> str:
    from mistralai.client import Mistral
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise ValueError("MISTRAL_API_KEY not set")
    client = Mistral(api_key=api_key)
    system = (
        SYSTEM_PROMPT
        + "\n\nDu har tilgang til følgende kampanjedata. Svar basert på disse dataene. "
        "Vær spesifikk, bruk tall, og gi handlingsrettede råd. "
        "Hold svar under 200 ord med mindre mer er etterspurt.\n\n"
        + context
    )
    response = client.chat.complete(
        model="mistral-small-latest",
        messages=[{"role": "system", "content": system}, *messages],
        temperature=0.3,
        max_tokens=700,
    )
    return response.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Public API — tries Groq → Gemini → Mistral automatically
# ---------------------------------------------------------------------------

def generate_insights(context: str, model: str = "llama-3.3-70b-versatile") -> dict:
    attempts = [
        ("Groq",    lambda: _groq_insights(context, model)),
        ("Gemini",  lambda: _gemini_insights(context)),
        ("Mistral", lambda: _mistral_insights(context)),
    ]
    last_error: Exception = RuntimeError("No AI provider configured")
    for name, fn in attempts:
        try:
            return fn()
        except Exception as e:
            last_error = e
    raise last_error


def answer_question(
    messages: list[dict],
    context: str,
    model: str = "llama-3.3-70b-versatile",
) -> str:
    attempts = [
        ("Groq",    lambda: _groq_answer(messages, context, model)),
        ("Gemini",  lambda: _gemini_answer(messages, context)),
        ("Mistral", lambda: _mistral_answer(messages, context)),
    ]
    last_error: Exception = RuntimeError("No AI provider configured")
    for name, fn in attempts:
        try:
            return fn()
        except Exception as e:
            last_error = e
    raise last_error
