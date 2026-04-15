"""Markedsinnsikt AI — Streamlit marketing analytics assistant."""

import os

import pandas as pd
import streamlit as st
import plotly.express as px

from data import generate_dataset
from ai_assistant import build_context, generate_insights, answer_question

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Markedsinnsikt AI",
    page_icon="📊",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Load data (cached)
# ---------------------------------------------------------------------------

@st.cache_data
def load_data() -> pd.DataFrame:
    return generate_dataset()


df = load_data()

# ---------------------------------------------------------------------------
# Sidebar — filters
# ---------------------------------------------------------------------------

st.sidebar.title("📊 Markedsinnsikt AI")
st.sidebar.markdown("*AI-powered marketing analytics*")
st.sidebar.divider()

client_options = ["All"] + sorted(df["client"].unique().tolist())
selected_client = st.sidebar.selectbox("Client", client_options)

filtered_df = df if selected_client == "All" else df[df["client"] == selected_client]

campaign_options = ["All"] + sorted(filtered_df["campaign"].unique().tolist())
selected_campaign = st.sidebar.selectbox("Campaign", campaign_options)

if selected_campaign != "All":
    filtered_df = filtered_df[filtered_df["campaign"] == selected_campaign]

channel_options = ["All"] + sorted(filtered_df["channel"].unique().tolist())
selected_channel = st.sidebar.selectbox("Channel", channel_options)

if selected_channel != "All":
    filtered_df = filtered_df[filtered_df["channel"] == selected_channel]

st.sidebar.divider()
st.sidebar.caption(f"{len(filtered_df)} rows in current view")

# Groq key input in sidebar
api_key_input = st.sidebar.text_input(
    "Groq API Key",
    type="password",
    value=os.getenv("GROQ_API_KEY", ""),
    help="Needed for AI insights and Q&A. Not stored.",
)
if api_key_input:
    os.environ["GROQ_API_KEY"] = api_key_input

model_choice = st.sidebar.selectbox(
    "Model",
    ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"],
    index=0,
)

# ---------------------------------------------------------------------------
# Main header
# ---------------------------------------------------------------------------

st.title("📊 Markedsinnsikt AI")
st.markdown("AI-powered marketing analytics assistant for multi-client, multi-channel campaigns.")
st.divider()

# ---------------------------------------------------------------------------
# KPI row
# ---------------------------------------------------------------------------

total_spend = filtered_df["spend"].sum()
total_revenue = filtered_df["revenue"].sum()
total_conversions = filtered_df["conversions"].sum()
total_clicks = filtered_df["clicks"].sum()
total_impressions = filtered_df["impressions"].sum()
avg_roas = total_revenue / total_spend if total_spend > 0 else 0
avg_ctr = total_clicks / total_impressions * 100 if total_impressions > 0 else 0

kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
kpi1.metric("Total Spend", f"NOK {total_spend:,.0f}")
kpi2.metric("Total Revenue", f"NOK {total_revenue:,.0f}")
kpi3.metric("Total Conversions", f"{total_conversions:,}")
kpi4.metric("Avg ROAS", f"{avg_roas:.2f}x")
kpi5.metric("Avg CTR", f"{avg_ctr:.2f}%")

st.divider()

# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

col_left, col_right = st.columns(2)

with col_left:
    st.subheader("ROAS by Channel")
    roas_by_channel = (
        filtered_df.groupby("channel")
        .apply(lambda x: (x["revenue"].sum() / x["spend"].sum()) if x["spend"].sum() > 0 else 0)
        .reset_index(name="roas")
    )
    fig_roas = px.bar(
        roas_by_channel,
        x="channel",
        y="roas",
        color="channel",
        color_discrete_map={
            "Google Ads": "#4285F4",
            "Meta Ads": "#1877F2",
            "TikTok Ads": "#010101",
        },
        text_auto=".2f",
        labels={"roas": "ROAS (x)", "channel": "Channel"},
    )
    fig_roas.update_traces(textposition="outside")
    fig_roas.update_layout(showlegend=False, yaxis_title="ROAS (x)", xaxis_title="")
    st.plotly_chart(fig_roas, use_container_width=True)

with col_right:
    st.subheader("Conversions by Campaign")
    conv_by_campaign = (
        filtered_df.groupby("campaign")["conversions"]
        .sum()
        .reset_index()
        .sort_values("conversions", ascending=True)
    )
    fig_conv = px.bar(
        conv_by_campaign,
        x="conversions",
        y="campaign",
        orientation="h",
        text_auto=True,
        labels={"conversions": "Conversions", "campaign": "Campaign"},
        color="conversions",
        color_continuous_scale="Blues",
    )
    fig_conv.update_layout(coloraxis_showscale=False, xaxis_title="Conversions", yaxis_title="")
    st.plotly_chart(fig_conv, use_container_width=True)

# Second row of charts
col_a, col_b = st.columns(2)

with col_a:
    st.subheader("Spend by Channel")
    spend_by_channel = filtered_df.groupby("channel")["spend"].sum().reset_index()
    fig_spend = px.pie(
        spend_by_channel,
        names="channel",
        values="spend",
        color="channel",
        color_discrete_map={
            "Google Ads": "#4285F4",
            "Meta Ads": "#1877F2",
            "TikTok Ads": "#69C9D0",
        },
    )
    fig_spend.update_traces(textposition="inside", textinfo="percent+label")
    st.plotly_chart(fig_spend, use_container_width=True)

with col_b:
    st.subheader("Weekly Spend Trend")
    weekly_spend = filtered_df.groupby("week")["spend"].sum().reset_index()
    fig_trend = px.line(
        weekly_spend,
        x="week",
        y="spend",
        markers=True,
        labels={"spend": "Spend (NOK)", "week": "Week"},
    )
    fig_trend.update_layout(xaxis_title="Week", yaxis_title="Spend (NOK)")
    st.plotly_chart(fig_trend, use_container_width=True)

st.divider()

# ---------------------------------------------------------------------------
# Raw data table
# ---------------------------------------------------------------------------

with st.expander("View raw data", expanded=False):
    st.dataframe(
        filtered_df[[
            "client", "campaign", "channel", "week", "spend", "impressions",
            "clicks", "conversions", "revenue", "roas", "ctr", "goal", "audience",
        ]].reset_index(drop=True),
        use_container_width=True,
    )

st.divider()

# ---------------------------------------------------------------------------
# AI Insights
# ---------------------------------------------------------------------------

st.subheader("🤖 AI Performance Insights")

if not os.getenv("GROQ_API_KEY"):
    st.info("Enter your OpenAI API key in the sidebar to enable AI features.")
else:
    if st.button("Generate Insights", type="primary"):
        with st.spinner("Analyzing campaign data..."):
            try:
                context = build_context(
                    df,
                    client_filter=selected_client,
                    campaign_filter=selected_campaign,
                    channel_filter=selected_channel,
                )
                insights = generate_insights(context, model=model_choice)
                st.session_state["last_insights"] = insights
                st.session_state["last_context"] = context
            except Exception as e:
                st.error(f"Error generating insights: {e}")

    if "last_insights" in st.session_state:
        st.markdown(st.session_state["last_insights"])

st.divider()

# ---------------------------------------------------------------------------
# Q&A Assistant
# ---------------------------------------------------------------------------

st.subheader("💬 Ask the AI Analyst")

if not os.getenv("GROQ_API_KEY"):
    st.info("Enter your OpenAI API key in the sidebar to enable AI features.")
else:
    example_questions = [
        "Which campaign has the best ROAS and why might that be?",
        "Which channel should we increase budget on?",
        "What is underperforming and what should we do about it?",
        "How does TikTok compare to Google Ads in terms of efficiency?",
        "Which client needs the most attention?",
    ]

    with st.expander("Example questions"):
        for q in example_questions:
            st.markdown(f"- {q}")

    user_question = st.text_area(
        "Your question",
        placeholder="e.g. Which campaign has the best ROAS and why?",
        height=80,
    )

    if st.button("Ask", type="primary") and user_question.strip():
        with st.spinner("Thinking..."):
            try:
                context = build_context(
                    df,
                    client_filter=selected_client,
                    campaign_filter=selected_campaign,
                    channel_filter=selected_channel,
                )
                answer = answer_question(user_question, context, model=model_choice)
                st.session_state["last_answer"] = answer
                st.session_state["last_question"] = user_question
            except Exception as e:
                st.error(f"Error: {e}")

    if "last_answer" in st.session_state:
        st.markdown(f"**Q: {st.session_state['last_question']}**")
        st.markdown(st.session_state["last_answer"])

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.divider()
st.caption("Markedsinnsikt AI — Built with Streamlit & Groq · Data is synthetic for demo purposes.")
