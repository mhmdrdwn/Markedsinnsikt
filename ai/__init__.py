"""ai package — re-exports all public AI assistant functions."""

from ai.insights import (
    build_context,
    generate_insights,
    answer_question,
    detect_anomalies,
    compute_trends,
    compute_goal_context,
    compute_audience_context,
    compute_predictions,
    compute_benchmark_context,
    generate_insights_with_meta,
    answer_question_with_tools,
    ObsRecord,
)

__all__ = [
    "build_context",
    "generate_insights",
    "answer_question",
    "detect_anomalies",
    "compute_trends",
    "compute_goal_context",
    "compute_audience_context",
    "compute_predictions",
    "compute_benchmark_context",
    "generate_insights_with_meta",
    "answer_question_with_tools",
    "ObsRecord",
]
