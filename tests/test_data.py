"""Tests for the Robyn-based dataset loader."""

import sys
import os

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data import get_dataset

EXPECTED_COLUMNS = {
    "client", "campaign", "channel", "week", "week_date",
    "spend", "revenue", "roas", "impressions", "clicks",
    "ctr", "conversions", "goal", "audience", "ad_text",
}
EXPECTED_CHANNELS = {"Google Ads", "Meta Ads", "TikTok Ads"}
EXPECTED_CLIENTS  = {"Haaland", "Salah", "ViníJr"}
EXPECTED_GOALS    = {"Brand Awareness", "Lead Generation", "Direct Sales", "App Installs"}


@pytest.fixture(scope="module")
def df():
    return get_dataset()


def test_returns_dataframe(df):
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0


def test_all_columns_present(df):
    assert EXPECTED_COLUMNS.issubset(set(df.columns))


def test_channels(df):
    assert set(df["channel"].unique()) == EXPECTED_CHANNELS


def test_clients(df):
    assert set(df["client"].unique()) == EXPECTED_CLIENTS


def test_goals(df):
    assert set(df["goal"].unique()) == EXPECTED_GOALS


def test_date_range(df):
    # Robyn covers Nov 2015 – Nov 2019
    assert df["week_date"].min() >= "2015-01-01"
    assert df["week_date"].max() <= "2020-01-01"


def test_week_count(df):
    # 208 raw Robyn weeks; after filtering zero-spend rows ≥ 100 distinct weeks
    assert df["week"].nunique() >= 100


def test_no_negative_spend(df):
    assert (df["spend"] > 0).all()


def test_no_negative_revenue(df):
    assert (df["revenue"] >= 0).all()


def test_roas_in_realistic_range(df):
    # Derived from spend × channel ROAS factor with seasonal variation
    assert df["roas"].min() >= 0.5
    assert df["roas"].max() <= 10.0


def test_dtypes(df):
    assert df["spend"].dtype == float
    assert df["revenue"].dtype == float
    assert df["roas"].dtype == float
    assert df["impressions"].dtype == "int64"
    assert df["clicks"].dtype == "int64"
    assert df["conversions"].dtype == "int64"
    assert df["week"].dtype == "int64"


def test_reproducibility():
    df1 = get_dataset()
    df2 = get_dataset()
    pd.testing.assert_frame_equal(df1, df2)


def test_each_client_has_campaigns(df):
    for client in EXPECTED_CLIENTS:
        camps = df[df["client"] == client]["campaign"].nunique()
        assert camps >= 1, f"{client} has no campaigns"


def test_each_channel_has_multiple_weeks(df):
    for ch in EXPECTED_CHANNELS:
        weeks = df[df["channel"] == ch]["week"].nunique()
        assert weeks >= 10, f"{ch} has only {weeks} weeks"
