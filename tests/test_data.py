"""Tests for data.py — dataset generation."""

import pandas as pd
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data import generate_dataset, CLIENTS, CAMPAIGNS, CHANNELS


def test_dataset_shape():
    df = generate_dataset()
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0


def test_dataset_columns():
    df = generate_dataset()
    expected = {"client", "campaign", "channel", "week", "week_date",
                "spend", "impressions", "clicks", "conversions",
                "revenue", "roas", "ctr", "ad_text", "audience", "goal"}
    assert expected.issubset(set(df.columns))


def test_all_clients_present():
    df = generate_dataset()
    assert set(df["client"].unique()) == set(CLIENTS)


def test_all_channels_present():
    df = generate_dataset()
    assert set(df["channel"].unique()) == set(CHANNELS)


def test_no_negative_spend_or_revenue():
    df = generate_dataset()
    assert (df["spend"] >= 0).all()
    assert (df["revenue"] >= 0).all()


def test_campaigns_match_clients():
    df = generate_dataset()
    for client, campaigns in CAMPAIGNS.items():
        client_campaigns = set(df[df["client"] == client]["campaign"].unique())
        assert client_campaigns.issubset(set(campaigns)), (
            f"{client} has unexpected campaigns: {client_campaigns - set(campaigns)}"
        )


def test_reproducibility():
    df1 = generate_dataset()
    df2 = generate_dataset()
    pd.testing.assert_frame_equal(df1, df2)
