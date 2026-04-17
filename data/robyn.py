"""Load Meta's Robyn MMM sample dataset and reshape to the app schema.

The pre-built parquet file (robyn_weekly.parquet) is committed to the repo
so the app always starts without a network call. If the file is missing for
any reason (fresh clone without LFS, manual deletion) it is re-downloaded
and rebuilt automatically.

What's real vs. fabricated
---------------------------
REAL (from Robyn dt_simulated_weekly, 208 weeks, 2015-2019):
  - Weekly spend per channel (Facebook → Meta Ads, Search → Google Ads)
  - Facebook impressions + Search clicks

FABRICATED (deterministic, seeded):
  - TikTok Ads spend (20 % of Meta spend — TikTok didn't exist in 2015)
  - Revenue: spend × channel ROAS factor with seasonal variation
    (Robyn's raw revenue = total business revenue, ~200x ROAS if used directly)
  - Campaign / client / audience / goal breakdown
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

# Committed parquet — always the primary source
_PARQUET_PATH = Path(__file__).parent / "robyn_weekly.parquet"

# Fallback: original RData from Meta's GitHub
_ROBYN_URL = (
    "https://github.com/facebookexperimental/Robyn"
    "/raw/main/R/data/dt_simulated_weekly.RData"
)

_SEED = 7391

# Realistic per-channel ROAS (search > social > awareness)
_CHANNEL_ROAS = {
    "Google Ads": 5.2,
    "Meta Ads":   3.1,
    "TikTok Ads": 2.0,
}

_CAMPAIGNS: dict[str, list[tuple[str, str]]] = {
    "Haaland": [("Haaland – Brand Q1",    "Brand Awareness"),
                ("Haaland – Direct Sales", "Direct Sales")],
    "Nansen":  [("Nansen – Lead Gen",     "Lead Generation"),
                ("Nansen – App Install",  "App Installs")],
    "Solberg": [("Solberg – Direct Sales", "Direct Sales")],
}

_CLIENTS   = list(_CAMPAIGNS.keys())
_AUDIENCES = ["18-34 Urban", "35-54 Suburban", "25-44 High Income", "Retargeting", "Lookalike"]
_AD_TEXTS  = [
    "Prøv gratis i 30 dager", "Bestill nå – begrenset tilbud",
    "Se hvorfor 10 000+ kunder velger oss",
    "Eksklusivt tilbud denne uken", "Gratis frakt over NOK 500",
]
_CONV_RATE = {
    "Direct Sales": 0.040, "Lead Generation": 0.060,
    "App Installs": 0.080, "Brand Awareness": 0.008,
}


# ---------------------------------------------------------------------------
# Download + reshape (only needed when parquet is missing)
# ---------------------------------------------------------------------------

def _download_raw() -> pd.DataFrame:
    try:
        import pyreadr
        import requests
    except ImportError as exc:
        raise ImportError(
            "pyreadr and requests are needed to rebuild the dataset. "
            "Run: pip install pyreadr requests"
        ) from exc

    resp = requests.get(_ROBYN_URL, timeout=60)
    resp.raise_for_status()
    with tempfile.NamedTemporaryFile(suffix=".RData", delete=False) as f:
        f.write(resp.content)
        tmp = f.name
    try:
        result = pyreadr.read_r(tmp)
        return result[list(result.keys())[0]]
    finally:
        os.unlink(tmp)


def _reshape(raw: pd.DataFrame) -> pd.DataFrame:
    np.random.seed(_SEED)
    raw = raw.copy()
    raw["DATE"] = pd.to_datetime(raw["DATE"])
    raw = raw.sort_values("DATE").reset_index(drop=True)

    rows: list[dict] = []
    for idx, wrow in raw.iterrows():
        week_num  = int(idx) + 1
        week_date = wrow["DATE"].strftime("%Y-%m-%d")

        meta_spend    = max(float(wrow.get("facebook_S",      0)), 0)
        google_spend  = max(float(wrow.get("search_S",        0)), 0)
        meta_imp      = max(float(wrow.get("facebook_I",      0)), 0)
        google_clicks = max(float(wrow.get("search_clicks_P", 0)), 0)
        tiktok_spend  = meta_spend * 0.20

        channel_raw = {
            "Google Ads": dict(spend=google_spend,
                               impressions=google_clicks / 0.032 if google_clicks > 0 else google_spend * 80,
                               clicks=google_clicks),
            "Meta Ads":   dict(spend=meta_spend,   impressions=meta_imp,         clicks=meta_imp * 0.018),
            "TikTok Ads": dict(spend=tiktok_spend, impressions=tiktok_spend*120, clicks=tiktok_spend*1.5),
        }

        # Seasonal ROAS variation (same formula as original build)
        week_phase = week_num / 52.0 * 2 * np.pi
        roas_mult = (
            0.82
            + 0.22 * np.sin(week_phase)
            + 0.08 * np.sin(week_phase * 2.7 + 1.1)
        )
        channel_rev = {
            name: ch["spend"] * _CHANNEL_ROAS[name] * (
                roas_mult + 0.04 * np.sin(week_num * 0.41 + hash(name) % 6)
            )
            for name, ch in channel_raw.items()
        }

        primary = _CLIENTS[(week_num - 1) % len(_CLIENTS)]
        for ch_name, ch_data in channel_raw.items():
            for client, campaigns in _CAMPAIGNS.items():
                client_w = 0.55 if client == primary else 0.45 / (len(_CLIENTS) - 1)
                for camp_name, goal in campaigns:
                    camp_w = client_w / len(campaigns)
                    spend  = ch_data["spend"]       * camp_w
                    if spend <= 0:
                        continue
                    rev    = channel_rev[ch_name]   * camp_w
                    impr   = ch_data["impressions"] * camp_w
                    clk    = ch_data["clicks"]      * camp_w
                    conv   = clk * _CONV_RATE.get(goal, 0.03)
                    roas   = rev / spend if spend > 0 else 0.0
                    ctr    = clk / impr * 100 if impr > 0 else 0.0

                    det = week_num * 31 + hash(camp_name) + hash(ch_name)
                    rows.append({
                        "client":      client,
                        "campaign":    camp_name,
                        "channel":     ch_name,
                        "week":        week_num,
                        "week_date":   week_date,
                        "spend":       round(spend,  2),
                        "revenue":     round(rev,    2),
                        "roas":        round(roas,   4),
                        "impressions": int(round(impr, 0)),
                        "clicks":      int(round(clk,  0)),
                        "ctr":         round(ctr,    4),
                        "conversions": int(round(conv, 0)),
                        "goal":        goal,
                        "audience":    _AUDIENCES[det % len(_AUDIENCES)],
                        "ad_text":     _AD_TEXTS[(det // 7) % len(_AD_TEXTS)],
                    })

    return pd.DataFrame(rows)


def _build_and_save() -> pd.DataFrame:
    """Download Robyn data, reshape, and save to robyn_weekly.parquet."""
    raw = _download_raw()
    df  = _reshape(raw)
    df.to_parquet(_PARQUET_PATH, index=False)
    return df


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def load_robyn_dataset() -> pd.DataFrame:
    """Return the Robyn-based dataset, building it from source if needed."""
    if _PARQUET_PATH.exists():
        return pd.read_parquet(_PARQUET_PATH)
    return _build_and_save()
