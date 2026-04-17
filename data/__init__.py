"""Dataset loader — always returns the Robyn-based dataset."""

from __future__ import annotations

import pandas as pd


def get_dataset() -> pd.DataFrame:
    """Return the Robyn MMM dataset (real Meta spend/revenue data, 208 weeks)."""
    from data.robyn import load_robyn_dataset
    return load_robyn_dataset()


if __name__ == "__main__":
    df = get_dataset()
    print(f"Shape: {df.shape}")
    print(f"Date range: {df.week_date.min()} → {df.week_date.max()}")
    print("Clients: ", df["client"].unique().tolist())
    print("Channels:", df["channel"].unique().tolist())
