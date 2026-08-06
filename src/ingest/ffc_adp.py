"""Pull historical ADP from the Fantasy Football Calculator public REST API and cache
to data/raw/ffc_adp.parquet.

Format is full PPR (not half-PPR): FFC's half-PPR ADP series only goes back to
~2024, which isn't enough history for the 2013-2025 temporal train/tune/test split
this project uses. Decided with the project owner -- see constants.py for the
matching note on the panel's start year.
"""

import argparse
import os
import time

import polars as pl
import requests

from src.ingest.constants import RAW_DIR, SEASONS

FFC_URL = "https://fantasyfootballcalculator.com/api/v1/adp/ppr"
LEAGUE_TEAMS = 12


def _pull_season(season: int) -> pl.DataFrame:
    resp = requests.get(
        FFC_URL,
        params={"teams": LEAGUE_TEAMS, "year": season, "position": "all"},
        timeout=30,
    )
    resp.raise_for_status()
    payload = resp.json()
    if payload.get("status") != "Success":
        raise RuntimeError(f"FFC API returned non-success status for {season}: {payload.get('status')}")
    players = payload.get("players", [])
    if not players:
        raise RuntimeError(f"FFC API returned zero players for {season} -- check the year is covered.")
    df = pl.DataFrame(players)
    df = df.with_columns(pl.lit(season).alias("season"))
    return df


def pull_ffc_adp(refresh: bool = False) -> pl.DataFrame:
    path = os.path.join(RAW_DIR, "ffc_adp.parquet")
    if os.path.exists(path) and not refresh:
        return pl.read_parquet(path)

    season_frames = []
    for season in SEASONS:
        season_frames.append(_pull_season(season))
        time.sleep(0.5)  # be polite to a free public API

    df = pl.concat(season_frames, how="diagonal_relaxed")
    os.makedirs(RAW_DIR, exist_ok=True)
    df.write_parquet(path)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pull and cache FFC ADP data.")
    parser.add_argument(
        "--refresh", action="store_true", help="Re-pull from network even if cached."
    )
    args = parser.parse_args()
    result = pull_ffc_adp(refresh=args.refresh)
    print(f"Pulled ADP for {result.select('season').n_unique()} seasons, {result.height} total rows.")
