"""Pull raw nflverse tables via nflreadpy and cache each to data/raw/*.parquet.

Every function here hits the network once (unless `refresh=True` or the cache file is
missing) and writes the untouched raw table straight to parquet. No filtering, joining,
or column derivation happens here -- that's build_panel.py's job. Keeping this split
means a bug in the join logic never requires re-pulling data from the network.
"""

import argparse
import os

import nflreadpy as nfl
import polars as pl

from src.ingest.constants import RAW_DIR, SEASONS


def _cached_pull(filename: str, pull_fn, refresh: bool) -> pl.DataFrame:
    path = os.path.join(RAW_DIR, filename)
    if os.path.exists(path) and not refresh:
        return pl.read_parquet(path)
    df = pull_fn()
    os.makedirs(RAW_DIR, exist_ok=True)
    df.write_parquet(path)
    return df


def pull_player_stats(refresh: bool = False) -> pl.DataFrame:
    """Season-level (regular season) player stats: targets, carries, receptions,
    yards, TDs, games, etc. One row per (player, season) -- nflverse has already
    resolved players who changed teams mid-season to a single `recent_team`."""
    return _cached_pull(
        "player_stats_season.parquet",
        lambda: nfl.load_player_stats(SEASONS, summary_level="reg"),
        refresh,
    )


def pull_rosters(refresh: bool = False) -> pl.DataFrame:
    """Season-level roster snapshot: one row per (player, season), team/position/status
    as of that player's last roster entry for the season, plus birth_date and years_exp.
    This is the base spine for the panel because it includes players who never recorded
    a stat -- player_stats only has rows for players with at least one counting stat."""
    return _cached_pull(
        "rosters_season.parquet",
        lambda: nfl.load_rosters(SEASONS),
        refresh,
    )


def pull_rosters_weekly(refresh: bool = False) -> pl.DataFrame:
    """Weekly roster status (ACT/RES/PUP/etc.), used to compute games_active."""
    return _cached_pull(
        "rosters_weekly.parquet",
        lambda: nfl.load_rosters_weekly(SEASONS),
        refresh,
    )


def pull_snap_counts(refresh: bool = False) -> pl.DataFrame:
    """Weekly offensive/defensive/special-teams snap counts, keyed by pfr_player_id
    (not gsis_id) -- bridged to gsis_id via the players master table in build_panel.py."""
    return _cached_pull(
        "snap_counts.parquet",
        lambda: nfl.load_snap_counts(SEASONS),
        refresh,
    )


def pull_team_stats(refresh: bool = False) -> pl.DataFrame:
    """Season-level (regular season) team offensive/defensive totals, used to derive
    team pass attempts / rush attempts / plays for Step 3's team volume projection."""
    return _cached_pull(
        "team_stats_season.parquet",
        lambda: nfl.load_team_stats(SEASONS, summary_level="reg"),
        refresh,
    )


def pull_players_master(refresh: bool = False) -> pl.DataFrame:
    """nflverse's master player table: one row per player, keyed by gsis_id, with a
    near-complete gsis_id<->pfr_id crosswalk (needed to join snap_counts) and static
    per-player attributes (birth_date, draft_round, draft_pick, draft_year)."""
    return _cached_pull(
        "players_master.parquet",
        lambda: nfl.load_players(),
        refresh,
    )


def pull_schedules(refresh: bool = False) -> pl.DataFrame:
    """Game-level schedule (home/away team per week). Needed because rosters_weekly's
    ACT status persists through a team's bye week -- without the real schedule, a
    games_active count built from ACT weeks alone silently counts the bye week as an
    extra "active" game."""
    return _cached_pull(
        "schedules.parquet",
        lambda: nfl.load_schedules(SEASONS),
        refresh,
    )


def pull_ff_playerids(refresh: bool = False) -> pl.DataFrame:
    """Community-maintained cross-platform fantasy ID file (dynastyprocess-style),
    keyed by gsis_id with a curated `merge_name` field. Used as the primary bridge for
    matching Fantasy Football Calculator ADP rows to gsis_id -- its merge_name already
    handles many nickname/formatting mismatches a naive normalizer would miss."""
    return _cached_pull(
        "ff_playerids.parquet",
        lambda: nfl.load_ff_playerids(),
        refresh,
    )


def pull_ff_rankings(refresh: bool = False) -> pl.DataFrame:
    """Full historical archive of FantasyPros Expert Consensus Rankings (ECR), scraped
    weekly since 2019 and exposed by nflreadpy/ffverse. Used as the source for Step 2's
    baseline_consensus -- pick the snapshot closest before Aug 1 of the target season to
    keep it leakage-safe. Large (~1.8M rows across all ranking types/dates); cached like
    everything else so this only hits the network once."""
    return _cached_pull(
        "ff_rankings_all.parquet",
        lambda: nfl.load_ff_rankings("all"),
        refresh,
    )


def pull_all(refresh: bool = False) -> None:
    pull_player_stats(refresh)
    pull_rosters(refresh)
    pull_rosters_weekly(refresh)
    pull_snap_counts(refresh)
    pull_team_stats(refresh)
    pull_players_master(refresh)
    pull_ff_playerids(refresh)
    pull_schedules(refresh)
    pull_ff_rankings(refresh)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pull and cache raw nflverse data.")
    parser.add_argument(
        "--refresh", action="store_true", help="Re-pull from network even if cached."
    )
    args = parser.parse_args()
    pull_all(refresh=args.refresh)
