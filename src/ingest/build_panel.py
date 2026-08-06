"""Build the player-season panel (Step 1 of PLAN.md).

One row per (gsis_id, season), QB/RB/WR/TE, 2013-2025. The roster snapshot is the base
spine -- not player_stats -- specifically so that zero-opportunity rostered players
stay in the panel (a player who was on a 53-man roster all year but never touched the
ball must show up with zeroes, not be silently absent; see CLAUDE.md's zero-inflation
trap). Every numeric stat column is filled to 0 for players who never appear in the
source stat/snap tables, which is a real 0, not missing data.

This panel stores realized, end-of-season outcomes for season t. It is NOT yet a
leakage-safe feature set -- that distinction (features knowable as of Aug 1 of t+1,
built from season-t history) is Step 4's job, in src/features/, not this file's.
"""

import datetime as dt
import os

import polars as pl

from src.ingest.constants import PROCESSED_DIR, RESULTS_DIR, POSITIONS, SEASONS, season_length_expr
from src.ingest.id_matching import build_name_index, match_adp_to_gsis
from src.ingest.pull_raw import (
    pull_ff_playerids,
    pull_player_stats,
    pull_players_master,
    pull_rosters,
    pull_rosters_weekly,
    pull_schedules,
    pull_snap_counts,
    pull_team_stats,
)
from src.ingest.ffc_adp import pull_ffc_adp
from src.ingest.team_codes import canonicalize_team


def _age_as_of_sept1(birth_date: pl.Expr, season: pl.Expr) -> pl.Expr:
    """Age in years as of September 1 of the given season -- the standard convention
    for "age entering the season" in NFL analysis (season kicks off early Sept)."""
    sept1 = pl.date(season, 9, 1)
    return (sept1 - birth_date).dt.total_days() / 365.25


def build_spine(rosters: pl.DataFrame, players_master: pl.DataFrame) -> pl.DataFrame:
    rosters = (
        rosters.filter(pl.col("position").is_in(POSITIONS))
        .filter(pl.col("season").is_in(SEASONS))
        .with_columns(canonicalize_team(pl.col("team")))
        .select(
            "gsis_id",
            "season",
            "team",
            "position",
            "week",
            pl.col("full_name").alias("roster_name"),
            pl.col("birth_date").alias("roster_birth_date"),
            "years_exp",
        )
    )
    # load_rosters(season) gives one row PER TEAM-STINT, not one row per player-season
    # -- a player traded mid-season (e.g. Rob Housler, CLE->CHI in 2015) gets two rows.
    # Keep the highest-week row per (gsis_id, season) as that player's team of record
    # for the season, matching how player_stats.recent_team resolves the same case.
    rosters = rosters.sort(["gsis_id", "season", "week"]).unique(
        subset=["gsis_id", "season"], keep="last", maintain_order=True
    )
    dupes = rosters.group_by("gsis_id", "season").len().filter(pl.col("len") > 1)
    if dupes.height > 0:
        raise ValueError(f"rosters_season still has duplicate (gsis_id, season) rows:\n{dupes}")

    # A handful of practice-/development-squad players (status DEV/RSN/CUT) never
    # receive a gsis_id because they never appeared in an actual NFL game. gsis_id is
    # the join key for every other table in this panel (stats, draft capital, ADP),
    # so these rows are dead ends regardless; they also aren't "zero-opportunity
    # rostered" in PLAN.md's sense, which means the 53-man active roster, not the
    # practice squad. Drop them, loudly.
    null_id = rosters.filter(pl.col("gsis_id").is_null())
    if null_id.height > 0:
        print(
            f"[build_spine] dropping {null_id.height} roster rows with no gsis_id "
            f"(practice/development squad players who never appeared in a game): "
            f"{null_id.select('roster_name', 'season').to_series(0).to_list()}"
        )
        rosters = rosters.filter(pl.col("gsis_id").is_not_null())

    master = players_master.select(
        "gsis_id",
        pl.col("display_name"),
        pl.col("pfr_id"),
        # players_master stores birth_date as an ISO string; rosters_season stores it
        # as a native Date. Parse to Date before coalescing the two or polars upcasts
        # the merged column to String and every downstream date arithmetic breaks.
        pl.col("birth_date").str.to_date("%Y-%m-%d", strict=False).alias("master_birth_date"),
        "draft_year",
        "draft_round",
        "draft_pick",
    )

    spine = rosters.join(master, on="gsis_id", how="left")
    spine = spine.with_columns(
        pl.coalesce(["display_name", "roster_name"]).alias("display_name"),
        pl.coalesce(["master_birth_date", "roster_birth_date"]).alias("birth_date"),
    )
    spine = spine.with_columns(
        _age_as_of_sept1(pl.col("birth_date"), pl.col("season")).alias("age")
    )
    return spine.select(
        "gsis_id",
        "season",
        "team",
        "position",
        "display_name",
        "birth_date",
        "age",
        "years_exp",
        "draft_year",
        "draft_round",
        "draft_pick",
        "pfr_id",
    )


def build_stats(player_stats: pl.DataFrame) -> pl.DataFrame:
    stats = player_stats.filter(pl.col("player_id").is_not_null())
    stats = stats.filter(pl.col("position").is_in(POSITIONS))
    stats = stats.filter(pl.col("season").is_in(SEASONS))

    stat_cols = [
        "games",
        "completions",
        "attempts",
        "passing_yards",
        "passing_tds",
        "passing_interceptions",
        "carries",
        "rushing_yards",
        "rushing_tds",
        "receptions",
        "targets",
        "receiving_yards",
        "receiving_tds",
        "fantasy_points",
        "fantasy_points_ppr",
        # Needed for src/scoring.py's config-driven points calculation (Step 2) --
        # nflreadpy's own fantasy_points_ppr bakes in its own scoring assumptions,
        # which is exactly what CLAUDE.md's "never hardcode league scoring" rule
        # exists to avoid relying on.
        "fumbles_lost_total",
        "passing_2pt_conversions",
        "rushing_2pt_conversions",
        "receiving_2pt_conversions",
    ]
    # Defensive: a player could in principle have more than one reg-season row for a
    # season (e.g. a mid-season position change); sum stats and take the first team.
    stats = stats.group_by(["player_id", "season"]).agg(
        [pl.col(c).sum().alias(c) for c in stat_cols]
        + [pl.col("recent_team").first().alias("recent_team")]
    )
    stats = stats.with_columns(
        (
            pl.col("passing_2pt_conversions")
            + pl.col("rushing_2pt_conversions")
            + pl.col("receiving_2pt_conversions")
        ).alias("two_pt_conversions")
    ).drop(["passing_2pt_conversions", "rushing_2pt_conversions", "receiving_2pt_conversions"])

    return stats.rename(
        {
            "player_id": "gsis_id",
            "games": "games_played",
            "completions": "pass_completions",
            "attempts": "pass_attempts",
            "passing_interceptions": "pass_interceptions",
            "fumbles_lost_total": "fumbles_lost",
        }
    ).with_columns(canonicalize_team(pl.col("recent_team")))


def build_snaps(snap_counts: pl.DataFrame, players_master: pl.DataFrame) -> pl.DataFrame:
    sc = snap_counts.filter(pl.col("game_type") == "REG").filter(
        pl.col("season").is_in(SEASONS)
    )
    bridge = players_master.select("pfr_id", "gsis_id").drop_nulls()
    before = sc.height
    sc = sc.join(bridge, left_on="pfr_player_id", right_on="pfr_id", how="inner")
    dropped = before - sc.height
    if dropped > 0:
        print(
            f"[build_snaps] {dropped}/{before} weekly snap rows had no pfr_id -> gsis_id "
            "bridge and were dropped (mostly players never rostered as QB/RB/WR/TE)."
        )
    return sc.group_by(["gsis_id", "season"]).agg(
        pl.col("offense_snaps").sum().alias("season_offense_snaps")
    )


def build_games_active(rosters_weekly: pl.DataFrame, schedules: pl.DataFrame) -> pl.DataFrame:
    # rosters_weekly's ACT status persists through a team's bye week (verified: Tom
    # Brady, 2013, shows ACT for all 17 weeks of a season NE played only 16 games).
    # Counting ACT weeks directly overcounts games_active by one for every full-season
    # player pre-2021 (17-week season, 16 games). Restrict to weeks the player's team
    # actually had a game.
    sched = schedules.filter(pl.col("game_type") == "REG").filter(
        pl.col("season").is_in(SEASONS)
    )
    team_week_games = pl.concat(
        [
            sched.select("season", "week", pl.col("home_team").alias("team")),
            sched.select("season", "week", pl.col("away_team").alias("team")),
        ]
    ).with_columns(canonicalize_team(pl.col("team"))).unique()

    rw = rosters_weekly.filter(pl.col("game_type") == "REG").filter(
        pl.col("season").is_in(SEASONS)
    ).with_columns(canonicalize_team(pl.col("team")))

    rw_played = rw.filter(pl.col("status") == "ACT").join(
        team_week_games, on=["season", "week", "team"], how="inner"
    )
    return rw_played.group_by(["gsis_id", "season"]).agg(
        pl.col("week").n_unique().alias("games_active")
    )


def build_team_totals(team_stats: pl.DataFrame) -> pl.DataFrame:
    ts = team_stats.filter(pl.col("season").is_in(SEASONS)).with_columns(
        canonicalize_team(pl.col("team"))
    )
    ts = ts.with_columns(
        (pl.col("attempts") + pl.col("sacks_suffered")).alias("team_pass_attempts"),
        pl.col("carries").alias("team_rush_attempts"),
    )
    # team_plays is scrimmage plays from the box score (pass attempts + sacks + rush
    # attempts) and will systematically undercount true offensive snaps by ~20-60 per
    # season -- it excludes QB kneels and pre-snap-penalty no-plays, which still count
    # as a snap. Verified: every full-snap QB and iron-man WR/TE in the panel has
    # season_offense_snaps > team_plays for their own team. Fine as "plays" for Step 3's
    # team volume projection; do NOT use this as the denominator for a Step 4
    # snap_share feature without accounting for the gap, or shares will exceed 1.0.
    ts = ts.with_columns(
        (pl.col("team_pass_attempts") + pl.col("team_rush_attempts")).alias("team_plays")
    )
    return ts.select("season", "team", "team_pass_attempts", "team_rush_attempts", "team_plays")


def build_adp(
    ffc_adp: pl.DataFrame, ff_playerids: pl.DataFrame, spine: pl.DataFrame
) -> tuple[pl.DataFrame, pl.DataFrame]:
    ffc = (
        ffc_adp.filter(pl.col("position").is_in(POSITIONS))
        .filter(pl.col("season").is_in(SEASONS))
        .with_columns(canonicalize_team(pl.col("team")))
    )
    name_index = build_name_index(
        spine.select("gsis_id", "season", "position", "team", "display_name")
    )
    matched, unmatched = match_adp_to_gsis(ffc, ff_playerids, name_index)
    adp = matched.select(
        "gsis_id",
        "season",
        pl.col("adp").alias("ffc_adp"),
        pl.col("adp_formatted").alias("ffc_adp_formatted"),
        pl.col("times_drafted").alias("ffc_times_drafted"),
        pl.col("stdev").alias("ffc_adp_stdev"),
    )
    return adp, unmatched


def build_panel(refresh: bool = False) -> pl.DataFrame:
    rosters = pull_rosters(refresh)
    rosters_weekly = pull_rosters_weekly(refresh)
    players_master = pull_players_master(refresh)
    player_stats = pull_player_stats(refresh)
    snap_counts = pull_snap_counts(refresh)
    team_stats = pull_team_stats(refresh)
    ff_playerids = pull_ff_playerids(refresh)
    ffc_adp = pull_ffc_adp(refresh)
    schedules = pull_schedules(refresh)

    spine = build_spine(rosters, players_master)
    stats = build_stats(player_stats)
    snaps = build_snaps(snap_counts, players_master)
    games_active = build_games_active(rosters_weekly, schedules)
    team_totals = build_team_totals(team_stats)
    adp, unmatched_adp = build_adp(ffc_adp, ff_playerids, spine)

    panel = spine.join(stats, on=["gsis_id", "season"], how="left")
    # team: prefer where the player actually accrued stats (handles in-season trades
    # more meaningfully than the roster snapshot's final-week team); fall back to the
    # roster snapshot for zero-opportunity players who never appear in player_stats.
    panel = panel.with_columns(pl.coalesce(["recent_team", "team"]).alias("team")).drop(
        "recent_team"
    )

    panel = panel.join(snaps, on=["gsis_id", "season"], how="left")
    panel = panel.join(games_active, on=["gsis_id", "season"], how="left")
    panel = panel.join(team_totals, on=["season", "team"], how="left")
    panel = panel.join(adp, on=["gsis_id", "season"], how="left")

    zero_fill_cols = [
        "games_played",
        "pass_completions",
        "pass_attempts",
        "passing_yards",
        "passing_tds",
        "pass_interceptions",
        "carries",
        "rushing_yards",
        "rushing_tds",
        "receptions",
        "targets",
        "receiving_yards",
        "receiving_tds",
        "fantasy_points",
        "fantasy_points_ppr",
        "fumbles_lost",
        "two_pt_conversions",
        "season_offense_snaps",
        "games_active",
    ]
    panel = panel.with_columns([pl.col(c).fill_null(0) for c in zero_fill_cols])

    # Defensive clip: nflreadpy's season-level `games` column has at least one known
    # bad value (Rashid Shaheed, 2025, games=18 in an 18-week/17-game season --
    # confirmed present in the raw source table itself, not introduced by this
    # pipeline). A single-season regular-season schedule is 16 games through 2020 and
    # 17 from 2021 on; clip both games_played and games_active to that ceiling so one
    # upstream data glitch can't silently corrupt the Step 6 availability model.
    max_games = season_length_expr()
    over_cap = panel.filter(
        (pl.col("games_played") > max_games) | (pl.col("games_active") > max_games)
    )
    if over_cap.height > 0:
        print(
            f"[build_panel] clipping games_played/games_active to the season's game "
            f"count for {over_cap.height} row(s): "
            f"{over_cap.select('display_name', 'season', 'games_played', 'games_active').rows()}"
        )
    panel = panel.with_columns(
        pl.min_horizontal("games_played", max_games).alias("games_played"),
        pl.min_horizontal("games_active", max_games).alias("games_active"),
    )

    missing_team_totals = panel.filter(pl.col("team_plays").is_null())
    if missing_team_totals.height > 0:
        raise ValueError(
            "Some player-seasons joined to no team-level totals -- likely a team code "
            f"mismatch. Examples:\n{missing_team_totals.select('season', 'team').unique().head(10)}"
        )

    panel = panel.sort(["season", "position", "team", "display_name"])

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    panel.write_parquet(os.path.join(PROCESSED_DIR, "player_season_panel.parquet"))

    os.makedirs(RESULTS_DIR, exist_ok=True)
    unmatched_adp.sort(["season", "adp"]).write_csv(
        os.path.join(RESULTS_DIR, "unmatched_adp_players.csv")
    )
    if unmatched_adp.height > 0:
        print(
            f"[build_adp] {unmatched_adp.height} FFC ADP rows (of {unmatched_adp.height + adp.height}) "
            f"could not be matched to a gsis_id -- see results/unmatched_adp_players.csv"
        )

    return panel


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build the player-season panel.")
    parser.add_argument("--refresh", action="store_true", help="Re-pull raw data from network.")
    args = parser.parse_args()

    result = build_panel(refresh=args.refresh)
    print(f"Panel built: {result.height} rows, {result.width} columns.")
    print(f"Seasons: {result.select('season').min().item()}-{result.select('season').max().item()}")
    print(result.group_by("position").len().sort("position"))
