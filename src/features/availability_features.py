"""Step 6 of PLAN.md: availability features. games_missed_t1/t2 are the "games missed
in the prior two seasons" PLAN.md names explicitly; age and position come straight from
the panel. All knowable as of Aug 1 of the season being predicted.
"""

import polars as pl

from src.ingest.constants import season_length_expr


def _fill_within_career_gaps(panel: pl.DataFrame) -> pl.DataFrame:
    """gsis_id, season, games_played for every season between each player's first and
    last panel appearance -- including seasons they're entirely absent from the panel
    (out of the league that year: unsigned, retired-and-returned, season-ending injury
    with no roster spot), filled with games_played=0.

    Without this, a player with a within-career gap (verified with Hunter Renfrow, who
    has no 2024 row and returned in 2025) would get games_missed_t1=0 -- "fully
    available" -- for the season he was completely out of football, exactly backwards
    for an availability-risk feature. Only fills gaps *within* a player's known career
    span; doesn't guess at pre-draft or post-retirement seasons, which aren't gaps.
    """
    span = panel.group_by("gsis_id").agg(
        pl.col("season").min().alias("min_season"), pl.col("season").max().alias("max_season")
    )
    full_grid = span.select(
        "gsis_id",
        pl.int_ranges(pl.col("min_season"), pl.col("max_season") + 1).alias("season"),
    ).explode("season")

    real = panel.select("gsis_id", "season", "games_played")
    return full_grid.join(real, on=["gsis_id", "season"], how="left").with_columns(
        pl.col("games_played").fill_null(0)
    )


def build_availability_features(panel: pl.DataFrame) -> pl.DataFrame:
    """gsis_id, season, position, age, games_missed_t1, games_missed_t2,
    games_played_t1 -- predicting season t+1's games_played. 0 for a player with no
    season-t/t-1 row at all (a rookie, or before their first panel appearance): a
    neutral "no known injury history" default, same fill choice used for missing
    prior-season features throughout Steps 4-5, not an attempt to guess at unknown
    history.

    games_played_t1 (season t's raw games played, not derived) is exposed alongside
    games_missed_t1 specifically so a "carried forward" naive baseline can use it
    directly -- deriving it as season_length(t+1) - games_missed_t1 would silently
    mix two different season lengths across the 2020->2021 boundary, when the league
    expanded from 16 to 17 games.
    """
    filled = _fill_within_career_gaps(panel)
    games = filled.with_columns(
        (season_length_expr() - pl.col("games_played")).alias("games_missed")
    ).select("gsis_id", "season", "games_missed", "games_played")

    gm1 = games.with_columns((pl.col("season") + 1).alias("season")).rename(
        {"games_missed": "games_missed_t1", "games_played": "games_played_t1"}
    )
    gm2 = games.with_columns((pl.col("season") + 2).alias("season")).rename(
        {"games_missed": "games_missed_t2"}
    ).drop("games_played")

    out = panel.select("gsis_id", "season", "position", "age")
    out = out.join(gm1, on=["gsis_id", "season"], how="left")
    out = out.join(gm2, on=["gsis_id", "season"], how="left")
    return _fill_availability_feature_nulls(out)


def _fill_availability_feature_nulls(out: pl.DataFrame) -> pl.DataFrame:
    return out.with_columns(
        # games_missed_t1/t2 default to 0 as a neutral "no known injury history" model
        # input (Steps 4-5's rookie-defaults precedent). games_played_t1 separately
        # defaults to 0 for the *naive baseline* only, where a rookie has no true prior
        # season to carry forward -- an intentionally dumb baseline, same spirit as
        # Step 2's baseline_ppg being 0 for rookies. The two aren't meant to agree.
        pl.col("games_missed_t1").fill_null(0),
        pl.col("games_missed_t2").fill_null(0),
        pl.col("games_played_t1").fill_null(0),
    )


def build_future_availability_features(
    panel: pl.DataFrame, future_population: pl.DataFrame
) -> pl.DataFrame:
    """Same columns as build_availability_features, for a live not-yet-played season's
    roster population instead of panel's own rows. games_missed_t1/t2 still come from
    panel (the player's real, already-completed 2025/2024 seasons); age/position come
    from the live population, since a not-yet-played season has no panel row of its own
    to source them from."""
    filled = _fill_within_career_gaps(panel)
    games = filled.with_columns(
        (season_length_expr() - pl.col("games_played")).alias("games_missed")
    ).select("gsis_id", "season", "games_missed", "games_played")

    gm1 = games.with_columns((pl.col("season") + 1).alias("season")).rename(
        {"games_missed": "games_missed_t1", "games_played": "games_played_t1"}
    )
    gm2 = games.with_columns((pl.col("season") + 2).alias("season")).rename(
        {"games_missed": "games_missed_t2"}
    ).drop("games_played")

    out = future_population.select("gsis_id", "season", "position", "age")
    out = out.join(gm1, on=["gsis_id", "season"], how="left")
    out = out.join(gm2, on=["gsis_id", "season"], how="left")
    return _fill_availability_feature_nulls(out)
