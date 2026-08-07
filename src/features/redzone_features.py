"""Red-zone share and aDOT (average depth of target) features -- added to give
the usage-share models a signal beyond total target/rush share, which treats a
5-yard checkdown and a 20-yard shot (or a red-zone look and a garbage-time one)
identically. Both are built from play-by-play (yardline_100, air_yards), which
the panel and player_stats-based features don't carry.

Same as-of-Aug-1 discipline as every other feature here: all of it is built
from season t and earlier, shifted forward, never season t+1's own outcome.
"""

import polars as pl

RED_ZONE_YARDLINE_MAX = 20

RED_ZONE_SHARE_STATS = {
    "red_zone_target_share": "redzone_targets",
    "red_zone_rush_share": "redzone_carries",
}


def build_redzone_counts(pbp: pl.DataFrame) -> pl.DataFrame:
    """gsis_id, season, redzone_targets, redzone_carries -- counted from plays
    with yardline_100 <= 20 (distance to the end zone), the standard red-zone
    definition. A target is any pass play with a recorded receiver, regardless
    of completion -- matching how targets are counted everywhere else in this
    project (Step 1's panel, Step 4's target_share)."""
    rz = pbp.filter(pl.col("yardline_100") <= RED_ZONE_YARDLINE_MAX)

    targets = (
        rz.filter((pl.col("pass") == 1) & pl.col("receiver_player_id").is_not_null())
        .group_by(["receiver_player_id", "season"])
        .agg(pl.len().alias("redzone_targets"))
        .rename({"receiver_player_id": "gsis_id"})
    )
    carries = (
        rz.filter((pl.col("rush") == 1) & pl.col("rusher_player_id").is_not_null())
        .group_by(["rusher_player_id", "season"])
        .agg(pl.len().alias("redzone_carries"))
        .rename({"rusher_player_id": "gsis_id"})
    )
    return targets.join(carries, on=["gsis_id", "season"], how="full", coalesce=True).with_columns(
        pl.col("redzone_targets").fill_null(0), pl.col("redzone_carries").fill_null(0)
    )


def build_redzone_shares(panel: pl.DataFrame, pbp: pl.DataFrame) -> pl.DataFrame:
    """gsis_id, season, team, position, plus red_zone_target_share and
    red_zone_rush_share -- each player's share of their TEAM's red-zone
    targets/carries that season. Denominator is summed across this project's
    QB/RB/WR/TE scope, same convention as Step 4's build_shares."""
    counts = build_redzone_counts(pbp)
    base = panel.select("gsis_id", "season", "team", "position").join(
        counts, on=["gsis_id", "season"], how="left"
    ).with_columns(pl.col("redzone_targets").fill_null(0), pl.col("redzone_carries").fill_null(0))

    team_totals = base.group_by(["team", "season"]).agg(
        pl.col("redzone_targets").sum().alias("team_total_redzone_targets"),
        pl.col("redzone_carries").sum().alias("team_total_redzone_carries"),
    )
    out = base.join(team_totals, on=["team", "season"], how="left")
    return out.with_columns(
        pl.when(pl.col("team_total_redzone_targets") > 0)
        .then(pl.col("redzone_targets") / pl.col("team_total_redzone_targets"))
        .otherwise(0.0)
        .alias("red_zone_target_share"),
        pl.when(pl.col("team_total_redzone_carries") > 0)
        .then(pl.col("redzone_carries") / pl.col("team_total_redzone_carries"))
        .otherwise(0.0)
        .alias("red_zone_rush_share"),
    ).select(
        "gsis_id", "season", "team", "position", "red_zone_target_share", "red_zone_rush_share"
    )


def build_ewma_redzone_shares(redzone_shares: pl.DataFrame) -> pl.DataFrame:
    """gsis_id, season(=t+1), ewma_red_zone_target_share, ewma_red_zone_rush_share
    -- 2:1-weighted blend of season t and t-1, mirroring Step 4's
    build_ewma_prior_shares exactly (same rationale: a player's own red-zone
    role over their last two seasons, regardless of team)."""
    t1 = redzone_shares.select(
        "gsis_id", "season",
        pl.col("red_zone_target_share").alias("rz_target_share_t1"),
        pl.col("red_zone_rush_share").alias("rz_rush_share_t1"),
    ).with_columns((pl.col("season") + 1).alias("season"))
    t2 = redzone_shares.select(
        "gsis_id", "season",
        pl.col("red_zone_target_share").alias("rz_target_share_t2"),
        pl.col("red_zone_rush_share").alias("rz_rush_share_t2"),
    ).with_columns((pl.col("season") + 2).alias("season"))

    out = t1.join(t2, on=["gsis_id", "season"], how="full", coalesce=True)
    for stat in ["target", "rush"]:
        t1_col, t2_col = f"rz_{stat}_share_t1", f"rz_{stat}_share_t2"
        out = out.with_columns(
            pl.when(pl.col(t1_col).is_not_null() & pl.col(t2_col).is_not_null())
            .then(2 / 3 * pl.col(t1_col) + 1 / 3 * pl.col(t2_col))
            .when(pl.col(t1_col).is_not_null())
            .then(pl.col(t1_col))
            .otherwise(0.0)
            .alias(f"ewma_red_zone_{stat}_share")
        )
    return out.select(
        "gsis_id", "season", "ewma_red_zone_target_share", "ewma_red_zone_rush_share"
    )


def build_season_adot(pbp: pl.DataFrame) -> pl.DataFrame:
    """gsis_id, season, adot, adot_targets -- average depth of target (mean
    air_yards across every target with a recorded air_yards value that
    season), plus the target count it's averaged over (needed downstream to
    weight a career-cumulative average correctly). ~15.6% of pass plays have
    null air_yards (spikes, some incompletions with no charted depth) --
    excluded from both the average and the target count, not treated as 0
    (an uncharted depth is missing data, not a zero-yard target)."""
    targets = pbp.filter((pl.col("pass") == 1) & pl.col("receiver_player_id").is_not_null())
    charted = targets.filter(pl.col("air_yards").is_not_null())
    return charted.group_by(["receiver_player_id", "season"]).agg(
        pl.col("air_yards").mean().alias("adot"),
        pl.len().alias("adot_targets"),
    ).rename({"receiver_player_id": "gsis_id"})


def build_career_adot(pbp: pl.DataFrame) -> pl.DataFrame:
    """gsis_id, season, career_adot -- cumulative-through-`season` average
    depth of target, weighted by each season's target count (not a naive
    average of season-level aDOTs, which would let a 3-target cup-of-coffee
    season count as much as a 120-target starter season). Callers shift
    season+1 to use as a feature for predicting the next season, same pattern
    as Step 5's build_career_cumulative."""
    season_adot = build_season_adot(pbp).sort(["gsis_id", "season"])
    cum = season_adot.with_columns(
        (pl.col("adot") * pl.col("adot_targets")).alias("_air_yards_sum")
    ).with_columns(
        pl.col("_air_yards_sum").cum_sum().over("gsis_id").alias("career_air_yards_sum"),
        pl.col("adot_targets").cum_sum().over("gsis_id").alias("career_adot_targets"),
    )
    return cum.with_columns(
        (pl.col("career_air_yards_sum") / pl.col("career_adot_targets")).alias("career_adot")
    ).select("gsis_id", "season", "career_adot", "career_adot_targets")
