"""Step 4 of PLAN.md: usage share features. target_share and rush_share are the
outcome labels this step predicts; everything else here is an input feature for
predicting season t+1, built only from season t and earlier -- knowable as of Aug 1 of
t+1. See build_usage_training_table() for the as-of assertion.

Scope note: team_total_targets/team_total_carries (the share denominators) are summed
across this project's QB/RB/WR/TE panel only, per CLAUDE.md's Step 1 scope decision.
Fullbacks and other positions occasionally draw a target or carry; excluding them means
shares here are computed against a very slightly smaller pie than the true team total.
Negligible in practice (FB usage is a handful of plays a year on most teams) and
consistent with the project's existing position scope, not a new gap.
"""

import polars as pl


def build_shares(panel: pl.DataFrame) -> pl.DataFrame:
    """gsis_id, season, team, position, target_share, rush_share -- the realized shares
    for that player-season. Used both as the training label (for season t+1) and as the
    raw series the EWMA prior-share and vacated-share features are built from (season t
    and earlier)."""
    team_totals = panel.group_by(["team", "season"]).agg(
        pl.col("targets").sum().alias("team_total_targets"),
        pl.col("carries").sum().alias("team_total_carries"),
    )
    out = panel.join(team_totals, on=["team", "season"], how="left")
    out = out.with_columns(
        pl.when(pl.col("team_total_targets") > 0)
        .then(pl.col("targets") / pl.col("team_total_targets"))
        .otherwise(0.0)
        .alias("target_share"),
        pl.when(pl.col("team_total_carries") > 0)
        .then(pl.col("carries") / pl.col("team_total_carries"))
        .otherwise(0.0)
        .alias("rush_share"),
    )
    return out.select("gsis_id", "season", "team", "position", "target_share", "rush_share")


def build_ewma_prior_shares(shares: pl.DataFrame) -> pl.DataFrame:
    """gsis_id, season(=t+1), ewma_target_share, ewma_rush_share -- a 2:1-weighted blend
    of the player's own target_share/rush_share in season t and season t-1 (their most
    recent two seasons of NFL usage, regardless of team -- team-change effects are
    handled by a separate team_change_flag feature, not by discounting here).

    0 for a player with no season-t row at all (a true rookie entering t+1, or a player
    who left the league after t-1): this is a real "no prior usage" signal, not a
    missing-data problem. CLAUDE.md's rookie trap is handled downstream by blending in
    consensus for rookies, not by faking this feature.
    """
    t1 = shares.select(
        "gsis_id", "season", pl.col("target_share").alias("target_share_t1"),
        pl.col("rush_share").alias("rush_share_t1"),
    ).with_columns((pl.col("season") + 1).alias("season"))
    t2 = shares.select(
        "gsis_id", "season", pl.col("target_share").alias("target_share_t2"),
        pl.col("rush_share").alias("rush_share_t2"),
    ).with_columns((pl.col("season") + 2).alias("season"))

    out = t1.join(t2, on=["gsis_id", "season"], how="full", coalesce=True)
    out = out.with_columns(
        # t1 missing (no season-t row) but t2 present is a genuine gap year (e.g. missed
        # a season to injury and dropped off the active roster) -- treat as no-prior-data
        # rather than silently weighting an even-older season at full strength.
        pl.when(pl.col("target_share_t1").is_not_null() & pl.col("target_share_t2").is_not_null())
        .then(2 / 3 * pl.col("target_share_t1") + 1 / 3 * pl.col("target_share_t2"))
        .when(pl.col("target_share_t1").is_not_null())
        .then(pl.col("target_share_t1"))
        .otherwise(0.0)
        .alias("ewma_target_share"),
        pl.when(pl.col("rush_share_t1").is_not_null() & pl.col("rush_share_t2").is_not_null())
        .then(2 / 3 * pl.col("rush_share_t1") + 1 / 3 * pl.col("rush_share_t2"))
        .when(pl.col("rush_share_t1").is_not_null())
        .then(pl.col("rush_share_t1"))
        .otherwise(0.0)
        .alias("ewma_rush_share"),
    )
    # target_share_t1/rush_share_t1 (the single prior season, unblended) are also
    # exposed here -- not used by the EWMA feature itself, but this is the natural place
    # to compute them once, and src/models/usage_share.py uses them as the "naive
    # carryforward" baseline to check the LightGBM model actually adds value.
    out = out.with_columns(
        pl.col("target_share_t1").fill_null(0.0),
        pl.col("rush_share_t1").fill_null(0.0),
    )
    return out.select(
        "gsis_id", "season", "ewma_target_share", "ewma_rush_share",
        "target_share_t1", "rush_share_t1",
    )


def build_vacated_share(shares: pl.DataFrame) -> pl.DataFrame:
    """team, position, season(=t+1), vacated_target_share, vacated_rush_share -- summed
    target_share/rush_share, as of season t, of same-position players who were on the
    team in season t but are on a different team (or out of this panel entirely -- cut,
    retired) in season t+1. Applies identically to every player of that (team, position)
    in season t+1, since it describes an opportunity opening on the roster, not anything
    about a specific player."""
    cur = shares.select("gsis_id", "season", "team", "position", "target_share", "rush_share")
    nxt = shares.select(
        "gsis_id", (pl.col("season") - 1).alias("season"), pl.col("team").alias("team_next")
    )
    merged = cur.join(nxt, on=["gsis_id", "season"], how="left")
    departed = merged.filter(
        pl.col("team_next").is_null() | (pl.col("team_next") != pl.col("team"))
    )
    vacated = departed.group_by(["team", "position", "season"]).agg(
        pl.col("target_share").sum().alias("vacated_target_share"),
        pl.col("rush_share").sum().alias("vacated_rush_share"),
    )
    # `season` here is the departure season (t); the feature describes what's newly
    # open for the roster entering season t+1.
    return vacated.with_columns((pl.col("season") + 1).alias("season"))


def build_snap_share(panel: pl.DataFrame, team_offensive_plays: pl.DataFrame) -> pl.DataFrame:
    """gsis_id, season, snap_share -- player's season_offense_snaps over the team's true
    offensive play count from play-by-play (src/features/team_volume.py's build_pace),
    NOT the box-score team_plays column, which Step 1 already flagged as an undercount
    (excludes penalty no-plays that still count as a snap) that would push shares above
    1.0 for high-snap players."""
    out = panel.select("gsis_id", "season", "team", "season_offense_snaps").join(
        team_offensive_plays.select("team", "season", "offensive_plays"),
        on=["team", "season"],
        how="left",
    )
    return out.with_columns(
        pl.when(pl.col("offensive_plays") > 0)
        .then(pl.col("season_offense_snaps") / pl.col("offensive_plays"))
        .otherwise(0.0)
        .alias("snap_share")
    ).select("gsis_id", "season", "snap_share")
