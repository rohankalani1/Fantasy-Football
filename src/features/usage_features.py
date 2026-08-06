"""Step 4 of PLAN.md: usage share features. target_share and rush_share are the
outcome labels this step predicts; everything else here is an input feature for
predicting season t+1, built only from season t and earlier -- knowable as of Aug 1 of
t+1. See build_usage_training_table() for the as-of assertion.

Extended in Step 7 (src/models/combine.py) with passer_share (a QB's share of team pass
attempts) -- passing volume wasn't part of the original Step 4 design at all, an
architecture gap only visible once Step 7 tried to combine everything into points.
SHARE_STATS drives build_shares/build_ewma_prior_shares/build_vacated_share generically
so passer_share reuses the exact same logic as target_share/rush_share rather than a
third hand-copied block.

Scope note: the share denominators are summed across this project's QB/RB/WR/TE panel
only, per CLAUDE.md's Step 1 scope decision. Fullbacks and other positions occasionally
draw a target or carry; excluding them means shares here are computed against a very
slightly smaller pie than the true team total. Negligible in practice (FB usage is a
handful of plays a year on most teams) and consistent with the project's existing
position scope, not a new gap.
"""

import polars as pl

SHARE_STATS = {
    "target_share": "targets",
    "rush_share": "carries",
    "passer_share": "pass_attempts",
}


def build_shares(panel: pl.DataFrame) -> pl.DataFrame:
    """gsis_id, season, team, position, plus every SHARE_STATS share -- the realized
    shares for that player-season. Used both as the training label (for season t+1) and
    as the raw series the EWMA prior-share and vacated-share features are built from
    (season t and earlier)."""
    numerator_cols = sorted(set(SHARE_STATS.values()))
    team_totals = panel.group_by(["team", "season"]).agg(
        [pl.col(c).sum().alias(f"team_total_{c}") for c in numerator_cols]
    )
    out = panel.join(team_totals, on=["team", "season"], how="left")
    out = out.with_columns(
        [
            pl.when(pl.col(f"team_total_{num_col}") > 0)
            .then(pl.col(num_col) / pl.col(f"team_total_{num_col}"))
            .otherwise(0.0)
            .alias(share)
            for share, num_col in SHARE_STATS.items()
        ]
    )
    return out.select("gsis_id", "season", "team", "position", *SHARE_STATS.keys())


def build_ewma_prior_shares(shares: pl.DataFrame) -> pl.DataFrame:
    """gsis_id, season(=t+1), plus ewma_{share} and {share}_t1 for every SHARE_STATS
    share -- a 2:1-weighted blend of the player's own share in season t and season t-1
    (their most recent two seasons of NFL usage, regardless of team -- team-change
    effects are handled by a separate team_change_flag feature, not by discounting
    here). {share}_t1 (the single prior season, unblended) is also exposed -- not used
    by the EWMA feature itself, but src/models/usage_share.py uses it as the "naive
    carryforward" baseline to check the LightGBM model actually adds value.

    0 for a player with no season-t row at all (a true rookie entering t+1, or a player
    who left the league after t-1): this is a real "no prior usage" signal, not a
    missing-data problem. CLAUDE.md's rookie trap is handled downstream by blending in
    consensus for rookies, not by faking this feature.
    """
    t1 = shares.select(
        "gsis_id", "season", *[pl.col(s).alias(f"{s}_t1") for s in SHARE_STATS]
    ).with_columns((pl.col("season") + 1).alias("season"))
    t2 = shares.select(
        "gsis_id", "season", *[pl.col(s).alias(f"{s}_t2") for s in SHARE_STATS]
    ).with_columns((pl.col("season") + 2).alias("season"))

    out = t1.join(t2, on=["gsis_id", "season"], how="full", coalesce=True)
    ewma_exprs = []
    for s in SHARE_STATS:
        t1_col, t2_col = f"{s}_t1", f"{s}_t2"
        # t1 missing (no season-t row) but t2 present is a genuine gap year (e.g.
        # missed a season to injury and dropped off the active roster) -- treat as
        # no-prior-data rather than silently weighting an even-older season at full
        # strength.
        ewma_exprs.append(
            pl.when(pl.col(t1_col).is_not_null() & pl.col(t2_col).is_not_null())
            .then(2 / 3 * pl.col(t1_col) + 1 / 3 * pl.col(t2_col))
            .when(pl.col(t1_col).is_not_null())
            .then(pl.col(t1_col))
            .otherwise(0.0)
            .alias(f"ewma_{s}")
        )
    out = out.with_columns(ewma_exprs)
    out = out.with_columns([pl.col(f"{s}_t1").fill_null(0.0) for s in SHARE_STATS])

    return out.select(
        "gsis_id", "season",
        *[f"ewma_{s}" for s in SHARE_STATS],
        *[f"{s}_t1" for s in SHARE_STATS],
    )


def build_vacated_share(shares: pl.DataFrame) -> pl.DataFrame:
    """team, position, season(=t+1), plus vacated_{share} for every SHARE_STATS share --
    summed share, as of season t, of same-position players who were on the team in
    season t but are on a different team (or out of this panel entirely -- cut, retired)
    in season t+1. Applies identically to every player of that (team, position) in
    season t+1, since it describes an opportunity opening on the roster, not anything
    about a specific player."""
    cur = shares.select("gsis_id", "season", "team", "position", *SHARE_STATS.keys())
    nxt = shares.select(
        "gsis_id", (pl.col("season") - 1).alias("season"), pl.col("team").alias("team_next")
    )
    merged = cur.join(nxt, on=["gsis_id", "season"], how="left")
    departed = merged.filter(
        pl.col("team_next").is_null() | (pl.col("team_next") != pl.col("team"))
    )
    vacated = departed.group_by(["team", "position", "season"]).agg(
        [pl.col(s).sum().alias(f"vacated_{s}") for s in SHARE_STATS]
    )
    # `season` here is the departure season (t); the feature describes what's newly
    # open for the roster entering season t+1.
    return vacated.with_columns((pl.col("season") + 1).alias("season"))


def build_future_vacated_share(
    shares: pl.DataFrame, future_population: pl.DataFrame, departure_season: int
) -> pl.DataFrame:
    """Same definition as build_vacated_share, for the live not-yet-played-season entry
    point: "team in departure_season+1" comes from the live current-season roster
    population instead of a historical shares row, since a not-yet-played season has no
    row in `shares` to look up."""
    cur = shares.filter(pl.col("season") == departure_season).select(
        "gsis_id", "team", "position", *SHARE_STATS.keys()
    )
    next_team = future_population.select("gsis_id", pl.col("team").alias("team_next"))
    merged = cur.join(next_team, on="gsis_id", how="left")
    departed = merged.filter(
        pl.col("team_next").is_null() | (pl.col("team_next") != pl.col("team"))
    )
    vacated = departed.group_by(["team", "position"]).agg(
        [pl.col(s).sum().alias(f"vacated_{s}") for s in SHARE_STATS]
    )
    return vacated.with_columns(pl.lit(departure_season + 1).alias("season"))


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
