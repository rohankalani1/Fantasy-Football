"""Step 3 of PLAN.md: team-season input features for the team volume model --
pace, pass-rate-over-expected (PROE), and head-coach-change flag, all built from raw
play-by-play data.

Every function here returns one row per (team, season) using only that season's own
completed data -- these become trailing (t-1, t-2) features when assembled into the
training table in src/models/team_volume.py, which is where the actual "knowable as of
Aug 1" assertion belongs (it depends on which season is being predicted, not on how
these per-season summaries are computed).
"""

import polars as pl

from src.ingest.constants import SEASONS
from src.ingest.team_codes import canonicalize_team

# Offensive scrimmage plays only -- excludes kickoffs/punts/FG/XP (special teams reps,
# not offensive tempo) and qb_kneel/qb_spike (clock-killing, not representative of a
# team's real pace). This is the standard "pace" convention used by public tempo
# analytics (e.g. rbsdm, Football Outsiders), not nflfastR's own special_teams_play
# flag, which inconsistently tags field goals as non-special-teams.
_PACE_PLAY_TYPES = ["pass", "run", "no_play"]


def build_pace(pbp: pl.DataFrame) -> pl.DataFrame:
    """gsis-team-season pace: offensive scrimmage plays per game."""
    reg = pbp.filter(
        (pl.col("season_type") == "REG")
        & pl.col("posteam").is_not_null()
        & pl.col("play_type").is_in(_PACE_PLAY_TYPES)
    ).with_columns(canonicalize_team(pl.col("posteam")).alias("team"))

    per_team_season = reg.group_by(["team", "season"]).agg(
        pl.len().alias("offensive_plays"),
        pl.col("game_id").n_unique().alias("games"),
    )
    return per_team_season.with_columns(
        (pl.col("offensive_plays") / pl.col("games")).alias("pace")
    ).select("team", "season", "pace", "games", "offensive_plays")


def build_proe(pbp: pl.DataFrame) -> pl.DataFrame:
    """Team-season PROE: mean pass_oe (nflfastR's own expected-pass model output) over
    plays where it's defined -- already restricted to genuine pass/run decisions
    (excludes kneels, spikes, special teams, and most penalty-negated no-plays)."""
    reg = pbp.filter(
        (pl.col("season_type") == "REG") & pl.col("pass_oe").is_not_null()
    ).with_columns(canonicalize_team(pl.col("posteam")).alias("team"))

    return reg.group_by(["team", "season"]).agg(pl.col("pass_oe").mean().alias("proe"))


def build_hc_change(pbp: pl.DataFrame) -> pl.DataFrame:
    """Team-season head coach (Week 1 of record) and a flag for whether it differs from
    the prior season's HC. Flag is 0 (not 1) when there's no prior-season row in this
    panel -- 2013 is the first season covered, so this only affects 2013 rows, which are
    never used to predict an even-earlier season anyway."""
    reg = pbp.filter(pl.col("season_type") == "REG")
    home = reg.select(
        "season", "week", pl.col("home_team").alias("team"), pl.col("home_coach").alias("coach")
    )
    away = reg.select(
        "season", "week", pl.col("away_team").alias("team"), pl.col("away_coach").alias("coach")
    )
    coaches = pl.concat([home, away]).with_columns(canonicalize_team(pl.col("team"))).drop_nulls(
        "coach"
    )
    # Week 1 coach is "the" HC of record for the season -- knowable as of Aug 1, unlike
    # a coach installed after an in-season firing.
    week1 = (
        coaches.sort(["team", "season", "week"])
        .unique(subset=["team", "season"], keep="first")
        .select("team", "season", "coach")
    )

    prior = week1.select("team", "season", pl.col("coach").alias("prior_coach")).with_columns(
        (pl.col("season") + 1).alias("season")
    )
    out = week1.join(prior, on=["team", "season"], how="left")
    return out.with_columns(
        pl.when(pl.col("prior_coach").is_not_null())
        .then((pl.col("coach") != pl.col("prior_coach")).cast(pl.Int8))
        .otherwise(0)
        .alias("hc_change_flag")
    ).select("team", "season", "coach", "hc_change_flag")


def build_team_volume_features(pbp: pl.DataFrame) -> pl.DataFrame:
    pbp = pbp.filter(pl.col("season").is_in(SEASONS))
    pace = build_pace(pbp)
    proe = build_proe(pbp)
    hc = build_hc_change(pbp)

    out = pace.join(proe, on=["team", "season"], how="left")
    out = out.join(hc, on=["team", "season"], how="left")
    return out.sort(["season", "team"])
