"""Live player population for a not-yet-played season (2026) -- the prerequisite Steps
4-6 never needed historically, since the panel only ever covered seasons that had
already been played. Every model's "predict for the future" entry point predicts FOR
this population, not for panel's own rows.

Mirrors src/ingest/build_panel.py's build_spine() (team/position/age/draft-capital
spine), but for a single live season pulled outside the frozen historical SEASONS
window, and with one necessary difference: nflreadpy's players_master file lags the
current year's draft class (verified directly: 0 rows with draft_year==2026 as of this
writing), so draft_round/draft_pick for brand-new rookies falls back to the roster
pull's own `draft_number` (overall pick), which IS current. Every other player (anyone
with at least one prior season) still gets their real draft_round/draft_pick from
players_master, unchanged.
"""

import polars as pl

from src.ingest.constants import POSITIONS
from src.ingest.pull_raw import pull_players_master, pull_rosters_for_season
from src.ingest.team_codes import canonicalize_team


def _age_as_of_sept1(birth_date: pl.Expr, season: int) -> pl.Expr:
    sept1 = pl.date(season, 9, 1)
    return (sept1 - birth_date).dt.total_days() / 365.25


def _draft_round_from_overall_pick(pick: pl.Expr) -> pl.Expr:
    """Modern NFL drafts run 32 picks/round with compensatory picks appended at the end
    of rounds 3-7, so this slightly overstates round for a late compensatory pick in
    practice -- an approximation, not exact, but draft_round is a shallow-tree secondary
    signal alongside the exact draft_pick number, not the sole draft-capital feature."""
    return ((pick - 1) // 32 + 1).clip(upper_bound=7)


def build_future_population(season: int, refresh: bool = False) -> pl.DataFrame:
    """gsis_id, season, team, position, display_name, age, years_exp, draft_round,
    draft_pick for every QB/RB/WR/TE on a season's live roster snapshot. `season` is a
    not-yet-played season (2026), so this is NOT drawn from the historical panel -- it's
    the live "who do we predict for" population every future-season model function
    joins onto.
    """
    rosters = pull_rosters_for_season(season, refresh)
    rosters = rosters.filter(pl.col("position").is_in(POSITIONS)).with_columns(
        canonicalize_team(pl.col("team"))
    )

    null_id = rosters.filter(pl.col("gsis_id").is_null())
    if null_id.height > 0:
        print(
            f"[build_future_population] dropping {null_id.height} roster row(s) with no "
            f"gsis_id (no cross-platform ID yet): "
            f"{null_id.select('full_name').to_series().to_list()}"
        )
        rosters = rosters.filter(pl.col("gsis_id").is_not_null())

    dupes = rosters.group_by("gsis_id").len().filter(pl.col("len") > 1)
    if dupes.height > 0:
        raise ValueError(f"rosters_{season} has duplicate gsis_id rows:\n{dupes}")

    master = pull_players_master(refresh).select("gsis_id", "draft_round", "draft_pick")
    out = rosters.join(master, on="gsis_id", how="left")

    out = out.with_columns(
        pl.when(pl.col("draft_round").is_null() & pl.col("draft_number").is_not_null())
        .then(_draft_round_from_overall_pick(pl.col("draft_number")))
        .otherwise(pl.col("draft_round"))
        .alias("draft_round"),
        pl.when(pl.col("draft_pick").is_null() & pl.col("draft_number").is_not_null())
        .then(pl.col("draft_number"))
        .otherwise(pl.col("draft_pick"))
        .alias("draft_pick"),
        _age_as_of_sept1(pl.col("birth_date"), season).alias("age"),
        pl.lit(season).alias("season"),
    )

    # A handful of camp bodies (verified: 18/914 QB/RB/WR/TE rows) have no birth_date
    # yet in nflreadpy's live roster feed -- a null age would otherwise NaN out every
    # downstream model for that row (age is a required, non-droppable feature in both
    # the availability and usage-share models). Filling with that row's own
    # position's median age within this same live population is a neutral "typical
    # player at this position" default, same spirit as this project's other
    # no-real-signal-available fallbacks -- these are deep-roster longshots, not
    # players whose exact age materially changes the board.
    out = out.with_columns(
        pl.col("age").fill_null(pl.col("age").median().over("position"))
    )

    return out.select(
        "gsis_id", "season", "team", "position",
        pl.col("full_name").alias("display_name"),
        "age", "years_exp", "draft_round", "draft_pick",
    ).sort(["position", "team", "display_name"])
