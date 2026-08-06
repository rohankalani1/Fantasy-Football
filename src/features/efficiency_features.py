"""Step 5 of PLAN.md: efficiency features. Career-to-date cumulative opportunity counts
and rates -- NOT recency-weighted like Step 4's usage-share EWMA. The entire point of
empirical Bayes shrinkage here is that more career opportunities means a more reliable
estimate of a player's true rate, so pooling the full career (not just recent seasons)
is deliberate, not an oversight.

Extended in Step 7 (src/models/combine.py) with catch_rate (receptions/target -- PPR
scoring needs it and Step 5 originally didn't model it) and three QB passing rate stats
(ypa, pass_td_rate, int_rate -- passing production wasn't part of the original Steps 4-5
design at all, an architecture gap only visible once Step 7 tried to combine everything
into points). Both build_career_cumulative and build_season_rates are driven directly by
RATE_STATS so adding a stat here doesn't require hand-editing column lists in two places.
"""

import polars as pl

RATE_STATS = {
    "ypt": ("targets", "receiving_yards"),
    "td_rate_receiving": ("targets", "receiving_tds"),
    "catch_rate": ("targets", "receptions"),
    "ypc": ("carries", "rushing_yards"),
    "td_rate_rushing": ("carries", "rushing_tds"),
    "ypa": ("pass_attempts", "passing_yards"),
    "pass_td_rate": ("pass_attempts", "passing_tds"),
    "int_rate": ("pass_attempts", "pass_interceptions"),
}


def _opportunity_and_numerator_cols() -> list[str]:
    cols = set()
    for opp_col, num_col in RATE_STATS.values():
        cols.add(opp_col)
        cols.add(num_col)
    return sorted(cols)


def build_career_cumulative(panel: pl.DataFrame) -> pl.DataFrame:
    """gsis_id, season, plus career_{col} for every opportunity/numerator column
    RATE_STATS references -- cumulative through `season` inclusive ("career as of the
    end of this season"). Callers shift season+1 to use as a feature for predicting the
    next season, same pattern as every other career/prior-season feature in this
    project.
    """
    cols = _opportunity_and_numerator_cols()
    sorted_panel = panel.sort(["gsis_id", "season"])
    out = sorted_panel.with_columns(
        [pl.col(c).cum_sum().over("gsis_id").alias(f"career_{c}") for c in cols]
    )
    return out.select("gsis_id", "season", *[f"career_{c}" for c in cols])


def build_season_rates(panel: pl.DataFrame) -> pl.DataFrame:
    """gsis_id, season, position, opportunity columns, plus every RATE_STATS rate
    (null, not 0, when the player had 0 opportunities that season -- an undefined rate
    is not the same as a 0 rate, and averaging in a fake 0 would bias every downstream
    mean). Used both as training labels and as the raw per-season observations the
    empirical Bayes prior-fitting procedure (src/models/efficiency.py) needs.
    """
    opp_cols = sorted({opp_col for opp_col, _ in RATE_STATS.values()})
    rate_exprs = [
        pl.when(pl.col(opp_col) > 0)
        .then(pl.col(num_col) / pl.col(opp_col))
        .alias(stat)
        for stat, (opp_col, num_col) in RATE_STATS.items()
    ]
    return panel.select("gsis_id", "season", "position", *opp_cols, *rate_exprs)
