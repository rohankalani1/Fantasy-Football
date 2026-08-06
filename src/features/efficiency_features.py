"""Step 5 of PLAN.md: efficiency features. Career-to-date cumulative opportunity counts
and rates -- NOT recency-weighted like Step 4's usage-share EWMA. The entire point of
empirical Bayes shrinkage here is that more career opportunities means a more reliable
estimate of a player's true rate, so pooling the full career (not just recent seasons)
is deliberate, not an oversight.
"""

import polars as pl

RATE_STATS = {
    "ypt": ("targets", "receiving_yards"),
    "td_rate_receiving": ("targets", "receiving_tds"),
    "ypc": ("carries", "rushing_yards"),
    "td_rate_rushing": ("carries", "rushing_tds"),
}


def build_career_cumulative(panel: pl.DataFrame) -> pl.DataFrame:
    """gsis_id, season, plus career_{targets,receiving_yards,receiving_tds,carries,
    rushing_yards,rushing_tds} -- cumulative through `season` inclusive ("career as of
    the end of this season"). Callers shift season+1 to use as a feature for predicting
    the next season, same pattern as every other career/prior-season feature in this
    project.
    """
    sorted_panel = panel.sort(["gsis_id", "season"])
    out = sorted_panel.with_columns(
        pl.col("targets").cum_sum().over("gsis_id").alias("career_targets"),
        pl.col("receiving_yards").cum_sum().over("gsis_id").alias("career_receiving_yards"),
        pl.col("receiving_tds").cum_sum().over("gsis_id").alias("career_receiving_tds"),
        pl.col("carries").cum_sum().over("gsis_id").alias("career_carries"),
        pl.col("rushing_yards").cum_sum().over("gsis_id").alias("career_rushing_yards"),
        pl.col("rushing_tds").cum_sum().over("gsis_id").alias("career_rushing_tds"),
    )
    return out.select(
        "gsis_id", "season", "career_targets", "career_receiving_yards", "career_receiving_tds",
        "career_carries", "career_rushing_yards", "career_rushing_tds",
    )


def build_season_rates(panel: pl.DataFrame) -> pl.DataFrame:
    """gsis_id, season, position, plus the 4 realized single-season rate stats (null,
    not 0, when the player had 0 opportunities that season -- an undefined rate is not
    the same as a 0 rate, and averaging in a fake 0 would bias every downstream mean).
    Used both as training labels and as the raw per-season observations the empirical
    Bayes prior-fitting procedure (src/models/efficiency.py) needs.
    """
    return panel.select(
        "gsis_id", "season", "position", "targets", "carries",
        pl.when(pl.col("targets") > 0).then(pl.col("receiving_yards") / pl.col("targets")).alias("ypt"),
        pl.when(pl.col("targets") > 0).then(pl.col("receiving_tds") / pl.col("targets")).alias("td_rate_receiving"),
        pl.when(pl.col("carries") > 0).then(pl.col("rushing_yards") / pl.col("carries")).alias("ypc"),
        pl.when(pl.col("carries") > 0).then(pl.col("rushing_tds") / pl.col("carries")).alias("td_rate_rushing"),
    )
