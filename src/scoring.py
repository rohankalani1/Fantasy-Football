"""Config-driven fantasy point scoring.

CLAUDE.md: "Keep league scoring rules as configuration, never hardcoded constants."
This is the one place point values are read from config/scoring.yaml and turned into a
`points` column. Everything downstream -- Step 2's baselines, Step 7's combine step,
Step 9's draft board -- should compute points by calling `compute_fantasy_points`, never
by re-deriving a formula or reusing nflreadpy's own `fantasy_points_ppr` (which bakes in
its own scoring assumptions that may not match the real league).
"""

import polars as pl
import yaml

_REQUIRED_STAT_COLS = [
    "passing_yards",
    "passing_tds",
    "pass_interceptions",
    "rushing_yards",
    "rushing_tds",
    "receptions",
    "receiving_yards",
    "receiving_tds",
    "fumbles_lost",
    "two_pt_conversions",
]


def load_scoring_config(path: str = "config/scoring.yaml") -> dict:
    with open(path) as f:
        config = yaml.safe_load(f)
    # Bonuses aren't implemented below -- fail loudly rather than silently ignore a
    # configured bonus if someone adds one later without also adding the code for it.
    if config.get("bonuses"):
        raise NotImplementedError(
            "config/scoring.yaml declares non-empty 'bonuses', but compute_fantasy_points "
            "does not implement yardage-milestone bonuses yet. Add that logic before "
            "using a config with bonuses configured."
        )
    return config


def compute_fantasy_points(df: pl.DataFrame, scoring: dict) -> pl.DataFrame:
    """Adds a `points` column computed from raw counting stats per the scoring config.

    Expects one row per player-season (or per player-week -- any grain with the raw
    counting stat columns present) and does not aggregate; sum first if you need a
    season total from weekly rows.
    """
    missing = [c for c in _REQUIRED_STAT_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"compute_fantasy_points: missing required columns {missing}")

    passing = scoring["passing"]
    rushing = scoring["rushing"]
    receiving = scoring["receiving"]

    points_expr = (
        pl.col("passing_yards") / passing["yards_per_point"]
        + pl.col("passing_tds") * passing["td"]
        + pl.col("pass_interceptions") * passing["interception"]
        + pl.col("rushing_yards") / rushing["yards_per_point"]
        + pl.col("rushing_tds") * rushing["td"]
        + pl.col("receiving_yards") / receiving["yards_per_point"]
        + pl.col("receiving_tds") * receiving["td"]
        + pl.col("receptions") * receiving["reception"]
        + pl.col("fumbles_lost") * scoring["fumbles"]["lost"]
        + pl.col("two_pt_conversions") * scoring["two_point_conversion"]
    )
    return df.with_columns(points_expr.alias("points"))
