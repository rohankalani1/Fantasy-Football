"""Shared constants for panel construction.

Panel starts at 2013, not the 2012 named in PLAN.md, because nflreadpy's snap-count
data (needed for the usage-share model's snap_share feature) has no coverage before
2013. Decided with the project owner rather than silently dropping 2012 or leaving
every snap column null for that one season.
"""

START_SEASON = 2013
END_SEASON = 2025
SEASONS = list(range(START_SEASON, END_SEASON + 1))

# QB/RB/WR/TE only: the opportunity x efficiency x games decomposition this project
# is built around doesn't apply to K/DST, which score on unrelated mechanics.
POSITIONS = ["QB", "RB", "WR", "TE"]

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
RESULTS_DIR = "results"


def season_length_expr(season_col: str = "season"):
    """Regular-season game count: 17 from 2021 on (the league expanded that year), 16
    before. Same rule build_panel.py already uses to clip games_played/games_active;
    pulled out here so Step 6's beta-binomial availability model (which needs it as the
    Binomial's n) doesn't silently duplicate-and-drift from that definition."""
    import polars as pl

    return pl.when(pl.col(season_col) >= 2021).then(17).otherwise(16)
