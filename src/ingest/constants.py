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
