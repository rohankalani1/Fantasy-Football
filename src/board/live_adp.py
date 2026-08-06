"""Live current-season ADP as a reference column on the draft board -- lets the owner
see at a glance where the model disagrees with the market, which is exactly where the
project's edge (or a modeling mistake) would show up. Reuses the same two-stage
name-matching machinery src/ingest/id_matching.py already built for historical FFC ADP,
since FFC has no ID crosswalk of its own regardless of season.
"""

import polars as pl

from src.ingest.ffc_adp import pull_ffc_adp_for_season
from src.ingest.id_matching import build_name_index, match_adp_to_gsis
from src.ingest.pull_raw import pull_ff_playerids
from src.ingest.team_codes import canonicalize_team


def build_live_adp(future_population: pl.DataFrame, season: int, refresh: bool = False) -> pl.DataFrame:
    """gsis_id, live_adp, bye_week for every live FFC ADP row matched to the season's
    roster population. Unmatched rows are logged, never silently dropped -- CLAUDE.md's
    "log every unmatched draftable player loudly" rule."""
    ffc = pull_ffc_adp_for_season(season, refresh).with_columns(canonicalize_team(pl.col("team")))
    ff_playerids = pull_ff_playerids(refresh)
    name_index = build_name_index(
        future_population.select("gsis_id", "season", "position", "team", "display_name")
    )
    matched, unmatched = match_adp_to_gsis(ffc, ff_playerids, name_index)
    if unmatched.height > 0:
        print(
            f"[build_live_adp] {unmatched.height}/{ffc.height} live {season} FFC ADP rows "
            f"could not be matched to a gsis_id: {unmatched.select('name', 'position').rows()}"
        )
    return matched.select(
        "gsis_id", pl.col("adp").alias("live_adp"), pl.col("bye").alias("bye_week")
    ).unique(subset=["gsis_id"], keep="first")
