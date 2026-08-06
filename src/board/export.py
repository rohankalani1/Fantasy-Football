"""Step 9 of PLAN.md: draft board. Assembles the live-season projections into a final,
ranked, tiered CSV -- VOR, tiers, live ADP as a reference column (where the model and
the market disagree is exactly where the project's edge shows up), manual overrides
applied. One row per player in the live roster population, sorted by VOR; a real draft
only needs the top ~150-200 rows, but the rest is left in so any rostered name is
searchable.
"""

import os

import polars as pl

from src.board.live_adp import build_live_adp
from src.board.overrides import apply_manual_overrides
from src.board.vor import build_draft_board, load_roster_requirements
from src.features.current_roster import build_future_population
from src.ingest.constants import RESULTS_DIR
from src.models.combine import FUTURE_SEASON

PROJECTIONS_PATH_TEMPLATE = os.path.join(RESULTS_DIR, "projections_{season}.parquet")


def main(season: int = FUTURE_SEASON, refresh: bool = False) -> pl.DataFrame:
    projections = pl.read_parquet(PROJECTIONS_PATH_TEMPLATE.format(season=season))
    future_population = build_future_population(season, refresh)
    projections = projections.join(
        future_population.select("gsis_id", "display_name"), on="gsis_id", how="left"
    )

    # Overrides applied before VOR/tiers so a late-breaking adjustment can actually move
    # a player's tier and the replacement-level calculation, not just their raw total.
    projections = apply_manual_overrides(projections)

    roster_requirements = load_roster_requirements()
    board = build_draft_board(projections, roster_requirements)

    live_adp = build_live_adp(future_population, season, refresh)
    board = board.join(live_adp, on="gsis_id", how="left")

    out_cols = [
        "overall_rank", "display_name", "position", "team", "tier",
        "points_total_pred", "points_per_game_pred", "vor",
        "position_rank", "live_adp", "bye_week", "manual_adjustment", "override_note",
    ]
    board = board.select(out_cols).sort("overall_rank")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, f"draft_board_{season}.csv")
    board.write_csv(out_path)
    print(f"Wrote {out_path}: {board.height} rows")
    return board


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build Step 9's draft board.")
    parser.add_argument("--season", type=int, default=FUTURE_SEASON)
    parser.add_argument("--refresh", action="store_true", help="Re-pull raw data from network.")
    args = parser.parse_args()
    main(season=args.season, refresh=args.refresh)
