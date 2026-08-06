"""Step 9 of PLAN.md: manual override file. Late-August news (trades, injuries, depth
chart surprises) applied as a point adjustment at load time, so the board can reflect
same-day news on draft morning without rerunning the whole pipeline.

config/manual_overrides.csv columns: display_name, team (optional tiebreaker),
points_adjustment (added to points_total_pred, can be negative), note (free text,
carried through to the final board for the owner's own reference).

Matched by normalized display name, not gsis_id -- a human editing this file at 6am on
draft day won't have gsis_ids memorized. `team` is optional and only used to
disambiguate two same-named players; most rows won't need it.
"""

import os

import polars as pl

from src.ingest.id_matching import normalize_name

OVERRIDES_PATH = "config/manual_overrides.csv"


def apply_manual_overrides(board: pl.DataFrame, overrides_path: str = OVERRIDES_PATH) -> pl.DataFrame:
    if not os.path.exists(overrides_path):
        return board.with_columns(
            pl.lit(0.0).alias("manual_adjustment"), pl.lit(None, dtype=pl.String).alias("override_note")
        )

    overrides = pl.read_csv(overrides_path)
    if overrides.height == 0:
        return board.with_columns(
            pl.lit(0.0).alias("manual_adjustment"), pl.lit(None, dtype=pl.String).alias("override_note")
        )

    overrides = overrides.with_columns(
        pl.col("display_name").map_elements(normalize_name, return_dtype=pl.String).alias("norm_name")
    )
    if "note" not in overrides.columns:
        overrides = overrides.with_columns(pl.lit(None, dtype=pl.String).alias("note"))

    board_keyed = board.with_columns(
        pl.col("display_name").map_elements(normalize_name, return_dtype=pl.String).alias("norm_name")
    )

    join_cols = ["norm_name", "team"] if "team" in overrides.columns else ["norm_name"]
    overrides_for_join = overrides.select(*join_cols, "points_adjustment", "note")

    unmatched = overrides.join(board_keyed.select(join_cols).unique(), on=join_cols, how="anti")
    if unmatched.height > 0:
        print(
            f"[apply_manual_overrides] {unmatched.height} override row(s) matched no "
            f"player on the board: {unmatched.select('display_name').to_series().to_list()}"
        )

    out = board_keyed.join(overrides_for_join, on=join_cols, how="left")
    out = out.with_columns(
        pl.col("points_adjustment").fill_null(0.0).alias("manual_adjustment"),
        pl.col("note").alias("override_note"),
    )
    out = out.with_columns(
        (pl.col("points_total_pred") + pl.col("manual_adjustment")).clip(lower_bound=0.0).alias("points_total_pred")
    )
    return out.drop(["norm_name", "points_adjustment", "note"])
