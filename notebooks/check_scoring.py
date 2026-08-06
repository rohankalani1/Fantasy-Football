"""Sanity-check src/scoring.py's config-driven points against nflreadpy's own
fantasy_points_ppr. They should be very close (both are standard full-PPR formulas) but
not necessarily identical -- this is a bug-catching check, not an equality assertion."""
import polars as pl

from src.scoring import compute_fantasy_points, load_scoring_config

pl.Config.set_tbl_rows(30)

panel = pl.read_parquet("data/processed/player_season_panel.parquet")
scoring = load_scoring_config()
panel = compute_fantasy_points(panel, scoring)

panel = panel.with_columns((pl.col("points") - pl.col("fantasy_points_ppr")).alias("diff"))

print("=== diff = our points - nflreadpy fantasy_points_ppr ===")
print(panel.select(
    pl.col("diff").mean().alias("mean"),
    pl.col("diff").median().alias("median"),
    pl.col("diff").abs().mean().alias("mean_abs"),
    pl.col("diff").abs().max().alias("max_abs"),
    pl.col("diff").std().alias("std"),
))

print("\n=== correlation ===")
print(panel.select(pl.corr("points", "fantasy_points_ppr").alias("pearson_r")))

print("\n=== biggest absolute discrepancies ===")
print(panel.sort("diff", descending=True).head(5).select(
    "display_name", "season", "position", "points", "fantasy_points_ppr", "diff",
    "fumbles_lost", "two_pt_conversions"
))
print(panel.sort("diff").head(5).select(
    "display_name", "season", "position", "points", "fantasy_points_ppr", "diff",
    "fumbles_lost", "two_pt_conversions"
))

print("\n=== spot check: 2023 top 5 by our points ===")
print(panel.filter(pl.col("season") == 2023).sort("points", descending=True).head(5).select(
    "display_name", "position", "points", "fantasy_points_ppr", "games_played"
))
