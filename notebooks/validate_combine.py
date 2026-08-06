"""Spot-check Step 7's combine output -- sanity, not source of truth."""
import polars as pl

from src.models.baselines import build_baselines

pl.Config.set_tbl_rows(25)
pl.Config.set_tbl_cols(15)

panel = pl.read_parquet("data/processed/player_season_panel.parquet")
combined = pl.read_parquet("results/combined_projections.parquet")
names = panel.select("gsis_id", "season", "display_name")
combined = combined.join(names, on=["gsis_id", "season"])

print("=== negative points anywhere? ===")
print(combined.filter(pl.col("points_total_pred") < 0).height, "(should be 0)")

print("\n=== 2025 top 20 overall, all positions ===")
print(combined.filter(pl.col("season") == 2025).sort("points_total_pred", descending=True).head(20).select(
    "display_name", "position", "points_total_pred", "points_per_game_pred", "rookie_blend_applied"
))

print("\n=== rookie blend: applied to how many 2025 rows? ===")
r2025 = combined.filter(pl.col("season") == 2025)
print(r2025.select(pl.col("rookie_blend_applied").sum()).item(), "/", r2025.height)

print("\n=== accuracy check: 2025 draftable players, points_total_pred vs actual (all positions) ===")
with_actual = build_baselines(panel).select("gsis_id", "season", "points", "ffc_adp")
check = combined.join(with_actual, on=["gsis_id", "season"], how="inner").filter(
    (pl.col("season") == 2025) & pl.col("ffc_adp").is_not_null()
)
mae = (check["points_total_pred"] - check["points"]).abs().mean()
print(f"MAE on 2025 draftable players: {mae:.2f} points (n={check.height})")
print(check.select(pl.corr("points_total_pred", "points").alias("pearson_r")))

print("\n=== per-position MAE ===")
print(check.group_by("position").agg(
    (pl.col("points_total_pred") - pl.col("points")).abs().mean().alias("mae"), pl.len()
).sort("position"))
