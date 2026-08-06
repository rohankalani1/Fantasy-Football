"""Spot-check Step 5's efficiency shrinkage model -- sanity, not source of truth."""
import polars as pl

from src.models.efficiency import build_shrinkage_predictions, fit_all_shrinkage_params

pl.Config.set_tbl_rows(20)

panel = pl.read_parquet("data/processed/player_season_panel.parquet")
params = fit_all_shrinkage_params(panel)
preds = build_shrinkage_predictions(panel, params)
names = panel.select("gsis_id", "season", "display_name")

print("=== high-career-volume WR should show minimal shrinkage (shrunk ~= own career rate) ===")
sample = preds.filter((pl.col("season") == 2025) & (pl.col("career_targets") > 1000)).join(
    names, on=["gsis_id", "season"]
).select("display_name", "career_targets", "ypt_career_rate", "ypt_shrunk")
print(sample.sort("career_targets", descending=True).head(8))

print()
print("=== low-career-volume player should show heavy shrinkage toward position mean ===")
sample2 = preds.filter((pl.col("season") == 2025) & (pl.col("career_targets") > 0) & (pl.col("career_targets") < 30) & (pl.col("position")=="WR")).join(
    names, on=["gsis_id", "season"]
).select("display_name", "career_targets", "ypt_career_rate", "ypt_shrunk")
print(sample2.sort("career_targets").head(8))

print()
print("=== TD-rate regression: players with a hot TD-rate season should see next season pulled toward the mean ===")
season_rates = panel.select("gsis_id", "season", "targets",
    pl.when(pl.col("targets") > 15).then(pl.col("receiving_tds") / pl.col("targets")).alias("season_td_rate")
).drop_nulls("season_td_rate")
hot = season_rates.filter(pl.col("season_td_rate") > 0.15).sort("season_td_rate", descending=True).head(5)
hot_next = hot.with_columns((pl.col("season") + 1).alias("next_season")).join(
    preds.select("gsis_id", pl.col("season").alias("next_season"), "td_rate_receiving_shrunk"),
    on=["gsis_id", "next_season"], how="left",
).join(names, on=["gsis_id", "season"])
actual_next = panel.select("gsis_id", pl.col("season").alias("next_season"), "targets",
    pl.when(pl.col("targets") > 0).then(pl.col("receiving_tds") / pl.col("targets")).alias("actual_next_td_rate")
)
hot_next = hot_next.join(actual_next.select("gsis_id", "next_season", "actual_next_td_rate"), on=["gsis_id", "next_season"], how="left")
print(hot_next.select("display_name", "season", "season_td_rate", "td_rate_receiving_shrunk", "actual_next_td_rate"))

print()
print("=== leakage check: career_targets for (gsis_id, season) should equal cumsum through season-1 ===")
check = preds.filter(pl.col("season") == 2025).select("gsis_id", "career_targets").join(
    panel.filter(pl.col("season") <= 2024).group_by("gsis_id").agg(pl.col("targets").sum().alias("manual_cum")),
    on="gsis_id", how="inner"
)
mismatch = check.filter((pl.col("career_targets") - pl.col("manual_cum")).abs() > 0)
print(f"mismatches: {mismatch.height} / {check.height} (should be 0)")
