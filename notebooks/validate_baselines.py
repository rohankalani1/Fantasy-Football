"""Spot-check Step 2 baselines against real-world knowledge -- sanity, not source of truth."""
import polars as pl

from src.models.baselines import build_baselines

pl.Config.set_tbl_rows(40)
pl.Config.set_tbl_cols(20)

panel = pl.read_parquet("data/processed/player_season_panel.parquet")
pwb = build_baselines(panel)

print("=== 2024: CMC (torn achilles/PCL, missed almost all of 2024) ===")
print(pwb.filter((pl.col("display_name").str.contains("McCaffrey")) & (pl.col("season").is_in([2023, 2024])))
      .select("display_name", "season", "games_played", "points", "baseline_ppg", "baseline_consensus", "ffc_adp"))

print("\n=== 2024: Bijan Robinson (2023 rookie -> 2024 breakout) ===")
print(pwb.filter((pl.col("display_name").str.contains("Bijan")) & (pl.col("season").is_in([2023, 2024])))
      .select("display_name", "season", "games_played", "points", "baseline_ppg", "baseline_consensus", "ffc_adp"))

print("\n=== 2025: Ja'Marr Chase (elite, consistent) ===")
print(pwb.filter((pl.col("display_name").str.contains("Chase")) & (pl.col("season").is_in([2024, 2025])))
      .select("display_name", "season", "games_played", "points", "baseline_ppg", "baseline_consensus", "ffc_adp"))

print("\n=== true rookies in 2024 (years_exp==0) -- baseline_ppg should be exactly 0 ===")
rookies_2024 = pwb.filter((pl.col("season") == 2024) & (pl.col("years_exp") == 0) & pl.col("ffc_adp").is_not_null())
print("count:", rookies_2024.height, " | count with baseline_ppg != 0:",
      rookies_2024.filter(pl.col("baseline_ppg") != 0).height)
print(rookies_2024.sort("ffc_adp").head(8).select("display_name", "position", "baseline_ppg", "baseline_consensus", "ffc_adp", "points"))

print("\n=== baseline_consensus direction check: best (most negative ecr) should be stars ===")
print(pwb.filter(pl.col("season") == 2024).sort("baseline_consensus", descending=True).head(10)
      .select("display_name", "position", "baseline_consensus", "ffc_adp", "points"))

print("\n=== does baseline_consensus correlate sensibly with ffc_adp? (both should roughly agree on stars) ===")
print(pwb.filter((pl.col("season") == 2024) & pl.col("ffc_adp").is_not_null())
      .select(pl.corr("baseline_consensus", "baseline_adp").alias("corr_consensus_vs_adp")))
