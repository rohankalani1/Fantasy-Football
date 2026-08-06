"""Scratch validation of the Step 1 panel -- sanity checks, not the source of truth."""
import polars as pl

pl.Config.set_tbl_rows(30)
pl.Config.set_tbl_cols(40)

panel = pl.read_parquet("data/processed/player_season_panel.parquet")

print("=== shape ===")
print(panel.shape)

print("\n=== dupes on (gsis_id, season)? ===")
dupes = panel.group_by(["gsis_id", "season"]).len().filter(pl.col("len") > 1)
print(dupes.height, "duplicate keys")

print("\n=== null rate per column ===")
null_rates = panel.null_count() / panel.height
print(null_rates.transpose(include_header=True, header_name="column", column_names=["null_rate"]).sort("null_rate", descending=True))

print("\n=== games_played / games_active range ===")
print(panel.select(
    pl.col("games_played").max().alias("games_played_max"),
    pl.col("games_active").max().alias("games_active_max"),
    pl.col("games_played").min().alias("games_played_min"),
    pl.col("games_active").min().alias("games_active_min"),
))

print("\n=== rows with null gsis_id (should be 0 -- primary key of the base spine) ===")
print(panel.filter(pl.col("gsis_id").is_null()).select("display_name","season","team","position"))
print("rows with games_active > games_played + 1 (should be rare/explainable):",
      panel.filter(pl.col("games_active") > pl.col("games_played") + 1).height)

print("\n=== age range ===")
print(panel.select(pl.col("age").min().alias("age_min"), pl.col("age").max().alias("age_max"), pl.col("age").mean().alias("age_mean")))

print("\n=== who has games_played == 18? (impossible in a single reg season) ===")
print(panel.filter(pl.col("games_played") >= 18).select("gsis_id","display_name","season","team","position","games_played"))

print("\n=== snaps vs team_plays sanity ===")
bad_snaps = panel.filter(pl.col("season_offense_snaps") > pl.col("team_plays"))
print("player snaps > team plays (should be 0 or explainable):", bad_snaps.height)
if bad_snaps.height > 0:
    print(bad_snaps.select("display_name","season","team","position","season_offense_snaps","team_plays"))

print("\n=== team_plays sanity (should be roughly 950-1150 per team-season) ===")
print(panel.select("season","team","team_plays").unique().select(
    pl.col("team_plays").min().alias("min"), pl.col("team_plays").max().alias("max"), pl.col("team_plays").mean().alias("mean")
))

print("\n=== Rob Housler 2015 (mid-season trade CLE->CHI) ===")
print(panel.filter((pl.col("display_name").str.contains("Housler")) & (pl.col("season")==2015)).select(
    "display_name","season","team","position","games_played","targets","receptions"
))

print("\n=== Christian McCaffrey career (spot check known player) ===")
print(panel.filter(pl.col("display_name").str.contains("McCaffrey")).select(
    "display_name","season","team","age","years_exp","draft_round","draft_pick","games_played","carries","rushing_yards","receptions","receiving_yards","ffc_adp"
))

print("\n=== a zero-opportunity rostered player example ===")
zero_opp = panel.filter((pl.col("games_played")==0) & (pl.col("years_exp") > 0)).head(5)
print(zero_opp.select("display_name","season","team","position","years_exp","games_played","targets","carries"))

print("\n=== rookie-year draft capital spot check (2023 rookie WRs) ===")
print(panel.filter((pl.col("season")==2023) & (pl.col("position")=="WR") & (pl.col("years_exp")==0)).select(
    "display_name","draft_round","draft_pick","targets","receptions","ffc_adp"
).sort("draft_pick").head(10))

print("\n=== ADP match rate ===")
adp_present = panel.filter(pl.col("ffc_adp").is_not_null()).height
print(f"{adp_present} / {panel.height} rows have an FFC ADP")

print("\n=== fantasy_points_ppr vs ffc_adp sanity (top 15 by adp, 2023) ===")
print(panel.filter((pl.col("season")==2023) & pl.col("ffc_adp").is_not_null())
      .sort("ffc_adp").head(15)
      .select("display_name","position","ffc_adp","fantasy_points_ppr","games_played"))
