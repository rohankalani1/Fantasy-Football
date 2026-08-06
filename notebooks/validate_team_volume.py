"""Spot-check Step 3's team volume model -- sanity, not source of truth."""
import polars as pl

from src.features.team_volume import build_team_volume_features
from src.ingest.pull_raw import pull_pbp
from src.models.team_volume import assemble_training_table, fit_team_volume_model

pl.Config.set_tbl_rows(20)

table = assemble_training_table()

print("=== leakage check: pace_t1 for (team, season) should equal realized pace for (team, season-1) ===")
feats = build_team_volume_features(pull_pbp())
check = table.select("team", "season", "pace_t1").join(
    feats.select("team", pl.col("season").alias("season"), pl.col("pace").alias("pace_realized")).with_columns(
        (pl.col("season") + 1).alias("season")
    ),
    on=["team", "season"],
    how="inner",
)
mismatch = check.filter((pl.col("pace_t1") - pl.col("pace_realized")).abs() > 1e-9)
print("mismatches:", mismatch.height, "(should be 0)")

print("\n=== no duplicate (team, season) rows ===")
dupes = table.group_by(["team", "season"]).len().filter(pl.col("len") > 1)
print("dupes:", dupes.height, "(should be 0)")

print("\n=== row count sanity (32 teams x 13 seasons = 416 max) ===")
print(table.shape)

results, models = fit_team_volume_model(table)

print("\n=== 2025 test set: predicted vs actual, a few well-known teams ===")
test = table.filter(pl.col("season") == 2025).drop_nulls(
    ["pace_t1", "pace_t2", "proe_t1", "proe_t2", "hc_change_flag", "vegas_win_total"]
)
X = test.select(["pace_t1", "pace_t2", "proe_t1", "proe_t2", "hc_change_flag", "vegas_win_total"]).to_numpy()
test = test.with_columns(
    pl.Series("team_plays_pred", models["team_plays"].predict(X)),
    pl.Series("pass_rate_pred", models["pass_rate"].predict(X)),
)
print(test.filter(pl.col("team").is_in(["BAL", "CIN", "PHI", "KC", "DEN"])).select(
    "team", "team_plays", "team_plays_pred", "pass_rate", "pass_rate_pred", "vegas_win_total"
))

print("\n=== overall 2025 correlation (pred vs actual) ===")
print(test.select(
    pl.corr("team_plays_pred", "team_plays").alias("team_plays_corr"),
    pl.corr("pass_rate_pred", "pass_rate").alias("pass_rate_corr"),
))
