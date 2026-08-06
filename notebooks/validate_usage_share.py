"""Spot-check Step 4's usage share models -- sanity, not source of truth."""
import polars as pl

from src.models.usage_share import (
    RUSH_SHARE_POSITIONS,
    TARGET_SHARE_POSITIONS,
    TEST_SEASON,
    _fit_one_share_model,
    _to_lgb_frame,
    assemble_usage_training_table,
)

pl.Config.set_tbl_rows(20)

panel = pl.read_parquet("data/processed/player_season_panel.parquet")
table = assemble_usage_training_table(panel)
names = panel.select("gsis_id", "season", "display_name")

print("=== leakage check: ewma_target_share for (gsis_id, season) should only use season < season ===")
shares_direct = table.select("gsis_id", "season", "target_share")
check = table.select("gsis_id", "season", "ewma_target_share").join(
    shares_direct, on=["gsis_id", "season"], how="inner"
)
# ewma should never exactly equal the CURRENT season's own realized share for a player
# with real usage in both years (that would indicate the join used season t instead of
# t-1/t-2) -- spot check a large sample of nonzero rows instead of asserting equality
# never happens by chance.
nonzero = check.filter((pl.col("ewma_target_share") > 0.05) & (pl.col("target_share") > 0.05))
suspicious = nonzero.filter((pl.col("ewma_target_share") - pl.col("target_share")).abs() < 1e-9)
print(f"rows where ewma exactly equals current-season target_share: {suspicious.height} / {nonzero.height} (should be ~0, coincidental matches only)")

print()
r_ts, model_ts = _fit_one_share_model(table, "target_share", TARGET_SHARE_POSITIONS)
r_rs, model_rs = _fit_one_share_model(table, "rush_share", RUSH_SHARE_POSITIONS)

test_ts = table.filter((pl.col("position").is_in(TARGET_SHARE_POSITIONS)) & (pl.col("season") == TEST_SEASON)).drop_nulls(
    [c for c in table.columns if c not in ("target_share", "rush_share")]
)
X_ts = _to_lgb_frame(test_ts)
test_ts = test_ts.with_columns(pl.Series("target_share_pred", model_ts.predict(X_ts)))
test_ts = test_ts.join(names, on=["gsis_id", "season"])

print("=== 2025 target_share: predicted vs actual, top 10 by prediction ===")
print(test_ts.sort("target_share_pred", descending=True).head(10).select(
    "display_name", "team", "position", "target_share_pred", "target_share", "ewma_target_share", "vacated_target_share"
))

print()
print("=== Seattle WRs 2025 (post Metcalf/Lockett departure -- large vacated_target_share) ===")
print(test_ts.filter((pl.col("team") == "SEA") & (pl.col("position") == "WR")).select(
    "display_name", "target_share_pred", "target_share", "ewma_target_share", "vacated_target_share"
).sort("target_share_pred", descending=True))

test_rs = table.filter((pl.col("position").is_in(RUSH_SHARE_POSITIONS)) & (pl.col("season") == TEST_SEASON)).drop_nulls(
    [c for c in table.columns if c not in ("target_share", "rush_share")]
)
X_rs = _to_lgb_frame(test_rs)
test_rs = test_rs.with_columns(pl.Series("rush_share_pred", model_rs.predict(X_rs)))
test_rs = test_rs.join(names, on=["gsis_id", "season"])

print()
print("=== 2025 rush_share: predicted vs actual, top 10 by prediction ===")
print(test_rs.sort("rush_share_pred", descending=True).head(10).select(
    "display_name", "team", "position", "rush_share_pred", "rush_share", "ewma_rush_share"
))

print()
print("=== rookies (years_exp == 0) in 2025 -- should show near-zero pred (Phase 1 rookie gap, expected) ===")
rookies = test_ts.filter(pl.col("years_exp") == 0).sort("draft_pick").head(8)
print(rookies.select("display_name", "draft_pick", "target_share_pred", "target_share"))
