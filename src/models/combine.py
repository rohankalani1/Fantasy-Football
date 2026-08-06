"""Step 7 of PLAN.md: combine. points = team_volume x usage_share x efficiency x
games_played, with league scoring read from config/scoring.yaml.

Assembles every component built so far -- Step 3 (team volume), Step 4/7 (usage share:
target/rush/passer), Step 5/7 (efficiency: 8 empirical-Bayes-shrunk rates), Step 6
(availability) -- into projected raw counting stats, then applies the league's real
scoring config (never a hardcoded formula) to get total points and points per game.

fumbles_lost and two_pt_conversions are NOT modeled and are projected as 0 -- both are
low-frequency events (a handful of plays a season for even high-volume players) worth
under 3% of a typical season's points; not worth a 9th/10th empirical Bayes shrinkage
model in Phase 1. Documented here as a deliberate scope decision, not an oversight.
"""

import os

import polars as pl

from src.ingest.constants import PROCESSED_DIR, RESULTS_DIR, SEASONS
from src.models.availability import build_training_table as build_availability_table
from src.models.availability import fit_availability_model
from src.models.availability import predict as predict_availability
from src.models.efficiency import build_shrinkage_predictions, fit_all_shrinkage_params
from src.models.rookie_blend import blend_rookie_projections
from src.models.team_volume import assemble_training_table as build_team_volume_table
from src.models.team_volume import fit_team_volume_model
from src.models.team_volume import project_historical as project_team_volume_historical
from src.models.usage_share import (
    PASSER_SHARE_POSITIONS,
    RUSH_SHARE_POSITIONS,
    TARGET_SHARE_POSITIONS,
    _fit_one_share_model,
    assemble_usage_training_table,
    predict_shares,
)
from src.scoring import compute_fantasy_points, load_scoring_config

# Columns that must be 0, not null, before multiplying -- a null propagates through a
# product even when the other factor is legitimately 0 (e.g. TE rush_share_pred=0 but
# ypc_shrunk=null since TEs aren't in Step 5's ypc STAT_POSITIONS; 0 * null = null, not
# the 0 contribution it should be).
_ZERO_FILL_BEFORE_MULTIPLY = [
    "team_plays_pred", "pass_rate_pred",
    "target_share_pred", "rush_share_pred", "passer_share_pred",
    "ypt_shrunk", "td_rate_receiving_shrunk", "catch_rate_shrunk",
    "ypc_shrunk", "td_rate_rushing_shrunk",
    "ypa_shrunk", "pass_td_rate_shrunk", "int_rate_shrunk",
]


def assemble_component_predictions(panel: pl.DataFrame, refresh: bool = False) -> pl.DataFrame:
    """One row per (gsis_id, season) with every component's prediction joined
    together. Coverage starts ~2015, same as every individual component (2 years of
    trailing history needed before any of them produce a prediction)."""
    team_vol_table = build_team_volume_table(refresh)
    _, team_vol_models = fit_team_volume_model(team_vol_table)
    team_vol_pred = project_team_volume_historical(team_vol_table, team_vol_models)

    usage_table = assemble_usage_training_table(panel, refresh)
    usage_models = {}
    for target, positions in [
        ("target_share", TARGET_SHARE_POSITIONS),
        ("rush_share", RUSH_SHARE_POSITIONS),
        ("passer_share", PASSER_SHARE_POSITIONS),
    ]:
        _, model = _fit_one_share_model(usage_table, target, positions)
        usage_models[target] = model
    usage_pred = predict_shares(usage_table, usage_models)

    eb_params = fit_all_shrinkage_params(panel)
    efficiency_pred = build_shrinkage_predictions(panel, eb_params)

    avail_table = build_availability_table(panel)
    avail_fit = fit_availability_model(avail_table)
    avail_pred = predict_availability(avail_table, avail_fit)

    base = panel.select("gsis_id", "season", "team", "position")
    out = base.join(team_vol_pred, on=["team", "season"], how="left")
    out = out.join(usage_pred.drop("position"), on=["gsis_id", "season"], how="left")
    out = out.join(efficiency_pred.drop("position"), on=["gsis_id", "season"], how="left")
    out = out.join(
        avail_pred.select("gsis_id", "season", "pred_games_played", "pred_availability_prob"),
        on=["gsis_id", "season"], how="left",
    )
    return out.with_columns([pl.col(c).fill_null(0.0) for c in _ZERO_FILL_BEFORE_MULTIPLY])


def project_raw_stats(predictions: pl.DataFrame) -> pl.DataFrame:
    """Multiplies team volume x usage share x efficiency into projected raw counting
    stats, in the exact column names src/scoring.py's compute_fantasy_points expects."""
    out = predictions.with_columns(
        (pl.col("team_plays_pred") * pl.col("pass_rate_pred")).alias("team_pass_attempts_pred"),
        (pl.col("team_plays_pred") * (1 - pl.col("pass_rate_pred"))).alias("team_rush_attempts_pred"),
    )
    out = out.with_columns(
        (pl.col("team_pass_attempts_pred") * pl.col("target_share_pred")).alias("targets_pred"),
        (pl.col("team_rush_attempts_pred") * pl.col("rush_share_pred")).alias("carries_pred"),
        (pl.col("team_pass_attempts_pred") * pl.col("passer_share_pred")).alias("pass_attempts_pred"),
    )
    out = out.with_columns(
        (pl.col("targets_pred") * pl.col("catch_rate_shrunk")).alias("receptions"),
        (pl.col("targets_pred") * pl.col("ypt_shrunk")).alias("receiving_yards"),
        (pl.col("targets_pred") * pl.col("td_rate_receiving_shrunk")).alias("receiving_tds"),
        (pl.col("carries_pred") * pl.col("ypc_shrunk")).alias("rushing_yards"),
        (pl.col("carries_pred") * pl.col("td_rate_rushing_shrunk")).alias("rushing_tds"),
        (pl.col("pass_attempts_pred") * pl.col("ypa_shrunk")).alias("passing_yards"),
        (pl.col("pass_attempts_pred") * pl.col("pass_td_rate_shrunk")).alias("passing_tds"),
        (pl.col("pass_attempts_pred") * pl.col("int_rate_shrunk")).alias("pass_interceptions"),
        pl.lit(0.0).alias("fumbles_lost"),
        pl.lit(0.0).alias("two_pt_conversions"),
    )
    return out


def compute_projected_points(predictions: pl.DataFrame, scoring: dict) -> pl.DataFrame:
    raw_stats = project_raw_stats(predictions)
    scored = compute_fantasy_points(raw_stats, scoring)
    return scored.with_columns(
        pl.col("points").alias("points_total_pred"),
        pl.when(pl.col("pred_games_played") > 0)
        .then(pl.col("points") / pl.col("pred_games_played"))
        .otherwise(0.0)
        .alias("points_per_game_pred"),
    ).drop("points")


def main(refresh: bool = False) -> pl.DataFrame:
    panel = pl.read_parquet(os.path.join(PROCESSED_DIR, "player_season_panel.parquet"))
    predictions = assemble_component_predictions(panel, refresh)
    scoring = load_scoring_config()
    combined = compute_projected_points(predictions, scoring)
    combined = blend_rookie_projections(combined, panel, refresh)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "combined_projections.parquet")
    combined.write_parquet(out_path)
    print(f"Wrote {out_path}: {combined.height} rows")
    return combined


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Step 7's combine step.")
    parser.add_argument("--refresh", action="store_true", help="Re-pull raw data from network.")
    args = parser.parse_args()
    main(refresh=args.refresh)
