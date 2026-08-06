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

from src.features.current_roster import build_future_population
from src.ingest.constants import PROCESSED_DIR, RESULTS_DIR, SEASONS, season_length_expr
from src.models.availability import build_future_availability_table
from src.models.availability import build_training_table as build_availability_table
from src.models.availability import fit_availability_model
from src.models.availability import predict as predict_availability
from src.models.efficiency import (
    build_future_shrinkage_predictions,
    build_shrinkage_predictions,
    fit_all_shrinkage_params,
)
from src.models.rookie_blend import blend_rookie_projections, blend_future_rookie_projections
from src.models.team_volume import assemble_training_table as build_team_volume_table
from src.models.team_volume import fit_team_volume_model
from src.models.team_volume import project_future_season
from src.models.team_volume import project_historical as project_team_volume_historical
from src.models.usage_share import (
    PASSER_SHARE_POSITIONS,
    RUSH_SHARE_POSITIONS,
    TARGET_SHARE_POSITIONS,
    _fit_one_share_model,
    assemble_future_usage_table,
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


def _fit_all_models(panel: pl.DataFrame, refresh: bool = False) -> dict:
    """Fits every component model exactly once, on the historical panel -- shared by
    both assemble_component_predictions (historical rows) and
    assemble_future_component_predictions (a live not-yet-played season's roster), so
    the two entry points score with the literal same fitted models rather than two
    separately-refit copies."""
    team_vol_table = build_team_volume_table(refresh)
    _, team_vol_models = fit_team_volume_model(team_vol_table)

    usage_table = assemble_usage_training_table(panel, refresh)
    usage_models = {}
    for target, positions in [
        ("target_share", TARGET_SHARE_POSITIONS),
        ("rush_share", RUSH_SHARE_POSITIONS),
        ("passer_share", PASSER_SHARE_POSITIONS),
    ]:
        _, model = _fit_one_share_model(usage_table, target, positions)
        usage_models[target] = model

    eb_params = fit_all_shrinkage_params(panel)

    avail_table = build_availability_table(panel)
    avail_fit = fit_availability_model(avail_table)

    return {
        "team_vol_models": team_vol_models,
        "usage_models": usage_models,
        "eb_params": eb_params,
        "avail_fit": avail_fit,
    }


def assemble_component_predictions(panel: pl.DataFrame, refresh: bool = False) -> pl.DataFrame:
    """One row per (gsis_id, season) with every component's prediction joined
    together. Coverage starts ~2015, same as every individual component (2 years of
    trailing history needed before any of them produce a prediction)."""
    models = _fit_all_models(panel, refresh)

    team_vol_table = build_team_volume_table(refresh)
    team_vol_pred = project_team_volume_historical(team_vol_table, models["team_vol_models"])

    usage_table = assemble_usage_training_table(panel, refresh)
    usage_pred = predict_shares(usage_table, models["usage_models"])

    efficiency_pred = build_shrinkage_predictions(panel, models["eb_params"])

    avail_table = build_availability_table(panel)
    avail_pred = predict_availability(avail_table, models["avail_fit"])

    base = panel.select("gsis_id", "season", "team", "position")
    out = base.join(team_vol_pred, on=["team", "season"], how="left")
    out = out.join(usage_pred.drop("position"), on=["gsis_id", "season"], how="left")
    out = out.join(efficiency_pred.drop("position"), on=["gsis_id", "season"], how="left")
    out = out.join(
        avail_pred.select("gsis_id", "season", "pred_games_played", "pred_availability_prob"),
        on=["gsis_id", "season"], how="left",
    )
    return out.with_columns([pl.col(c).fill_null(0.0) for c in _ZERO_FILL_BEFORE_MULTIPLY])


def assemble_future_component_predictions(
    panel: pl.DataFrame, season: int, refresh: bool = False
) -> pl.DataFrame:
    """Same shape as assemble_component_predictions, for a live not-yet-played `season`
    (2026) instead of panel's own historical rows -- one row per player on the live
    roster population, using the exact same fitted models."""
    models = _fit_all_models(panel, refresh)
    future_population = build_future_population(season, refresh)

    team_vol_pred = project_future_season(models["team_vol_models"], season).select(
        "team", "team_plays_pred", "pass_rate_pred"
    )

    usage_table = assemble_future_usage_table(panel, season, refresh)
    usage_pred = predict_shares(usage_table, models["usage_models"])

    efficiency_pred = build_future_shrinkage_predictions(
        panel, future_population, models["eb_params"]
    )

    avail_table = build_future_availability_table(panel, future_population)
    avail_pred = predict_availability(avail_table, models["avail_fit"])

    base = future_population.select("gsis_id", "season", "team", "position")
    out = base.join(team_vol_pred, on="team", how="left")
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
    """project_raw_stats/compute_fantasy_points give a full-season-equivalent point
    total implied by the player's projected share/efficiency alone -- with no
    availability term at all, contradicting CLAUDE.md's own
    "opportunities x efficiency x games played" thesis. Bug found and fixed here:
    pred_games_played was previously used only to *divide* into points_per_game_pred,
    never to scale points_total_pred down for elevated missed-game risk (verified: an
    injury-prone player's points_per_game_pred was actually being inflated by this,
    since a smaller denominator was applied to an undiminished numerator -- backwards).

    Scaling the full-season-equivalent total by pred_games_played / season_length
    folds in the beta-binomial availability model's own risk assessment, and as a
    side effect makes points_per_game_pred = points_total_pred / pred_games_played
    collapse back to points / season_length -- a clean "expected rate when playing"
    that's now correctly independent of the player's own availability risk, rather
    than being inflated by it.
    """
    raw_stats = project_raw_stats(predictions)
    scored = compute_fantasy_points(raw_stats, scoring)
    availability_factor = pl.col("pred_games_played") / season_length_expr()
    return scored.with_columns(
        (pl.col("points") * availability_factor).alias("points_total_pred"),
    ).with_columns(
        pl.when(pl.col("pred_games_played") > 0)
        .then(pl.col("points_total_pred") / pl.col("pred_games_played"))
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


FUTURE_SEASON = 2026


def main_future(season: int = FUTURE_SEASON, refresh: bool = False) -> pl.DataFrame:
    """The live projection entry point: same combine pipeline as main(), but for
    `season`'s not-yet-played roster population instead of panel's historical rows."""
    panel = pl.read_parquet(os.path.join(PROCESSED_DIR, "player_season_panel.parquet"))
    future_population = build_future_population(season, refresh)
    predictions = assemble_future_component_predictions(panel, season, refresh)
    scoring = load_scoring_config()
    combined = compute_projected_points(predictions, scoring)
    combined = blend_future_rookie_projections(combined, panel, future_population, season, refresh)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, f"projections_{season}.parquet")
    combined.write_parquet(out_path)
    print(f"Wrote {out_path}: {combined.height} rows")
    return combined


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Step 7's combine step.")
    parser.add_argument("--refresh", action="store_true", help="Re-pull raw data from network.")
    parser.add_argument(
        "--future", action="store_true",
        help=f"Project the live {FUTURE_SEASON} season instead of scoring historical seasons.",
    )
    args = parser.parse_args()
    if args.future:
        main_future(refresh=args.refresh)
    else:
        main(refresh=args.refresh)
