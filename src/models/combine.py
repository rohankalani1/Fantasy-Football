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
from src.models.calibration import (
    CALIBRATION_TRAIN_SEASONS,
    apply_calibration,
    build_calibration_table,
    fit_calibration_curve,
    load_curve,
    save_curve,
)
from src.models.availability import build_training_table as build_availability_table
from src.models.availability import fit_availability_model
from src.models.availability import predict as predict_availability
from src.models.efficiency import (
    build_future_shrinkage_predictions,
    build_shrinkage_predictions,
    fit_all_shrinkage_params,
)
from src.features.team_volume import build_sack_rate_projection
from src.ingest.pull_raw import pull_team_stats
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

# sack_rate_pred is deliberately NOT in _ZERO_FILL_BEFORE_MULTIPLY -- a 0 default
# would mean "no sacks," which understates true attempts and silently re-introduces
# the exact overstatement bug this feature exists to fix. build_sack_rate_projection
# already falls back to the league mean for any team/season with real trailing data;
# this is a last-resort defensive fallback for a join miss that should never happen
# with real historical team codes, set to the modern NFL's typical sack rate.
_LEAGUE_MEAN_SACK_RATE_FALLBACK = 0.065


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
    sack_rate_pred = build_sack_rate_projection(pull_team_stats(refresh))

    usage_table = assemble_usage_training_table(panel, refresh)
    usage_pred = predict_shares(usage_table, models["usage_models"])

    efficiency_pred = build_shrinkage_predictions(panel, models["eb_params"])

    avail_table = build_availability_table(panel)
    avail_pred = predict_availability(avail_table, models["avail_fit"])

    base = panel.select("gsis_id", "season", "team", "position")
    out = base.join(team_vol_pred, on=["team", "season"], how="left")
    out = out.join(sack_rate_pred, on=["team", "season"], how="left")
    out = out.join(usage_pred.drop("position"), on=["gsis_id", "season"], how="left")
    out = out.join(efficiency_pred.drop("position"), on=["gsis_id", "season"], how="left")
    out = out.join(
        avail_pred.select("gsis_id", "season", "pred_games_played", "pred_availability_prob"),
        on=["gsis_id", "season"], how="left",
    )
    out = out.with_columns(pl.col("sack_rate_pred").fill_null(_LEAGUE_MEAN_SACK_RATE_FALLBACK))
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
    sack_rate_pred = build_sack_rate_projection(pull_team_stats(refresh)).filter(
        pl.col("season") == season
    ).select("team", "sack_rate_pred")

    usage_table = assemble_future_usage_table(panel, season, refresh)
    usage_pred = predict_shares(usage_table, models["usage_models"])

    efficiency_pred = build_future_shrinkage_predictions(
        panel, future_population, models["eb_params"]
    )

    avail_table = build_future_availability_table(panel, future_population)
    avail_pred = predict_availability(avail_table, models["avail_fit"])

    base = future_population.select("gsis_id", "season", "team", "position")
    out = base.join(team_vol_pred, on="team", how="left")
    out = out.join(sack_rate_pred, on="team", how="left")
    out = out.join(usage_pred.drop("position"), on=["gsis_id", "season"], how="left")
    out = out.join(efficiency_pred.drop("position"), on=["gsis_id", "season"], how="left")
    out = out.join(
        avail_pred.select("gsis_id", "season", "pred_games_played", "pred_availability_prob"),
        on=["gsis_id", "season"], how="left",
    )
    out = out.with_columns(pl.col("sack_rate_pred").fill_null(_LEAGUE_MEAN_SACK_RATE_FALLBACK))
    return out.with_columns([pl.col(c).fill_null(0.0) for c in _ZERO_FILL_BEFORE_MULTIPLY])


_SHARE_COLS_TO_NORMALIZE = ["passer_share_pred"]


def normalize_shares_to_one(predictions: pl.DataFrame) -> pl.DataFrame:
    """target_share_pred/rush_share_pred/passer_share_pred are each fit as an
    independent per-player regression, not a constrained multinomial/softmax, so
    nothing forces a team's players to sum to 1.0 -- diagnosed directly: only ~25-36%
    of historical team-seasons land within +/-5% of 1.0 for any of the three shares,
    and the tails are large (e.g. WAS 2019 passer_share sums to 1.81, NO 2015
    target_share sums to 0.65). Unlike the receiving/passing yards gap (two
    independently-*noisy* efficiency estimates with no principled reason to trust one
    over the other -- tested, reverted), this is a hard constraint violation with only
    one correct answer: a team's own players cannot combine for more or less than
    100% of the team's own volume. Rescaling every player's share by their team's
    total (a simplex projection, not a damped blend) is the actual fix, applied
    before team volume is multiplied through so it's load-bearing for every
    downstream counting stat, not just yards.

    Tested all 3 shares in isolation via paired bootstrap (see git log): only
    passer_share_pred showed a real, consistent, significant win (pooled QB rank
    Spearman +0.112, p=0.04; +0.077 tune -> +0.152 test -- same direction, growing
    out of sample). target_share/rush_share renormalization sign-flipped or hurt
    between tune and test -- a QB team only carries 2-4 players, so its share-sum is a
    low-noise correction target; WR/RB/TE teams carry 8-12+, so the same correction
    divides by a noisier aggregate and adds noise instead of removing bias. Only
    passer_share_pred is normalized as a result; the other two are listed in the
    diagnostic above but deliberately left alone."""
    cols = _SHARE_COLS_TO_NORMALIZE
    sum_aliases = [f"_{c}_sum" for c in cols]
    team_totals = predictions.group_by(["team", "season"]).agg(
        [pl.col(c).sum().alias(f"_{c}_sum") for c in cols]
    )
    out = predictions.join(team_totals, on=["team", "season"], how="left")
    out = out.with_columns(
        [
            pl.when(pl.col(f"_{c}_sum") > 0)
            .then(pl.col(c) / pl.col(f"_{c}_sum"))
            .otherwise(pl.col(c))
            .alias(c)
            for c in cols
        ]
    )
    return out.drop(sum_aliases)


def project_raw_stats(predictions: pl.DataFrame) -> pl.DataFrame:
    """Multiplies team volume x usage share x efficiency into projected raw counting
    stats, in the exact column names src/scoring.py's compute_fantasy_points expects.

    team_plays_pred x pass_rate_pred gives team DROPBACKS (Step 3's own target
    definition includes sacks -- team_pass_attempts = attempts + sacks_suffered, see
    build_team_totals). Bug found and fixed here: target_share_pred and
    passer_share_pred were being applied directly to that dropback-inclusive total,
    but their own denominators (team_total_targets, and player-level pass_attempts)
    both exclude sacks -- verified directly that this overstated every team's true
    pass attempts by a real, consistent amount (median 39 attempts/team/season,
    matching real NFL sack totals almost exactly, not noise), which flowed straight
    through into overstated targets/pass attempts and therefore receiving and
    passing yards/TDs/INTs leaguewide. Converting dropbacks to true pass attempts via
    the projected sack rate before applying either share fixes this at the source."""
    predictions = normalize_shares_to_one(predictions)
    out = predictions.with_columns(
        (pl.col("team_plays_pred") * pl.col("pass_rate_pred")).alias("team_dropbacks_pred"),
        (pl.col("team_plays_pred") * (1 - pl.col("pass_rate_pred"))).alias("team_rush_attempts_pred"),
    )
    out = out.with_columns(
        (pl.col("team_dropbacks_pred") * (1 - pl.col("sack_rate_pred"))).alias("team_pass_attempts_pred"),
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
    combined_raw = compute_projected_points(predictions, scoring)

    # Rookie blend runs BEFORE calibration, not after: it replaces the model's
    # own (known-bad, CLAUDE.md-flagged) rookie predictions with a consensus-
    # anchored estimate, so calibration -- which corrects compounding-shrinkage
    # compression in the *model's own* veteran predictions -- fits and applies
    # against the already-sensible post-rookie-blend population instead of
    # partly against raw rookie noise. Tried the reverse order first: it let a
    # handful of rookies who also ranked near the top of their position get a
    # calibration boost before being blended, which changed their blended
    # result and cost 2024's QB Spearman ~0.03 versus running calibration in
    # isolation -- small, but avoidable, so the order was fixed instead of
    # accepted.
    combined_raw = blend_rookie_projections(combined_raw, panel, refresh)

    # Fit and apply the post-hoc points calibration (see src/models/calibration.py):
    # the raw model systematically underpredicts elite players' season points, a
    # compounding-shrinkage effect confirmed on the 2024/2025 backtest. Fit on
    # 2023-2024 (out-of-sample from every component), saved so main_future() can
    # reuse the exact same curve without recomputing the whole historical fit.
    calib_table = build_calibration_table(combined_raw, panel, refresh)
    curve = fit_calibration_curve(calib_table, CALIBRATION_TRAIN_SEASONS)
    save_curve(curve)
    combined = apply_calibration(combined_raw, curve)

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
    combined_raw = compute_projected_points(predictions, scoring)

    # Rookie blend before calibration -- see main()'s docstring comment for why.
    combined_raw = blend_future_rookie_projections(combined_raw, panel, future_population, season, refresh)

    # Reuses the curve main() already fit on 2023-2024 and saved -- avoids
    # recomputing the entire historical team_volume/usage_share/efficiency/
    # availability pipeline a second time just to get the correction.
    curve = load_curve()
    combined = apply_calibration(combined_raw, curve)

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
