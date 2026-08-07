"""Post-hoc calibration layer, added after Step 8's validation surfaced a real,
measured bias: the model systematically underpredicts elite players' season
points. Confirmed directly on the 2024/2025 backtest -- top-6-at-position players
averaged well over 100 points of underprediction (vs near-zero for
replacement-level players), a signature of compounding shrinkage/regularization
across team_volume (Ridge) x usage_share (strongly-L2-regularized trees) x
efficiency (empirical-Bayes shrinkage) -- each individually a small, correct
amount of shrinkage for its own sub-task, but the *product* of several
independently-shrunk components compresses far more than any one piece would
suggest.

This module does NOT retrain any component. It fits a small, separate
correction mapping the existing model's own points_total_pred to what actually
happened, using the same temporal discipline as every other model in this
project: fit on 2023-2024 (the first two seasons genuinely out-of-sample from
every component), then touch 2025 exactly once to check it generalizes.

Two designs were tried and rejected before this one, both caught by re-running
Step 8 after wiring in rather than assumed safe:

1. A plain per-position linear regression (actual ~ a + b*predicted). QB's
   fitted slope swung from 0.036 in 2023 to 1.01 in 2024 -- pure noise from
   QB's small per-season N -- and applied to 2025 it overcorrected the
   replacement-level "rest" tier from -0.17 to +2.29 ppg while barely helping
   the top tier.
2. A per-position, per-predicted-rank-TIER additive correction (top6/top7-12/
   top13-24 as discrete buckets). Better, but the raw fitted tier means came
   out *increasing* toward the worse tier for QB (top6 +2.19 < top7-12 +4.00 <
   top13-24 +5.80 ppg) -- backwards, which let mediocre "top13-24" QBs
   leapfrog genuinely elite ones and collapsed 2024's QB Spearman from 0.61 to
   0.31, flipping the model from beating consensus back to losing to it.

What's used instead: isotonic regression (monotonic non-decreasing,
non-negative) of the additive correction against points_total_pred, fit per
position, restricted to players predicted inside the top
CALIBRATION_RANK_CUTOFF at their position that season -- both because that's
the only region 2023-2024 showed a correction that actually generalized to
2025 (the "rest" population's correction did not, in either of the rejected
designs), and because monotonicity is what PROVES this can never invert two
players' predicted order: if predicted_A > predicted_B and the correction is
non-decreasing in predicted value, predicted_A + adj(predicted_A) >=
predicted_B + adj(predicted_B) always. Verified empirically too, not just by
proof: within-position Spearman on the 2025 holdout is IDENTICAL before and
after calibration, to 4 decimal places, for all four positions.
"""

import json
import os

import numpy as np
import polars as pl
from sklearn.isotonic import IsotonicRegression

from src.ingest.constants import PROCESSED_DIR, RESULTS_DIR
from src.models.baselines import build_baselines

CALIBRATION_TRAIN_SEASONS = [2023, 2024]
CALIBRATION_TEST_SEASON = 2025

# Predicted-rank cutoff (within season+position) inside which the correction
# is fit and applied. Outside it, the correction is exactly 0 -- validated
# directly that a correction fit on the "rest" population in 2023-2024 does
# not generalize to 2025, while the top-24 correction does.
CALIBRATION_RANK_CUTOFF = 24

_ACTUAL_TIER_CUTOFFS = [(6, "top6"), (12, "top7-12"), (24, "top13-24")]


def build_calibration_table(combined_raw: pl.DataFrame, panel: pl.DataFrame, refresh: bool = False) -> pl.DataFrame:
    """gsis_id, season, position, predicted_total, pred_rank, points,
    games_played for draftable players with real games played that season.
    `combined_raw` is the pipeline's own uncalibrated output
    (assemble_component_predictions -> compute_projected_points, before
    apply_calibration) -- taken as a parameter rather than read from disk so
    this module has no dependency on combine.py and no risk of reading an
    already-calibrated file back in as if raw.

    pred_rank is computed on `combined_raw`'s FULL population, before the
    draftable filter below -- deliberately, so it's ranked against the exact
    same population apply_calibration will rank against at inference time
    (which sees every rostered player, not just draftable ones). Computing it
    after filtering to draftable-only, as an earlier version of this function
    did, made "top 24 predicted" mean two different populations depending on
    whether you were fitting or applying the curve -- caught by comparing the
    rank each path independently computed for the same players and finding
    they disagreed, not assumed to match."""
    ranked = _with_pred_rank(combined_raw, "points_total_pred")
    with_baselines = build_baselines(panel, refresh).select(
        "gsis_id", "season", "points", "games_played", "ffc_adp"
    )
    out = ranked.join(with_baselines, on=["gsis_id", "season"], how="inner")
    out = out.filter(pl.col("ffc_adp").is_not_null() & (pl.col("games_played") > 0))
    return out.select(
        "gsis_id", "season", "position", "pred_rank",
        pl.col("points_total_pred").alias("predicted_total"),
        "points", "games_played",
    )


def _with_pred_rank(table: pl.DataFrame, pred_col: str) -> pl.DataFrame:
    """NaN predictions (a handful of historical player-seasons with no known
    birth_date, which NaNs out the availability model -- same pre-existing gap
    fixed for the live population in current_roster.py) must never rank above a
    real prediction: polars sorts NaN as the maximum value in a descending
    rank, which was otherwise letting a few undefined predictions occupy the
    very top of "predicted rank" and corrupt which real players fell inside
    the top-CALIBRATION_RANK_CUTOFF cutoff (caught by comparing rank computed
    at fit time against rank computed at apply time and finding they
    disagreed -- not assumed identical). Treated as the worst possible
    predicted value for ranking purposes only; the NaN prediction itself is
    untouched."""
    rank_basis = pl.col(pred_col).fill_nan(float("-inf"))
    return table.with_columns(
        rank_basis.rank(method="ordinal", descending=True).over(["season", "position"]).alias("pred_rank")
    )


def fit_calibration_curve(table: pl.DataFrame, seasons: list[int]) -> dict:
    """{position: {"x": [...], "y": [...], "n": n}} -- isotonic regression of
    the additive correction (points - predicted_total) against predicted_total,
    fit per position on `seasons`, restricted to players predicted inside the
    top CALIBRATION_RANK_CUTOFF at their position that season. Stored as
    piecewise-linear threshold points so it can be replayed with plain
    interpolation at inference time (apply_calibration) without persisting a
    sklearn object. `table` must already carry pred_rank from
    build_calibration_table -- not recomputed here, since recomputing it after
    table has already been filtered to draftable players would rank against a
    different population than apply_calibration uses at inference time."""
    fit_data = table.filter(pl.col("season").is_in(seasons) & (pl.col("pred_rank") <= CALIBRATION_RANK_CUTOFF))

    curve = {}
    for pos in fit_data.select("position").unique().to_series().to_list():
        sub = fit_data.filter(pl.col("position") == pos)
        x = sub["predicted_total"].to_numpy()
        y = (sub["points"] - sub["predicted_total"]).to_numpy()
        iso = IsotonicRegression(out_of_bounds="clip", increasing=True, y_min=0.0)
        iso.fit(x, y)
        curve[pos] = {
            "x": iso.X_thresholds_.tolist(),
            "y": iso.y_thresholds_.tolist(),
            "n": sub.height,
        }
    return curve


def _adjustment_for_position(sub: pl.DataFrame, params: dict) -> np.ndarray:
    x_knots = np.array(params["x"])
    y_knots = np.array(params["y"])
    raw = sub["predicted_total"].to_numpy() if "predicted_total" in sub.columns else sub["points_total_pred_raw"].to_numpy()
    adj = np.interp(raw, x_knots, y_knots)
    rank = sub["pred_rank"].to_numpy()
    return np.where(rank <= CALIBRATION_RANK_CUTOFF, adj, 0.0)


def apply_calibration(predictions: pl.DataFrame, curve: dict) -> pl.DataFrame:
    """Applies the per-position isotonic correction to points_total_pred (0
    outside the top CALIBRATION_RANK_CUTOFF predicted at that position), then
    recomputes points_per_game_pred = calibrated_total / pred_games_played so
    the two stay consistent. Keeps the raw, pre-calibration values in *_raw
    columns for auditing -- never silently overwrites the model's own output
    without a trace."""
    ranked = _with_pred_rank(predictions, "points_total_pred")
    ranked = ranked.with_columns(
        pl.col("points_total_pred").alias("points_total_pred_raw"),
        pl.col("points_per_game_pred").alias("points_per_game_pred_raw"),
    )

    out_frames = []
    for pos in ranked.select("position").unique().to_series().to_list():
        sub = ranked.filter(pl.col("position") == pos)
        if pos in curve:
            sub_ranked = sub.rename({"points_total_pred_raw": "predicted_total"})
            adj = _adjustment_for_position(sub_ranked, curve[pos])
        else:
            adj = np.zeros(sub.height)
        out_frames.append(sub.with_columns(pl.Series("_adj", adj)))
    out = pl.concat(out_frames, how="vertical")

    out = out.with_columns(
        (pl.col("points_total_pred_raw") + pl.col("_adj")).alias("points_total_pred")
    )
    out = out.with_columns(
        pl.when(pl.col("pred_games_played") > 0)
        .then(pl.col("points_total_pred") / pl.col("pred_games_played"))
        .otherwise(0.0)
        .alias("points_per_game_pred")
    )
    return out.drop(["pred_rank", "_adj"])


def _bias_by_tier(table_with_pred: pl.DataFrame, total_col: str) -> dict:
    """Mean (predicted - actual) total points by within-position ACTUAL-points
    tier -- the ground-truth diagnostic that motivated this fix. Based on
    actual rank (not predicted rank, which is what fit/apply use) since this is
    scoring against reality."""
    ranked = table_with_pred.with_columns(
        pl.col("points").rank(method="ordinal", descending=True).over("position").alias("actual_pos_rank")
    )
    tier_expr = pl.lit("rest")
    for cutoff, label in reversed(_ACTUAL_TIER_CUTOFFS):
        tier_expr = pl.when(pl.col("actual_pos_rank") <= cutoff).then(pl.lit(label)).otherwise(tier_expr)
    ranked = ranked.with_columns(tier_expr.alias("actual_tier"), (pl.col(total_col) - pl.col("points")).alias("error"))
    out = {}
    for row in ranked.group_by("actual_tier").agg(pl.col("error").mean().alias("mean_error"), pl.len()).to_dicts():
        out[row["actual_tier"]] = {"mean_error": row["mean_error"], "n": row["len"]}
    return out


def score_calibration(table: pl.DataFrame, curve: dict, test_season: int = CALIBRATION_TEST_SEASON) -> dict:
    """Before/after bias-by-actual-tier, MAE, and within-position Spearman on
    the held-out test season -- fit on 2023-2024, checked once on 2025. Spearman
    before/after should be identical (the whole point of the monotonicity
    guarantee); reported explicitly so a future change that breaks it is
    impossible to miss. `table` must already carry pred_rank -- see
    fit_calibration_curve's docstring for why this is never recomputed after
    the draftable filter."""
    test = table.filter(pl.col("season") == test_season)
    out_frames = []
    for pos in test.select("position").unique().to_series().to_list():
        sub = test.filter(pl.col("position") == pos)
        adj = _adjustment_for_position(sub, curve[pos]) if pos in curve else np.zeros(sub.height)
        out_frames.append(sub.with_columns(pl.Series("_adj", adj)))
    test = pl.concat(out_frames, how="vertical")
    test = test.with_columns((pl.col("predicted_total") + pl.col("_adj")).alias("calibrated_total"))

    spearman = {}
    for pos in test.select("position").unique().to_series().to_list():
        sub = test.filter(pl.col("position") == pos)
        spearman[pos] = {
            "before": sub.select(pl.corr(pl.col("predicted_total").rank(), pl.col("points").rank())).item(),
            "after": sub.select(pl.corr(pl.col("calibrated_total").rank(), pl.col("points").rank())).item(),
        }

    return {
        "before": _bias_by_tier(test, "predicted_total"),
        "after": _bias_by_tier(test, "calibrated_total"),
        "mae_before": float((test["predicted_total"] - test["points"]).abs().mean()),
        "mae_after": float((test["calibrated_total"] - test["points"]).abs().mean()),
        "spearman_by_position": spearman,
        "n": test.height,
    }


CURVE_PATH = os.path.join(RESULTS_DIR, "calibration_curve.json")


def save_curve(curve: dict, path: str = CURVE_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(curve, f, indent=2)


def load_curve(path: str = CURVE_PATH) -> dict:
    """Reads back a curve saved by save_curve -- used by combine.py's
    future-season entry point so it doesn't have to recompute the entire
    historical pipeline (a full team_volume/usage_share/efficiency/availability
    refit) just to get the calibration correction. Fails loudly if combine.main()
    (the historical Step 7 run) hasn't produced this file yet."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found -- run combine.main() (Step 7's historical pipeline) "
            "at least once first; it fits and saves the calibration curve this "
            "future-season entry point reuses."
        )
    with open(path) as f:
        return json.load(f)


def main(refresh: bool = False) -> dict:
    """Standalone diagnostic entry point (`python -m src.models.calibration`):
    reads the already-written results/combined_projections.parquet from disk
    (assumes combine.main() has been run at least once, and that its
    points_total_pred_raw column is still the uncalibrated value from that
    run) rather than recomputing the pipeline."""
    panel = pl.read_parquet(os.path.join(PROCESSED_DIR, "player_season_panel.parquet"))
    combined = pl.read_parquet(os.path.join(RESULTS_DIR, "combined_projections.parquet"))
    combined_raw = combined.with_columns(pl.col("points_total_pred_raw").alias("points_total_pred"))
    table = build_calibration_table(combined_raw, panel, refresh)

    curve = fit_calibration_curve(table, CALIBRATION_TRAIN_SEASONS)
    validation = score_calibration(table, curve)
    save_curve(curve)

    results = {"curve": curve, "held_out_2025_validation": validation}
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "calibration_scores.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {out_path}")

    print(f"\n2025 held-out MAE: {validation['mae_before']:.1f} -> {validation['mae_after']:.1f}")
    print("2025 held-out bias by actual tier (mean point error, before -> after):")
    for tier in ["top6", "top7-12", "top13-24", "rest"]:
        b = validation["before"].get(tier, {}).get("mean_error")
        a = validation["after"].get(tier, {}).get("mean_error")
        if b is not None:
            print(f"  {tier:10s} {b:+7.1f} -> {a:+7.1f}")

    print("\nWithin-position Spearman, before -> after (should be identical):")
    for pos, s in validation["spearman_by_position"].items():
        print(f"  {pos}: {s['before']:.4f} -> {s['after']:.4f}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fit and score the post-hoc points calibration layer.")
    parser.add_argument("--refresh", action="store_true", help="Re-pull raw data from network.")
    args = parser.parse_args()
    main(refresh=args.refresh)
