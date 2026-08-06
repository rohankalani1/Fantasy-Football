"""Step 3 of PLAN.md: team volume projection -- per-team projected 2026 offensive plays
and pass/run split, from prior-two-seasons pace/PROE, a head-coach-change flag, and
Vegas preseason win total.

Small N by design: 32 teams x ~11 usable seasons once both trailing-season features
exist (features start in 2015 -- 2013 and 2014 don't have two full prior seasons in this
panel) is ~350 rows. Per CLAUDE.md's small-N guidance this uses a regularized linear
model (Ridge), not gradient boosting.

Every feature here is knowable well before Aug 1 of the season it predicts: pace/PROE
come from *completed* prior seasons (t-1, t-2), NFL head-coach hiring is essentially
always finished by February, and win-total futures are posted by books in early summer.
Season t's own realized team_plays/pass_rate is the label, never a feature.
"""

import json
import os

import numpy as np
import polars as pl
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.features.team_volume import build_team_volume_features
from src.ingest.build_panel import build_team_totals
from src.ingest.constants import RESULTS_DIR, SEASONS
from src.ingest.pull_raw import pull_pbp, pull_team_stats
from src.ingest.vegas_win_totals import pull_vegas_win_totals

FEATURE_COLS = ["pace_t1", "pace_t2", "proe_t1", "proe_t2", "hc_change_flag", "vegas_win_total"]
TARGET_COLS = ["team_plays", "pass_rate"]

TRAIN_MAX_SEASON = 2022
TUNE_SEASONS = [2023, 2024]
TEST_SEASON = 2025
_ALPHA_GRID = [0.01, 0.1, 1.0, 10.0, 50.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0]

CURRENT_SEASON_INPUTS_PATH = "config/current_season_inputs.csv"
PROJECTION_SEASON = 2026


def assemble_training_table(refresh: bool = False) -> pl.DataFrame:
    pbp = pull_pbp(refresh)
    team_stats = pull_team_stats(refresh)

    feats = build_team_volume_features(pbp)
    vegas = pull_vegas_win_totals(refresh)
    totals = build_team_totals(team_stats).filter(pl.col("season").is_in(SEASONS))

    trailing1 = (
        feats.select("team", "season", "pace", "proe")
        .with_columns((pl.col("season") + 1).alias("season"))
        .rename({"pace": "pace_t1", "proe": "proe_t1"})
    )
    trailing2 = (
        feats.select("team", "season", "pace", "proe")
        .with_columns((pl.col("season") + 2).alias("season"))
        .rename({"pace": "pace_t2", "proe": "proe_t2"})
    )
    hc = feats.select("team", "season", "hc_change_flag")

    table = totals.join(trailing1, on=["team", "season"], how="left")
    table = table.join(trailing2, on=["team", "season"], how="left")
    table = table.join(hc, on=["team", "season"], how="left")
    table = table.join(vegas, on=["team", "season"], how="left")

    table = table.with_columns(
        (pl.col("team_pass_attempts") / pl.col("team_plays")).alias("pass_rate")
    )
    return table.sort(["season", "team"])


def fit_team_volume_model(table: pl.DataFrame) -> tuple[dict, dict]:
    """Temporal split per CLAUDE.md: train <=2022, tune on 2023-2024 (alpha selection),
    touch 2025 exactly once as the final test, refitting on train+tune first."""
    complete = table.drop_nulls(FEATURE_COLS + TARGET_COLS)

    train = complete.filter(pl.col("season") <= TRAIN_MAX_SEASON)
    tune = complete.filter(pl.col("season").is_in(TUNE_SEASONS))
    test = complete.filter(pl.col("season") == TEST_SEASON)

    X_train = train.select(FEATURE_COLS).to_numpy()
    X_tune = tune.select(FEATURE_COLS).to_numpy()
    X_test = test.select(FEATURE_COLS).to_numpy()
    X_train_tune = np.vstack([X_train, X_tune])

    results = {}
    models = {}
    for target in TARGET_COLS:
        y_train = train.select(target).to_numpy().ravel()
        y_tune = tune.select(target).to_numpy().ravel()
        y_test = test.select(target).to_numpy().ravel()
        y_train_tune = np.concatenate([y_train, y_tune])

        # Features span wildly different scales (pace ~60-70, PROE +-10, win_total
        # 3.5-12.5, hc_change_flag 0/1) -- unscaled Ridge penalizes large-scale features
        # far more harshly than small-scale ones for the same alpha, which silently
        # distorts which features get shrunk. Standardize inside the pipeline (fit on
        # train only, to avoid leaking tune/test statistics into the scaler).
        best_alpha, best_tune_mae = None, np.inf
        for alpha in _ALPHA_GRID:
            model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
            model.fit(X_train, y_train)
            mae = float(np.mean(np.abs(model.predict(X_tune) - y_tune)))
            if mae < best_tune_mae:
                best_tune_mae, best_alpha = mae, alpha

        final_model = make_pipeline(StandardScaler(), Ridge(alpha=best_alpha))
        final_model.fit(X_train_tune, y_train_tune)
        test_mae = float(np.mean(np.abs(final_model.predict(X_test) - y_test)))
        naive_mae = float(np.mean(np.abs(y_test - y_train_tune.mean())))

        ridge_step = final_model.named_steps["ridge"]
        results[target] = {
            "best_alpha": best_alpha,
            "tune_mae": best_tune_mae,
            "test_mae": test_mae,
            "naive_mean_baseline_test_mae": naive_mae,
            "n_train_tune": len(y_train_tune),
            "n_test": len(y_test),
            "standardized_coefficients": dict(zip(FEATURE_COLS, ridge_step.coef_.tolist())),
            "intercept": float(ridge_step.intercept_),
        }
        models[target] = final_model

    return results, models


def project_historical(table: pl.DataFrame, models: dict) -> pl.DataFrame:
    """Applies the fitted models to every (team, season) row in the already-assembled
    training table, including seasons the models themselves were trained/tuned on.

    This exists for Step 4's "projected team pass rate from Step 3" feature: using the
    *realized* pass_rate for historical seasons instead of what this model would have
    projected is a real leakage violation (the realized value isn't knowable as of Aug 1
    -- only a projection is), and it would also create a train/inference mismatch (clean
    ground truth at Step 4 training time, a noisy model prediction at Step 4 inference
    time). Every historical row gets the same kind of feature 2026 will get: this
    model's own output, not the answer key."""
    complete = table.drop_nulls(FEATURE_COLS)
    X = complete.select(FEATURE_COLS).to_numpy()
    complete = complete.with_columns(
        pl.Series("team_plays_pred", models["team_plays"].predict(X)),
        pl.Series("pass_rate_pred", models["pass_rate"].predict(X)),
    )
    return complete.select("team", "season", "team_plays_pred", "pass_rate_pred")


def project_future_season(
    models: dict[str, Ridge], season: int, inputs_path: str = CURRENT_SEASON_INPUTS_PATH
) -> pl.DataFrame:
    """Projects team_plays/pass_rate for a not-yet-played `season` using season-1 and
    season-2 pace/PROE (already realized) plus manually supplied hc_change_flag and
    vegas_win_total for `season` itself (config/current_season_inputs.csv) -- neither is
    derivable from historical data since `season` hasn't been played yet. Rows for teams
    with a blank input are left with a null projection, not silently guessed."""
    current_inputs = pl.read_csv(inputs_path)

    feats = build_team_volume_features(pull_pbp())
    pace_t1 = feats.filter(pl.col("season") == season - 1).select(
        "team", pl.col("pace").alias("pace_t1"), pl.col("proe").alias("proe_t1")
    )
    pace_t2 = feats.filter(pl.col("season") == season - 2).select(
        "team", pl.col("pace").alias("pace_t2"), pl.col("proe").alias("proe_t2")
    )

    proj = pace_t1.join(pace_t2, on="team", how="full", coalesce=True)
    proj = proj.join(current_inputs, on="team", how="left")

    complete = proj.drop_nulls(FEATURE_COLS)
    incomplete = proj.filter(~pl.col("team").is_in(complete.select("team").to_series().to_list()))
    if incomplete.height > 0:
        print(
            f"[project_future_season] {incomplete.height} team(s) missing hc_change_flag/"
            f"vegas_win_total in {inputs_path} -- left unprojected: "
            f"{incomplete.select('team').to_series().to_list()}"
        )

    if complete.height == 0:
        return proj.with_columns(
            pl.lit(None, dtype=pl.Float64).alias("team_plays_pred"),
            pl.lit(None, dtype=pl.Float64).alias("pass_rate_pred"),
        )

    X = complete.select(FEATURE_COLS).to_numpy()
    complete = complete.with_columns(
        pl.Series("team_plays_pred", models["team_plays"].predict(X)),
        pl.Series("pass_rate_pred", models["pass_rate"].predict(X)),
    )
    complete = complete.with_columns(
        (pl.col("team_plays_pred") * pl.col("pass_rate_pred")).alias("team_pass_attempts_pred"),
        (pl.col("team_plays_pred") * (1 - pl.col("pass_rate_pred"))).alias("team_rush_attempts_pred"),
    )
    return pl.concat([complete, incomplete], how="diagonal_relaxed").sort("team")


def main(refresh: bool = False) -> dict:
    table = assemble_training_table(refresh)
    results, models = fit_team_volume_model(table)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "team_volume_scores.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {RESULTS_DIR}/team_volume_scores.json")

    projection = project_future_season(models, PROJECTION_SEASON)
    projection.write_csv(os.path.join(RESULTS_DIR, f"team_volume_projection_{PROJECTION_SEASON}.csv"))
    print(f"Wrote {RESULTS_DIR}/team_volume_projection_{PROJECTION_SEASON}.csv")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fit and score the Step 3 team volume model.")
    parser.add_argument("--refresh", action="store_true", help="Re-pull raw data from network.")
    args = parser.parse_args()
    main(refresh=args.refresh)
