"""Step 4 of PLAN.md: usage share model -- the core. Predicts each player's share of
team targets and share of team carries for season t+1.

Two combined-group LightGBM models (confirmed with the league owner, resolving an
apparent contradiction in PLAN.md between "per position group" and listing "position"
itself as a feature): target_share is trained on WR+RB+TE together, rush_share on
RB+QB+WR together, with position as a categorical feature in each -- this pools more
data per model than four fully separate per-position models would, which matters given
CLAUDE.md's small-N warning.

A third model, passer_share (QB only), was added in Step 7 -- passing volume wasn't
part of the original Step 4 design, an architecture gap only visible once Step 7 tried
to combine everything into points. Single-position by necessity (only QBs throw
passes in this project's scope), but built with the exact same feature set and fitting
procedure as the other two.

Every feature is knowable as of Aug 1 of the season being predicted: EWMA prior shares,
games played, snap share, and vacated share all come from season t and t-1 (completed
seasons); draft capital/years_exp/age/position are static facts about the player
entering the target season; team_pass_rate_projected is Step 3's own model *output* for
the target season, never the realized ground truth (see src/models/team_volume.py's
project_historical for why that distinction is a real leakage issue, not pedantry).
"""

import json
import os

import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl

from src.features.current_roster import build_future_population
from src.features.team_volume import build_pace
from src.features.usage_features import (
    build_ewma_prior_shares,
    build_future_vacated_share,
    build_shares,
    build_snap_share,
    build_vacated_share,
)
from src.ingest.constants import PROCESSED_DIR, RESULTS_DIR, SEASONS
from src.ingest.pull_raw import pull_pbp
from src.models.team_volume import assemble_training_table as build_team_volume_table
from src.models.team_volume import fit_team_volume_model, project_future_season, project_historical

FEATURE_COLS = [
    "ewma_target_share",
    "ewma_rush_share",
    "ewma_passer_share",
    "games_played_t1",
    "games_played_t2",
    "snap_share_t1",
    "vacated_target_share",
    "vacated_rush_share",
    "vacated_passer_share",
    "draft_round",
    "draft_pick",
    "years_exp",
    "age",
    "team_change_flag",
    "position",
    "team_pass_rate_projected",
]
CATEGORICAL_FEATURES = ["position"]

# draft_round/draft_pick are legitimately null for undrafted players (~44% of the
# panel) -- NOT missing data to drop. LightGBM handles NaN features natively (a real
# split branch, not an error), so these two are deliberately excluded from the
# drop_nulls gate below. Bug found and fixed here: the original drop_nulls covered
# every FEATURE_COLS entry, which silently dropped every undrafted player from
# training AND zeroed their predicted share at inference (verified: Austin Ekeler,
# James Robinson, Phillip Lindsay all showed target_share_pred=0.0 for their entire
# careers in the pre-fix output) -- a real accuracy bug, not a hypothetical.
_REQUIRED_NON_NULL_FEATURES = [c for c in FEATURE_COLS if c not in ("position", "draft_round", "draft_pick")]

TARGET_SHARE_POSITIONS = ["WR", "RB", "TE"]
RUSH_SHARE_POSITIONS = ["RB", "QB", "WR"]
PASSER_SHARE_POSITIONS = ["QB"]

TRAIN_MAX_SEASON = 2022
TUNE_SEASONS = [2023, 2024]
TEST_SEASON = 2025

# PLAN.md: "shallow trees (max_depth 3-4), strong L2, early stopping on 2023-2024."
_PARAM_GRID = [
    {"max_depth": 3, "reg_lambda": 5.0},
    {"max_depth": 3, "reg_lambda": 20.0},
    {"max_depth": 4, "reg_lambda": 5.0},
    {"max_depth": 4, "reg_lambda": 20.0},
]


def assemble_usage_training_table(panel: pl.DataFrame, refresh: bool = False) -> pl.DataFrame:
    """One row per (gsis_id, season) for QB/RB/WR/TE, season in SEASONS. target_share and
    rush_share are the season's own realized shares (the label); every other column is a
    feature built only from season-1/season-2 data, static player facts, or Step 3's own
    projection -- never this season's own realized outcome."""
    shares = build_shares(panel)
    ewma = build_ewma_prior_shares(shares)
    vacated = build_vacated_share(shares)

    pace = build_pace(pull_pbp(refresh))
    snap_share = (
        build_snap_share(panel, pace)
        .with_columns((pl.col("season") + 1).alias("season"))
        .rename({"snap_share": "snap_share_t1"})
    )

    team_vol_table = build_team_volume_table(refresh)
    _, team_vol_models = fit_team_volume_model(team_vol_table)
    pass_rate_proj = project_historical(team_vol_table, team_vol_models).select(
        "team", "season", pl.col("pass_rate_pred").alias("team_pass_rate_projected")
    )

    base = panel.select(
        "gsis_id", "season", "team", "position", "draft_round", "draft_pick",
        "years_exp", "age",
    )

    gp1 = (
        panel.select("gsis_id", "season", "games_played")
        .with_columns((pl.col("season") + 1).alias("season"))
        .rename({"games_played": "games_played_t1"})
    )
    gp2 = (
        panel.select("gsis_id", "season", "games_played")
        .with_columns((pl.col("season") + 2).alias("season"))
        .rename({"games_played": "games_played_t2"})
    )
    team_prior = (
        panel.select("gsis_id", "season", "team")
        .with_columns((pl.col("season") + 1).alias("season"))
        .rename({"team": "team_t1"})
    )

    out = base.join(
        shares.select("gsis_id", "season", "target_share", "rush_share", "passer_share"),
        on=["gsis_id", "season"], how="left",
    )
    out = out.join(ewma, on=["gsis_id", "season"], how="left")
    out = out.join(gp1, on=["gsis_id", "season"], how="left")
    out = out.join(gp2, on=["gsis_id", "season"], how="left")
    out = out.join(team_prior, on=["gsis_id", "season"], how="left")
    out = out.join(snap_share, on=["gsis_id", "season"], how="left")
    out = out.join(vacated, on=["team", "position", "season"], how="left")
    out = out.join(pass_rate_proj, on=["team", "season"], how="left")
    out = _fill_usage_feature_nulls(out)
    return out.filter(pl.col("season").is_in(SEASONS)).sort(["season", "team", "position"])


def _fill_usage_feature_nulls(out: pl.DataFrame) -> pl.DataFrame:
    """team_change_flag plus the null-fill pass shared by the historical training table
    and the live future-season table: ewma's own fill_null only covers rows that exist
    in its output; joining it onto a wider (every QB/RB/WR/TE row) population
    reintroduces nulls for rows absent from ewma entirely, so these need filling again
    here too."""
    return out.with_columns(
        pl.when(pl.col("team_t1").is_not_null())
        .then((pl.col("team") != pl.col("team_t1")).cast(pl.Int8))
        .otherwise(0)
        .alias("team_change_flag"),
        pl.col("ewma_target_share").fill_null(0.0),
        pl.col("ewma_rush_share").fill_null(0.0),
        pl.col("ewma_passer_share").fill_null(0.0),
        pl.col("target_share_t1").fill_null(0.0),
        pl.col("rush_share_t1").fill_null(0.0),
        pl.col("passer_share_t1").fill_null(0.0),
        pl.col("games_played_t1").fill_null(0),
        pl.col("games_played_t2").fill_null(0),
        pl.col("snap_share_t1").fill_null(0.0),
        pl.col("vacated_target_share").fill_null(0.0),
        pl.col("vacated_rush_share").fill_null(0.0),
        pl.col("vacated_passer_share").fill_null(0.0),
    )


def assemble_future_usage_table(
    panel: pl.DataFrame, season: int, refresh: bool = False
) -> pl.DataFrame:
    """Same feature set and columns as assemble_usage_training_table, keyed to the live
    `season` roster population instead of panel's own historical rows -- there is no
    label column here (target_share/rush_share/passer_share for `season` don't exist
    yet), only the inputs predict_shares() needs. Every input is either a real,
    already-completed prior season (ewma shares, games played, snap share, all from
    panel through season-1) or Step 3's own future-season projection (never realized
    ground truth), matching this project's as-of-Aug-1 discipline for a live season
    exactly the same way it already does for historical ones.
    """
    future_population = build_future_population(season, refresh)

    shares = build_shares(panel)
    ewma = build_ewma_prior_shares(shares).filter(pl.col("season") == season)
    vacated = build_future_vacated_share(shares, future_population, season - 1)

    pace = build_pace(pull_pbp(refresh))
    snap_share = (
        build_snap_share(panel, pace)
        .with_columns((pl.col("season") + 1).alias("season"))
        .rename({"snap_share": "snap_share_t1"})
        .filter(pl.col("season") == season)
    )

    team_vol_table = build_team_volume_table(refresh)
    _, team_vol_models = fit_team_volume_model(team_vol_table)
    pass_rate_proj = project_future_season(team_vol_models, season).select(
        "team", pl.col("pass_rate_pred").alias("team_pass_rate_projected")
    )

    base = future_population.select(
        "gsis_id", "season", "team", "position", "draft_round", "draft_pick",
        "years_exp", "age",
    )

    gp1 = (
        panel.select("gsis_id", "season", "games_played")
        .with_columns((pl.col("season") + 1).alias("season"))
        .rename({"games_played": "games_played_t1"})
        .filter(pl.col("season") == season)
    )
    gp2 = (
        panel.select("gsis_id", "season", "games_played")
        .with_columns((pl.col("season") + 2).alias("season"))
        .rename({"games_played": "games_played_t2"})
        .filter(pl.col("season") == season)
    )
    team_prior = (
        panel.select("gsis_id", "season", "team")
        .with_columns((pl.col("season") + 1).alias("season"))
        .rename({"team": "team_t1"})
        .filter(pl.col("season") == season)
    )

    out = base.join(ewma, on=["gsis_id", "season"], how="left")
    out = out.join(gp1, on=["gsis_id", "season"], how="left")
    out = out.join(gp2, on=["gsis_id", "season"], how="left")
    out = out.join(team_prior, on=["gsis_id", "season"], how="left")
    out = out.join(snap_share, on=["gsis_id", "season"], how="left")
    out = out.join(vacated, on=["team", "position", "season"], how="left")
    out = out.join(pass_rate_proj, on="team", how="left")
    out = _fill_usage_feature_nulls(out)
    return out.sort(["team", "position"])


def _to_lgb_frame(df: pl.DataFrame) -> pd.DataFrame:
    pdf = df.select(FEATURE_COLS).to_pandas()
    for c in CATEGORICAL_FEATURES:
        pdf[c] = pdf[c].astype("category")
    return pdf


def _fit_one_share_model(table: pl.DataFrame, target: str, positions: list[str]) -> tuple[dict, lgb.LGBMRegressor]:
    complete = table.filter(pl.col("position").is_in(positions)).drop_nulls(
        _REQUIRED_NON_NULL_FEATURES + [target]
    )
    train = complete.filter(pl.col("season") <= TRAIN_MAX_SEASON)
    tune = complete.filter(pl.col("season").is_in(TUNE_SEASONS))
    test = complete.filter(pl.col("season") == TEST_SEASON)

    X_train, y_train = _to_lgb_frame(train), train.select(target).to_numpy().ravel()
    X_tune, y_tune = _to_lgb_frame(tune), tune.select(target).to_numpy().ravel()
    X_test, y_test = _to_lgb_frame(test), test.select(target).to_numpy().ravel()

    best_params, best_iter, best_mae = None, None, np.inf
    for params in _PARAM_GRID:
        model = lgb.LGBMRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=params["max_depth"],
            reg_lambda=params["reg_lambda"],
            num_leaves=2 ** params["max_depth"],
            min_child_samples=20,
            verbosity=-1,
        )
        model.fit(
            X_train, y_train,
            eval_X=X_tune, eval_y=y_tune,
            categorical_feature=CATEGORICAL_FEATURES,
            callbacks=[lgb.early_stopping(30, verbose=False)],
        )
        mae = float(np.mean(np.abs(model.predict(X_tune) - y_tune)))
        if mae < best_mae:
            best_mae, best_params, best_iter = mae, params, model.best_iteration_

    # Refit on train+tune at the chosen depth/L2 and the tune-selected iteration count,
    # matching Step 3's discipline: tune only selects hyperparameters, then the final
    # model is refit on all non-test data before the single held-out 2025 test.
    X_train_tune = pd.concat([X_train, X_tune], ignore_index=True)
    y_train_tune = np.concatenate([y_train, y_tune])
    final_model = lgb.LGBMRegressor(
        n_estimators=best_iter,
        learning_rate=0.05,
        max_depth=best_params["max_depth"],
        reg_lambda=best_params["reg_lambda"],
        num_leaves=2 ** best_params["max_depth"],
        min_child_samples=20,
        verbosity=-1,
    )
    final_model.fit(X_train_tune, y_train_tune, categorical_feature=CATEGORICAL_FEATURES)
    test_pred = final_model.predict(X_test)
    test_mae = float(np.mean(np.abs(test_pred - y_test)))

    # Naive baseline: last season's raw (unweighted) share, carried forward -- same
    # "dumb baseline" discipline as Step 2, applied locally to this component. The
    # EWMA feature itself is already a 2:1 t1/t2 blend, so the naive reference point
    # uses the raw single-season t1 column instead, not the (smarter) EWMA feature.
    prior_col = f"{target}_t1"
    naive_pred = test.select(prior_col).to_numpy().ravel()
    naive_mae = float(np.mean(np.abs(naive_pred - y_test)))

    results = {
        "best_params": best_params,
        "best_iteration": best_iter,
        "tune_mae": best_mae,
        "test_mae": test_mae,
        "naive_carryforward_test_mae": naive_mae,
        "n_train_tune": len(y_train_tune),
        "n_test": len(y_test),
        "feature_importance": dict(
            zip(FEATURE_COLS, final_model.feature_importances_.tolist())
        ),
    }
    return results, final_model


_MODEL_POSITIONS = {
    "target_share": TARGET_SHARE_POSITIONS,
    "rush_share": RUSH_SHARE_POSITIONS,
    "passer_share": PASSER_SHARE_POSITIONS,
}


def predict_shares(table: pl.DataFrame, models: dict) -> pl.DataFrame:
    """Applies each fitted share model to every row of `table` it applies to (rows
    outside that model's position group, or with a null required feature, get 0 --
    e.g. a TE's rush_share_pred is 0 because TEs essentially never rush, not because
    the model failed). Used by Step 7's combine to turn share models into projected
    raw counting stats for every player-season at once.

    Clipped to [0, 1]: LightGBM is an unconstrained regressor, and for deep-bench
    players near 0 it can (and does -- verified directly: 117 rows with a negative
    target_share_pred, 536 with negative rush_share_pred) predict a small negative
    share as ordinary regression noise. A negative share means negative projected
    targets/carries downstream in Step 7's combine, which can produce a negative
    total-season point projection for a real (if irrelevant) player -- nonsensical,
    not a modeling nuance worth preserving.
    """
    out = table.select("gsis_id", "season", "position")
    for target, positions in _MODEL_POSITIONS.items():
        eligible = table.filter(pl.col("position").is_in(positions)).drop_nulls(
            _REQUIRED_NON_NULL_FEATURES
        )
        X = _to_lgb_frame(eligible)
        raw_pred = np.clip(models[target].predict(X), 0.0, 1.0)
        pred = eligible.select("gsis_id", "season").with_columns(
            pl.Series(f"{target}_pred", raw_pred)
        )
        out = out.join(pred, on=["gsis_id", "season"], how="left")
        out = out.with_columns(pl.col(f"{target}_pred").fill_null(0.0))
    return out


def main(refresh: bool = False) -> dict:
    panel = pl.read_parquet(os.path.join(PROCESSED_DIR, "player_season_panel.parquet"))
    table = assemble_usage_training_table(panel, refresh)

    results = {}
    models = {}
    for target, positions in [
        ("target_share", TARGET_SHARE_POSITIONS),
        ("rush_share", RUSH_SHARE_POSITIONS),
        ("passer_share", PASSER_SHARE_POSITIONS),
    ]:
        r, model = _fit_one_share_model(table, target, positions)
        results[target] = r
        models[target] = model

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "usage_share_scores.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {RESULTS_DIR}/usage_share_scores.json")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fit and score the Step 4 usage share models.")
    parser.add_argument("--refresh", action="store_true", help="Re-pull raw data from network.")
    args = parser.parse_args()
    main(refresh=args.refresh)
