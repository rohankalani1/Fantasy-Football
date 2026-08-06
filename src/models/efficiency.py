"""Step 5 of PLAN.md: efficiency model. Empirical Bayes shrinkage of each player's
career rate toward a position-and-role mean, for 8 rate stats: yards per target (ypt),
touchdown rate on targets (td_rate_receiving), catch rate (receptions/target), yards
per carry (ypc), touchdown rate on carries (td_rate_rushing), and three QB passing
stats added in Step 7 (yards/attempt, passing TD rate, INT rate) -- passing production
wasn't part of the original Steps 4-5 design, a gap only visible once Step 7 tried to
combine everything into actual fantasy points.

"Position-and-role mean" = the shrinkage prior is fit separately per (position, stat)
group -- a WR's rushing YPC (jet sweeps) is a different population from a workhorse
RB's, so they get different priors, not one pooled rushing mean.

"Position-and-role mean" = the shrinkage prior is fit separately per (position, stat)
group -- a WR's rushing YPC (jet sweeps) is a different population from a workhorse
RB's, so they get different priors, not one pooled rushing mean.

Shrinkage weight k is derived via method-of-moments empirical Bayes, not chosen by
hand: a one-way random-effects variance decomposition estimates how much of the
observed spread in players' rates is real (between-player) skill differences versus
opportunity-driven sampling noise. This is what lets the model discover for itself
whether touchdown rate really does need near-total shrinkage (CLAUDE.md's claim,
verified empirically here rather than assumed) or not.

PLAN.md requires demonstrating that shrinkage beats gradient boosting on this
component -- see fit_comparison_gbm() and results/efficiency_scores.json.
"""

import json
import os
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl

from src.features.efficiency_features import RATE_STATS, build_career_cumulative, build_season_rates
from src.ingest.constants import PROCESSED_DIR, RESULTS_DIR, SEASONS

STAT_POSITIONS = {
    "ypt": ["WR", "RB", "TE"],
    "td_rate_receiving": ["WR", "RB", "TE"],
    "catch_rate": ["WR", "RB", "TE"],
    "ypc": ["RB", "QB", "WR"],
    "td_rate_rushing": ["RB", "QB", "WR"],
    "ypa": ["QB"],
    "pass_td_rate": ["QB"],
    "int_rate": ["QB"],
}

TRAIN_MAX_SEASON = 2022
TUNE_SEASONS = [2023, 2024]
TEST_SEASON = 2025

# Method-of-moments EB needs the whole training window's per-season observations (not
# just career-cumulative snapshots) to estimate within-player variance -- see
# _fit_shrinkage_params.
_MIN_SEASONS_FOR_WITHIN_PLAYER_VARIANCE = 2

# A naive per-season-observation v_stat estimate is dominated by tiny-sample flukes
# (e.g. a 3-target game going for 27 YPT contributes a huge, correctly-in-expectation
# but wildly noisy term) -- verified by hand: at no minimum-opportunity floor, WR YPT's
# v_stat estimate was inflated ~4x by a handful of sub-10-target seasons, driving tau^2
# negative even though YPT clearly has real persistent signal. Search increasing
# opportunity floors and use the smallest one that yields a stable, positive tau^2,
# maximizing retained sample size while keeping the estimate valid. If no floor in the
# range works, that itself is evidence there's no real signal above sampling noise even
# in well-measured seasons -- full shrinkage is the correct, not a failed, outcome.
_MIN_OPPORTUNITY_CANDIDATES = [20, 30, 50, 80, 120, 150, 200]


def _fit_shrinkage_params(season_rates: pl.DataFrame, opp_col: str, rate_col: str) -> tuple[float, float, int]:
    """Method-of-moments empirical Bayes: returns (prior_mean, k, min_opportunity_used)
    for one rate stat, fit on season_rates (already filtered to the training window and
    one position/stat group).

    v_stat (opportunity-driven sampling variance per unit of exposure) is estimated
    directly from how much a player's own season-level rate bounces around THEIR OWN
    multi-season mean, weighted by that season's opportunity count -- not assumed from
    a Binomial/Poisson formula, since yards-per-target isn't cleanly either. Players
    with only one training-season observation can't inform this (their deviation from
    their own single-season mean is trivially 0), so they're excluded from the v_stat
    estimate specifically, though they still count toward the overall mean/variance.

    tau^2 (true between-player variance) is backed out via the total-variance identity
    Var(r) = tau^2 + E[v_stat/n]. If v_stat alone already explains the observed
    variance, tau^2 is ~0 and k is effectively infinite -- total shrinkage.
    """
    # The pooled mean needs far less data than the variance-component estimate below,
    # so it's computed once, unconditionally, from every available opportunity>0 row --
    # not gated behind the same sample-size search as k. Without this, a stat/position
    # combo too sparse for ANY threshold candidate to pass (e.g. WR carries: a 20+
    # carry WR season is rare) would return no prior mean at all, leaving the
    # shrinkage formula undefined for that group.
    all_df = season_rates.filter(pl.col(opp_col) > 0).drop_nulls(rate_col)
    all_n = all_df[opp_col].to_numpy().astype(float)
    all_r = all_df[rate_col].to_numpy().astype(float)
    mu = float((all_r * all_n).sum() / all_n.sum())

    for min_opp in _MIN_OPPORTUNITY_CANDIDATES:
        df = season_rates.filter(pl.col(opp_col) >= min_opp).drop_nulls(rate_col)
        if df.height < 30:
            continue
        n = df[opp_col].to_numpy().astype(float)
        r = df[rate_col].to_numpy().astype(float)
        v_obs = float(((r - mu) ** 2 * n).sum() / n.sum())

        player_n_seasons = df.group_by("gsis_id").len()
        multi_season_players = player_n_seasons.filter(
            pl.col("len") >= _MIN_SEASONS_FOR_WITHIN_PLAYER_VARIANCE
        ).select("gsis_id")
        df_multi = df.join(multi_season_players, on="gsis_id", how="inner")
        if df_multi.height < 10:
            continue

        player_means = df_multi.group_by("gsis_id").agg(
            (pl.col(rate_col) * pl.col(opp_col)).sum().alias("_x"), pl.col(opp_col).sum().alias("_n")
        ).with_columns((pl.col("_x") / pl.col("_n")).alias("player_mu"))
        df_multi = df_multi.join(player_means.select("gsis_id", "player_mu"), on="gsis_id")

        n_m = df_multi[opp_col].to_numpy().astype(float)
        r_m = df_multi[rate_col].to_numpy().astype(float)
        mu_m = df_multi["player_mu"].to_numpy().astype(float)
        v_stat = float((n_m * (r_m - mu_m) ** 2).sum() / len(n_m))

        mean_inv_n = float((1.0 / n).mean())
        tau2 = v_obs - v_stat * mean_inv_n

        if tau2 > 0:
            return mu, v_stat / tau2, min_opp

    # No floor produced a positive tau^2 (either genuinely no signal above sampling
    # noise, or too little data at every floor to tell -- both get the same honest
    # answer): full shrinkage to the pooled mean.
    return mu, 1e6, _MIN_OPPORTUNITY_CANDIDATES[-1]


def fit_all_shrinkage_params(panel: pl.DataFrame) -> dict:
    """{(stat, position): {"prior_mean": mu, "k": k}} for all 4 stats x their applicable
    positions, fit on TRAIN-window seasons only (<=TRAIN_MAX_SEASON) -- these are
    population-level statistics, and using tune/test-season data to fit them would leak
    future-season information into the prior used to score those same seasons, exactly
    the same leakage class CLAUDE.md warns about for player-level features."""
    season_rates = build_season_rates(panel).filter(pl.col("season") <= TRAIN_MAX_SEASON)

    params = {}
    for stat, (opp_col, _) in RATE_STATS.items():
        for pos in STAT_POSITIONS[stat]:
            sub = season_rates.filter(pl.col("position") == pos)
            mu, k, min_opp_used = _fit_shrinkage_params(sub, opp_col, stat)
            params[(stat, pos)] = {"prior_mean": mu, "k": k, "min_opportunity_used": min_opp_used}
    return params


def build_shrinkage_predictions(panel: pl.DataFrame, params: dict) -> pl.DataFrame:
    """gsis_id, season, position, plus {stat}_shrunk for all 4 stats -- predicting
    season t+1 using career-cumulative opportunity/rate as of the player's most recent
    prior season (inclusive).

    NOT a naive "season - 1" lookup: a player with a gap year (out of the league, e.g.
    injury/release, then back the following season -- verified with Hunter Renfrow, who
    has no 2024 row and returned in 2025) has no career-cumulative row at exactly
    season t, so an exact-match join on season+1 would silently null out (then
    zero-fill) his entire real career history right when he returns. join_asof with
    strategy="backward" instead finds the closest prior season that actually exists for
    that player, however far back the gap goes.
    """
    career = build_career_cumulative(panel)
    base = panel.select("gsis_id", "season", "position")

    # Both sides are genuinely sorted by (gsis_id, season) here (verified: 0/731
    # mismatches against a manual cumsum check) -- polars just can't confirm per-group
    # sortedness cheaply when a "by" key is given, hence the warning it emits.
    career_shifted = career.with_columns((pl.col("season") + 1).alias("season")).sort(
        ["gsis_id", "season"]
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Sortedness of columns cannot be checked")
        out = base.sort(["gsis_id", "season"]).join_asof(
            career_shifted, by="gsis_id", on="season", strategy="backward"
        )

    for c in [
        "career_targets", "career_receiving_yards", "career_receiving_tds",
        "career_carries", "career_rushing_yards", "career_rushing_tds",
    ]:
        out = out.with_columns(pl.col(c).fill_null(0))

    for stat, (opp_col, num_col) in RATE_STATS.items():
        career_opp_col = f"career_{opp_col}"
        career_num_col = f"career_{num_col}"
        out = out.with_columns(
            pl.when(pl.col(career_opp_col) > 0)
            .then(pl.col(career_num_col) / pl.col(career_opp_col))
            .otherwise(0.0)
            .alias(f"_{stat}_career_rate")
        )
        pred_expr = pl.lit(None, dtype=pl.Float64)
        for pos in STAT_POSITIONS[stat]:
            mu, k = params[(stat, pos)]["prior_mean"], params[(stat, pos)]["k"]
            n = pl.col(career_opp_col)
            shrunk = (n / (n + k)) * pl.col(f"_{stat}_career_rate") + (k / (n + k)) * mu
            pred_expr = pl.when(pl.col("position") == pos).then(shrunk).otherwise(pred_expr)
        out = out.with_columns(pred_expr.alias(f"{stat}_shrunk"))

    unique_opp_cols = sorted({f"career_{opp}" for opp, _ in RATE_STATS.values()})
    out = out.rename({f"_{stat}_career_rate": f"{stat}_career_rate" for stat in RATE_STATS})
    return out.select(
        "gsis_id", "season", "position",
        *[f"{stat}_shrunk" for stat in RATE_STATS],
        *[f"{stat}_career_rate" for stat in RATE_STATS],
        *unique_opp_cols,
    )


def score_shrinkage(panel: pl.DataFrame, predictions: pl.DataFrame) -> dict:
    """MAE of the shrinkage predictions against actual season t+1 rates, restricted to
    players with real opportunities that season -- a 0/0 undefined rate isn't a
    meaningful evaluation point (this project's zero-opportunity-rostered-players rule
    is about not filtering the *panel*, not about pretending an undefined rate is 0)."""
    actual = build_season_rates(panel)
    opp_cols = sorted({opp_col for opp_col, _ in RATE_STATS.values()})
    joined = predictions.join(
        actual.select("gsis_id", "season", *opp_cols, *RATE_STATS.keys()),
        on=["gsis_id", "season"], how="inner",
    )
    results = {}
    for stat, (opp_col, _) in RATE_STATS.items():
        sub = joined.filter(pl.col(opp_col) > 0).drop_nulls([f"{stat}_shrunk", stat])
        tune = sub.filter(pl.col("season").is_in(TUNE_SEASONS))
        test = sub.filter(pl.col("season") == TEST_SEASON)
        results[stat] = {
            "tune_mae": float((tune[f"{stat}_shrunk"] - tune[stat]).abs().mean()) if tune.height else None,
            "test_mae": float((test[f"{stat}_shrunk"] - test[stat]).abs().mean()) if test.height else None,
            "n_tune": tune.height,
            "n_test": test.height,
        }
    return results


_GBM_FEATURE_COLS = ["career_rate", "career_opportunities", "position", "age", "years_exp"]
_GBM_CATEGORICAL = ["position"]
_GBM_PARAM_GRID = [
    {"max_depth": 3, "reg_lambda": 5.0},
    {"max_depth": 3, "reg_lambda": 20.0},
    {"max_depth": 4, "reg_lambda": 5.0},
]


def fit_comparison_gbm(panel: pl.DataFrame, predictions: pl.DataFrame, stat: str) -> dict:
    """LightGBM comparison model for one rate stat, using the same information the
    shrinkage estimator sees (career rate, career opportunity count, position) plus age
    and years_exp -- PLAN.md explicitly requires demonstrating shrinkage beats gradient
    boosting on this component, and that comparison has to be apples-to-apples on
    inputs to mean anything."""
    opp_col, _ = RATE_STATS[stat]
    positions = STAT_POSITIONS[stat]
    career_opp_col = f"career_{opp_col}"

    static = panel.select("gsis_id", "season", "age", "years_exp")
    table = predictions.select(
        "gsis_id", "season", "position", f"{stat}_career_rate", career_opp_col
    ).rename({f"{stat}_career_rate": "career_rate", career_opp_col: "career_opportunities"})
    table = table.join(static, on=["gsis_id", "season"], how="left")

    actual = build_season_rates(panel).select("gsis_id", "season", opp_col, stat)
    table = table.join(actual, on=["gsis_id", "season"], how="inner")
    table = table.filter(pl.col("position").is_in(positions) & (pl.col(opp_col) > 0)).drop_nulls(
        _GBM_FEATURE_COLS + [stat]
    )

    train = table.filter(pl.col("season") <= TRAIN_MAX_SEASON)
    tune = table.filter(pl.col("season").is_in(TUNE_SEASONS))
    test = table.filter(pl.col("season") == TEST_SEASON)

    def to_frame(df):
        pdf = df.select(_GBM_FEATURE_COLS).to_pandas()
        for c in _GBM_CATEGORICAL:
            pdf[c] = pdf[c].astype("category")
        return pdf

    X_train, y_train = to_frame(train), train[stat].to_numpy()
    X_tune, y_tune = to_frame(tune), tune[stat].to_numpy()
    X_test, y_test = to_frame(test), test[stat].to_numpy()

    best_params, best_iter, best_mae = None, None, np.inf
    for p in _GBM_PARAM_GRID:
        model = lgb.LGBMRegressor(
            n_estimators=500, learning_rate=0.05, max_depth=p["max_depth"],
            reg_lambda=p["reg_lambda"], num_leaves=2 ** p["max_depth"],
            min_child_samples=20, verbosity=-1,
        )
        model.fit(
            X_train, y_train, eval_X=X_tune, eval_y=y_tune,
            categorical_feature=_GBM_CATEGORICAL,
            callbacks=[lgb.early_stopping(30, verbose=False)],
        )
        mae = float(np.mean(np.abs(model.predict(X_tune) - y_tune)))
        if mae < best_mae:
            best_mae, best_params, best_iter = mae, p, model.best_iteration_

    X_train_tune = pd.concat([X_train, X_tune], ignore_index=True)
    y_train_tune = np.concatenate([y_train, y_tune])
    final_model = lgb.LGBMRegressor(
        n_estimators=best_iter, learning_rate=0.05, max_depth=best_params["max_depth"],
        reg_lambda=best_params["reg_lambda"], num_leaves=2 ** best_params["max_depth"],
        min_child_samples=20, verbosity=-1,
    )
    final_model.fit(X_train_tune, y_train_tune, categorical_feature=_GBM_CATEGORICAL)
    test_mae = float(np.mean(np.abs(final_model.predict(X_test) - y_test)))

    return {
        "best_params": best_params, "tune_mae": best_mae, "test_mae": test_mae,
        "n_train_tune": len(y_train_tune), "n_test": len(y_test),
    }


def main(refresh: bool = False) -> dict:
    panel = pl.read_parquet(os.path.join(PROCESSED_DIR, "player_season_panel.parquet"))
    eb_params = fit_all_shrinkage_params(panel)
    predictions = build_shrinkage_predictions(panel, eb_params)
    shrinkage_scores = score_shrinkage(panel, predictions)

    gbm_scores = {stat: fit_comparison_gbm(panel, predictions, stat) for stat in RATE_STATS}

    results = {
        "shrinkage_params": {
            f"{stat}|{pos}": v for (stat, pos), v in eb_params.items()
        },
        "shrinkage_scores": shrinkage_scores,
        "gbm_comparison_scores": gbm_scores,
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "efficiency_scores.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {RESULTS_DIR}/efficiency_scores.json")

    for stat in RATE_STATS:
        s_mae = shrinkage_scores[stat]["test_mae"]
        g_mae = gbm_scores[stat]["test_mae"]
        winner = "shrinkage" if s_mae < g_mae else "GBM"
        print(f"{stat:20s} shrinkage_test_mae={s_mae:.4f}  gbm_test_mae={g_mae:.4f}  winner={winner}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fit and score the Step 5 efficiency models.")
    parser.add_argument("--refresh", action="store_true", help="Re-pull raw data from network.")
    args = parser.parse_args()
    main(refresh=args.refresh)
