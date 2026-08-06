"""Step 6 of PLAN.md: availability model. Beta-binomial regression over games played
(0-17), parameterized by age, position, and games missed in the prior two seasons.

One pooled model across QB/RB/WR/TE with position as a categorical feature, matching
Step 4's precedent (small-N pooling beats four separate per-position fits). Fit from
scratch via maximum likelihood (scipy.optimize), not a package -- consistent with how
Step 5's empirical Bayes estimator was built directly rather than reached for a library,
and there's no natural off-the-shelf beta-binomial GLM in the scikit-learn/statsmodels
stack this project already depends on.

The mean of the Beta-Binomial (the predicted availability probability p) is a logistic
regression on the features; a single global concentration parameter S controls how
tightly a player's realized games_played clusters around n*p -- S is fit jointly with
the regression coefficients, not assumed.
"""

import json
import os

import numpy as np
import polars as pl
from scipy.optimize import minimize
from scipy.special import betaln

from src.features.availability_features import build_availability_features
from src.ingest.constants import PROCESSED_DIR, RESULTS_DIR, SEASONS, season_length_expr

FEATURE_COLS = ["intercept", "age", "games_missed_t1", "games_missed_t2", "is_RB", "is_WR", "is_TE"]
_STANDARDIZE_COLS = ["age", "games_missed_t1", "games_missed_t2"]

TRAIN_MAX_SEASON = 2022
TUNE_SEASONS = [2023, 2024]
TEST_SEASON = 2025


def _design_matrix(df: pl.DataFrame) -> np.ndarray:
    pdf = df.select(
        pl.lit(1.0).alias("intercept"),
        "age", "games_missed_t1", "games_missed_t2",
        (pl.col("position") == "RB").cast(pl.Float64).alias("is_RB"),
        (pl.col("position") == "WR").cast(pl.Float64).alias("is_WR"),
        (pl.col("position") == "TE").cast(pl.Float64).alias("is_TE"),
    )
    return pdf.select(FEATURE_COLS).to_numpy()


def _neg_log_likelihood(params: np.ndarray, X: np.ndarray, k: np.ndarray, n: np.ndarray) -> float:
    n_coef = X.shape[1]
    coef = params[:n_coef]
    log_s = params[n_coef]
    s = np.exp(log_s)

    linear = X @ coef
    p = 1.0 / (1.0 + np.exp(-linear))
    alpha = p * s
    beta = (1.0 - p) * s

    # Binomial(n, k) coefficient term is dropped -- constant w.r.t. parameters, doesn't
    # affect the argmin.
    ll = betaln(k + alpha, n - k + beta) - betaln(alpha, beta)
    return -np.sum(ll)


def fit_availability_model(table: pl.DataFrame) -> dict:
    """Fits on TRAIN-window seasons only (<=TRAIN_MAX_SEASON), restricted to the
    draftable universe (non-null ffc_adp that season -- same definition Step 2 used).

    This restriction is load-bearing, not cosmetic: verified directly on the data that
    ~47% of the full panel are deep-bench/practice-squad-tier rows with no real role,
    where games_played is driven by roster-depth decisions, not health -- age/position/
    injury-history can't explain "will a third-stringer ever get on the field," but
    "did he play last season" (the naive baseline) predicts that persistence almost
    perfectly by construction, which was making the model lose to the naive baseline by
    ~35-40% before this fix. Restricted to draftable players, mean games_played is
    nearly identical whether or not they played last season (13.16 vs 13.35) --
    confirming the remaining variation in that population really is availability risk,
    the thing this model is meant to predict.

    Standardizes the continuous features using train-only mean/std (fit on train,
    applied everywhere) so the optimizer isn't working with wildly different feature
    scales (age ~20-40 vs games_missed ~0-17)."""
    train = table.filter((pl.col("season") <= TRAIN_MAX_SEASON) & pl.col("ffc_adp").is_not_null())
    before = train.height
    train = train.drop_nulls(["age", "games_missed_t1", "games_missed_t2"])
    if train.height < before:
        print(
            f"[fit_availability_model] dropping {before - train.height}/{before} training "
            "rows with null age (a pre-existing panel gap from Step 1 -- missing birth_date)"
        )

    norm = {}
    train_std = train
    for c in _STANDARDIZE_COLS:
        mean, std = float(train[c].mean()), float(train[c].std())
        norm[c] = {"mean": mean, "std": std}
        train_std = train_std.with_columns(((pl.col(c) - mean) / std).alias(c))

    X = _design_matrix(train_std)
    k = train["games_played"].to_numpy().astype(float)
    n = train["season_length"].to_numpy().astype(float)

    x0 = np.zeros(len(FEATURE_COLS) + 1)
    x0[0] = np.log(k.sum() / (n.sum() - k.sum() + 1e-9))  # rough intercept guess
    x0[-1] = np.log(50.0)  # starting concentration guess

    result = minimize(
        _neg_log_likelihood, x0, args=(X, k, n), method="L-BFGS-B",
        options={"maxiter": 2000},
    )
    if not result.success:
        raise RuntimeError(f"Beta-binomial MLE did not converge: {result.message}")

    coef = result.x[: len(FEATURE_COLS)]
    concentration = float(np.exp(result.x[len(FEATURE_COLS)]))

    return {
        "coefficients": dict(zip(FEATURE_COLS, coef.tolist())),
        "concentration": concentration,
        "normalization": norm,
        "converged": bool(result.success),
        "n_train": len(k),
    }


def predict(table: pl.DataFrame, fit: dict) -> pl.DataFrame:
    """Adds `pred_availability_prob` and `pred_games_played` (= prob * season_length,
    a continuous expectation -- Step 7's combine multiplication wants E[games_played],
    not a rounded integer)."""
    std_table = table
    for c in _STANDARDIZE_COLS:
        m, s = fit["normalization"][c]["mean"], fit["normalization"][c]["std"]
        std_table = std_table.with_columns(((pl.col(c) - m) / s).alias(c))

    X = _design_matrix(std_table)
    coef = np.array([fit["coefficients"][c] for c in FEATURE_COLS])
    p = 1.0 / (1.0 + np.exp(-(X @ coef)))

    return table.with_columns(
        pl.Series("pred_availability_prob", p),
    ).with_columns(
        (pl.col("pred_availability_prob") * pl.col("season_length")).alias("pred_games_played")
    )


def build_training_table(panel: pl.DataFrame) -> pl.DataFrame:
    """Includes ffc_adp so fit/score can restrict to the draftable universe (see
    fit_availability_model's docstring for why) without restricting predict() itself --
    at real draft time we want an availability projection for every player being
    considered, not just ones who happened to have an ADP in a past season."""
    feats = build_availability_features(panel)
    labels = panel.select("gsis_id", "season", "games_played", "ffc_adp")
    out = feats.join(labels, on=["gsis_id", "season"], how="left")
    out = out.with_columns(season_length_expr().alias("season_length"))
    return out.filter(pl.col("season").is_in(SEASONS))


def score(table_with_preds: pl.DataFrame) -> dict:
    """MAE of predicted vs actual games_played, and vs a naive "prior season carried
    forward" baseline, on tune (2023-2024) and test (2025) -- restricted to the same
    draftable universe the model was fit on (see fit_availability_model)."""
    naive = table_with_preds.rename({"games_played_t1": "naive_pred_games_played"})
    naive = naive.filter(pl.col("ffc_adp").is_not_null())
    results = {}
    for label, seasons in [("tune", TUNE_SEASONS), ("test", [TEST_SEASON])]:
        sub = naive.filter(pl.col("season").is_in(seasons))
        before = sub.height
        # drop_nulls alone isn't enough here: a null `age` produces a NaN prediction
        # (float NaN, not a polars null) once it flows through standardization and the
        # sigmoid -- the two are distinct in polars and drop_nulls only catches nulls.
        sub = sub.drop_nulls(["pred_games_played", "games_played", "naive_pred_games_played"])
        sub = sub.filter(~pl.col("pred_games_played").is_nan())
        if sub.height < before:
            print(
                f"[score:{label}] dropping {before - sub.height}/{before} rows with a null/NaN "
                "prediction (null age -- see fit_availability_model) from evaluation"
            )
        model_mae = float((sub["pred_games_played"] - sub["games_played"]).abs().mean())
        naive_mae = float((sub["naive_pred_games_played"] - sub["games_played"]).abs().mean())
        results[label] = {"model_mae": model_mae, "naive_mae": naive_mae, "n": sub.height}
    return results


def main() -> dict:
    panel = pl.read_parquet(os.path.join(PROCESSED_DIR, "player_season_panel.parquet"))
    table = build_training_table(panel)
    fit = fit_availability_model(table)
    predictions = predict(table, fit)
    scores = score(predictions)

    results = {"fit": fit, "scores": scores}
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "availability_scores.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {RESULTS_DIR}/availability_scores.json")

    for label in ["tune", "test"]:
        s = scores[label]
        winner = "model" if s["model_mae"] < s["naive_mae"] else "naive"
        print(f"{label}: model_mae={s['model_mae']:.3f} naive_mae={s['naive_mae']:.3f} winner={winner}")

    return results


if __name__ == "__main__":
    main()
