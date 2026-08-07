"""Statistical evaluation of the final (post-calibration) model against the
baselines, on 2024 and 2025 draftable players -- exploration/reporting only,
never the source of truth (that's src/models/validate.py's Step 8 numbers).

Goes beyond Step 8's point-estimate Spearman with: point-accuracy metrics
(MAE/RMSE/R2), bootstrap confidence intervals on Spearman (small-N means a
point estimate alone overstates precision), and a paired bootstrap
significance test of model vs. each baseline (is the observed edge real given
this project's own explicit small-N warning, or could a sample this size show
this gap by chance).
"""

import numpy as np
import polars as pl
from scipy.stats import rankdata

pl.Config.set_tbl_rows(30)

N_BOOTSTRAP = 5000
RNG = np.random.default_rng(42)

df = pl.read_csv("results/historical_predictions_2024_2025.csv")
df = df.with_columns((-pl.col("ffc_adp")).alias("baseline_adp"))

POSITIONS = ["QB", "RB", "WR", "TE"]
PRED_COLS = {
    "model": "points_total_pred",
    "consensus": "baseline_consensus",
    "adp": "baseline_adp",
}


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx = rankdata(x)
    ry = rankdata(y)
    return float(np.corrcoef(rx, ry)[0, 1])


def bootstrap_spearman_ci(actual: np.ndarray, pred: np.ndarray, n_boot: int = N_BOOTSTRAP) -> tuple[float, float, float]:
    """(point estimate, 2.5th pctile, 97.5th pctile) via resampling players
    with replacement -- the appropriate small-N substitute for a point
    estimate alone, which implies false precision at n~15-40 per position."""
    n = len(actual)
    point = spearman(actual, pred)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        idx = RNG.integers(0, n, n)
        boots[i] = spearman(actual[idx], pred[idx])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return point, float(lo), float(hi)


def paired_bootstrap_test(actual: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray, n_boot: int = N_BOOTSTRAP) -> dict:
    """One-sided test of H1: Spearman(pred_a, actual) > Spearman(pred_b, actual),
    resampling players (not predictions) with replacement so both correlations
    are recomputed on the exact same resampled players each draw -- the correct
    paired design, since model and baseline are scored on the identical
    players, not independent samples."""
    n = len(actual)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = RNG.integers(0, n, n)
        diffs[i] = spearman(actual[idx], pred_a[idx]) - spearman(actual[idx], pred_b[idx])
    point_diff = spearman(actual, pred_a) - spearman(actual, pred_b)
    p_one_sided = float(np.mean(diffs <= 0))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"diff": point_diff, "ci_lo": float(lo), "ci_hi": float(hi), "p_one_sided": p_one_sided}


def point_accuracy(actual: np.ndarray, pred: np.ndarray) -> dict:
    err = pred - actual
    r = float(np.corrcoef(pred, actual)[0, 1])
    return {
        "n": len(actual),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "bias": float(np.mean(err)),
        "pearson_r": r,
        "r_squared": r ** 2,
    }


print("=" * 78)
print("POINT-ACCURACY METRICS (model predictions vs. actual season points)")
print("=" * 78)
for season in [2024, 2025, "pooled"]:
    sub = df if season == "pooled" else df.filter(pl.col("season") == season)
    print(f"\n--- {season} ---")
    rows = []
    for pos in POSITIONS + ["ALL"]:
        s = sub if pos == "ALL" else sub.filter(pl.col("position") == pos)
        m = point_accuracy(s["points"].to_numpy(), s["points_total_pred"].to_numpy())
        rows.append({"position": pos, **m})
    print(pl.DataFrame(rows).select(
        "position", "n", pl.col("mae").round(1), pl.col("rmse").round(1),
        pl.col("bias").round(1), pl.col("pearson_r").round(3), pl.col("r_squared").round(3),
    ))

print()
print("=" * 78)
print(f"RANK ACCURACY: SPEARMAN WITH 95% BOOTSTRAP CI (n_boot={N_BOOTSTRAP})")
print("=" * 78)
for season in [2024, 2025]:
    sub = df.filter(pl.col("season") == season)
    print(f"\n--- {season} ---")
    for pos in POSITIONS:
        s = sub.filter(pl.col("position") == pos)
        actual = s["points"].to_numpy()
        point, lo, hi = bootstrap_spearman_ci(actual, s["points_total_pred"].to_numpy())
        print(f"  {pos}: rho={point:.3f}  95% CI [{lo:.3f}, {hi:.3f}]  n={s.height}")

print()
print("=" * 78)
print("SIGNIFICANCE: IS THE MODEL'S RANK EDGE OVER EACH BASELINE REAL?")
print("Paired bootstrap, H1: model's Spearman > baseline's Spearman (one-sided)")
print("=" * 78)
for season in [2024, 2025]:
    sub = df.filter(pl.col("season") == season)
    print(f"\n--- {season} ---")
    for baseline_name in ["consensus", "adp"]:
        print(f"  vs. {baseline_name}:")
        for pos in POSITIONS + ["ALL (mean of positions)"]:
            if pos == "ALL (mean of positions)":
                # match Step 8's own definition: mean of per-position diffs,
                # each position resampled independently within the same draw
                n_per_pos = {p: sub.filter(pl.col("position") == p).height for p in POSITIONS}
                diffs = np.zeros(N_BOOTSTRAP)
                point_total = 0.0
                for p in POSITIONS:
                    s = sub.filter(pl.col("position") == p)
                    actual = s["points"].to_numpy()
                    pred_model = s["points_total_pred"].to_numpy()
                    pred_base = s[PRED_COLS[baseline_name]].to_numpy()
                    n = len(actual)
                    for i in range(N_BOOTSTRAP):
                        idx = RNG.integers(0, n, n)
                        diffs[i] += (spearman(actual[idx], pred_model[idx]) - spearman(actual[idx], pred_base[idx]))
                    point_total += spearman(actual, pred_model) - spearman(actual, pred_base)
                diffs /= len(POSITIONS)
                point_total /= len(POSITIONS)
                p_val = float(np.mean(diffs <= 0))
                lo, hi = np.percentile(diffs, [2.5, 97.5])
                print(f"    {pos:24s} diff={point_total:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  p={p_val:.3f}")
            else:
                s = sub.filter(pl.col("position") == pos)
                actual = s["points"].to_numpy()
                res = paired_bootstrap_test(
                    actual, s["points_total_pred"].to_numpy(), s[PRED_COLS[baseline_name]].to_numpy()
                )
                sig = "*" if res["p_one_sided"] < 0.05 else " "
                print(f"    {pos:24s} diff={res['diff']:+.3f}  95% CI [{res['ci_lo']:+.3f}, {res['ci_hi']:+.3f}]  p={res['p_one_sided']:.3f} {sig}")

print("\n(* = p < 0.05, one-sided, that the model's rank accuracy beats that baseline)")

# --- persist a compact summary for reference ---
import json

summary = {"point_accuracy": {}, "spearman_ci": {}, "significance": {}}
for season in [2024, 2025, "pooled"]:
    sub = df if season == "pooled" else df.filter(pl.col("season") == season)
    summary["point_accuracy"][str(season)] = {
        pos: point_accuracy(
            (sub if pos == "ALL" else sub.filter(pl.col("position") == pos))["points"].to_numpy(),
            (sub if pos == "ALL" else sub.filter(pl.col("position") == pos))["points_total_pred"].to_numpy(),
        )
        for pos in POSITIONS + ["ALL"]
    }
for season in [2024, 2025]:
    sub = df.filter(pl.col("season") == season)
    summary["spearman_ci"][str(season)] = {}
    for pos in POSITIONS:
        s = sub.filter(pl.col("position") == pos)
        point, lo, hi = bootstrap_spearman_ci(s["points"].to_numpy(), s["points_total_pred"].to_numpy())
        summary["spearman_ci"][str(season)][pos] = {"rho": point, "ci_lo": lo, "ci_hi": hi, "n": s.height}

with open("results/statistical_evaluation.json", "w") as f:
    json.dump(summary, f, indent=2)
print("\nWrote results/statistical_evaluation.json")
