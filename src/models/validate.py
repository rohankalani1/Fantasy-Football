"""Step 8 of PLAN.md: validate. Within-position Spearman rank correlation over
draftable players, versus both Step 2 baselines, on 2024 and 2025, plus ADP itself as a
third reference point. This is the actual moment of truth for the whole project: every
component from Steps 3-7 individually beat its own local naive baseline, but that says
nothing about whether the *combined* system beats the real bar -- consensus/ADP's
~0.40-0.48 Spearman from Step 2.
"""

import json
import os

import polars as pl

from src.eval import spearman_by_position
from src.ingest.constants import PROCESSED_DIR, RESULTS_DIR
from src.models.baselines import build_baselines

TARGET_SEASONS = [2024, 2025]


def build_validation_table(panel: pl.DataFrame, refresh: bool = False) -> pl.DataFrame:
    combined = pl.read_parquet(os.path.join(RESULTS_DIR, "combined_projections.parquet"))
    with_baselines = build_baselines(panel, refresh)
    return combined.join(
        with_baselines.select(
            "gsis_id", "season", "points", "baseline_ppg", "baseline_consensus", "ffc_adp"
        ),
        on=["gsis_id", "season"], how="inner",
    ).with_columns((-pl.col("ffc_adp")).alias("baseline_adp"))


def score_model(validation_table: pl.DataFrame) -> dict:
    results = {}
    for season in TARGET_SEASONS:
        sub = validation_table.filter(
            (pl.col("season") == season) & pl.col("ffc_adp").is_not_null()
        )
        season_results = {
            "n_draftable_by_position": {
                row["position"]: row["len"] for row in sub.group_by("position").len().to_dicts()
            }
        }
        for name, pred_col in [
            ("model", "points_total_pred"),
            ("baseline_ppg", "baseline_ppg"),
            ("baseline_consensus", "baseline_consensus"),
            ("adp_reference", "baseline_adp"),
        ]:
            spearman_df = spearman_by_position(sub, pred_col, "points")
            season_results[name] = {
                row["position"]: row["spearman"] for row in spearman_df.to_dicts()
            }
        results[str(season)] = season_results
    return results


def main(refresh: bool = False) -> dict:
    panel = pl.read_parquet(os.path.join(PROCESSED_DIR, "player_season_panel.parquet"))
    validation_table = build_validation_table(panel, refresh)
    results = score_model(validation_table)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "step8_validation.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {out_path}")

    for season in TARGET_SEASONS:
        r = results[str(season)]
        print(f"\n{season}: model={r['model']['ALL']:.4f}  "
              f"consensus={r['baseline_consensus']['ALL']:.4f}  "
              f"adp={r['adp_reference']['ALL']:.4f}  "
              f"ppg={r['baseline_ppg']['ALL']:.4f}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Step 8's validation.")
    parser.add_argument("--refresh", action="store_true", help="Re-pull raw data from network.")
    args = parser.parse_args()
    main(refresh=args.refresh)
