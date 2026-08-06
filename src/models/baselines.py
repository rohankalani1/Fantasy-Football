"""Step 2 of PLAN.md: baselines, scored before any modeling begins. Every future model
in this project must beat both of these, out of sample, on 2024 and 2025 individually, or
it does not ship.

- baseline_ppg: prior-season points per game, carried forward. 0 for rookies (no prior
  row) and for anyone with 0 games played the prior season -- an intentionally dumb
  baseline; it isn't supposed to know about injuries or role changes.
- baseline_consensus: FantasyPros' public Expert Consensus Rankings, snapshotted just
  before Aug 1 of the target season (see src/ingest/consensus_ecr.py).

Both are also compared against the raw FFC ADP itself, per CLAUDE.md Step 8's "report
ADP as a third reference point."
"""

import json
import os

import polars as pl

from src.eval import spearman_by_position
from src.ingest.consensus_ecr import match_ecr_to_gsis, pull_ecr_snapshot
from src.ingest.constants import PROCESSED_DIR, RESULTS_DIR
from src.ingest.pull_raw import pull_ff_playerids, pull_ff_rankings
from src.scoring import compute_fantasy_points, load_scoring_config

TARGET_SEASONS = [2024, 2025]
# "Knowable as of August 1" per CLAUDE.md -- the cutoff is exclusive (snapshot strictly
# before this date), so there's no same-day ambiguity about what was public yet.
ECR_CUTOFFS = {2024: "2024-08-01", 2025: "2025-08-01"}


def build_baseline_ppg(panel_with_points: pl.DataFrame) -> pl.DataFrame:
    """gsis_id, season, baseline_ppg -- season t's prediction uses season (t-1)'s
    points / games_played."""
    prior = panel_with_points.select("gsis_id", "season", "points", "games_played").with_columns(
        (pl.col("season") + 1).alias("season"),
        pl.when(pl.col("games_played") > 0)
        .then(pl.col("points") / pl.col("games_played"))
        .otherwise(0.0)
        .alias("baseline_ppg"),
    ).select("gsis_id", "season", "baseline_ppg")

    out = panel_with_points.select("gsis_id", "season").join(
        prior, on=["gsis_id", "season"], how="left"
    )
    return out.with_columns(pl.col("baseline_ppg").fill_null(0.0))


def build_baseline_consensus(refresh: bool = False) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Returns (baseline, unmatched). baseline has gsis_id, season, baseline_consensus
    (= -ECR, so higher is better -- consistent direction with baseline_ppg)."""
    ff_rankings = pull_ff_rankings(refresh)
    ff_playerids = pull_ff_playerids(refresh)

    snapshots = [
        pull_ecr_snapshot(ff_rankings, season, cutoff) for season, cutoff in ECR_CUTOFFS.items()
    ]
    ecr = pl.concat(snapshots, how="vertical")
    matched, unmatched = match_ecr_to_gsis(ecr, ff_playerids)

    deduped = matched.unique(subset=["gsis_id", "season"], keep="first")
    if deduped.height < matched.height:
        print(
            f"[build_baseline_consensus] {matched.height - deduped.height} duplicate "
            "(gsis_id, season) ECR rows collapsed -- investigate if this grows large."
        )

    baseline = deduped.with_columns((-pl.col("ecr")).alias("baseline_consensus")).select(
        "gsis_id", "season", "baseline_consensus"
    )
    return baseline, unmatched


def build_baselines(panel: pl.DataFrame, refresh: bool = False) -> pl.DataFrame:
    scoring = load_scoring_config()
    panel = compute_fantasy_points(panel, scoring)

    baseline_ppg = build_baseline_ppg(panel)
    baseline_consensus, unmatched_ecr = build_baseline_consensus(refresh)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    unmatched_ecr.sort(["season", "ecr"]).write_csv(
        os.path.join(RESULTS_DIR, "unmatched_ecr_players.csv")
    )
    if unmatched_ecr.height > 0:
        print(
            f"[build_baseline_consensus] {unmatched_ecr.height} ECR rows could not be "
            "matched to a gsis_id -- see results/unmatched_ecr_players.csv"
        )

    out = panel.join(baseline_ppg, on=["gsis_id", "season"], how="left")
    out = out.join(baseline_consensus, on=["gsis_id", "season"], how="left")
    out = out.with_columns((-pl.col("ffc_adp")).alias("baseline_adp"))
    return out


def score_baselines(panel_with_baselines: pl.DataFrame) -> dict:
    """Within-position Spearman vs actual total season points, over the ADP-defined
    draftable universe (non-null ffc_adp that season), on 2024 and 2025 separately."""
    results = {}
    for season in TARGET_SEASONS:
        sub = panel_with_baselines.filter(
            (pl.col("season") == season) & pl.col("ffc_adp").is_not_null()
        )
        season_results = {
            "n_draftable_by_position": {
                row["position"]: row["len"] for row in sub.group_by("position").len().to_dicts()
            }
        }
        for name, pred_col in [
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
    panel_with_baselines = build_baselines(panel, refresh)
    results = score_baselines(panel_with_baselines)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "baselines.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {out_path}")
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build and score Step 2 baselines.")
    parser.add_argument("--refresh", action="store_true", help="Re-pull raw data from network.")
    args = parser.parse_args()
    main(refresh=args.refresh)
