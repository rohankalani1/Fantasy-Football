"""FantasyPros consensus expert rankings (ECR), matched to gsis_id, for Step 2's
baseline_consensus.

Snapshot selection is "closest scrape_date strictly before Aug 1 of the target season" --
these are genuine historical scrapes (nflreadpy's load_ff_rankings('all') archive goes
back to 2019), not backfilled after the fact, so picking a pre-season snapshot is
leakage-safe per CLAUDE.md's "knowable as of August 1" rule.

Matching uses ff_playerids' fantasypros_id <-> gsis_id crosswalk, a clean numeric-ID
join -- unlike FFC's ADP data, which has no ID crosswalk at all and needed the two-stage
name matcher in id_matching.py.
"""

import polars as pl

_ECR_PAGE_TYPES = {"QB": "redraft-qb", "RB": "redraft-rb", "WR": "redraft-wr", "TE": "redraft-te"}


def latest_snapshot_before(ff_rankings: pl.DataFrame, cutoff: str) -> str:
    """Latest scrape_date strictly before `cutoff` (YYYY-MM-DD string)."""
    dates = (
        ff_rankings.filter(pl.col("scrape_date") < cutoff)
        .select("scrape_date")
        .unique()
        .sort("scrape_date", descending=True)
    )
    if dates.height == 0:
        raise ValueError(f"No ECR snapshot found before {cutoff}")
    return dates.item(0, 0)


def pull_ecr_snapshot(ff_rankings: pl.DataFrame, target_season: int, cutoff: str) -> pl.DataFrame:
    """One row per player in the redraft-PPR ECR snapshot closest before `cutoff`,
    tagged with `target_season` (the season this snapshot is predicting)."""
    snap_date = latest_snapshot_before(ff_rankings, cutoff)
    pos_map = {v: k for k, v in _ECR_PAGE_TYPES.items()}
    snap = ff_rankings.filter(
        (pl.col("scrape_date") == snap_date)
        & (pl.col("page_type").is_in(list(_ECR_PAGE_TYPES.values())))
        & (pl.col("ecr_type") == "rp")  # redraft, PPR
    )
    return snap.with_columns(
        pl.col("id").cast(pl.Int64).alias("fantasypros_id"),
        pl.col("page_type").replace(pos_map).alias("position"),
        pl.lit(target_season).alias("season"),
        pl.lit(snap_date).alias("ecr_snapshot_date"),
    ).select("fantasypros_id", "position", "season", "ecr_snapshot_date", "player", "ecr")


def match_ecr_to_gsis(
    ecr: pl.DataFrame, ff_playerids: pl.DataFrame
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Returns (matched, unmatched). matched has a gsis_id column; unmatched does not."""
    bridge = (
        ff_playerids.select("fantasypros_id", "gsis_id")
        .drop_nulls()
        .unique(subset=["fantasypros_id", "gsis_id"])
    )
    # A handful of fantasypros_id values map to >1 distinct gsis_id in the crosswalk
    # (e.g. two different defensive players sharing a stale ID) -- drop those as
    # ambiguous rather than guess. None observed among QB/RB/WR/TE as of this writing,
    # but check defensively rather than assume it stays that way.
    dupe_ids = bridge.group_by("fantasypros_id").len().filter(pl.col("len") > 1).select("fantasypros_id")
    if dupe_ids.height > 0:
        bridge = bridge.join(dupe_ids, on="fantasypros_id", how="anti")

    joined = ecr.join(bridge, on="fantasypros_id", how="left")
    matched = joined.filter(pl.col("gsis_id").is_not_null())
    unmatched = joined.filter(pl.col("gsis_id").is_null()).drop("gsis_id")
    return matched, unmatched
