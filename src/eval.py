"""Evaluation metric for every phase-1 step (CLAUDE.md Step 8): within-position Spearman
rank correlation over draftable players. Not MAE on season totals -- that's dominated by
injury noise and by the already-correctly-priced top 20 picks.

No scipy dependency: Spearman is just the Pearson correlation of rank-transformed values,
and polars' `.rank()` already does average-rank tie handling (matching scipy's default),
so this is implemented directly on top of polars.
"""

import polars as pl


def spearman_by_position(
    df: pl.DataFrame,
    pred_col: str,
    actual_col: str,
    position_col: str = "position",
) -> pl.DataFrame:
    """Returns one row per position (Spearman(pred_col, actual_col) within that position)
    plus an "ALL" row that is the unweighted mean of the per-position correlations --
    deliberately not a single pooled Spearman across positions, since raw point totals
    aren't comparable across positions (a QB1 season and a TE1 season score very
    differently) and pooling would let position mix drive the number.
    """
    results = []
    for pos in sorted(df.select(position_col).unique().to_series().to_list()):
        sub = df.filter(pl.col(position_col) == pos).drop_nulls([pred_col, actual_col])
        n = sub.height
        if n < 2:
            results.append({"position": pos, "n": n, "spearman": None})
            continue
        ranked = sub.select(
            pl.col(pred_col).rank().alias("pred_rank"),
            pl.col(actual_col).rank().alias("actual_rank"),
        )
        r = ranked.select(pl.corr("pred_rank", "actual_rank")).item()
        results.append({"position": pos, "n": n, "spearman": r})

    out = pl.DataFrame(results, schema={"position": pl.String, "n": pl.Int64, "spearman": pl.Float64})
    mean_r = out.select(pl.col("spearman").mean()).item()
    total_n = out.select(pl.col("n").sum()).item()
    overall = pl.DataFrame(
        [{"position": "ALL", "n": total_n, "spearman": mean_r}],
        schema={"position": pl.String, "n": pl.Int64, "spearman": pl.Float64},
    )
    return pl.concat([out, overall])
