"""Step 9 of PLAN.md: draft board. Value over replacement using 12 teams and the
league's starting lineup requirements (config/roster_requirements.yaml -- never
hardcoded, per CLAUDE.md), with a hardcoded-for-now replacement level: the best
projected point total among players who would NOT be drafted as a starter, per
position. Clusters players into tiers within each position by projected-points gaps.
"""

import numpy as np
import polars as pl
import yaml


def load_roster_requirements(path: str = "config/roster_requirements.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def compute_replacement_levels(
    board: pl.DataFrame, roster_requirements: dict, points_col: str = "points_total_pred"
) -> dict[str, float]:
    """{position: replacement_level_points} -- the projected points of the best player
    at that position who would NOT be drafted as a starter across a 12-team league.

    Flex allocation is greedy, not a fixed per-position split: after each position's
    dedicated starting slots are set aside, the flex spots go to whichever remaining
    flex-eligible players have the highest projected points regardless of position --
    matching what actually happens in a real draft (the better player fills flex, not a
    predetermined RB/WR/TE ratio).
    """
    teams = roster_requirements["teams"]
    starters = roster_requirements["starters"]
    flex_count = roster_requirements["flex"]["count"] * teams
    flex_eligible = roster_requirements["flex"]["eligible"]

    ranked = board.select("gsis_id", "position", points_col).sort(points_col, descending=True)
    ranked = ranked.with_columns(
        pl.col(points_col).rank(method="ordinal", descending=True).over("position").alias("pos_rank")
    )

    n_starters_by_pos = {pos: count * teams for pos, count in starters.items()}

    leftover_frames = [
        ranked.filter((pl.col("position") == pos) & (pl.col("pos_rank") > n))
        for pos, n in n_starters_by_pos.items()
    ]
    flex_pool = pl.concat(leftover_frames, how="vertical").filter(
        pl.col("position").is_in(flex_eligible)
    ).sort(points_col, descending=True)

    flex_taken = flex_pool.head(flex_count)
    flex_by_pos = dict(
        zip(*flex_taken.group_by("position").len().to_dict(as_series=False).values())
    ) if flex_taken.height > 0 else {}

    replacement_levels = {}
    for pos, n_starters in n_starters_by_pos.items():
        total_drafted = n_starters + flex_by_pos.get(pos, 0)
        pos_rows = ranked.filter(pl.col("position") == pos).sort(points_col, descending=True)
        idx = min(total_drafted, pos_rows.height - 1)
        replacement_levels[pos] = float(pos_rows[points_col][idx])
    return replacement_levels


def add_vor(
    board: pl.DataFrame, replacement_levels: dict, points_col: str = "points_total_pred"
) -> pl.DataFrame:
    repl_expr = pl.lit(None, dtype=pl.Float64)
    for pos, level in replacement_levels.items():
        repl_expr = pl.when(pl.col("position") == pos).then(level).otherwise(repl_expr)
    return board.with_columns(repl_expr.alias("replacement_level")).with_columns(
        (pl.col(points_col) - pl.col("replacement_level")).alias("vor")
    )


def assign_tiers(board: pl.DataFrame, value_col: str = "vor", min_value: float = 0.0) -> pl.DataFrame:
    """Gap-based tiering within each position: among players at or above `min_value`
    (replacement level, VOR=0, by default -- i.e. the draftable-or-better population),
    sort by `value_col` descending and start a new tier whenever the drop to the next
    player is unusually large relative to that position's typical gap (mean + 1 stdev
    of consecutive gaps) -- a data-driven break point instead of picking a fixed number
    of tiers by hand, consistent with this project's preference for fitting thresholds
    from the data (Step 3's alpha grid, Step 5's variance-based shrinkage) over
    hand-picked constants.

    Restricted to the draftable-or-better population deliberately: computing gap
    statistics over the full rostered population (hundreds of near-identical
    replacement-level bench players) makes typical gaps tiny and over-segments the
    handful of players who actually matter into far more tiers than a draft board
    should show. Everyone below `min_value` collapses into one explicit final
    "waiver/undraftable" tier instead.
    """
    out_frames = []
    for pos in board.select("position").unique().to_series().to_list():
        sub = board.filter(pl.col("position") == pos).sort(value_col, descending=True)
        drafted = sub.filter(pl.col(value_col) >= min_value)
        rest = sub.filter(pl.col(value_col) < min_value)

        values = drafted[value_col].to_numpy()
        if len(values) <= 1:
            tiers = [1] * len(values)
        else:
            gaps = -np.diff(values)
            threshold = gaps.mean() + gaps.std()
            tiers = [1]
            current_tier = 1
            for gap in gaps:
                if gap > threshold:
                    current_tier += 1
                tiers.append(current_tier)
        drafted = drafted.with_columns(pl.Series("tier", tiers, dtype=pl.Int64))

        waiver_tier = (max(tiers) + 1) if tiers else 1
        rest = rest.with_columns(pl.lit(waiver_tier, dtype=pl.Int64).alias("tier"))

        out_frames.append(pl.concat([drafted, rest], how="vertical"))
    return pl.concat(out_frames, how="vertical")


def build_draft_board(
    projections: pl.DataFrame,
    roster_requirements: dict,
    points_col: str = "points_total_pred",
) -> pl.DataFrame:
    """Full Step 9 pipeline: replacement level -> VOR -> tiers -> overall/position rank,
    sorted by VOR descending."""
    replacement_levels = compute_replacement_levels(projections, roster_requirements, points_col)
    board = add_vor(projections, replacement_levels, points_col)
    board = assign_tiers(board, "vor")
    board = board.sort("vor", descending=True)
    board = board.with_columns(
        (pl.int_range(1, board.height + 1)).alias("overall_rank"),
        pl.col("vor").rank(method="ordinal", descending=True).over("position").cast(pl.Int64).alias("position_rank"),
    )
    return board
