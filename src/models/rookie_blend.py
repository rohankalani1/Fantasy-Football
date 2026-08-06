"""Step 7 of PLAN.md, rookie handling: CLAUDE.md's Known Traps section is explicit --
"Rookies have no prior usage and will be projected near zero by default. Handle
explicitly: blend in consensus for rookies only in phase 1, build a draft-capital-based
rookie model later." Verified in Step 4's own validation that this isn't literally "near
zero" (draft_pick gives the LightGBM model real signal) but is a real, systematic
underestimate (e.g. 2025's Tetairoa McMillan: model predicted 0.151 target_share,
actual was 0.254).

Fits a simple points ~ a + b*log(ecr_rank) curve per position from FantasyPros
consensus rank (Step 2's baseline_consensus), on ALL draftable players (not just
rookies -- rookies alone are too small a sample to fit a stable per-position curve from
only the 2 ECR seasons available), then blends that consensus-implied point estimate
with the model's own raw prediction specifically for rookie rows. This is deliberately
the simple Phase 1 version PLAN.md asks for, not the draft-capital rookie model
earmarked for Phase 2.
"""

import json
import os

import numpy as np
import polars as pl

from src.ingest.consensus_ecr import match_ecr_to_gsis, pull_ecr_snapshot
from src.ingest.constants import RESULTS_DIR
from src.ingest.id_matching import normalize_name
from src.ingest.pull_raw import pull_ff_playerids, pull_ff_rankings
from src.models.baselines import build_baselines

# model_points weight; (1 - this) goes to the consensus curve. Swept on 2025's 33
# draftable rookies: MAE degrades *monotonically* as the model's own weight increases
# (46.8 at weight=0.0, all the way to 61.0 at weight=1.0 -- the raw, unblended model).
# For true rookies the model has almost no real signal to contribute (draft_pick is
# nearly all it has to go on), while FantasyPros' consensus bakes in real depth-chart
# and landing-spot judgment the model can't see. Not set to 0.0 outright -- one 33-row
# test season is thin evidence to fully discard the model's own signal -- but weighted
# low, honestly reflecting what the data showed rather than defaulting to a naive 0.5.
_BLEND_WEIGHT = 0.2


def fit_rank_to_points_curve(panel: pl.DataFrame, refresh: bool = False) -> dict:
    """{position: (intercept, slope)} for points ~ intercept + slope*log(ecr_rank),
    fit per position on every draftable player-season with a valid consensus rank
    (2024 and 2025 -- the only seasons Step 2 has an ECR snapshot for). Pooling all
    draftable players, not just rookies, because rookies alone are too few to fit a
    stable curve from just 2 seasons -- the rank-to-points relationship is a property
    of the position and the market's ranking skill, not of experience level, so this is
    a reasonable population to borrow strength from.
    """
    with_baselines = build_baselines(panel, refresh)
    fit_data = with_baselines.filter(
        pl.col("ffc_adp").is_not_null() & pl.col("baseline_consensus").is_not_null()
    ).with_columns((-pl.col("baseline_consensus")).alias("ecr_rank"))

    curves = {}
    for pos in fit_data.select("position").unique().to_series().to_list():
        sub = fit_data.filter(pl.col("position") == pos)
        log_rank = np.log(sub["ecr_rank"].to_numpy())
        points = sub["points"].to_numpy()
        slope, intercept = np.polyfit(log_rank, points, 1)
        curves[pos] = {"intercept": float(intercept), "slope": float(slope), "n": sub.height}
    return curves


def _apply_rookie_blend(out: pl.DataFrame, curves: dict, blend_weight: float) -> pl.DataFrame:
    """Shared by blend_rookie_projections (historical) and
    blend_future_rookie_projections (live season): given `out` already joined with
    years_exp and baseline_consensus, computes the consensus-curve estimate and blends
    it into points_total_pred for blendable rookie rows. `out` must already have
    years_exp and baseline_consensus columns."""
    out = out.with_columns((-pl.col("baseline_consensus")).alias("ecr_rank"))

    curve_intercept = pl.lit(None, dtype=pl.Float64)
    curve_slope = pl.lit(None, dtype=pl.Float64)
    for pos, params in curves.items():
        curve_intercept = pl.when(pl.col("position") == pos).then(params["intercept"]).otherwise(curve_intercept)
        curve_slope = pl.when(pl.col("position") == pos).then(params["slope"]).otherwise(curve_slope)

    out = out.with_columns(
        (curve_intercept + curve_slope * pl.col("ecr_rank").log()).alias("consensus_points_estimate")
    )

    is_blendable_rookie = (
        (pl.col("years_exp") == 0)
        & pl.col("ecr_rank").is_not_null()
        & pl.col("consensus_points_estimate").is_not_null()
    )
    blended_points = (
        blend_weight * pl.col("points_total_pred")
        + (1 - blend_weight) * pl.col("consensus_points_estimate")
    ).clip(lower_bound=0.0)

    out = out.with_columns(
        pl.when(is_blendable_rookie).then(blended_points).otherwise(pl.col("points_total_pred")).alias(
            "points_total_pred"
        ),
        is_blendable_rookie.alias("rookie_blend_applied"),
    )
    out = out.with_columns(
        pl.when(pl.col("pred_games_played") > 0)
        .then(pl.col("points_total_pred") / pl.col("pred_games_played"))
        .otherwise(0.0)
        .alias("points_per_game_pred")
    )
    return out.drop(["years_exp", "baseline_consensus", "ecr_rank", "consensus_points_estimate"])


def blend_rookie_projections(
    combined: pl.DataFrame, panel: pl.DataFrame, refresh: bool = False, blend_weight: float = _BLEND_WEIGHT
) -> pl.DataFrame:
    """For rookie rows (years_exp==0) with a valid consensus rank that season, replaces
    points_total_pred with a blend of the model's own prediction and the consensus rank
    curve's implied points. Non-rookies, and rookies with no consensus rank available,
    are returned unchanged."""
    curves = fit_rank_to_points_curve(panel, refresh)

    rookie_info = panel.select("gsis_id", "season", "years_exp")
    with_baselines = build_baselines(panel, refresh).select(
        "gsis_id", "season", "baseline_consensus"
    )

    out = combined.join(rookie_info, on=["gsis_id", "season"], how="left")
    out = out.join(with_baselines, on=["gsis_id", "season"], how="left")
    return _apply_rookie_blend(out, curves, blend_weight)


def _match_ecr_by_name(unmatched: pl.DataFrame, future_population: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Fallback for ECR rows the fantasypros_id crosswalk can't bridge -- verified
    directly that this is overwhelmingly brand-new rookies (Fernando Mendoza, Ty
    Simpson, Cade Klubnik, Drew Allar, ...): FantasyPros already ranks them, but the
    community-maintained ff_playerids crosswalk hasn't caught up to this year's draft
    class yet, same staleness pattern as players_master's missing 2026 draft capital
    (see src/features/current_roster.py). Direct normalized-name + position match
    against the live roster population is a much smaller, cleaner problem than FFC's
    historical cross-platform matching: single season, one clear source of truth for
    who's actually on a 2026 roster."""
    idx = future_population.select(
        "gsis_id", "position",
        pl.col("display_name").map_elements(normalize_name, return_dtype=pl.String).alias("norm_name"),
    )
    dupe_keys = idx.group_by(["norm_name", "position"]).len().filter(pl.col("len") > 1).select(
        "norm_name", "position"
    )
    idx = idx.join(dupe_keys, on=["norm_name", "position"], how="anti")

    keyed = unmatched.with_columns(
        pl.col("player").map_elements(normalize_name, return_dtype=pl.String).alias("norm_name")
    )
    joined = keyed.join(idx, on=["norm_name", "position"], how="left")
    matched = joined.filter(pl.col("gsis_id").is_not_null()).drop("norm_name")
    still_unmatched = joined.filter(pl.col("gsis_id").is_null()).drop("norm_name", "gsis_id")
    return matched, still_unmatched


def build_future_consensus(
    season: int, future_population: pl.DataFrame, refresh: bool = False
) -> pl.DataFrame:
    """baseline_consensus for a live not-yet-played season, using the same
    leakage-safe "closest scrape strictly before Aug 1" cutoff Step 2's
    build_baseline_consensus already uses for historical seasons -- live current-season
    ECR data already exists (verified directly: a real 2026-07-31 scrape, strictly
    before the 2026-08-01 cutoff), so this isn't a backfill, just the same rule applied
    to a season that's still in progress rather than already complete. Falls back to a
    name-based match (see _match_ecr_by_name) for rows the ID crosswalk misses --
    without it, the vast majority of the current rookie class would silently lose the
    consensus safety net CLAUDE.md's rookie-handling rule specifically calls for."""
    ff_rankings = pull_ff_rankings(refresh)
    ff_playerids = pull_ff_playerids(refresh)
    ecr = pull_ecr_snapshot(ff_rankings, season, f"{season}-08-01")
    matched, unmatched = match_ecr_to_gsis(ecr, ff_playerids)

    if unmatched.height > 0:
        name_matched, still_unmatched = _match_ecr_by_name(unmatched, future_population)
        matched = pl.concat([matched, name_matched], how="diagonal_relaxed")
        if still_unmatched.height > 0:
            print(
                f"[build_future_consensus] {still_unmatched.height} {season} ECR rows "
                f"still unmatched after name fallback: "
                f"{still_unmatched.select('player', 'position').rows()}"
            )

    deduped = matched.unique(subset=["gsis_id", "season"], keep="first")
    return deduped.with_columns((-pl.col("ecr")).alias("baseline_consensus")).select(
        "gsis_id", "season", "baseline_consensus"
    )


def blend_future_rookie_projections(
    combined: pl.DataFrame, panel: pl.DataFrame, future_population: pl.DataFrame,
    season: int, refresh: bool = False, blend_weight: float = _BLEND_WEIGHT,
) -> pl.DataFrame:
    """Same blend as blend_rookie_projections, for the live `season` roster population.
    The rank-to-points curve itself is still fit on historical (2024/2025) draftable
    players -- that relationship doesn't change just because the target season hasn't
    been played yet."""
    curves = fit_rank_to_points_curve(panel, refresh)

    rookie_info = future_population.select("gsis_id", "season", "years_exp")
    with_baselines = build_future_consensus(season, future_population, refresh)

    out = combined.join(rookie_info, on=["gsis_id", "season"], how="left")
    out = out.join(with_baselines, on=["gsis_id", "season"], how="left")
    return _apply_rookie_blend(out, curves, blend_weight)
