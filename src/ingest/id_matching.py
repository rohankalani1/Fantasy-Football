"""Match Fantasy Football Calculator ADP rows to nflverse gsis_id.

FFC doesn't participate in any cross-platform ID crosswalk, so this is fundamentally
name-based. Two stages, in order:

1. Bridge through the community-maintained ff_playerids table (merge_name + position),
   which already has gsis_id and has handled a lot of nickname/formatting quirks
   (e.g. "Gabe" vs "Gabriel" Davis). Only accepted if it resolves to exactly one
   candidate AND that candidate actually appears on a roster the same season+position
   -- otherwise the bridge is stale or ambiguous and we fall through to stage 2.
2. Direct match against this project's own roster spine (gsis_id, season, position,
   team, display_name) on normalized name + position + season, using team as a
   tiebreaker when more than one player shares a normalized name and position in the
   same season.

Anything left unresolved is reported, never silently dropped -- a silent drop here
means a real player missing from the draft board.
"""

import re

import polars as pl

_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}
_PUNCT_RE = re.compile(r"[^a-z0-9\s-]")

# FFC lists some players under a nickname or informal first name nflverse doesn't use
# (and the ff_playerids bridge doesn't catch either), so no amount of punctuation/
# suffix normalization resolves them. Found by inspecting the unmatched report: these
# recurred across every season the player was drafted, not a one-off typo, so they're
# worth a permanent entry rather than a per-season manual override. Keep this list
# small and evidence-based -- add to it only when the unmatched report shows a repeat
# offender, not preemptively.
MANUAL_NAME_ALIASES: dict[str, str] = {
    "hollywood brown": "marquise brown",
    "joshua palmer": "josh palmer",
}


def normalize_name(name: str | None) -> str | None:
    if name is None:
        return None
    s = name.lower()
    s = MANUAL_NAME_ALIASES.get(s, s)
    s = _PUNCT_RE.sub("", s)
    tokens = [t for t in re.split(r"[\s-]+", s) if t and t not in _SUFFIXES]
    return "".join(tokens)


def build_name_index(roster_spine: pl.DataFrame) -> pl.DataFrame:
    """roster_spine: gsis_id, season, position, team, display_name -> adds norm_name."""
    return roster_spine.with_columns(
        pl.col("display_name")
        .map_elements(normalize_name, return_dtype=pl.String)
        .alias("norm_name")
    )


def _stage1_bridge(ffc: pl.DataFrame, ff_playerids: pl.DataFrame, name_index: pl.DataFrame) -> pl.DataFrame:
    """Returns ffc rows with a `gsis_id` column populated where stage 1 resolves cleanly."""
    bridge = (
        ff_playerids.filter(pl.col("gsis_id").is_not_null())
        .with_columns(
            pl.col("merge_name")
            .map_elements(normalize_name, return_dtype=pl.String)
            .alias("norm_name")
        )
        .select("norm_name", "position", "gsis_id")
        .unique()
    )
    # Drop (norm_name, position) keys that map to more than one gsis_id -- ambiguous bridge.
    dupe_keys = (
        bridge.group_by(["norm_name", "position"])
        .len()
        .filter(pl.col("len") > 1)
        .select("norm_name", "position")
    )
    bridge = bridge.join(dupe_keys, on=["norm_name", "position"], how="anti")

    ffc_keyed = ffc.with_columns(
        pl.col("name").map_elements(normalize_name, return_dtype=pl.String).alias("norm_name")
    )
    candidate = ffc_keyed.join(bridge, on=["norm_name", "position"], how="left")

    # Validate: the bridged gsis_id must actually be rostered at that position+season.
    valid_gsis_season = name_index.select("gsis_id", "season", "position").unique()
    candidate = candidate.join(
        valid_gsis_season.with_columns(pl.lit(True).alias("_valid")),
        on=["gsis_id", "season", "position"],
        how="left",
    )
    candidate = candidate.with_columns(
        pl.when(pl.col("_valid").is_not_null()).then(pl.col("gsis_id")).otherwise(None).alias("gsis_id")
    ).drop("_valid")
    return candidate


def _stage2_direct(unresolved: pl.DataFrame, name_index: pl.DataFrame) -> pl.DataFrame:
    """Direct normalized-name + position + season match against the roster spine,
    with team as a tiebreaker when multiple players share the key."""
    idx = name_index.select("gsis_id", "season", "position", "team", "norm_name")

    unresolved_keyed = unresolved.drop("gsis_id").with_columns(
        pl.col("name").map_elements(normalize_name, return_dtype=pl.String).alias("norm_name")
    )

    joined = unresolved_keyed.join(idx, on=["norm_name", "position", "season"], how="left", suffix="_idx")

    match_counts = (
        joined.filter(pl.col("gsis_id").is_not_null())
        .group_by(["name", "season", "position"])
        .agg(pl.col("gsis_id").n_unique().alias("n_candidates"))
    )
    joined = joined.join(match_counts, on=["name", "season", "position"], how="left")

    # Unique match: keep as-is. Multiple candidates: keep only rows where team also
    # matches; if that still leaves != 1 row, drop to unmatched.
    unique_matches = joined.filter(pl.col("n_candidates") == 1)
    ambiguous = joined.filter(pl.col("n_candidates") > 1)
    team_resolved = ambiguous.filter(pl.col("team") == pl.col("team_idx"))
    team_resolved_counts = team_resolved.group_by(["name", "season", "position"]).len()
    team_resolved = team_resolved.join(
        team_resolved_counts.filter(pl.col("len") == 1).select("name", "season", "position"),
        on=["name", "season", "position"],
        how="inner",
    )

    resolved = pl.concat(
        [unique_matches, team_resolved], how="diagonal_relaxed"
    ).unique(subset=["player_id", "season"])

    resolved = resolved.drop(["n_candidates", "team_idx", "norm_name"])
    return resolved


def match_adp_to_gsis(
    ffc: pl.DataFrame, ff_playerids: pl.DataFrame, name_index: pl.DataFrame
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Returns (matched, unmatched). matched has a gsis_id column; unmatched does not."""
    stage1 = _stage1_bridge(ffc, ff_playerids, name_index)
    stage1_matched = stage1.filter(pl.col("gsis_id").is_not_null()).drop("norm_name")
    stage1_unresolved = stage1.filter(pl.col("gsis_id").is_null()).drop("norm_name")

    stage2_matched = _stage2_direct(stage1_unresolved, name_index)

    matched = pl.concat([stage1_matched, stage2_matched], how="diagonal_relaxed")
    matched_keys = matched.select("player_id", "season").unique()
    unmatched = ffc.join(matched_keys, on=["player_id", "season"], how="anti")
    return matched, unmatched
