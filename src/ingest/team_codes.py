"""Canonicalize team abbreviations across sources.

nflverse tables are NOT internally consistent about team codes: player_stats and
team_stats already use modern codes retroactively (LA, LAC, LV, ARI, BAL, CLE, HOU,
JAX, WAS for every season), but rosters/rosters_weekly mix in old-style codes
(ARZ, BLT, CLV, HST, SD, SL/STL, OAK) depending on the season, and the Fantasy
Football Calculator API uses yet another variant (LAR instead of LA). A naive
(season, team) join across these tables silently drops rows for exactly the
franchises that relocated or whose code changed -- found by diffing the team
vocab of every raw table against player_stats/team_stats before writing any join.

Canonical target = whatever player_stats/team_stats already use.
"""

import polars as pl

TEAM_ALIASES: dict[str, str | None] = {
    "ARZ": "ARI",
    "BLT": "BAL",
    "CLV": "CLE",
    "HST": "HOU",
    "JAC": "JAX",
    "WSH": "WAS",
    "SD": "LAC",
    "STL": "LA",
    "SL": "LA",
    "LAR": "LA",
    "OAK": "LV",
    "FA": None,  # free agent at ADP time -- no team
}


def canonicalize_team(expr: pl.Expr) -> pl.Expr:
    return expr.replace(TEAM_ALIASES)
