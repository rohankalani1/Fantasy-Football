"""Vegas preseason win totals, 2013-2025, scraped from covers.com/sportsoddshistory's
two static decade-summary pages (not a JS app -- plain server-rendered HTML tables, one
row per current franchise covering every year back to that decade regardless of the
team's name/location at the time, e.g. the "Los Angeles Rams" row also holds their
St. Louis-era win totals).

No public API exists for this data anywhere in nflverse/nflreadpy (see the project
history around Step 3 for what was checked and ruled out). The 2026 season is not yet
populated on this site as of this writing -- see config/current_season_inputs.csv for
the manual fill-in used for the live draft-night projection.
"""

import os
import re

import nflreadpy as nfl
import polars as pl
import requests

from src.ingest.constants import RAW_DIR
from src.ingest.team_codes import TEAM_ALIASES

_DECADE_URLS = {
    "2010s": "https://www.covers.com/sportsoddshistory/nfl-regular-season-win-total-results-by-team-2010s/",
    "2020s": "https://www.covers.com/sportsoddshistory/nfl-regular-season-win-total-results-by-team/",
}
_HEADERS = {"User-Agent": "Mozilla/5.0"}

# Number of year columns on each decade page, in header order starting at the decade's
# first year -- needed because rows are sliced positionally (see _parse_decade_table).
_DECADE_YEAR_COUNTS = {2010: 10, 2020: 7}

_ROW_RE = re.compile(r"<tr>(.*?)</tr>", re.DOTALL)
_TEAM_NAME_RE = re.compile(r"Team=[^\"]*\">([^<]+)</a>")
_CELL_RE = re.compile(r"<td[^>]*>(.*?)</td>", re.DOTALL)
# A year cell's line is its leading numeric token (or "?" if the book never posted a
# line that year, e.g. Colts 2011 after Peyton Manning's injury). NOT anchored to a
# trailing <br/> -- some cells lack the <br/>-separated actual-result half entirely, and
# requiring it silently drops the cell, which desyncs every later cell's year index.
_LINE_RE = re.compile(r"^\s*([\d.]+|\?)")


def _team_name_to_abbr() -> dict[str, str]:
    """Full team name (as used by covers.com, e.g. "Los Angeles Rams") -> canonical abbr."""
    teams = nfl.load_teams().select("team_abbr", "team_name").to_dicts()
    # Canonical = any abbr that isn't itself an alias key (i.e. survives canonicalize_team) --
    # keeps e.g. "LA"/"Los Angeles Rams" and drops the stale "LAR"/"STL" duplicate rows.
    alias_keys = set(TEAM_ALIASES.keys())
    return {
        row["team_name"]: row["team_abbr"]
        for row in teams
        if row["team_abbr"] not in alias_keys
    }


def _fetch_decade_html(decade: str, refresh: bool = False) -> str:
    path = os.path.join(RAW_DIR, f"vegas_win_totals_{decade}.html")
    if os.path.exists(path) and not refresh:
        with open(path, encoding="utf-8") as f:
            return f.read()
    resp = requests.get(_DECADE_URLS[decade], headers=_HEADERS, timeout=30)
    resp.raise_for_status()
    os.makedirs(RAW_DIR, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(resp.text)
    return resp.text


def _parse_decade_table(html: str, start_year: int, name_to_abbr: dict[str, str]) -> pl.DataFrame:
    n_year_cols = _DECADE_YEAR_COUNTS[start_year]
    start = html.find('<table class="soh1">')
    end = html.find("</table>", start) + len("</table>")
    table = html[start:end]

    rows = []
    unmatched_teams = set()
    for row_html in _ROW_RE.findall(table):
        team_match = _TEAM_NAME_RE.search(row_html)
        if not team_match:
            continue  # header rows have no team link
        team_name = team_match.group(1).strip()
        abbr = name_to_abbr.get(team_name)
        if abbr is None:
            unmatched_teams.add(team_name)
            continue

        # First <td> is the team-name cell itself; the next n_year_cols are the year
        # columns, in order -- sliced positionally so a malformed cell (e.g. no line
        # posted that year) can't shift every later year's index.
        cells = _CELL_RE.findall(row_html)
        year_cells = cells[1 : 1 + n_year_cols]
        if len(year_cells) != n_year_cols:
            raise ValueError(
                f"{team_name}: expected {n_year_cols} year cells, found {len(year_cells)} "
                f"-- covers.com's table layout may have changed."
            )
        for i, cell in enumerate(year_cells):
            # A cell with no <br/> has no posted line at all -- just a bare actual-wins
            # number (e.g. Colts 2011, after Peyton Manning's injury pulled the market).
            # Confirmed by inspecting the raw HTML directly: normal cells are always
            # "{line} <br/>{actual} &nbsp;{O/U/P}"; line-withdrawn cells are just
            # "{actual} &nbsp;" with no <br/>. Treat both this and "?" (line not yet
            # posted, e.g. future seasons) as no data, not as a real win total.
            if "<br" not in cell:
                continue
            line_match = _LINE_RE.match(cell)
            if line_match is None:
                raise ValueError(f"{team_name}, {start_year + i}: cell didn't match expected format: {cell!r}")
            line = line_match.group(1)
            if line == "?":
                continue
            rows.append({"team": abbr, "season": start_year + i, "vegas_win_total": float(line)})

    if unmatched_teams:
        raise ValueError(f"Unrecognized team names on covers.com page: {unmatched_teams}")

    return pl.DataFrame(rows, schema={"team": pl.String, "season": pl.Int64, "vegas_win_total": pl.Float64})


# A win total outside this range is a source data-entry error, not a real line -- e.g.
# Philadelphia's 2023 cell literally reads "115" on covers.com (obviously meant "11.5").
# Confirmed by inspecting the raw HTML directly, not guessed: no decimal-shift "fix" is
# applied, the row is just dropped and logged loudly, per CLAUDE.md's rule against
# silent data drops.
_PLAUSIBLE_WIN_TOTAL_RANGE = (1.0, 17.0)


def pull_vegas_win_totals(refresh: bool = False) -> pl.DataFrame:
    name_to_abbr = _team_name_to_abbr()

    html_2010s = _fetch_decade_html("2010s", refresh)
    html_2020s = _fetch_decade_html("2020s", refresh)

    totals_2010s = _parse_decade_table(html_2010s, 2010, name_to_abbr)
    totals_2020s = _parse_decade_table(html_2020s, 2020, name_to_abbr)

    totals = pl.concat([totals_2010s, totals_2020s]).unique(subset=["team", "season"]).sort(
        ["season", "team"]
    )

    lo, hi = _PLAUSIBLE_WIN_TOTAL_RANGE
    bad = totals.filter((pl.col("vegas_win_total") < lo) | (pl.col("vegas_win_total") > hi))
    if bad.height > 0:
        print(
            f"[pull_vegas_win_totals] dropping {bad.height} row(s) with an implausible "
            f"win total (source data-entry error, not fixed/guessed): {bad.rows()}"
        )
        totals = totals.filter((pl.col("vegas_win_total") >= lo) & (pl.col("vegas_win_total") <= hi))

    return totals
