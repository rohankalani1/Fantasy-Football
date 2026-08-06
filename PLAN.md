# Build plan

Read `CLAUDE.md` first for project rules. This document is the step-by-step spec.

## Architecture

```
Player-season panel (2012-2025)          nflreadpy + FFC historical ADP
                |
     +----------+----------+
     |          |          |
 Usage share  Efficiency  Availability   GBM+Dirichlet | Bayes shrinkage | hazard
     |          |          |
     +----------+----------+
                |
     Monte Carlo season simulator        10k correlated seasons per player
                |
     Draft simulator                     ADP bots, endogenous replacement level
                |
     Tiered draft board
```

## Phase 1 — ship before the draft (late August 2026)

Target: a working board that beats both baselines. Point estimates only.

### Step 0 — Environment
`uv` project. Dependencies: `nflreadpy`, `polars`, `duckdb`, `lightgbm`, `scikit-learn`,
`numpy`, `requests`. Pull raw nflverse data once, cache to `data/raw/*.parquet`.

### Step 1 — Player-season panel
One row per player-season, 2012-2025. Columns: targets, carries, receptions, receiving and
rushing yards, touchdowns, snaps, games played, games active, team, age, position, draft
round and pick, plus team-level offensive totals (plays, pass attempts, rush attempts).

Attach historical ADP by season from the Fantasy Football Calculator REST API. Match on the
nflverse cross-platform ID mapping, fall back to normalized name plus position plus team,
and emit a report of unmatched draftable players.

Include zero-opportunity rostered seasons. Do not filter on production.

### Step 2 — Baselines (do this before any modeling)
- `baseline_ppg`: prior-season points per game, carried forward, zero for rookies.
- `baseline_consensus`: scraped public consensus projections.

Score both on 2024 and 2025 using the metric in Step 8. Record the numbers in
`results/baselines.json`. These are the bar.

### Step 3 — Team volume projection
Per team, project 2026 total offensive plays and pass/run split. Inputs: prior two seasons'
pace and pass rate over expected, head coach and offensive coordinator change flags, Vegas
preseason win total. Output: projected team pass attempts and rush attempts.

### Step 4 — Usage share model (the core)
Predict each player's share of team targets and share of team carries for season t+1.

Target variable is the share, not the raw count. LightGBM regressor per position group.

Features (all as-of August 1 of season t+1):
- two-year exponentially weighted prior target share and rush share
- games played in seasons t and t-1
- snap share in season t
- **vacated share**: summed target/rush share of players at the same position who left the
  team between seasons — this is one of the highest-signal features and is fiddly to build
- draft round and overall pick, years of experience, age
- team-change flag, position, projected team pass rate from Step 3

Shallow trees (max_depth 3-4), strong L2, early stopping on the 2023-2024 validation seasons.

### Step 5 — Efficiency model
Yards per target, yards per carry, touchdowns per opportunity (separately for rushing and
receiving). Empirical Bayes: shrink each player's career rate toward his position-and-role
mean, with shrinkage weight a function of career opportunity count.

Shrink touchdown rate hard — close to fully. Year-over-year TD rate correlation is near zero,
and ADP's overreaction to prior-season touchdown luck is a primary source of edge.

Do not use gradient boosting here. Demonstrate in the writeup that shrinkage beats GBM on
this component; that comparison is itself a result.

### Step 6 — Availability model
Phase 1: beta-binomial over games played (0-17), parameterized by age, position, and games
missed in the prior two seasons.

### Step 7 — Combine
`points = team_volume x usage_share x efficiency x games_played`, with league scoring read
from a config file. Output projected total points and points per game per player.

### Step 8 — Validate
Within-position Spearman rank correlation over draftable players, versus both Step 2
baselines, on 2024 and 2025. Also report the same metric for ADP itself as a third reference
point. If the model loses to consensus, fit a blend weight and report the blend.

### Step 9 — Draft board
Value over replacement using 12 teams and the league's starting lineup requirements, with a
hardcoded replacement level for now. Cluster into tiers. Export to CSV.

Add a manual override file: a CSV of player-level adjustments applied at load time, so late
August news (trades, injuries, depth chart surprises) can be reflected on draft morning
without a rerun.

## Phase 2 — build in-season (September onward)

- **Dirichlet usage shares.** Model within-team shares as Dirichlet so they sum to 1. This
  enforces that teammates compete for a fixed pie and induces negative correlation between
  them for free when sampling. Fit the concentration parameter to observed dispersion.
- **Discrete-time hazard model** for games played, replacing the beta-binomial.
- **Monte Carlo season simulator.** 10k draws. Sample team volume at the team level (positive
  within-team correlation), Dirichlet shares jointly per team, availability per player,
  efficiency per player. Output full distributions: p10/p50/p90 and probability of finishing
  top-N at position.
- **Draft simulator.** Bots draft from ADP with Gaussian noise; use the real ADP standard
  deviation published by Fantasy Football Calculator rather than a guess. Replacement level
  becomes endogenous — derived from what actually goes undrafted in simulated leagues, rather
  than assumed.
- **Rookie model** using draft capital, college production, and landing spot, replacing the
  phase 1 consensus blend.
- **In-season validation.** Track weekly how the preseason projections held up. This is a
  stronger portfolio artifact than a backtest alone.

## Repo layout

```
data/raw/          cached parquet from nflreadpy and FFC
data/processed/    the player-season panel
src/ingest/        data pulls and ID matching
src/features/      feature builders, each with an as-of date assertion
src/models/        usage, efficiency, availability
src/sim/           Monte Carlo and draft simulator (phase 2)
src/board/         VOR, tiering, CSV export
results/           baseline and model scores
config/            league scoring, roster requirements
notebooks/         exploration only, never the source of truth
```
