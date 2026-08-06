# Fantasy football projection engine

## What this project is

A from-scratch fantasy football projection system for the 2026 NFL season, targeting a
12-team PPR/half-PPR redraft league. Two goals, equally weighted:

1. A draft board the owner will actually use on draft night (late August 2026).
2. A portfolio piece demonstrating modeling depth — ML, Bayesian shrinkage, simulation.

The core thesis: fantasy points decompose into `opportunities x efficiency x games played`,
and those three components have very different predictability. Volume is sticky year over
year. Efficiency — especially touchdown rate — is mostly noise. Availability is partly
predictable with high variance. Most public projections (and ADP itself) extrapolate total
fantasy points directly, silently importing last season's touchdown luck. Modeling the three
components separately is where the edge comes from.

See `docs/PLAN.md` for the full build spec.

## Stack

- Python, `uv` for dependency management
- `nflreadpy` for NFL data. **NOT `nfl_data_py`** — it is deprecated and unmaintained.
  `nflreadpy` returns Polars DataFrames; call `.to_pandas()` where a library needs it.
- `polars` for data manipulation, `duckdb` for ad hoc querying
- `lightgbm` and `scikit-learn` for modeling
- `numpy` for the Monte Carlo layer
- Historical ADP from the Fantasy Football Calculator public REST API

## Non-negotiable rules

**Temporal validation only.** Never use random K-fold on player-season data. Train on
seasons <= 2022, tune on 2023-2024, touch 2025 exactly once as a final test.

**Every feature must be knowable as of August 1 of the target season.** This is the single
biggest failure mode in this project. Full-season snap counts, end-of-season depth charts,
and final team totals are all leaks. The feature builder must assert an as-of date on every
column. If out-of-sample Spearman exceeds ~0.75, assume leakage until proven otherwise.

**Baselines come first.** Before any modeling work, implement and score two dumb baselines:
(a) prior-season points per game carried forward, (b) scraped public consensus projections.
Every model must beat both, out of sample, or it does not ship. Not beating consensus is a
likely outcome and an acceptable result — the response is to blend, not to hide it.

**Evaluate on within-position Spearman rank correlation over draftable players.** Not MAE on
season totals; that metric is dominated by injury noise and by the top 20 picks, which are
already priced correctly. Accuracy in rounds 4-10 is what changes draft decisions.

**Cache raw data to parquet.** Pull from the network once. Never hit `nflreadpy` inside a
training loop or a hyperparameter sweep.

## Scope discipline

There is a hard deadline: the draft is in late August 2026. Ship the simple version first
(point estimates, hardcoded replacement level, a working board), then add the sophisticated
layers in-season. Do not start the Dirichlet sampler or the draft simulator until the basic
board exists end to end and beats the baselines. See the phase split in `docs/PLAN.md`.

## Known traps — check against these before declaring anything working

- **Player ID matching** across nflverse GSIS IDs and FFC name strings will consume more time
  than any model. Use `nflreadpy`'s cross-platform ID mapping as the bridge. Log every
  unmatched draftable player loudly — silent drops mean missing players on the board.
- **Zero-opportunity seasons must stay in the panel.** Filtering to players who scored points
  teaches the model that the floor is far higher than it is.
- **Rookies have no prior usage** and will be projected near zero by default. Handle
  explicitly: blend in consensus for rookies only in phase 1, build a draft-capital-based
  rookie model later. Document whichever is in use.
- **Zero inflation.** Most rostered players get near-zero usage. Restrict to a draftable
  universe or use a two-stage model, or the loss function optimizes for predicting zeros.
- **Small N.** ~250 draftable players x 13 seasons is ~3,000 rows. That is small for gradient
  boosting. Shallow trees, strong regularization, few features. A regularized linear model
  beating LightGBM is a real possible outcome and worth reporting, not hiding.
- **Projections are not decisions.** A perfect projection still does not say whether to take
  the RB or the WR at pick 17. The value-over-replacement and tiering layer is not optional.

## Working preferences

- Explain modeling tradeoffs in comments where a choice is non-obvious, especially any place
  a simpler estimator was chosen over a fancier one.
- Prefer readable code over clever code. This doubles as a portfolio artifact.
- Write the test/assertion for leakage at the same time as the feature, not afterward.
- Keep league scoring rules as configuration, never hardcoded constants.
