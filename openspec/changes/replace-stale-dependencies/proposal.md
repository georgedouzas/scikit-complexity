# Proposal

## Why

The project aims to include libraries that are frequently updated, but three runtime dependencies have published nothing
for three years: `tidynamics` (July 2023), `kramersmoyal` (August 2023), and `matching` (October 2023). A stale library can
block the whole package, as `deeptime` blocked Python 3.14, and nothing today states a limit or checks for one.

## What Changes

- State a limit: a runtime dependency has a release within the last 24 months, or the Project Profile records why it may
  stay. Check it when adding a dependency and before each release.
- Replace `tidynamics` in `DisplacementAnalysis` with a NumPy and SciPy computation of the mean squared displacement.
- Replace `kramersmoyal` in `KramersMoyalAnalysis` with a NumPy computation of the kernel-weighted conditional moments.
- Replace `matching` in `MatchingInteraction` with a private deferred-acceptance method for `StableMatchingSpec`.
- Keep the public results of these three classes, within numerical tolerance for the two analyses, and the same stable
  matching for the interaction.
- Remove the three libraries from the manifest, the lock file, and the Project Profile.

`deeptime` is out of scope, because `add-python-3-14-support` replaces it. `mrestimator` (June 2025), `pydmd` (August 2025),
`igraph` (October 2025), `nolds` (November 2025), `powerlaw` (January 2026), and `pysindy` (January 2026) are inside the
limit and stay.

## Capabilities

### New Capabilities

- `dependency-freshness`: the limit on how long a runtime dependency may go without a release, and when it is checked.

### Modified Capabilities

None. `agent-analysis`, `stochastic-analysis`, and the matching requirements state results rather than libraries, and their
behaviour does not change.

## Impact

- Source: `src/skcomplex/analysis/_displacement.py`, `src/skcomplex/analysis/_kramers_moyal.py`, and
  `src/skcomplex/core/interactions/_matching.py`, with their tests and reference values.
- Models that use `StableMatchingSpec` through `MatchingInteraction`, whose results must not change.
- Packaging: `pyproject.toml`, including the deptry list, and `pdm.lock`.
- Documentation: `docs/overview/user_guide.md`, `CONTRIBUTING.md`, and `openspec/constitution.md`.
- The project writes three small numerical algorithms itself, an exception to wrapping mature libraries that the design
  justifies for each.
