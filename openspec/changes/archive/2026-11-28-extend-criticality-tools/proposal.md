# Proposal

## Why

The self-organized criticality chapter of the Complexity Science tutorials filed issues #76 to #79. It falls back to
NumPy and SciPy for Dhar's lattice Green function, for the avalanches of the branching process, the forest fire and
the traffic jams, for the fast-time activity of a sandpile, and for the mean-field variants of the sandpile and of
Bak-Sneppen. Each of these is general enough to belong in the library.

## What Changes

- **Toppling matrix and Green function (#76):** a new analysis `abelian_sandpile(system, kind=...)` of the criticality
  subpackage returns the toppling matrix `Delta` of a `SandpileSystem` lattice for `kind='toppling'`, its inverse, the
  lattice Green function, for `kind='green'`, and Dhar's mean avalanche size `(1 / N) Σ_ij (Delta^-1)_ij` for
  `kind='mean-size'`.
- **Events of activity (#77):** an `UpdateLaw` takes an `activity`, an expression of the values of its member, and the
  run records every connected burst of activity in space and time as an event under the label of the law, with its
  size, duration, area and profile. Bursts that touch in space at one step or follow each other at neighbouring sites
  in consecutive steps are one event.
- **Avalanches, fires and jams (#77):** `SelfOrganizedBranchingSystem` records `avalanche` events and holds the
  dissipation of the last avalanche in the new state `tree.sigma`. `ForestFireSystem` and `HexagonalForestFireSystem`
  record `fire` events. `NagelSchreckenbergSystem` records `jam` events, the clusters of stopped cars.
- **Forest fire boundary (#77):** `ForestFireSystem(boundary=...)` takes `'periodic'`, the default and today's
  behaviour, `'fixed'` or `'reflecting'`, like the other lattice systems, and the docstring says what each means.
- **Avalanche profiles (#78):** every cascade record of `relax`, such as the avalanches of `SandpileSystem`, holds its
  profile, the number of active sites at every round of the relaxation. `events(kind='excursions')` then returns these
  profiles as the shapes, so `crackling_noise` collapses the shapes of sandpile avalanches.
- **Fast-time activity (#78):** `events(trajectory.events, kind='activity')` returns the activity of every round of
  every cascade in fast time, with one quiet round for every drive, the series whose power spectrum Downey computes.
- **Spectrum segments (#78):** `spectrum(kind='power', segment_length=...)` sets the length of the Welch segments, 1024
  by default as today.
- **Random neighbours (#79):** `SandpileSystem(neighbors='random')` topples to sites drawn at random every time, the
  rule R2-MF of Jensen, with the share `conservation` of the grains kept. `BakSneppenSystem(neighbors='random', k=K)`
  renews the least fit species and `K - 1` species drawn afresh at every update.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `lattice-dynamics`: relaxation records carry profiles, and update laws record events of activity.
- `criticality-analysis`: the toppling matrix, Green function and mean avalanche size, the fast-time activity series,
  and the collapse of sandpile avalanche shapes.
- `trajectory-analysis`: the segment length of power spectra.
- `model-catalog`: the events and options of the branching process, the forest fires, the traffic system, the sandpile
  and Bak-Sneppen.

## Impact

- Laws and engine: `core/components/laws/_update.py`, `core/components/laws/utils/_relaxations.py`,
  `core/engine/problems/_cascade_record_spec.py` and `_cascade_kernel.py`, a new private recorder of activity events in
  `core/engine/problems`, and the solvers `core/engine/solvers/_lattice.py` and the iteration solver that run update
  laws.
- Analysis: `core/analysis/_events.py` and `core/analysis/_spectrum.py`.
- Systems: `systems/physics/criticality/_criticality.py`, `_sandpiles.py` and a new `_abelian_sandpile.py`,
  `systems/physics/common/avalanches/_toppling.py`, `systems/engineering/traffic/_traffic.py` and
  `systems/biology/evolution/_coevolution.py`, with the `__init__` modules that re-export new names.
- Tests under the mirrored paths of `tests/`, the user guide page on lattice systems, and the gallery examples of the
  sandpile, the forest fire and the branching process.
- Issues #76 to #79 close with the change.
