# Proposal

## Why

Lattice models are the most common formalism across the references. They include:
- **Cellular automata:** elementary rules, Game of Life, majority rule, traffic, and excitable media (Charbonneau 2017
  ch. 2, 7, 11; Downey's *Think Complexity* ch. 5-6; Argun et al. 2021 ch. 4; Gros 2015 ch. 6).
- **Self-organised criticality:** sandpiles, forest fires, and earthquake models, which separate a slow drive from a
  fast relaxation (Charbonneau 2017 ch. 5-6, 8; Jensen 2023 ch. 12; Satz 2022 ch. 8).
- **Spin models:** Ising, XY, and Potts models sampled by Metropolis and Glauber dynamics (Argun et al. 2021 ch. 2;
  Jensen 2023 ch. 6; Lindgren 2024 ch. 5).

Ceccherini-Silberstein and Coornaert (2023 def. 1.4.1) define a cellular automaton over any lattice by a finite memory
set and a local rule on the ordered neighbourhood pattern. That rule is neither a sum of pair terms nor an agent rule,
so the framework cannot express these models today except as slow Mesa agents.

## What Changes

- Add lattice spaces: `Z^d` boxes with periodic, fixed, or reflecting edges, and neighbourhoods given by Moore, von
  Neumann, radius, or ordered custom offsets.
- Add cell groups: one member per lattice site, with boolean, categorical, integer, or real states.
- Add local rules: interactions that compute a cell's next value from its ordered neighbourhood pattern. A rule is
  given as a function, a lookup table, a Wolfram rule number, or a totalistic or outer-totalistic rule.
- Run rules vectorised over the whole lattice on the NumPy and Numba backends.
- Add update schedules: synchronous by default, plus random sequential, random site, and checkerboard updates, and
  stochastic rules that draw from the simulation's random stream.
- Add relaxation: a drive law applied once per step and relaxation laws iterated until the lattice is stable, with each
  relaxation recorded as an event of given size, duration, and area.
- Add Monte Carlo dynamics: an energy declared symbolically over neighbourhoods, proposed moves, and Metropolis or
  Glauber acceptance at a live temperature. One unit of time is one sweep.
- Log lattices to Rerun as images or tensors, colored by state.
- Add library models: elementary automata, Game of Life, majority rule, Langton's ant, Nagel-Schreckenberg traffic,
  the hodgepodge machine, Bak-Tang-Wiesenfeld sandpile, Drossel-Schwabl forest fire, Olami-Feder-Christensen
  earthquakes, Ising, Potts, and XY.

## Capabilities

### New Capabilities

- `lattice-dynamics`: lattice spaces, local rules, update schedules, relaxation, and Monte Carlo dynamics.

### Modified Capabilities

- `model-library`: lattice models.

## Impact

- `skcomplex.model` gains `LatticeSpace`, `Neighborhood`, `LocalRule`, `Energy`, and `Move`. A new form `Form.LATTICE`
  covers local rules, and a new form `Form.MONTE_CARLO` covers energies with moves. Both combine with the other forms
  through co-simulation.
- `skcomplex.solvers` gains the lattice stepper and the Monte Carlo stepper.
- Depends on `add-general-model-interface` for discrete domains and observables, and on `add-analysis-toolkit` for
  avalanche and cluster statistics in its acceptance tests.
