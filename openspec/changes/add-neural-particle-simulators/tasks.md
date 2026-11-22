# Tasks

## 1. Prerequisite

- [ ] 1.1 Check that `add-learning-foundation` is applied, so Equinox, `learned_function`, the shared Optax loop,
  `split_runs`, and `forecast_skill` exist, and stop if it is not

## 2. Particle analyses

- [ ] 2.1 Hoist the periodic radius search of `neighbors` to `_radius_pairs` in `src/skcomplex/core/analysis/_base.py` with the
  minimum-image displacements, verified by `tests/core/analysis/test_neighbors.py` and a new unit test of the pairs in
  `tests/core/analysis/test_base.py`
- [ ] 2.2 Add `mapping` and `forces` to `coarse_grain` and the field `forces` to its result, verified by
  `tests/core/analysis/test_coarse_grain.py` (centres of dimers across periodic sides, rows that do not sum to one)
- [ ] 2.3 Add `distribution_distance` with the three kinds in `src/skcomplex/core/analysis/_distribution_distance.py`, verified by
  `tests/core/analysis/test_distribution_distance.py` (identical sets, shifted sets, unequal sizes) and its doctest
- [ ] 2.4 Document bead mappings and distribution distances in the agent section of
  `docs/overview/user_guide/analyse.md` and verify its code blocks run

## 3. SPH dam break

- [ ] 3.1 Add `SphDamBreakSystem` in `src/skcomplex/systems/physics/fluids/_sph_dam_break.py`, with fluid and wall
  particles in one element, a pair interaction bound with `within=`, and a gravity interaction, verified by
  `tests/systems/physics/fluids/test_sph_dam_break.py` (surge front, mass and density) and its doctest
- [ ] 3.2 Register `sph_dam_break`, add its inventory entry and the `particle-surrogates` primitive, and verify
  `tests/systems/test_catalog.py` and `inventory_problems()` report no problem

## 4. Learned particle systems

- [ ] 4.1 Add `force_matching` in `src/skcomplex/core/analysis/_force_matching.py` with the linear Gaussian basis and its
  bead system, whose interaction states the fitted sum as a symbolic pair energy, verified by `tests/core/analysis/test_force_matching.py` (identity mapping on Lennard-Jones, mismatched
  forces) and its doctest
- [ ] 4.2 Add the network kind of `force_matching` with the repulsive prior, trained on the shared loop, whose system
  holds the prior plus a learned pair energy with the weights as an array parameter of its interaction, and the dimer
  scenario, marked slow if needed
- [ ] 4.3 Add `hamiltonian_network` in `src/skcomplex/core/analysis/_hamiltonian_network.py`, trained on the shared
  loop, and its system with a learned pair energy, verified by `tests/core/analysis/test_hamiltonian_network.py`
  (Lennard-Jones potential, invariance, learned pair energy on JAX, bounded energy error of a spring chain) and its
  doctest
- [ ] 4.4 Add `graph_network` in `src/skcomplex/core/analysis/_graph_network.py`, trained on the shared loop with runs
  held out by `split_runs`, its system with one update law, and `rollout_error` from `forecast_skill`, verified by
  `tests/core/analysis/test_graph_network.py` (rollout better than constant velocity on short dam-break runs, forecast
  equal to the system's run, missing positions) and its doctest
- [ ] 4.5 Export the four functions from `skcomplex.core.analysis`, add a section on learned particle systems to
  `docs/overview/user_guide/analyse.md`, and verify its code blocks run and `tests/core/analysis/test_discovery.py`
  passes

## 5. Gallery

- [ ] 5.1 Add the gallery example of `sph_dam_break` that trains a graph network on it and compares the rollouts by
  `distribution_distance`, verified by `gallery_tools coverage --strict` and a run of the example

## 6. Checks

- [ ] 6.1 Run `nox -s checks` and `nox -s tests` with the thread limits set to 1, and verify they pass
