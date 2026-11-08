# Tasks

## 1. Dependency

- [ ] 1.1 Add `equinox>=0.13.8` to the runtime dependencies in `pyproject.toml` unless an earlier change of the wave
  added it, and verify `pdm install` resolves and `python -c "import equinox"` runs

## 2. Particle analyses

- [ ] 2.1 Hoist the periodic radius search of `neighbors` to `_radius_pairs` in `core/analysis/_base.py` with the
  minimum-image displacements, verified by `tests/core/analysis/test_neighbors.py` and a new unit test of the pairs
- [ ] 2.2 Add `mapping` and `forces` to `coarse_grain` and the field `forces` to its result, verified by
  `tests/core/analysis/test_coarse_grain.py` (centres of dimers across periodic sides, rows that do not sum to one)
- [ ] 2.3 Add `distribution_distance` with the three kinds, verified by
  `tests/core/analysis/test_distribution_distance.py` (identical sets, shifted sets, unequal sizes) and its doctest
- [ ] 2.4 Document bead mappings and distribution distances in the agent section of
  `docs/overview/user_guide/analyse.md` and verify its code blocks run

## 3. SPH dam break

- [ ] 3.1 Add `SphDamBreakSystem` in `systems/physics/fluids/_sph_dam_break.py`, verified by
  `tests/systems/physics/fluids/test_sph_dam_break.py` (surge front, mass and density) and its doctest
- [ ] 3.2 Register `sph_dam_break`, add its inventory entry and the `particle-surrogates` primitive, and verify
  `tests/systems/test_catalog.py` and `inventory_problems()` report no problem

## 4. Learned particle systems

- [ ] 4.1 Add `force_matching` with the linear Gaussian basis and its symbolic bead system, verified by
  `tests/core/analysis/test_force_matching.py` (identity mapping on Lennard-Jones, mismatched forces) and its doctest
- [ ] 4.2 Add the network kind of `force_matching` with the repulsive prior, and the dimer scenario, marked slow if
  needed
- [ ] 4.3 Add `hamiltonian_network` and its continuous-time system, verified by
  `tests/core/analysis/test_hamiltonian_network.py` (Lennard-Jones potential, invariance, bounded energy error of a
  spring chain) and its doctest
- [ ] 4.4 Add `graph_network` and its discrete-time system, verified by `tests/core/analysis/test_graph_network.py`
  (rollout better than constant velocity on short dam-break runs, forecast equal to the system's run, missing
  positions) and its doctest
- [ ] 4.5 Export the four functions from `skcomplex.core.analysis`, add a section on learned particle systems to
  `docs/overview/user_guide/analyse.md`, and verify its code blocks run and `tests/core/analysis/test_discovery.py`
  passes

## 5. Gallery

- [ ] 5.1 Add the gallery example of `sph_dam_break` that trains a graph network on it and compares the rollouts by
  `distribution_distance`, verified by `gallery_tools coverage --strict` and a run of the example

## 6. Checks

- [ ] 6.1 Run `nox -s checks` and `nox -s tests` with the thread limits set to 1, and verify they pass
