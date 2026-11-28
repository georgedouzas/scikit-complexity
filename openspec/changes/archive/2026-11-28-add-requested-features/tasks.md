## 1. Energies and interaction graphs (#49, #52)

- [x] 1.1 Record the observable `energy` in `IsingSystem`, `PottsSystem` and `XYSystem`, tested in
      `tests/systems/physics/statistical_mechanics/test_spins.py`
- [x] 1.2 Return the interaction graph of fixed lattice and continuous spaces from `Trajectory.get_network`, tested in
      `tests/core/engine/runtime/test_trajectories.py`
- [x] 1.3 Add `neighbors(kind='radius', r_max=)` with pairs and degrees across periodic edges, tested in
      `tests/core/analysis/test_neighbors.py`

## 2. Options of existing systems (#50, #51, #53, #54, #55)

- [x] 2.1 Add `groups='sample'` to `GalamMajoritySystem`, with tests
- [x] 2.2 Let `LogisticMapSystem` take an array of initial values, with tests
- [x] 2.3 Add `time='discrete'`, `start=`, `absorbing=` and `floor=` to `RandomWalkSystem`, with tests
- [x] 2.4 Replace `red=` and `blue=` of `PolyaUrnSystem` with `balls=`, update every caller, with tests
- [x] 2.5 Add `factors=` and `weights=` to `GibratProcessSystem`, with tests

## 3. Finite systems and Boolean networks (#56, #58)

- [x] 3.1 Hold the nodes of `RandomBooleanNetworkSystem` in one group with the state `nodes.x`, and add
      `update='asynchronous'` and `annealed=True`, with tests
- [x] 3.2 Add `automaton(kind='attractors')` for finite automata and Boolean networks, tested in
      `tests/core/analysis/test_automata.py`
- [x] 3.3 Let `automaton(kind='difference')` read Boolean networks, tested in `tests/core/analysis/test_automata.py`

## 4. Langton's ant (#57)

- [x] 4.1 Add `colors=` and `start=`, record the position and heading, and update only the ant's cell per step,
      with tests
- [x] 4.2 Add the unbounded plane for `shape=None`, with a test that it matches a large torus

## 5. Close

- [x] 5.1 Run the convention checks, ruff, black, mypy and lint-imports on the changed files
- [x] 5.2 Run the test groups of the changed modules, push, and close #49 to #58 with a comment each
