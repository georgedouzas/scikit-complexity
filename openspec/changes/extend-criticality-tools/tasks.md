# Tasks

## 1. Profiles of cascades (#78)

- [ ] 1.1 Add `profile` to `_CascadeRecordSpec` and fill it in `CascadeKernel._relax`, and test the single toppling and
  the profiles of a sandpile in `tests/core/components/laws/utils/test_relaxations.py` and
  `tests/core/engine/solvers/test_lattice.py`
- [ ] 1.2 Return the profiles as shapes in `events(kind='excursions')` and add `events(kind='activity')`, tested in
  `tests/core/analysis/test_events.py` with the two avalanche example
- [ ] 1.3 Test that `crackling_noise` collapses the shapes of sandpile avalanches in
  `tests/systems/physics/criticality/test_crackling_noise.py`
- [ ] 1.4 Describe profiles and the activity series in the docstring of `relax` and in
  `docs/overview/user_guide/build_a_custom_system/lattice_systems.md`, and verify that the examples run

## 2. Spectrum segments (#78)

- [x] 2.1 Add `segment_length` to `spectrum`, with its check and its setting of the kind `'power'`, tested in
  `tests/core/analysis/test_spectrum.py`

## 3. Events of activity (#77)

- [ ] 3.1 Add `activity` to `UpdateLaw`, with the check that a law with an activity has a label, tested in
  `tests/core/components/laws/test_update.py`
- [ ] 3.2 Add the private `_ActivityRecorder` in `core/engine/problems/_activity_recorder.py`, which joins active members
  by the nearest neighbours of the space with `scipy.sparse.csgraph`, tested through the public run in
  `tests/core/engine/solvers/test_lattice.py` and `tests/core/engine/solvers/test_maps.py` with the runs of one member
  and the two fires
- [ ] 3.3 Lower the activity in the compiler and call the recorder from the `LatticeSolver` and the `IterationSolver`
  after every step and on the initial state, and verify that the events reach `trajectory.events` in the same tests
- [ ] 3.4 Document the activity in the docstring of `UpdateLaw` and in
  `docs/overview/user_guide/build_a_custom_system/lattice_systems.md`, and verify that the examples run

## 4. Fires, jams and branching avalanches (#77)

- [ ] 4.1 Label the update of `FireInteraction` `'fire'` with the activity of burning sites, add `boundary` to
  `ForestFireSystem`, and test the corner fire, the strip and the driven forest in
  `tests/systems/physics/criticality/test_criticality.py`
- [ ] 4.2 Give the movement law of `NagelSchreckenbergInteraction` the activity of stopped cars and the label `'jam'`,
  and test the dissolving queue in `tests/systems/engineering/traffic/test_traffic.py`
- [ ] 4.3 Give the update of `z` in `BranchingInteraction` the activity and the label `'avalanche'`, add the state
  `sigma`, and test the profiles and the balance of dissipation in `tests/systems/physics/criticality/test_sandpiles.py`
- [ ] 4.4 Update the docstrings of the three systems and the gallery examples
  `docs/examples/physics/criticality/plot_forest_fire.py` and `plot_self_organized_branching.py` to read the events,
  and verify that the examples run

## 5. Toppling matrix and Green function (#76)

- [ ] 5.1 Add `abelian_sandpile` and `AbelianSandpileResult` in `systems/physics/criticality/_abelian_sandpile.py` with
  the kinds `'toppling'`, `'green'` and `'mean-size'` and their errors, and re-export them from
  `skcomplex.systems.physics.criticality`
- [ ] 5.2 Test the three by three matrix, the inverse, the agreement with a simulated pile and the growth with the side
  in `tests/systems/physics/criticality/test_abelian_sandpile.py`
- [ ] 5.3 Show the mean size against a simulated pile in `docs/examples/physics/criticality/plot_sandpile.py`, and
  verify that the example runs

## 6. Random neighbours (#79)

- [ ] 6.1 Add `neighbors='random'` and `conservation` to `TopplingInteraction`, tested in
  `tests/systems/physics/common/avalanches/test_toppling.py`
- [ ] 6.2 Add `neighbors` and `conservation` to `SandpileSystem`, with the error for `conservation=1`, test the
  mean-field avalanches in `tests/systems/physics/criticality/test_criticality.py`, and test that `abelian_sandpile`
  refuses random neighbours in `tests/systems/physics/criticality/test_abelian_sandpile.py`
- [ ] 6.3 Add `neighbors='random'` and `k` to `ExtinctionInteraction` and `BakSneppenSystem`, and test the threshold
  `1 / k` in `tests/systems/biology/evolution/test_coevolution.py`
- [ ] 6.4 Document both options in the docstrings of the systems, and verify the doctests

## 7. Checks

- [ ] 7.1 Run `black` and `ruff check` on the changed files and fix every finding
- [ ] 7.2 Run `mypy`, `pydoclint` and `interrogate` on the changed packages and fix every finding
- [ ] 7.3 Run `lint-imports` and `python openspec/scripts/check_conventions.py` and fix every finding
- [ ] 7.4 Run the tests of the touched modules with the thread variables set to 1:
  `tests/core/components/laws/test_update.py`, `tests/core/components/laws/utils/test_relaxations.py`,
  `tests/core/engine/solvers/test_lattice.py`, `tests/core/engine/solvers/test_maps.py`,
  `tests/core/analysis/test_events.py`, `tests/core/analysis/test_spectrum.py`, `tests/systems/physics/criticality`,
  `tests/systems/physics/common/avalanches`, `tests/systems/engineering/traffic/test_traffic.py` and
  `tests/systems/biology/evolution/test_coevolution.py`
