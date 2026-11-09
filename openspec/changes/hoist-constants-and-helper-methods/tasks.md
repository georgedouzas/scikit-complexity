# Tasks

The answers to the open questions in design.md are assumed to be the recommended ones. Scripts live in the session
scratchpad, never in the repository. Every group ends with `ruff format`, `ruff check`, `mypy`, `lint-imports`, the
mirrored tests, and the doctests of its package passing, and one commit on `development`. When several agents run
tests at once, each sets `NUMBA_NUM_THREADS`, `OMP_NUM_THREADS`, and the BLAS thread counts to 1.

## 1. Measure and add the check

- [ ] 1.1 Confirm that `add-system-package`, `slim-core-elements-and-spaces`, `slim-core-interactions`, and
  `slim-core-laws` are applied, then run the scratchpad scan and record per package the private module functions by
  kind of reader, the private and misplaced constants, the gaps, and the unexported public functions. Verify the scan
  lists the cases named in design.md that still exist.
- [ ] 1.2 Record the union of every `__all__` of `src/skcomplex` in the scratchpad, to compare at the end. Verify it
  imports cleanly.
- [ ] 1.3 Add `test_module_constants_lead_their_module`, `test_constant_blocks_hold_no_gap`, and
  `test_classes_own_the_functions_they_read` to `tests/test_sources.py`, parametrised by an empty tuple of checked
  packages. Verify each fails on a small source written to `tmp_path` that breaks its rule and passes on one that
  follows it, and names the path, line, and name.
- [ ] 1.4 Write the two scratchpad scripts of design.md decision 9, one for constants and one for helpers. Verify each
  on one module by hand before the groups below use them.

## 2. Core components

- [ ] 2.1 In `core/components` (base, elements, spaces, interactions, laws, `laws/utils`, systems), hoist and rename
  constants, mark the derived values of `laws/utils` (the reductions and `STOP_GRADIENT`)
  and of the sampling interaction, and move every helper a class reads into its owner. Keep the Numba kernels of the
  mobility and sampling interactions as module functions. Verify the source tests pass with `skcomplex.core.components`
  added to the checked packages.
- [ ] 2.2 Fold the attribute docstrings of the package's constants into the docstrings that read them. Verify the docs
  build shows no missing text for the re-exported law utilities.

## 3. Core engine

- [ ] 3.1 In `engine/compile` and `engine/compile/passes`, rename the private constants, renaming `_MEMBERS` of
  `_layout.py` to `HIDDEN_MEMBERS`, and make the helpers of `compile/_complex.py` static methods of `LowerComplexPass`
  that `ObservableEvaluator` and `ReductionLowering` call through it. Move `probe_laws` and its namespace helper into
  `engine/_utils.py` and remove `compile/_probe.py`. Verify the compile tests and `lint-imports` pass.
- [ ] 3.2 In `engine/problems`, `engine/backends`, and `engine/solvers`, rename and hoist constants, keep every Numba
  kernel and `step_count` as module functions after the constants, and move the plain Python helpers into their
  classes. Mark `HELPERS_FINGERPRINT` as a derived value. Verify the backend and solver tests pass and a Numba run
  recompiles once and then hits the cache.
- [ ] 3.3 In `engine/runtime`, `engine/translators`, and `engine/viz`, rename and hoist constants and move the helpers
  that classes read. Keep the helpers of `simulate` that no class reads. Verify the runtime, translator, and viz tests
  pass with `skcomplex.core.engine` added to the checked packages.

## 4. Core analysis

- [ ] 4.1 Hoist the constants of every analysis module, including the late blocks of `_automata.py`, and fold their
  attribute docstrings into the module or function docstrings. Verify `test_module_constants_lead_their_module` and
  `test_constant_blocks_hold_no_gap` pass on the package.
- [ ] 4.2 Move every helper that a result class or another class of an analysis module reads into that class. Keep the
  helpers of the public analysis functions and of `_base.py` that no class reads. Mark `MITTAG_LEFFLER` as a derived
  value with its hooks beside it. Verify the analysis tests pass with `skcomplex.core.analysis` added to the checked
  packages.

## 5. Core experiments, specs, and the root modules

- [ ] 5.1 In `core/experiments`, move the helpers that `_CalibrationObjective` and the result classes read into them,
  and keep the run and design helpers that only the experiment functions read. Update the imports of
  `tests/core/experiments/test_calibration.py` to call the moved functions through their class. Verify the experiment
  tests, including the parallel ones, pass.
- [ ] 5.2 In `core/specs` and the root modules of `core`, rename the private dialects and mark them and `DIALECTS` as
  derived values. Verify the spec tests pass with `skcomplex.core.experiments`, `skcomplex.core.specs`, and the root
  modules added to the checked packages.

## 6. Systems

Each task applies both scripts to one field, moves the helpers of system classes into their owners under the rule of
design.md decision 4, marks the derived values that design.md names, and verifies the field's tests and the catalog
tests pass with the field added to the checked packages.

- [ ] 6.1 `systems/_catalog.py` and `_extras.py`: rename `_LISTED` and `_SYSTEM_CLASSES`, the latter as a derived
  value. Update `tests/systems/test_catalog.py` only where it imports a renamed name. Verify `all_systems` returns the
  same list as before.
- [ ] 6.2 Biology, including `_carbon_13_mfa.py` and `_logical_networks.py`, whose three constants drop the
  annotation that names a local alias.
- [ ] 6.3 Chemistry, including the shared diffusion helpers of `_reaction_diffusion.py`.
- [ ] 6.4 Computer science, including the landscapes of `_optimisers.py`.
- [ ] 6.5 Economics, including the predictors of `_el_farol.py`, the heuristics namespace, and `landscape` of
  `_sugarscape.py`, whose example moves into its mirrored test.
- [ ] 6.6 Engineering, including the plants of `_koopman.py`.
- [ ] 6.7 Mathematics.
- [ ] 6.8 Physics, including the LAMMPS tables, `_lattice_gases.py`, `_yang_mills.py`, and the unexported
  `corotational`, `upper_convected`, `spin_space`, `site_space`, and `velocity_of`, whose examples move into their
  mirrored tests.
- [ ] 6.9 Social.

## 7. Rules and integration

- [ ] 7.1 Replace the tuple of checked packages in `tests/test_sources.py` with the whole of `src/skcomplex`. Verify
  the three source tests pass.
- [ ] 7.2 State in `openspec/constitution.md` the block layout, derived values, the owner rule, and the three kinds of
  module functions that stay, and point to them from the module layout section of `CONTRIBUTING.md`. Verify the
  wording has no semicolon and no hard-coded count.
- [ ] 7.3 Compare the union of every `__all__` with the one recorded in task 1.2. Verify they are equal.
- [ ] 7.4 Run the full gate: `nox -s checks`, `nox -s tests`, and `nox -s docs`. Verify all pass and the test count is
  unchanged apart from the moved examples and the new source tests.

## Workflow follow-up

- Archive the change once the user has reviewed it and the gate passes.
