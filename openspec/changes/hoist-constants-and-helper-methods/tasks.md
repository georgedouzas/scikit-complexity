# Tasks

The tasks follow the user's decisions recorded in design.md. Scripts live in the session scratchpad, never in the
repository. Every group ends with `ruff format`, `ruff check`, `mypy`, `lint-imports`, the mirrored tests, and the
doctests of its package passing, and one commit on `development`. When several agents run tests at once, each sets
`NUMBA_NUM_THREADS`, `OMP_NUM_THREADS`, and the BLAS thread counts to 1.

A group that moves helpers also gives every method of its package a verb name, as design.md decision 6 says. The
names script proposes the verbs, a person reviews the list, and the script rewrites the definitions and every call
site in the whole tree.

## 1. Measure and add the checks

- [x] 1.1 Confirm that `add-system-package`, `slim-core-elements-and-spaces`, `slim-core-interactions`, and
  `slim-core-laws` are applied, then run the scratchpad scan over `src`, `tests`, and `docs` and record per package
  the private module functions by kind of reader, the private and misplaced constants, the gaps, the unexported public
  functions, the Sage factory calls, and the methods that begin with a noun. Verify the scan lists the cases named in
  design.md that still exist.
- [x] 1.2 Record the union of every `__all__` of `src/skcomplex` in the scratchpad, to compare at the end. Verify it
  imports cleanly.
- [x] 1.3 Add `test_module_constants_lead_their_module`, `test_constant_blocks_hold_no_gap`,
  `test_classes_own_the_functions_they_read`, `test_methods_begin_with_a_verb`, and
  `test_symbolic_functions_are_builtin_subclasses` to `tests/test_sources.py`, parametrised by an empty tuple of checked
  packages and folders, with the verb vocabulary built from the verbs the code uses. Verify each fails on a small source
  written to `tmp_path` that breaks its rule and passes on one that follows it, and names the file, line, and name.
- [x] 1.4 Write the constants, helpers, and names scripts of design.md decision 11. Verify each on one module by hand
  before the groups below use them.

## 2. Symbolic functions

- [x] 2.1 Rename `_Where`, `_Either`, `_Both`, `_Mod`, `_Derivative`, `_Lag`, `_Delayed`, `_Residence`, and
  `_StopGradient` to `_<Name>Function`. Verify the law utility tests pass.
- [x] 2.2 Convert the factory functions of `laws/utils` to `BuiltinFunction` subclasses: the reductions, with one
  subclass for the kinds that share `_print_reduction`, the neighbour, hyperedge, component, path, and shift functions
  with their hooks as `_print_` and `_eval_`, the delays with one subclass by order and kind, and the hook-less
  operators of `_operators.py`, `_cells.py`, `_integrals.py`, `_contractions.py`, `_phase.py`, and
  `_local_operators.py`, keeping each name, `nargs`, and `latex_name`. Verify each on numeric and symbolic arguments,
  and that the printer, compile, Einstein, and metric tests pass.
- [x] 2.3 Convert `MITTAG_LEFFLER` and the Caputo derivative of `analysis/_symbolic.py`, with `_evalf_` for the
  Mittag-Leffler hook, and the draw functions of `specs/_draw_spec.py`, with a `_derivative_` method that returns zero
  and one instance kept per label. Verify the symbolic analysis and draw tests pass and each draw label registers once.
- [x] 2.4 Add the symbolic function test's packages to the checked list. Verify it passes over `src`.

## 3. Core components

- [x] 3.1 In `core/components` (base, elements, spaces, interactions, laws, `laws/utils`, systems), hoist and rename
  constants, mark the derived values of `laws/utils` and of the sampling interaction, move every helper a class reads
  into its owner, and give the package's methods verb names. Keep the Numba kernels of the mobility and sampling
  interactions as module functions. Verify the source tests pass with `skcomplex.core.components` added to the checked
  packages.
- [x] 3.2 Fold the attribute docstrings of the package's constants into the docstrings that read them. Verify the docs
  build shows no missing text for the re-exported law utilities.

## 4. Core engine

- [ ] 4.1 In `engine/compile` and `engine/compile/passes`, rename the private constants, renaming `_MEMBERS` of
  `_layout.py` to `HIDDEN_MEMBERS`, and make the helpers of `compile/_complex.py` static methods with verb names of
  `LowerComplexPass` that `ObservableEvaluator` and `ReductionLowering` call through it. Move `probe_laws` and its
  namespace helper into `engine/_utils.py` and remove `compile/_probe.py`. Verify the compile tests and
  `lint-imports` pass.
- [ ] 4.2 In `engine/problems`, `engine/backends`, and `engine/solvers`, rename and hoist constants, keep every Numba
  kernel and `step_count` as module functions after the constants, and move the plain Python helpers into their
  classes. Mark `HELPERS_FINGERPRINT` as a derived value. Verify the backend and solver tests pass and a Numba run
  recompiles once and then hits the cache.
- [ ] 4.3 In `engine/runtime`, `engine/translators`, and `engine/viz`, rename and hoist constants and move the helpers
  that classes read. Keep the helpers of `simulate` that no class reads. Give the engine's methods verb names. Verify
  the runtime, translator, and viz tests pass with `skcomplex.core.engine` added to the checked packages.

## 5. Core analysis

- [ ] 5.1 Hoist the constants of every analysis module, including the late blocks of `_automata.py`, and fold their
  attribute docstrings into the module or function docstrings. Verify the two constants tests pass on the package.
- [ ] 5.2 Move every helper that a result class or another class of an analysis module reads into that class, with a
  verb name. Keep the helpers of the public analysis functions and of `_base.py` that no class reads. Verify the
  analysis tests pass with `skcomplex.core.analysis` added to the checked packages.

## 6. Core experiments, specs, and the root modules

- [ ] 6.1 In `core/experiments`, move the helpers that `_CalibrationObjective` and the result classes read into them,
  with verb names, and keep the run and design helpers that only the experiment functions read. Update the imports of
  `tests/core/experiments/test_calibration.py` to call the moved functions through their class. Verify the experiment
  tests, including the parallel ones, pass.
- [ ] 6.2 In `core/specs` and the root modules of `core`, give the methods verb names. `DialectSpec` and its dialects
  are gone with `move-recipes-to-systems`, so no derived value remains there. Verify the spec tests pass with
  `skcomplex.core.experiments`, `skcomplex.core.specs`, and the root modules added to the checked packages.

## 7. Systems

Each task applies the three scripts to one field, moves the helpers of system classes into their owners under the
rule of design.md decision 5 with verb names, marks the derived values that design.md names, and verifies the field's
tests and the catalog tests pass with the field added to the checked packages.

- [ ] 7.1 `systems/_catalog.py` and `_extras.py`: rename `_LISTED` and `_SYSTEM_CLASSES`, the latter as a derived
  value. Update `tests/systems/test_catalog.py` only where it imports a renamed name. Verify `all_systems` returns the
  same list as before.
- [ ] 7.2 Biology, including `_carbon_13_mfa.py` and `_logical_networks.py`, whose three constants drop the
  annotation that names a local alias.
- [ ] 7.3 Chemistry, including the shared diffusion helpers and `_noise` of `_reaction_diffusion.py`, which becomes
  `_get_noise`.
- [ ] 7.4 Computer science, including the landscapes of `_optimisers.py`.
- [ ] 7.5 Economics, including the predictors of `_el_farol.py`, the heuristics namespace, and `landscape` of
  `_sugarscape.py`, whose example moves into its mirrored test.
- [ ] 7.6 Engineering, including the plants of `_koopman.py`.
- [ ] 7.7 Mathematics.
- [ ] 7.8 Physics, including the LAMMPS tables, `_lattice_gases.py`, `_yang_mills.py`, and the unexported
  `corotational`, `upper_convected`, `spin_space`, `site_space`, and `velocity_of`, whose examples move into their
  mirrored tests.
- [ ] 7.9 Social.

## 8. Tests and documents

- [ ] 8.1 Apply the constants script to `tests`, moving the constants that sit below a definition and closing the gaps.
  Verify the two constants tests pass with `tests` added to the checked folders and the suite passes unchanged.
- [ ] 8.2 Apply the constants script to `docs`: the gallery examples, `gallery_tools`, `gallery_conf.py`,
  `gallery_hooks.py`, the generators, and `inventory_checks.py`, without `docs/generated`. Rename their private
  constants, put their type aliases after the constants, and check by hand each constant that reads the state of a
  statement, such as `SCRAPER` in `gallery_conf.py`. Verify the two constants tests pass with `docs` added and
  `nox -s docs` builds the gallery as before.

## 9. Rules and integration

- [ ] 9.1 Replace the tuple of checked packages and folders in `tests/test_sources.py` with every folder each test
  covers. Verify the five source tests pass.
- [ ] 9.2 State in `openspec/constitution.md` the block layout for every Python file, derived values, symbolic
  functions as `BuiltinFunction` subclasses, the owner rule, the verb vocabulary, and the three kinds of module
  functions that stay, and point to them from the module layout section of `CONTRIBUTING.md`. Verify the wording has
  no semicolon and no hard-coded count.
- [ ] 9.3 Compare the union of every `__all__` with the one recorded in task 1.2. Verify they are equal.
- [ ] 9.4 Run the full gate: `nox -s checks`, `nox -s tests`, and `nox -s docs`. Verify all pass and the test count is
  unchanged apart from the moved examples and the new source tests.

## Workflow follow-up

- Archive the change once the user has reviewed it and the gate passes.
