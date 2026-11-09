# Tasks

Every group leaves the tree green. Run the tests with `NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`
when other agents run tests at the same time.

## 1. Noise law with a Hurst index

- [x] 1.1 Add `NoiseKind.FRACTIONAL` to `core/types.py` and widen its docstring from Wiener processes to the noise
  processes of a system, and verify `pytest --doctest-modules src/skcomplex/core/types.py` passes
- [x] 1.2 Add `hurst=None` to `NoiseLaw` after `conserved`, move the range check and the docstring of
  `FractionalNoiseLaw` into it, and add the checks that reject a Hurst index with a process or on a conserved noise,
  and verify new tests in
  `tests/core/components/laws/test_noise.py` cover the range, the process, and the conserved checks
- [x] 1.3 In `CheckFractionalNoisePass`, `FractionalProblemPass`, and `PassCompiler._noise_needs`, test
  `law.hurst is not None` in place of the class, record `NoiseKind.FRACTIONAL` for such a law, and give the fix text of
  that kind the name of `FractionalEulerMaruyamaSolver`, and verify
  `pytest tests/core/engine/compile/passes/test_fractional_form.py` passes after its laws become `NoiseLaw(..., hurst=)`
- [x] 1.4 Set the noise support of `FractionalEulerMaruyamaSolver` to `{DIAGONAL, FRACTIONAL}` and remove
  `FractionalNoiseLaw` from its laws, set `EulerMaruyamaSolver`, `HeunSolver`, and `CoSimulationSolver` to every kind
  but `FRACTIONAL`, and remove `FractionalNoiseLaw` from the numpy and numba backends, and verify
  `pytest tests/core/engine/solvers/test_fractional_euler_maruyama.py` passes, including the wrong stepper error
- [x] 1.5 Emit `NoiseLaw(..., hurst=)` in `interactions/_rate.py` and `_relaxation.py` (if it still exists), drop
  `FractionalNoiseLaw` from their `law_classes` and from the docstring of `specs/_interaction_requirement_spec.py`, and
  verify `pytest tests/core/components/interactions` passes
- [x] 1.6 Update `anomalous_processes.md` and `kinds_of_dynamics.md` in `docs/overview/user_guide/build_a_custom_system`
  to write `NoiseLaw(..., hurst=)`, and verify their markdown doctests pass
- [x] 1.7 Delete `laws/_fractional_noise.py` and its re-export, and verify `grep -rn FractionalNoiseLaw src tests docs`
  finds nothing and `pytest tests/core` passes

## 2. Transition law with a source

- [x] 2.1 Add `source=None` to `TransitionLaw`, and move the symbolic-matrix check, the docstring, and the example of
  `EmissionLaw` into it, and verify new tests in `tests/core/components/laws` cover a source with a rectangular matrix
  and the function-matrix error
- [x] 2.2 In `MarkovPass`, handle a `TransitionLaw` with a source as it handles an `EmissionLaw`, and verify
  `pytest tests/core/engine/compile` passes
- [x] 2.3 Emit `TransitionLaw(observed, matrix, source=value)` in `interactions/_transition.py`, drop `EmissionLaw` from
  its `law_classes` and variant tags, and verify `pytest tests/core/components/interactions/test_variants.py` and the
  hidden Markov systems of `tests/systems/mathematics/stochastic_processes` pass
- [x] 2.4 Update `stochastic_maps.md` of the user guide and `tests/core/engine/compile/test_draw_lowering.py`, and
  verify the markdown doctests and the test pass
- [x] 2.5 Remove `EmissionLaw` from `MarkovPass`, `solvers/_maps.py`, `_cosimulation.py`, the numpy and numba
  backends, and `experiments/_replicate.py`, delete `laws/_emission.py` and its re-export, and verify
  `grep -rn EmissionLaw src tests docs` finds nothing and `pytest tests/core` passes

## 3. Bilinear law with a conserved loss

- [ ] 3.1 Add `conserved=False` to `BilinearLaw`, allow `second=None`, raise for a conserved law without outcomes,
  and move the docstring of the gain and loss equation of `EncounterLaw` into it, and verify
  `tests/core/components/laws/test_bilinear.py` holds the law tests of `test_encounter.py` for conserved laws and
  passes
- [ ] 3.2 In `CompileEncountersPass` and `PassCompiler._unnormalized`, read the kind of a `BilinearLaw` from
  `law.conserved` and the linear case from `law.second is None`, and list `BilinearLaw` in `StrangSplittingSolver`,
  and verify new cases of `tests/core/engine/compile/passes/test_compile_encounters.py` for conserved bilinear laws
  match the encounter cases
- [ ] 3.3 Emit conserved `BilinearLaw`s in `interactions/_encounter.py`, update the law in
  `tests/core/components/elements/test_density.py`, `tests/core/components/laws/utils/test_moments.py`, and
  `tests/core/engine/solvers/test_kinetics.py`, and verify `pytest tests/core tests/systems` passes for the kinetic
  systems with the trajectories they had
- [ ] 3.4 Update `kinetic_systems_of_active_particles.md` and `kinds_of_dynamics.md`, and verify their markdown
  doctests pass
- [ ] 3.5 Remove `KineticLaw` from `engine/types.py`, `_KINETIC_LAWS` and the `EncounterLaw` branches from the compile
  passes and `compile/_laws.py`, and `EncounterLaw` from the supports of `KineticSolver`, `RK4Solver`,
  `ScipyODESolver`, `StrangSplittingSolver`, and the numpy and numba backends, delete `laws/_encounter.py`, its
  re-export, and `tests/core/components/laws/test_encounter.py`, and verify `grep -rn EncounterLaw src tests docs`
  finds nothing and `pytest tests/core` passes

## 4. Array update law

- [ ] 4.1 Rename `DriveLaw` to `ArrayUpdateLaw` in `laws/_array_update.py`, with its docstring and doctest, and rename
  `tests/core/components/laws/test_drive.py` to `test_array_update.py`, and verify it passes
- [ ] 4.2 Rename every use in `laws/_base.py`, `interactions/_sampling.py`, `_cascade.py`, `compile/_forms.py`,
  `compile/_laws.py`, `compile/passes/_lattices.py`, `solvers/_lattice.py`, and the numpy and numba backends, and
  verify `pytest tests/core` passes
- [ ] 4.3 Rename the uses in `physics/common/spins/_spin_sampling.py` and `biology/systems_biology/_fuzzy_oil_drop.py`,
  and verify their system tests pass
- [ ] 4.4 Update `general_interactions.md`, `kinds_of_dynamics.md`, and `lattice_systems.md`, and verify their markdown
  doctests pass and `grep -rn DriveLaw src tests docs` finds nothing

## 5. Collision integral

- [ ] 5.1 Add `collision_integral(density, kernel)` to `laws/utils/_operators.py`, which calls the kernel with two
  private real symbols and returns the unevaluated call, re-export it, and verify a doctest and a test
  in `tests/core/components/laws/utils/test_operators.py` show the call of a constant kernel
- [ ] 5.2 In one step, remove `collision` and its re-export, remove `cross_section` from `RateLaw`, read the kernel
  from the call in `FieldLowering._collision`, raise for a kernel that reads a state or a parameter, rename the
  operator in `PHASE_OPERATOR_NAMES`, make `PassCompiler._collision_issue` look for the call, and emit
  `RateLaw(candidate, factor * collision_integral(candidate, self.cross_section))` in `EncounterInteraction._get_laws`,
  and verify `pytest tests/core/engine/compile tests/core/engine/problems/test_collision_kernel.py
  tests/core/components/interactions/test_encounter.py tests/systems/physics/fluids` passes, including the BKW
  scenario within 1e-4 and the conservation within 1e-12
- [ ] 5.3 Update `phase_space_fields.md`, and verify its markdown doctests pass and
  `grep -rnw "collision\|cross_section" src/skcomplex/core/components/laws` finds nothing

## 6. Series, distributed delay, and inverse Helmholtz

- [ ] 6.1 Rename the first argument of `series` to `x` and its `times` to `points`, move the docstring example of
  `lookup` into it, and fold `test_lookups.py` into `test_signals.py`, and verify
  `pytest tests/core/components/laws/utils/test_signals.py` passes
- [ ] 6.2 Replace `lookup` with `series` in `social/organizations/_traps`, `economics/resources/_food`,
  `_plantations`, `_extraction`, `economics/markets/_commodities`, `physics/climate/_pollution`,
  `engineering/processes/_systems`, and `biology/ecology/_factories`, with each table as a pair of tuples, and verify
  the tests of those systems pass with unchanged trajectories
- [ ] 6.3 Replace `delay3` and `smooth` with `distributed_delay` in `engineering/processes/_systems`, and verify
  `pytest tests/systems/engineering` passes with unchanged trajectories
- [ ] 6.4 Make the translator write `series` for lookups and `distributed_delay` for delays and smoothings, and verify
  `pytest tests/core/engine/translators/test_system_dynamics.py` passes, including `DELAY3` within 1e-6 of PySD
- [ ] 6.5 Delete `laws/utils/_lookups.py`, `delay1`, `delay3`, and `smooth`, with their re-exports, move the delay
  docstrings into `distributed_delay`, and update `test_delays.py`, and verify
  `pytest tests/core/components/laws/utils` passes
- [ ] 6.6 Give `inverse_helmholtz` the default screening `0`, remove `inverse_laplacian`, use `inverse_helmholtz` in
  `leray_projection`, `physics/active_matter/_nematics`, `physics/fluids/_kinetic_equations`, and
  `physics/fluids/_flows`, and update the tests that name it, and verify `pytest tests/core/components/laws/utils
  tests/core/engine tests/systems/physics` passes
- [ ] 6.7 Update `stocks_and_flows.md`, `fields.md`, `general_interactions.md`, and `phase_space_fields.md`, and verify
  their markdown doctests pass and `grep -rnw "lookup\|delay1\|delay3\|smooth\|inverse_laplacian"` over the law
  utilities, the systems, and the user guide finds no call

## 7. Private helpers and draw functions

- [ ] 7.1 Move `uniform`, `normal`, `bernoulli`, `categorical`, `draw`, `draws_in`, `DRAWS`, and `_call` from
  `specs/_draw_spec.py` to `laws/utils/_draws.py`, keep `DrawSpec` and `RELAXED_KINDS` in specs, and point every import
  at the new module, and verify `pytest tests/core/components/laws/utils/test_draws.py tests/core/engine` passes
- [ ] 7.2 Drop `draws_in`, the `*_in` functions, `delay_parts`, `has_operator`, `distribute_operators`,
  `coordinates_of`, `COORDINATE_PREFIX`, `INVERSION_BOUNDARIES`, `MAX_NEIGHBOR_HYPEREDGE_SIZE`, and
  `NEIGHBOR_REDUCTIONS` from `laws/utils/__init__.py`, import them from their private modules in the laws, the
  interactions, the analysis, and the engine, and point the tests at the private modules, and verify `pytest tests/core`
  passes
- [ ] 7.3 Make `random_pairing` the private method `MatchLaw._get_random_pairs`, move its test from `test_agents.py` to
  `test_match.py`, and verify both pass
- [ ] 7.4 Delete the empty `laws/utils/_tables.py`, and verify `python -c "import skcomplex.core.components.laws.utils"`
  succeeds
- [ ] 7.5 Update the module docstring of `laws/utils/__init__.py` and the user guide pages that list the utilities, and
  verify `grep` finds no privatised name in `docs/overview`

## 8. Integration

- [ ] 8.1 Run `grep -rnw` over `src`, `tests`, `docs`, and `README.md` for every removed name, `FractionalNoiseLaw`,
  `EmissionLaw`, `EncounterLaw`, `DriveLaw`, `KineticLaw`, `lookup`, `delay1`, `delay3`, `smooth`, `inverse_laplacian`,
  `random_pairing`, and `cross_section` on a law, and verify it finds no use
- [ ] 8.2 Run `nox -s checks` for `src`, `tests`, and `docs`, with ruff, mypy, and the import contracts, and fix every
  finding without a suppression
- [ ] 8.3 Run `nox -s tests` with the doctests, and verify it passes
- [ ] 8.4 Run `nox -s docs` and `nox -s gallery`, and verify the strict build and the inventory check pass with no
  change to `docs/overview/inventory.toml`
- [ ] 8.5 Commit the work as conventional `refactor!` commits whose bodies name the removed names and their
  replacements, and verify `nox -s changelog` lists them as breaking changes

## Workflow follow-up

- Sync the delta specs and archive the change after review.
