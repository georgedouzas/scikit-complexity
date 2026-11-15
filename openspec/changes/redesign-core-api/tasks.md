# Tasks

The phases run in order. Within a phase, tasks of different groups may run in parallel, at most four at a time, with
the thread caps for parallel test runs. The new API is built beside the old one until phase 5 deletes the old one.

## 1. Bases and the system as an estimator

- [x] 1.1 Give `BaseSystem` in `src/skcomplex/core/components/systems/_base.py` the main public method `simulate`, next to `prepare`, taking the run arguments `until`, `dt`, `initial_state`, `random_state`, `observables`, `observers`, `invariants`, `schedule`, `clock`, `checked`, `record_draws`, and `replay`. The parts keep their current `prepare` and `run` for the engine until task 6.4. Verify with tests in `tests/core/components/systems/test_base.py` that `simulate` exists on systems and returns a trajectory.
- [x] 1.2 Add `System` in `src/skcomplex/core/components/systems` with label, elements, interactions, nested systems, space, solver, and backend, and nested parameters `<label>__<parameter>` for `get_params` and `set_params`. Verify with tests for tuning a part, an unknown nested name, nested systems, and `clone`.
- [x] 1.3 Implement `System.prepare`, which checks every part, infers the problem from the law kinds, chooses the solver and backend, and sets `problem_`, `solver_`, and `backend_`, and `System.simulate(until, initial_state=None, random_state=None, observables=(), observers=())`, which returns a `Trajectory`. Verify with tests that one preparation simulates twice, that `simulate` before `prepare` and after `set_params` raises `NotPreparedError`.
- [x] 1.4 Make `BaseSystem` subclasses set `system_` in the hook `_prepare`, which the public `prepare` compiles, and delegate `simulate` to it. Verify with a test of a ready-made system whose `set_params(beta=0.5)` reaches the run.
- [x] 1.5 Expose `states_`, `parameters_`, `equations_`, and `form_` on prepared systems, and make the compiled problem private as `_problem`. Verify with tests for a damped spring and a static system.

## 2. Elements, states, and specs

- [x] 2.1 Remove `rate` from `StateSpec` and the compiler, and the aliases `State` and `Parameter`, keeping `Role`. Verify that the orders pass and the specs carry no `rate` and that `skcomplex.core.specs` exports no `State` or `Parameter`.
- [x] 2.2 Give `Element` the arguments `values=`, `size=`, and `space=`, with reserved names and per-member arrays, so one class covers named values, values from data, many members, and one member per site, and infer the states of elements with many members. Verify with tests for 1000 members with positions under gravity, an element from data, a wrong number of values, one member per site, and a reserved name.
- [x] 2.3 Reject elements whose `_prepare` sets `laws_`, and remove `visual=` from element constructors, moving the drawing settings to the visualisation observer keyed by path. Verify with tests in `tests/core/components/elements` and `tests/core/engine/viz`.
- [x] 2.4 Empty the packages `skcomplex.core.specs` and `skcomplex.core.components.specs` of everything but `DeclaredElement`, `StateSpec`, `ParameterSpec`, `VariableSpec`, `DeclaredVariableSpec`, `VisualSpec`, `ConnectorSpec`, and the specs they name, which task 6.1 deletes with the packages: move `RoleSpec` and `Role` to `skcomplex.core.components.interactions`, `Tags` to `skcomplex.core.components`, `ObservableSpec` to `skcomplex.core.engine.runtime` as `Observable`, the noise values to `skcomplex.core.components.laws`, schedule values to the solvers that read them, field values such as `SectorSpec` to their field packages, and every engine record into a private module of the lowest package that uses it. Verify with the full test suite, the import contracts, and a source test that every public spec is a value a system, a test, or a document reads.
- [x] 2.5 Make elements plain classes: work out states and parameters from the laws in `prepare`, read the order of every state from the highest `derivative` or `lag` its laws use, name derivatives with primes and past values with indices in initial states and trajectories, read the kind of a value from its Python type hint or value, and read a constraint from the allowed values in `Annotated`. Verify with tests for a particle at rest, a falling particle with `"ball.x'"`, Sprott's jerk equation, a difference equation with two lags, an order mismatch, a categorical state, and a nonnegative value set negative.
- [x] 2.6 Give interactions every state as a Sage function of the time `t`, and translate `x.diff(t, k)`, `x(t=t - k)`, and `x(t=t - tau)` into the internal derivatives, lags, and delays when laws are normalised. Verify with tests that a falling particle written with `x.diff(t, 2)`, a damping that reads `x.diff(t)`, a difference equation with `y(t=t - 2)`, and a delay with `x(t=t - tau)` compile and run.

## 3. Interactions and laws

- [x] 3.1 Name the law kinds `RateLaw`, `NoiseLaw`, `JumpLaw`, `UpdateLaw`, `ConstraintLaw`, `EventLaw`, and `ObjectiveLaw`, renaming `EquationLaw` to `ConstraintLaw` and `CrossingLaw` to `EventLaw` everywhere. Verify with the full test suite and `rg` finding neither old name.
- [x] 3.2 Make an interaction prepare against the namespace of its bound elements through `_prepare`, setting `laws_` from symbols so parameters stay symbolic, with the engine preparing a clone per binding and routing every place that reads laws through one helper, while interactions that still define `_get_laws` keep working until phase 5 rewrites them. Verify with tests that a prepared interaction holds its laws, that a system keeps its parameters symbolic, and that the user's interaction has no `laws_` after a system binds it.
- [ ] 3.3 Add the operators `laplacian`, `divergence`, `gradient`, and the reductions over neighbours as public functions laws call on states written in Sage. Verify with tests for the heat equation on a mesh and a sum over network neighbours.
- [ ] 3.4 Write network rewiring as `UpdateLaw` laws on the edges of a network space, and circuits as node elements with constraint balances. Verify with tests for rewiring and for Kirchhoff's current law at a node of three resistors.
- [ ] 3.5 Make the problem of `prepare` follow the table of `design.md`: ODE, SDE, jump process, DAE, map, hybrid, optimisation, sampling, and gradient flow, with the default solver of each and an error naming the supported solvers for an unsuitable one. Verify with tests for each problem and for a map prepared with an ODE solver.
- [x] 3.6 Add `ObjectiveLaw`, a scalar function of states that a static system minimises, compiled into a static problem and solved by a `MinimizeSolver` built on `scipy.optimize.minimize`, with constraint laws as equality constraints. Verify with tests for an unconstrained minimum, a minimum on a constraint, and an objective of several elements.

## 4. Field-specific components leave the core

- [ ] 4.1 Merge `MeshSpace` into `CellComplexSpace` and `BallNeighborhood` into `RadiusNeighborhood`, moving the engine code that keys on the merged classes. Verify with the tests of both and of the systems that use them.
- [ ] 4.2 Move `DissipatorLaw` and the quantum generator to `skcomplex.systems.physics.common.quantum` with the solver that lowers them, built on the public engine bases. Verify with the moved tests under `tests/systems/physics/common/quantum`.
- [ ] 4.3 Move `CascadeLaw` to `skcomplex.systems.physics.common.criticality`, `ProductionLaw`, `SubstitutionLaw`, `WordElement`, and `PatchElement` to `skcomplex.systems.general.rewriting`, and `BilinearLaw` to `skcomplex.systems.general.kinetics`, each with its solver. Verify with the moved tests of each package.
- [ ] 4.4 Move `MatchLaw` to `skcomplex.systems.general.matching` and `GradientStepLaw` to the learning package, both as `UpdateLaw` recipes, and the finite-volume scheme of `FluxLaw` to the fluids package as a solver setting. Verify with the moved tests.
- [ ] 4.5 Move `HamiltonianInteraction` and `EulerLagrangeInteraction`, with the energy parts of the invariants analysis, to `skcomplex.systems.physics.common.mechanics`, `GradientInteraction` to `skcomplex.systems.general.gradients`, `ExtremalLaw` to the evolution package of biology, `SimplexSpace` to `skcomplex.systems.social.common`, and `CayleySpace` with `MemoryNeighborhood` and `StateTranslationNeighborhood` and their engine support to `skcomplex.systems.computer_science.common.automata`. Verify with the moved tests and a source test that no module of the core names a moved class.

## 5. Systems migrate field by field

Each task rewrites the systems of its fields on the new API: elements as data with declared states, interactions whose `_prepare` sets `laws_` and that own their relation parameters, the seven law kinds, `_prepare` setting `system_` instead of `_build_system`, and `prepare` and `simulate` in tests, docstrings, and gallery examples. Each task verifies with the tests of its fields, its doctests, and a gallery sample of its fields.

- [ ] 5.1 Wave 1: `physics`.
- [ ] 5.2 Wave 1: `chemistry`, `engineering`, and `general`.
- [ ] 5.3 Wave 1: `mathematics`.
- [ ] 5.4 Wave 1: `computer_science`.
- [ ] 5.5 Wave 2: `biology`.
- [ ] 5.6 Wave 2: `economics`.
- [ ] 5.7 Wave 2: `social`.
- [ ] 5.8 Rewrite the experiment functions in `src/skcomplex/core/experiments` to clone, set parameters, prepare, and simulate, preparing once when only initial or random states vary. Verify with tests that an ensemble prepares once and a sweep leaves the user's system unchanged.

## 6. The old API goes

- [ ] 6.1 Delete the free function `simulate`, `CompositeSystem`, `GroupElement`, `FieldElement`, the times, the wrapper and binding interactions, the old law classes, `order` and the companion states, `divisor`, ports and `ConnectorSpec`, the user-facing `compiler` setting, `StateSpec`, `ParameterSpec`, `VariableSpec`, `DeclaredVariableSpec`, the specs they name, the emptied packages `skcomplex.core.specs` and `skcomplex.core.components.specs`, `order` and every other state flag, the public `derivative`, `lag`, and `delayed` of `laws.utils`, and the engine keys on the old law classes. Verify with the full test suite and with `rg` finding none of these names in `src`, `tests`, and `docs`.
- [ ] 6.2 Update `openspec/scripts/check_conventions.py`, the import-linter contracts in `pyproject.toml`, and the source tests to the new placement rule and the new verbs. Verify with `pdm checks conventions` and `lint-imports`.
- [ ] 6.3 Simplify the code that remains: no hidden conventions, no argument lists longer than the concept needs, no private hook a user implements. Verify with ruff, mypy, and a review of the constructors of every public class of the core.
- [ ] 6.4 Split the bases in `src/skcomplex/core/base.py` and `src/skcomplex/core/components/base.py`: `BaseComponent` gives label, parameters, cloning, tags, and `prepare` with no `run`, the fallback to `_get_laws` goes from every component, and `BaseConfig` keeps `prepare` and `run` for engine classes only. Verify with tests in `tests/core/test_base.py` and `tests/core/components/test_base.py` that list the methods of each base.

## 7. Documentation

- [ ] 7.1 Rewrite every docstring of the core in the scikit-learn layout of `design.md`, with no jargon: say explicitly what a thing does in plain words and show it with a short example, such as `Element('ball', values={'x': 10.0, 'm': 1.0})` in place of "an element known only as data". Apply the same rule to the guide, the README, error messages, and replies, and replace the test of the "Unlike" template by tests of the layout and of short examples. Verify with the docstring tests and the doctests.
- [ ] 7.2 Rewrite the README, the user guide, and the case studies on the new API, with the falling particle, the reaction network with two solvers, the heat equation, and the Ising model as the running examples. Verify with the documents test group and a strict documentation build.
- [ ] 7.3 Restate the placement rule in `openspec/constitution.md`, `CONTRIBUTING.md`, and the extension page of the user guide, and rewrite the scenarios of the main specs that call `simulate(` in the method form. Verify that the three texts state the same rule and that `rg "simulate\(" openspec/specs` finds only method calls.

## 8. Checks

- [ ] 8.1 Run the full nox sessions for quality, types, conventions, tests, and the gallery sample. Verify that all pass, and report the mypy errors that predate this change separately.

## Workflow follow-up

- Rebase the parked change `add-component-finder` onto the components this change keeps, then apply it.
- Archive this change after phase 8, and release it as a breaking release with `pdm release`.
