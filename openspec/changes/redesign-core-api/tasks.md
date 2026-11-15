# Tasks

The phases run in order. Within a phase, tasks of different groups may run in parallel, at most four at a time, with
the thread caps for parallel test runs. The new API is built beside the old one until phase 5 deletes the old one.

## 1. Bases and the system as an estimator

- [x] 1.1 Give `BaseSystem` in `src/skcomplex/core/components/systems/_base.py` the main public method `simulate`, next to `prepare`, taking the run arguments `until`, `dt`, `initial_state`, `random_state`, `observables`, `observers`, `invariants`, `schedule`, `clock`, `checked`, `record_draws`, and `replay`. The parts keep their current `prepare` and `run` for the engine until task 6.4. Verify with tests in `tests/core/components/systems/test_base.py` that `simulate` exists on systems and returns a trajectory.
- [x] 1.2 Add `System` in `src/skcomplex/core/components/systems` with label, elements, interactions, nested systems, space, solver, and backend, and nested parameters `<label>__<parameter>` for `get_params` and `set_params`. Verify with tests for tuning a part, an unknown nested name, nested systems, and `clone`.
- [x] 1.3 Implement `System.prepare`, which checks every part, infers the problem from the law kinds, chooses the solver and backend, and sets `problem_`, `solver_`, and `backend_`, and `System.simulate(until, initial_state=None, random_state=None, observables=(), observers=())`, which returns a `Trajectory`. Verify with tests that one preparation simulates twice, that `simulate` before `prepare` and after `set_params` raises `NotPreparedError`.
- [x] 1.4 Make `BaseSystem` subclasses set `system_` in the hook `_prepare`, which the public `prepare` compiles, and delegate `simulate` to it. Verify with a test of a ready-made system whose `set_params(beta=0.5)` reaches the run.

## 2. Elements, states, and specs

- [x] 2.1 Remove `rate` from `StateSpec` and the compiler, and the aliases `State` and `Parameter`, keeping `Role`. Verify that the orders pass and the specs carry no `rate` and that `skcomplex.core.specs` exports no `State` or `Parameter`.
- [ ] 2.2 Give `Element` and `DeclaredElement` the arguments `size=` and `space=`, with reserved names and per-member arrays, and make `DeclaredElement` take `values={...}`. Verify with tests for 1000 members with positions, a wrong number of values, one member per site, and a reserved name.
- [ ] 2.3 Reject elements that define `_get_laws`, and remove `visual=` from element constructors, moving the drawing settings to the visualisation observer keyed by path. Verify with tests in `tests/core/components/elements` and `tests/core/engine/viz`.
- [ ] 2.4 Make the engine records and the eleven specs of `skcomplex.core.components.specs` private dataclasses beside the code that uses them, move field values such as `HilbertSpec` and `PartnerSpec` to their field packages, and keep public only the specs users build or read. Verify with a test that lists the public names of `skcomplex.core.specs` and finds about fifteen.
- [ ] 2.5 Make elements plain classes: work out states and parameters from the laws in `prepare`, read the kind of a value from its Python type hint or value, and read a constraint from the allowed values in `Annotated`. Verify with tests for a particle at rest, a particle in motion, a categorical state, and a nonnegative value set negative.

## 3. Interactions and laws

- [ ] 3.1 Name the law kinds `RateLaw`, `NoiseLaw`, `JumpLaw`, `UpdateLaw`, `ConstraintLaw`, `EventLaw`, and `ObjectiveLaw`, renaming `EquationLaw` to `ConstraintLaw` and `CrossingLaw` to `EventLaw` everywhere. Verify with the full test suite and `rg` finding neither old name.
- [ ] 3.2 Give `BaseInteraction` the public method `get_laws(s)`, defined once, which checks the namespace and the laws and calls the abstract hook `_get_laws(s)` that every interaction implements, as `BaseSearchCV.fit` calls `_run_search` in scikit-learn, add `Interaction(label, roles={...}, laws=callable)`, and give the base the bindings of named roles, `among=` with `within=`, and `among=` with `along=`. Verify with tests for pairs within a cutoff, lattice neighbours, network edges, hyperedges on a lattice, and a state declared on an interaction.
- [ ] 3.3 Port `GradientInteraction`, `HamiltonianInteraction`, and `EulerLagrangeInteraction` to the new base and laws, and add the operators `laplacian`, `divergence`, `gradient`, and the reductions over neighbours as public functions. Verify with tests for a particle in a well, a pendulum from its Lagrangian, and the heat equation on a mesh.
- [ ] 3.4 Write network rewiring as `UpdateLaw` laws on the edges of a network space, and circuits as node elements with constraint balances. Verify with tests for rewiring and for Kirchhoff's current law at a node of three resistors.
- [ ] 3.5 Make the problem of `prepare` follow the table of `design.md`: ODE, SDE, jump process, DAE, map, hybrid, optimisation, sampling, and gradient flow, with the default solver of each and an error naming the supported solvers for an unsuitable one. Verify with tests for each problem and for a map prepared with an ODE solver.
- [ ] 3.6 Add `ObjectiveLaw`, a scalar function of states that a static system minimises, compiled into a static problem and solved by a `MinimizeSolver` built on `scipy.optimize.minimize`, with constraint laws as equality constraints. Verify with tests for an unconstrained minimum, a minimum on a constraint, and an objective of several elements.

## 4. Field-specific components leave the core

- [ ] 4.1 Audit every space and neighbourhood of the core with the placement rule, starting with `CayleySpace`, `PhaseSpace`, `SimplexSpace`, `CofaceNeighborhood`, `MemoryNeighborhood`, and `StateTranslationNeighborhood`, and record the decision for each in this change before moving any. Verify that the record names the fields that use each one.
- [ ] 4.2 Move `DissipatorLaw` and the quantum generator to `skcomplex.systems.physics.common.quantum` with the solver that lowers them, built on the public engine bases. Verify with the moved tests under `tests/systems/physics/common/quantum`.
- [ ] 4.3 Move `CascadeLaw` and `ExtremalLaw` to the criticality package, `ProductionLaw`, `SubstitutionLaw`, `WordElement`, and `PatchElement` to the package of rewriting systems, and `BilinearLaw` to the package of kinetic models, each with its solver. Verify with the moved tests of each package.
- [ ] 4.4 Move `MatchLaw` to `skcomplex.systems.general` and `GradientStepLaw` to the learning package, both as `UpdateLaw` recipes, and the finite-volume scheme of `FluxLaw` to the fluids package as a solver setting. Verify with the moved tests.
- [ ] 4.5 Move the spaces and neighbourhoods that the audit of 4.1 assigns to fields. Verify with the moved tests and a source test that no module of the core names a moved class.
- [ ] 4.6 Add the `Motion` interaction, whose law is `RateLaw(x, v)`, to `skcomplex.systems.physics.common.particles`. Verify with a test of free fall against its exact solution.

## 5. Systems migrate field by field

Each task rewrites the systems of its fields on the new API: elements as data with declared states, interactions with `_get_laws` and their relation parameters, the seven law kinds, `_prepare` setting `system_` instead of `_build_system`, and `prepare` and `simulate` in tests, docstrings, and gallery examples. Each task verifies with the tests of its fields, its doctests, and a gallery sample of its fields.

- [ ] 5.1 Wave 1: `physics`.
- [ ] 5.2 Wave 1: `chemistry`, `engineering`, and `general`.
- [ ] 5.3 Wave 1: `mathematics`.
- [ ] 5.4 Wave 1: `computer_science`.
- [ ] 5.5 Wave 2: `biology`.
- [ ] 5.6 Wave 2: `economics`.
- [ ] 5.7 Wave 2: `social`.
- [ ] 5.8 Rewrite the experiment functions in `src/skcomplex/core/experiments` to clone, set parameters, prepare, and simulate, preparing once when only initial or random states vary. Verify with tests that an ensemble prepares once and a sweep leaves the user's system unchanged.

## 6. The old API goes

- [ ] 6.1 Delete the free function `simulate`, `CompositeSystem`, `GroupElement`, `FieldElement`, the times, the wrapper and binding interactions, the old law classes, `order` and the companion states, `divisor`, ports and `ConnectorSpec`, the user-facing `compiler` setting, `StateSpec`, `ParameterSpec`, `VariableSpec`, `DeclaredVariableSpec`, `order` and every other state flag, and the engine keys on the old law classes. Verify with the full test suite and with `rg` finding none of these names in `src`, `tests`, and `docs`.
- [ ] 6.2 Update `openspec/scripts/check_conventions.py`, the import-linter contracts in `pyproject.toml`, and the source tests to the new placement rule and the new verbs. Verify with `pdm checks conventions` and `lint-imports`.
- [ ] 6.3 Simplify the code that remains: no hidden conventions, no argument lists longer than the concept needs, no private hook a user implements. Verify with ruff, mypy, and a review of the constructors of every public class of the core.
- [ ] 6.4 Split the bases in `src/skcomplex/core/base.py` and `src/skcomplex/core/components/base.py`: `BaseComponent` gives label, parameters, cloning, and tags with no `prepare` or `run`, and `BaseConfig` keeps `prepare` and `run` for engine classes only. Verify with tests in `tests/core/test_base.py` and `tests/core/components/test_base.py` that list the methods of each base.

## 7. Documentation

- [ ] 7.1 Rewrite every docstring of the core in the scikit-learn layout of `design.md`, and replace the test of the "Unlike" template by tests of the layout and of short examples. Verify with the docstring tests and the doctests.
- [ ] 7.2 Rewrite the README, the user guide, and the case studies on the new API, with the falling particle, the reaction network with two solvers, the heat equation, and the Ising model as the running examples. Verify with the documents test group and a strict documentation build.
- [ ] 7.3 Restate the placement rule in `openspec/constitution.md`, `CONTRIBUTING.md`, and the extension page of the user guide, and rewrite the scenarios of the main specs that call `simulate(` in the method form. Verify that the three texts state the same rule and that `rg "simulate\(" openspec/specs` finds only method calls.

## 8. Checks

- [ ] 8.1 Run the full nox sessions for quality, types, conventions, tests, and the gallery sample. Verify that all pass, and report the mypy errors that predate this change separately.

## Workflow follow-up

- Rebase the parked change `add-component-finder` onto the components this change keeps, then apply it.
- Archive this change after phase 8, and release it as a breaking release with `pdm release`.
