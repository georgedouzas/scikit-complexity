# Tasks

The phases run in order. Within a phase, tasks of different groups may run in parallel, at most four at a time, with
the thread caps for parallel test runs. The new API is built beside the old one until phase 5 deletes the old one.

## 1. Bases and the system as an estimator

- [ ] 1.1 Split the bases in `src/skcomplex/core/base.py` and `src/skcomplex/core/components/base.py`: `BaseComponent` gives label, parameters, cloning, and tags with no `prepare` or `run`, `BaseConfig` keeps `prepare` and `run` for engine classes, and `BaseSystem` defines `prepare` and `simulate`. Verify with tests in `tests/core/test_base.py` and `tests/core/components/test_base.py` that list the methods of each base.
- [ ] 1.2 Add `System` in `src/skcomplex/core/components/systems` with label, elements, interactions, nested systems, space, solver, and backend, and nested parameters `<label>__<parameter>` for `get_params` and `set_params`. Verify with tests for tuning a part, an unknown nested name, nested systems, and `clone`.
- [ ] 1.3 Implement `System.prepare`, which checks every part, infers the problem from the law kinds, chooses the solver and backend, and sets `problem_`, `solver_`, and `backend_`, and `System.simulate(until, initial_state=None, random_state=None, observables=(), observers=())`, which returns a `Trajectory`. Verify with tests that one preparation simulates twice, that `simulate` before `prepare` and after `set_params` raises `NotPreparedError`, and that preparing again after a change of values reuses the compiled problem.
- [ ] 1.4 Make `BaseSystem` subclasses call the public hook `build` from `prepare` and delegate `simulate` to the built system. Verify with a test of a ready-made system whose `set_params(beta=0.5)` reaches the run.

## 2. Elements, states, and specs

- [ ] 2.1 Reduce `StateSpec` to `domain`, `shape`, and `rate`, add the aliases `State`, `Parameter`, and `Role` in `src/skcomplex/core/specs`, and make the compiler treat `rate='v'` as `dx/dt = v`. Verify with tests in `tests/core/specs` for each alias and for a free particle that moves at constant velocity.
- [ ] 2.2 Give `Element` and `DeclaredElement` the arguments `size=` and `space=`, with reserved names and per-member arrays, and make `DeclaredElement` take `states={...}` and `parameters={...}`. Verify with tests for 1000 members with positions, a wrong number of values, one member per site, and a reserved name.
- [ ] 2.3 Reject elements that define `get_laws`, and remove `visual=` from element constructors, moving the drawing settings to the visualisation observer keyed by path. Verify with tests in `tests/core/components/elements` and `tests/core/engine/viz`.
- [ ] 2.4 Make the engine records and the eleven specs of `skcomplex.core.components.specs` private dataclasses beside the code that uses them, move field values such as `HilbertSpec` and `PartnerSpec` to their field packages, and keep public only the specs users build or read. Verify with a test that lists the public names of `skcomplex.core.specs` and finds about fifteen.

## 3. Interactions and laws

- [ ] 3.1 Add the seven law kinds `Rate`, `Noise`, `Jump`, `Update`, `Constraint`, `Event`, and `Objective` as frozen values in `src/skcomplex/core/components/laws`, each accepting symbols and callables, and lower each onto the existing compilers. Verify with one test per kind in `tests/core/components/laws`, including a `Jump` run as ODEs and as Gillespie draws.
- [ ] 3.2 Make `get_laws(s)` the public hook of `Interaction`, add `Interaction(label, roles={...}, laws=callable)`, and give the base the bindings of named roles, `among=` with `within=`, and `among=` with `along=`. Verify with tests for pairs within a cutoff, lattice neighbours, network edges, hyperedges on a lattice, and a state declared on an interaction.
- [ ] 3.3 Port `GradientInteraction`, `HamiltonianInteraction`, and `EulerLagrangeInteraction` to the new base and laws, and add the operators `laplacian`, `divergence`, `gradient`, and the reductions over neighbours as public functions. Verify with tests for a particle in a well, a pendulum from its Lagrangian, and the heat equation on a mesh.
- [ ] 3.4 Write network rewiring as `Update` laws on the edges of a network space, and circuits as node elements with constraint balances. Verify with tests for rewiring and for Kirchhoff's current law at a node of three resistors.
- [ ] 3.5 Make the problem of `prepare` follow the table of `design.md`: ODE, SDE, jump process, DAE, map, hybrid, optimisation, sampling, and gradient flow, with the default solver of each and an error naming the supported solvers for an unsuitable one. Verify with tests for each problem and for a map prepared with an ODE solver.

## 4. Field-specific components leave the core

- [ ] 4.1 Audit every space and neighbourhood of the core with the placement rule, starting with `CayleySpace`, `PhaseSpace`, `SimplexSpace`, `CofaceNeighborhood`, `MemoryNeighborhood`, and `StateTranslationNeighborhood`, and record the decision for each in this change before moving any. Verify that the record names the fields that use each one.
- [ ] 4.2 Move `DissipatorLaw` and the quantum generator to `skcomplex.systems.physics.common.quantum` with the solver or compiler that lowers them, built on the public engine bases. Verify with the moved tests under `tests/systems/physics/common/quantum`.
- [ ] 4.3 Move `CascadeLaw` and `ExtremalLaw` to the criticality package, `ProductionLaw`, `SubstitutionLaw`, `WordElement`, and `PatchElement` to the package of rewriting systems, and `BilinearLaw` to the package of kinetic models, each with its solver. Verify with the moved tests of each package.
- [ ] 4.4 Move `MatchLaw` to `skcomplex.systems.general` and `GradientStepLaw` to the learning package, both as `Update` recipes, and the finite-volume scheme of `FluxLaw` to the fluids package as a solver setting. Verify with the moved tests.
- [ ] 4.5 Move the spaces and neighbourhoods that the audit of 4.1 assigns to fields. Verify with the moved tests and a source test that no module of the core names a moved class.

## 5. Systems migrate field by field

Each task rewrites the systems of its fields on the new API: elements as data with declared states, interactions with `get_laws` and their relation parameters, the seven law kinds, `build` instead of `_build_system`, and `prepare` and `simulate` in tests, docstrings, and gallery examples. Each task verifies with the tests of its fields, its doctests, and a gallery sample of its fields.

- [ ] 5.1 Wave 1: `physics`.
- [ ] 5.2 Wave 1: `chemistry`, `engineering`, and `general`.
- [ ] 5.3 Wave 1: `mathematics`.
- [ ] 5.4 Wave 1: `computer_science`.
- [ ] 5.5 Wave 2: `biology`.
- [ ] 5.6 Wave 2: `economics`.
- [ ] 5.7 Wave 2: `social`.
- [ ] 5.8 Rewrite the experiment functions in `src/skcomplex/core/experiments` to clone, set parameters, prepare, and simulate, preparing once when only initial or random states vary. Verify with tests that an ensemble prepares once and a sweep leaves the user's system unchanged.

## 6. The old API goes

- [ ] 6.1 Delete the free function `simulate`, `CompositeSystem`, `GroupElement`, `FieldElement`, the times, the wrapper and binding interactions, the old law classes, `order` and the companion states, `divisor`, ports and `ConnectorSpec`, and the engine keys on the old law classes. Verify with the full test suite and with `rg` finding none of these names in `src`, `tests`, and `docs`.
- [ ] 6.2 Update `openspec/scripts/check_conventions.py`, the import-linter contracts in `pyproject.toml`, and the source tests to the new placement rule and the new verbs. Verify with `pdm checks conventions` and `lint-imports`.
- [ ] 6.3 Simplify the code that remains: no hidden conventions, no argument lists longer than the concept needs, no private hook a user implements. Verify with ruff, mypy, and a review of the constructors of every public class of the core.

## 7. Documentation

- [ ] 7.1 Rewrite every docstring of the core in the scikit-learn layout of `design.md`, and replace the test of the "Unlike" template by tests of the layout and of short examples. Verify with the docstring tests and the doctests.
- [ ] 7.2 Rewrite the README, the user guide, and the case studies on the new API, with the falling particle, the reaction network with two solvers, the heat equation, and the Ising model as the running examples. Verify with the documents test group and a strict documentation build.
- [ ] 7.3 Restate the placement rule in `openspec/constitution.md`, `CONTRIBUTING.md`, and the extension page of the user guide, and rewrite the scenarios of the main specs that call `simulate(` in the method form. Verify that the three texts state the same rule and that `rg "simulate\(" openspec/specs` finds only method calls.

## 8. Checks

- [ ] 8.1 Run the full nox sessions for quality, types, conventions, tests, and the gallery sample. Verify that all pass, and report the mypy errors that predate this change separately.

## Workflow follow-up

- Rebase the parked change `add-component-finder` onto the components this change keeps, then apply it.
- Archive this change after phase 8, and release it as a breaking release with `pdm release`.
