# Tasks

## 1. Concepts of the shared core

- [x] 1.1 Replace the closed enums `LawKind`, `Form`, `SpaceKind`, and the hand-kept trait tuples with open registries
  of named values in `base`, keeping the current member names as constants. Verify that mypy is clean and that the
  full suite passes unchanged.
- [ ] 1.2 Add the `Law` base in `laws/_law.py`, which imports only `base`. Make every law class an explicit frozen
  dataclass that derives from it and declares its `kind`, `time_class`, and `traits`. Make `Jump` reject options of
  the other clock. Verify with the "Every law shares the base" and "Option of the other clock" scenarios.
- [ ] 1.3 Add `values_of(law, *, include=...)` to `laws`. Replace the four law-value walkers in `compile` with it.
  Verify that the compile and validation tests pass.
- [ ] 1.4 Rewrite `Component` in `components/_component.py`, which imports only `base`. It has the explicit
  constructor, the class-level `declarations()` with derived order and lag companions, and `get_params`, and it
  drops `__init_subclass__`. Make `Element`, `Group`, `Interaction`, and `System` derive from it. Verify with the
  "Hierarchy", "No generated attributes", "Clone keeps declared values", and "System is a component" scenarios, and
  with the existing tests for order, lags, and `define`.
- [ ] 1.5 Make group creation check the size, the value names, and the shapes at once. Verify with the "Wrong number
  of values" and "Unknown value name" scenarios.

## 2. Formalism registration and the first moves

- [ ] 2.1 Add `formalisms/_formalism.py`, with `Formalism`, `register`, `registered_formalisms`, and the dependency
  check. Add a test formalism that registers a new law kind, a pass, and a solver. Verify with the "Registered
  contributions are listed", "Unregistered law", and "Law from an outside formalism" scenarios.
- [ ] 2.2 Add the import codemod script `tools/rewrite_imports.py`, which reads a table of old and new paths, and its
  test. Each move below rewrites models, examples, tests, and documentation with it. Verify on a planted fixture.
- [ ] 2.3 Move `topology` and `renewals` to `formalisms`, with their tests mirrored. Verify with the topology and
  renewal tests and the catalog conformance tests.
- [ ] 2.4 Move `groups`, `control`, and `fractional`. Verify with their tests and the catalog conformance tests.
- [ ] 2.5 Move `flows`, `kinetics`, and `particles`, with LAMMPS moving into `particles`. Verify with their tests and
  the catalog conformance tests.
- [ ] 2.6 Move `quantum` and `rewriting`. Verify with their tests and the catalog conformance tests.
- [ ] 2.7 Move `lattices`, then `fields` with its declared dependency on `lattices`. Verify with their tests, the
  "Declared dependency" scenario, and the catalog conformance tests.
- [ ] 2.8 Move `agents`, with Mesa moving into `agents`. Verify with the agent tests, the "Formalism imports",
  "Old paths are gone", "Formalism primitives by formalism", and "Backends in one package" scenarios, and the catalog
  conformance tests.

## 3. Compile pipeline

- [ ] 3.1 Add the `Compilation` state, the `Pass` contract, and the ordering of passes by their declared reads and
  writes. Verify with unit tests of the ordering and of a missing input.
- [ ] 3.2 Split `_compiler.py` into the general passes under `compile/passes/`, and register the formalism passes
  from their formalisms. Remove `_compiler.py`. Verify that every registered model compiles to the same problem as
  before, with the same equations and layout.
- [ ] 3.3 Add `plan_partitions` and use it in both validation and compilation. Make `compile_system` continue from the
  state that `validate` returns, so each compile flattens the system once. Verify with the "Validation and
  compilation agree" scenario and a test that counts flattenings.
- [ ] 3.4 Report backend substitutions and Numba fallbacks with `BackendWarning`, and record the backend of each part
  on the problem. Report invalid lattices as `space-invalid` issues. Verify with the "Backend substituted for a form",
  "Kernel falls back to Python", "Lattice with a bad boundary", and "Invalid space among other errors" scenarios.

## 4. Solvers and backends

- [ ] 4.1 Split `_steppers.py` into `_stepper.py`, `_ode.py`, `_sde.py`, `_maps.py`, `_jumps.py`, and `_hybrid.py`.
  Add `step_count`, and use it in the steppers, the Numba loop generator, and the LAMMPS backend. Make the inline
  loops call the shared step functions. Verify with the "Same seed on both backends" scenario and a parity test of
  every fixed-step stepper on both backends.
- [ ] 4.2 Merge `_numba_cells` into `_numba_neighbours`, so pair kernels use the bounded grid. Delete the unused
  `neighbor_cell`. Verify with the pair-kernel tests and a test with two particles 1 000 cutoffs apart in three
  dimensions.

## 5. Module order, hoisting, and names

- [ ] 5.1 Extend the layout test with the checks of the package-layout and formalism-plugins specs:
  - the order of definitions;
  - the imports of base-class modules;
  - imports guarded by type checks;
  - imports between packages and formalisms;
  - leaking exports.

  Verify with the "Upward import", "Helper between classes", "Base class with wide imports", "Guarded import",
  "Planted import", "Undeclared import", and "Compiler internal leaks" scenarios.
- [ ] 5.2 Reorder and hoist every core module until the layout test passes. Move each helper to the concept that owns
  it, and drop the internals from `__all__`. Verify with the layout test and the full suite.
- [ ] 5.3 Unify the public spelling to `neighbors`, `Neighborhood`, and `over='neighbors'`. Fold `neighbor_sum` and
  `adjacent` into `neighbors(..., reduce='sum')`, rename `reduce=None` on lattice neighbourhoods to `reduce='each'`,
  and rename the `neighborhood=` keyword to `over=`, with the codemod applied. Verify with the reduction tests and the
  catalog conformance tests.

## 6. Documentation and profile

- [ ] 6.1 Rewrite the user guide's sections on components, laws, and package layout, and add a section on formalisms
  and how to write one. Regenerate the compatibility page. Verify that the documentation test runs every Python block.
- [ ] 6.2 Amend the constitution's Project Profile:
  - the layout with `formalisms`;
  - the ecosystem contract without generated dataclass constructors;
  - the extension points.

  Bump its minor version. Verify by review against this design.

## 7. Gate

- [ ] 7.1 Run formatting, checks, the docs build, and the tests through nox. Compare every gallery example's
  recording with the cache from before the change, and explain each difference by a fixed defect. Verify that all of
  them pass.

## Workflow follow-up

- Update the designs and tasks of the 15 active changes to the new paths.
- Open the follow-up change for the performance findings: the complex check, kernel cache keys and eviction, compiled
  Heun and Milstein loops, and parallel neighbour searches.
- Open the follow-up change for the test-suite findings: slow markers, the avalanche test, and timing tests under the
  benchmark marker.
