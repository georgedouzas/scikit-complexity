# Tasks

## 1. Core base

- [x] 1.1 Regroup `core/base` into the modules of the design. Add `BaseDomain` with its kind, derive the domains from
  it, remove `AnyDomain`, and make validation read `domain.domain_kind`. Move the Hilbert and structure domains, and the
  errors of one formalism, to their formalisms. Verify with the scenarios "Formalism domain" and "Formalism types".
- [x] 1.2 Move the declaration code of `Component` into helpers of `core/base/_variables.py`. Add `BaseComponent`
  and `BaseInteraction`, derive the element and group classes from `BaseComponent`, and remove `Interaction` and
  `core/interactions`. Detach `System`, and rewrite every `isinstance` check against the base it means. Verify with
  the scenario "System and interaction".
- [x] 1.3 Move `Law` with its registry helpers, `Space`, and `Time` into `core/base` as `BaseLaw`, `BaseSpace`, and
  `BaseTime`.

## 2. Problem, solver, and backend bases

- [ ] 2.1 Move `Layout` and `Entry` to `core/base/_problems.py`. Add `BaseProblem`, derive `engine.compile.Problem`
  from it, and derive the kinetic, quantum, and rewriting problems from `BaseProblem` with only the fields they use.
- [ ] 2.2 Add the generic `BaseSolver`, and turn `Stepper` into `Solver(BaseSolver[Problem])`. Move
  `ExternalBackend` to `engine/solvers`. Verify with the scenario "Solver over a base problem".
- [ ] 2.3 Add the generic `BaseBackend`, with `Lowering`, to `core/base/_solvers.py`. Rename the engine bases with
  the `Base` prefix, and derive `SystemPass` from `BasePass`. Verify with the scenarios "Unrelated bases" and
  "Derived classes".

## 3. Names

- [ ] 3.1 Build the rename map from the class graph of `src`, `tests`, and `docs`, including names given to
  `define`, and apply it by whole words with a codemod kept in the scratchpad. The codemod refuses colliding names.
  Move the tests of every moved or merged module to the mirror of its new path. Verify with the scenarios "Base
  classes", "Old base names are gone", "Framework names", and "Model names".

## 4. Rules

- [ ] 4.1 Amend the constitution to 1.9.0, with:
  - the naming rule;
  - the bases, their parents, and what `core.base` holds;
  - the layout without `core/interactions`;
  - the base classes users subclass;
  - a base-package rule that says a base package imports no other package of `skcomplex`.
- [ ] 4.2 Update the inventory, the user guide, the README, and the import contracts for the new names and the
  removed package.

## 5. Gate

- [ ] 5.1 Run formatting, checks, the docs build, and tests through nox once, and verify that all of them pass.
