# Tasks

## 1. Core base

- [ ] 1.1 Split `core/base` into the modules of the design, merging `_terms.py`, `_support.py`, and `_forms.py`
  into `_kinds.py`, and `_domains.py` into `_domain.py`. Add `BaseDomain` with its kind, derive every domain from
  it, remove `AnyDomain`, and make validation read `domain.kind`. Verify with the scenario "Formalism domain".
- [ ] 1.2 Move the declaration code of `Component` into helpers of `core/base/_variables.py`. Add `BaseComponent`,
  `BaseInteraction`, and `BaseSystem`, and derive `Element`, `Group`, `Interaction`, and `System` from them. Rewrite
  every `isinstance` check against the base it means. Verify with the scenarios "System and interaction" and
  "Derived classes".
- [ ] 1.3 Move `Law` with its registry helpers, `Space`, and `Time` into `core/base` as `BaseLaw`, `BaseSpace`, and
  `BaseTime`. Rename every reference in `src`, `tests`, and `docs` with a codemod kept in the scratchpad, and move
  the tests with their modules. Verify with the scenarios "Base classes" and "Old base names are gone".
- [ ] 1.4 Move `HilbertSpace`, `HilbertFamily`, and `HilbertSpaceError` to `formalisms/quantum`. Move `Structure`,
  `RewritingError`, and `WordProblemError` to `formalisms/rewriting`, and `HorizonError` to `formalisms/control`.
  Move their tests with them. Verify with the scenario "Formalism types".

## 2. Problem, solver, and backend bases

- [ ] 2.1 Move `Layout` to `core/base/_layout.py`. Add `BaseProblem`, derive `engine.compile.Problem` from it, and
  derive the kinetic, quantum, and rewriting problems from `BaseProblem` with only the fields they use.
- [ ] 2.2 Add the generic `BaseSolver`, and rename `Stepper` to `Solver(BaseSolver[Problem])`. Move
  `ExternalBackend` to `engine/solvers` as `ExternalSolver`, and rename `LammpsBackend` to `LammpsSolver`. Verify
  with the scenario "Solver over a base problem".
- [ ] 2.3 Add `BaseBackend` with the `KernelSource` and `Lowering` protocols, and derive `NumpyBackend` and
  `NumbaBackend` from it. Derive `SystemPass` from `Pass`. Verify with the scenario "Unrelated bases".

## 3. Rules

- [ ] 3.1 Amend the constitution to 1.9.0. The project profile names the base classes, their parents, and what
  `core.base` holds, and the list of base classes users subclass reads `BaseLaw`, `BaseInteraction`, `BaseSolver`,
  and `BaseBackend`. Change the base-module rule so that it says a base package imports no other package of
  `skcomplex`.
- [ ] 3.2 Update the inventory, the user guide, and the README for the new names, and update the import contracts if
  a moved name needs it.

## 4. Gate

- [ ] 4.1 Run formatting, checks, the docs build, and tests through nox once, and verify that all of them pass.
