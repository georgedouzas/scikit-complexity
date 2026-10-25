# Tasks

## 1. Core base

- [ ] 1.1 Move `Component`, `Law` with its registry helpers, and `Space` into `core/base` as `BaseComponent`,
  `BaseLaw`, and `BaseSpace`. Rename every reference in `src`, `tests`, and `docs` with a codemod kept in the
  scratchpad, and move their tests into `tests/core/base`. Verify with the scenarios "Base classes", "Old base names
  are gone", and "Derived classes".
- [ ] 1.2 Split `core/base` into the modules of the design: `_kinds.py` from `_terms.py` and `_support.py`, and
  `_domain.py` from `_domains.py`. Add `BaseDomain` with its kind, derive every domain from it, remove `AnyDomain`,
  and make validation read `domain.kind`. Verify with the scenario "Formalism domain".
- [ ] 1.3 Move `Form`, `Supports`, `OrderKind`, `NoiseKind`, `NUMERIC_DOMAINS`, and `vocabulary` to
  `engine/compile/_support.py`. Verify with the scenario "Engine vocabulary".
- [ ] 1.4 Move `HilbertSpace`, `HilbertFamily`, and `HilbertSpaceError` to `formalisms/quantum`. Move `Structure`,
  `RewritingError`, and `WordProblemError` to `formalisms/rewriting`, and `HorizonError` to `formalisms/control`.
  Move their tests with them. Verify with the scenario "Formalism types".

## 2. Engine bases

- [ ] 2.1 Rename `Stepper` to `BaseSolver` and move it to `engine/solvers/_base.py`. Rename `Backend` to
  `BaseBackend` in `engine/compile/_backends.py`. Verify with the scenarios "Solver and backend bases" and "Old
  engine base names are gone".

## 3. Rules

- [ ] 3.1 Amend the constitution to 1.9.0. The project profile names the base classes and what `core.base` holds, and
  the list of base classes users subclass reads `BaseLaw`, `BaseSolver`, and `BaseBackend`. Change the base-module
  rule so that it says a base package imports no other package of `skcomplex`.
- [ ] 3.2 Update the inventory, the user guide, and the README for the new names, and update the import contracts if
  a moved name needs it.

## 4. Gate

- [ ] 4.1 Run formatting, checks, the docs build, and tests through nox once, and verify that all of them pass.
