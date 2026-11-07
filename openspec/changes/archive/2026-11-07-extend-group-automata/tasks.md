# Tasks

## 1. Core capabilities

- [x] 1.1 Add `StateTranslationNeighborhood` in `src/skcomplex/core/components/spaces/_state_translation.py`, the `_translation_table` hook on `BaseSpace`, and its `CayleySpace` implementation (verify: doctests and space tests pass)
- [x] 1.2 Read translated memory in `LocalRuleLaw.run` and in the compiled `RuleKernel`, with the compiler checks of schedules and degrees (verify: tests on finite groups and balls pass)
- [x] 1.3 Add `PowerSeriesSpec` in `src/skcomplex/core/specs/_power_series_spec.py`, its entry alphabet and layout shape (verify: doctests and spec tests pass)
- [x] 1.4 Let function local rules read and set states of several components (verify: kernel tests pass)
- [x] 1.5 Add core tests under `tests/core/` mirroring the changed modules (verify: pytest passes)

## 2. Catalog systems

- [x] 2.1 Add the self-indexed shift, power series, subtraction, group difference, and real quadratic automata (verify: tests pass)
- [x] 2.2 Add Muller's automaton, the surjective linear automaton, and Life in a tree on free groups (verify: tests pass)
- [x] 2.3 Add the nilpotent and involutive automata (verify: tests pass)
- [x] 2.4 Add the S-gap, context-free, and Morse subshifts and the adding machine (verify: tests pass)

## 3. Inventory, gallery, and documentation

- [x] 3.1 Add inventory entries citing the CC and EX sections (verify: `PYTHONPATH=docs python docs/inventory_checks.py` passes)
- [x] 3.2 Add a gallery example per system (verify: each runs and `python -m gallery_tools coverage --strict` passes)
- [x] 3.3 Document translated reads and power series states in the lattice systems guide (verify: the code block runs)

## 4. Checks

- [x] 4.1 Run ruff, black, docformatter, mypy, and pydoclint on the changed files, the catalog tests, and the doctests of the changed modules (verify: all pass)
