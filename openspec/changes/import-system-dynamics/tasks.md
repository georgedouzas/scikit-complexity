# Tasks

## 1. Dependency and fixtures

- [ ] 1.1 Add `pysd` to `[project] dependencies` with a compatible range, confirm which of its dependencies are new to
  the install, and record the libraries surveyed (PySD, and parsing Vensim and XMILE ourselves) in the change.
- [ ] 1.2 Add Vensim and XMILE fixtures under `tests/fixtures/system_dynamics`: teacup, SIR in both formats, a
  population with births and deaths, a `DELAY3` of a step, a population subscripted over three regions, and a file
  with `ALLOCATE BY PRIORITY` and `GET XLS DATA`. Store PySD's run of each runnable fixture as reference values.

## 2. Translation

- [ ] 2.1 Write `core/system/_sd_files.py` with `read_vensim` and `read_xmile`, which import PySD lazily, read the
  abstract model, map names to snake case, return the system and its settings, and shift `Time` by the start time.
  Verify with "Teacup runs like PySD" and "XMILE reads like Vensim".
- [ ] 2.2 Map stocks, constants, auxiliaries, and flows: split each stock's flow into signed terms, make transfers
  between stocks and with the environment, and gather the remaining terms in a rate law. Verify with "SIR flows are
  transfers" and "Loops of an imported model".
- [ ] 2.3 Map the built-in functions through one table of syntax structures: lookups, `IF THEN ELSE`, the input
  signals, delays and smoothing, the fixed delay, and the random functions. Verify with "Third-order delay".
- [ ] 2.4 Map variables subscripted over one range to groups and `SUM` to totals. Verify with "Regional populations".
- [ ] 2.5 Collect every untranslatable construct and raise one `ValidationError` that names each variable and
  construct. Verify with "Unsupported functions".

## 3. Documents and gate

- [ ] 3.1 Export the readers from `skcomplex.core.system`. Add a user-guide section on importing models and a gallery
  example that imports a published model, runs it, and plots the dominance of its loops.
- [ ] 3.2 Run formatting, checks, the docs build, and tests through nox once, and verify that all of them pass.
