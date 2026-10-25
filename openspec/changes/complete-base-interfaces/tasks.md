# Tasks

## 1. Values

- [ ] 1.1 Make `Term` a frozen dataclass with `register`, drop its magic methods, and rewrite every `is` comparison of
  a term as `==` with a codemod kept in the scratchpad. Verify with the scenario "Terms by value".
- [ ] 1.2 Move the checks of laws and records from `__post_init__` into `_check` hooks behind `BaseLaw.check()` and
  the builders of the records, and make validation call `check()` on every law. Verify with the scenario "Invalid
  law".

## 2. Family APIs

- [ ] 2.1 Spaces: give `BaseSpace` the API and hooks of the design, remove `SiteSpace`, make the child-only methods
  private, and give continuous spaces and meshes their trivial or real answers. Verify with the scenarios "Same
  questions for every space" and "A new child through hooks".
- [ ] 2.2 Problems: give `BaseProblem` the API of the design, remove the `getattr` fallbacks of the runtime, the
  visuals, and the analyses, and move `engine/analysis/_quantum.py` to `formalisms/quantum`. Verify with the scenario
  "Quantum run through the base".
- [ ] 2.3 Laws, domains, components, interactions, solvers, visuals, observers, and designs: add the APIs and hooks
  of the design, and make child-only methods private or move them to their users. Verify with the scenarios
  "Children add no public methods" and "Designs combine by name".
- [ ] 2.4 Surrogates: make `BaseSurrogate` an estimator with `fit` and `forecast`, and replace `fit_dmd`,
  `fit_sindy`, and `fit_reservoir` with `DMDSurrogate`, `SINDySurrogate`, and `ReservoirSurrogate`. Verify with the
  scenario "Fit and forecast".

## 3. Magic methods

- [ ] 3.1 Replace every remaining magic method outside the allowed four: named methods for `__call__`, parameter
  printing for `__repr__`, named properties for `__str__`, and value records or explicit keys for `__eq__` and
  `__hash__`. Verify with the scenario "Magic methods".

## 4. Rules and gate

- [ ] 4.1 Amend the constitution with the template-method rule, the allowed magic methods, and the split between
  value records and estimators, and update the user guide and the README for the new names.
- [ ] 4.2 Run formatting, checks, the docs build, and tests through nox once, and verify that all of them pass.
