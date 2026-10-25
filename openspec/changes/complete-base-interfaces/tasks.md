# Tasks

## 1. Values

- [ ] 1.1 Make `Term` a frozen dataclass with `register`, drop its magic methods, and rewrite every `is` comparison of
  a term as `==` with a codemod kept in the scratchpad. Verify with the scenario "Terms by value".
- [ ] 1.2 Give `BaseLaw`, `BaseDomain`, and `BaseProblem` their shared fields and `check()`, move every
  `__post_init__` check into `_check` or the builder of the record, and make validation call `check()` on every law.
  Verify with the scenario "Invalid law".

## 2. Verbs

- [ ] 2.1 Add `check`, `fit`, and `run` with their trivial hooks to every configurable base, and make every
  `__init__` store its arguments only, moving its checks to `_check` and its derived values to `_fit`. Verify with the
  scenarios "Invalid argument" and "Facts are fitted".
- [ ] 2.2 Move every family to the verbs with a codemod kept in the scratchpad:
  - spaces, removing `SiteSpace`;
  - times;
  - solvers;
  - backends, system compilers, and passes;
  - visuals, observers, and clocks;
  - designs, with `product` and `concat`.

  Make child-only methods private or move them to their users. Verify with the scenarios "One vocabulary", "A new
  child through hooks", and "Designs combine by name".
- [ ] 2.3 Make the runtime call `fit` before the first frame, `run` at every frame, and `run(..., final=True)` at the
  last. Remove its `getattr` fallbacks, and move `engine/analysis/_quantum.py` to `formalisms/quantum`. Verify with
  the scenario "Quantum run through the base".
- [ ] 2.4 Make the surrogates estimators with `fit` and `run`, replacing `fit_dmd`, `fit_sindy`, and `fit_reservoir`.
  Verify with the scenario "Fit and run".

## 3. Magic methods

- [ ] 3.1 Replace every remaining magic method outside the allowed four:
  - `run` or `evaluate` for `__call__`;
  - parameter or field printing for `__repr__`;
  - named properties for `__str__`;
  - value records or explicit keys for `__eq__` and `__hash__`.

  Verify with the scenario "Magic methods".

## 4. Rules and gate

- [ ] 4.1 Amend the constitution with the three verbs, trivial constructors and fitted attributes, the split between
  values and configurable objects, and the allowed magic methods. Update the user guide, the README, and the gallery
  for the new verbs.
- [ ] 4.2 Run formatting, checks, the docs build, and tests through nox once, and verify that all of them pass.
