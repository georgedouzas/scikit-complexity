# Tasks

## 1. Values

- [x] 1.1 Make `Term` a string of its value with `register`, drop its magic methods, and rewrite every `is` comparison of
  a term as `==` with a codemod kept in the scratchpad. Verify with the scenario "Terms by value".
- [x] 1.2 Give `BaseLaw`, `BaseDomain`, and `BaseProblem` their shared fields and `check()`, move every
  `__post_init__` check into `_check` or the builder of the record, and make validation call `check()` on every law.
  Verify with the scenario "Invalid law".

## 2. Verbs

- [x] 2.1 Add `check`, `prepare`, and `run` with their trivial hooks to every configurable base, and make every
  `__init__` store its arguments only, moving its checks to `_check` and its derived values to `_prepare`. Verify with the
  scenarios "Invalid argument" and "Facts are prepared".
- [x] 2.2 Move every family to the verbs with a codemod kept in the scratchpad:
  - spaces, removing `SiteSpace`;
  - times;
  - solvers;
  - backends, system compilers, and passes;
  - visuals, observers, and clocks;
  - designs, with `product` and `concat`.

  Make child-only methods private or move them to their users. Verify with the scenarios "One vocabulary", "A new
  child through hooks", and "Designs combine by name".
- [x] 2.3 Make the runtime call `prepare` before the first frame, `run` at every frame, and `run(..., final=True)` at the
  last. Remove its `getattr` fallbacks, and move `engine/analysis/_quantum.py` to `formalisms/quantum`. Verify with
  the scenario "Quantum run through the base".
- [x] 2.4 Make the surrogates estimators with `prepare` and `run`, replacing `fit_dmd`, `fit_sindy`, and `fit_reservoir`.
  Verify with the scenario "Prepare and run".

## 3. Magic methods

- [x] 3.1 Replace every magic method except `__init__` and the quantum operator arithmetic:
  - `run` or `evaluate` for `__call__`;
  - parameter or field printing for `__repr__`;
  - named properties for `__str__`;
  - value records or explicit keys for `__eq__` and `__hash__`;
  - `values`, `size`, `entries`, `offsets`, and `frames` for the collection protocol, rewriting every
    `trajectory[...]` in the library, the tests, the models, and the documents;
  - plain attributes for the symbol namespace, and a namespace of declared names for probing laws.

  Verify with the scenarios "Magic methods", "Reading a trajectory", and "Unknown symbol".

## 4. Rules and gate

- [x] 4.1 Amend the constitution with the three verbs, trivial constructors and prepared attributes, the split between
  values and configurable objects, and the allowed magic methods. Update the user guide, the README, and the gallery
  for the new verbs.
- [ ] 4.2 Run formatting, checks, the docs build, and tests through nox once, and verify that all of them pass.
