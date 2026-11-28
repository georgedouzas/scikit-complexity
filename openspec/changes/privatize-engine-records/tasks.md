## 1. Contract

- [x] 1.1 Audit which engine records a solver or backend written outside the library must read, and record the final
      public set in design.md
- [x] 1.2 Amend the constitution rule on dataclasses and raise its version

## 2. Problems package

- [ ] 2.1 Keep one module per problem kind in `core/engine/problems` with its private records, and move every other
      record to the module that builds or reads it
- [ ] 2.2 Turn lists of records of one kind that readers loop over into arrays or structured arrays
- [ ] 2.3 Replace the trivial records of the package with arguments or tuples

## 3. Compile, solvers, backends and translators

- [x] 3.1 Make the records of `compile` private, move them to their users, and remove the trivial ones
- [x] 3.2 Do the same for `solvers`, `backends` and `translators`

## 4. Records outside the engine

- [x] 4.1 Do the same for the private records of `core/analysis`, `core/components`, `core/experiments` and
      `systems/general/rewriting`
- [x] 4.2 Move any check of a user value that a record held to the `_check` of the public owner, with its message

## 5. Close

- [x] 5.1 Add the tests of the engine's public names and of trivial records, in the tests that mirror the packages
- [ ] 5.2 Run black, ruff, mypy, pydoclint, interrogate, lint-imports and the convention checks, and the test groups
      of every touched package
- [ ] 5.3 Run the full suite at the end of the change
