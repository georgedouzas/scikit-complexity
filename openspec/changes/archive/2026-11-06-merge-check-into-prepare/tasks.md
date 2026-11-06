# Tasks

## 1. Base classes

- [x] 1.1 Record the hashes of the seeded popularity-similarity networks and the compile time of `tests/systems/test_catalog.py` as the values to compare against; verify they repeat.
- [x] 1.2 Make `prepare` of `BaseConfig`, `BasePart`, and `BaseSpec` validate first, and remove `check`, `_is_checked`, `_checked`, and `NotCheckedError`; verify the scenarios of `base-interfaces` in `tests/core/test_base.py`, `tests/core/components/test_base.py`, and `tests/core/test_errors.py`.
- [x] 1.3 Replace `.check()` in `src` outside doctests with `._validated()` on the same object, including `ModelEntry` and the compiler's report; verify the core and engine tests.
- [x] 1.4 Compare the compile time with task 1.1, and move any validation that a hot path repeats; verify the time is within 5 percent.

## 2. Migration

- [x] 2.1 Apply the rewrites of the design to `src` doctests, `tests`, the guide, the README, and the gallery with a script kept out of the repository; verify no `.check()` or `check()` verb remains, reviewing any call on another library's object.
- [x] 2.2 Fix by hand each test that only asserted a valid object passes, giving `prepare` what it needs; verify the invalid setting and missing context scenarios.
- [x] 2.3 Rewrite the prose on verbs in the guide, the README, the docstrings of the bases, and `openspec/constitution.md`; verify `pdm checks docs` and the guide's blocks.
- [x] 2.4 Compare the seeded hashes with task 1.1; verify they are identical.

## 3. Gate

- [x] 3.1 Run formatting and all checks, and push to `development`; verify every CI job passes.
- [ ] 3.2 Run the full suite on Linux on a temporary branch and the full gallery build, since the rewrite invalidates its cache; verify both pass.
