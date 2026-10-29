## 1. Structure

- [x] 1.1 Make `BaseConfig` generic in its context and input. Type `_get_laws` on `BasePart` as taking the namespace and returning a sequence of parts. Remove the `TYPE_CHECKING` blocks of the core.
- [x] 1.2 Move every spec that names a class of a package into that package's `specs` package, and rewrite the imports. Remove every `TYPE_CHECKING` block and `exclude_type_checking_imports`. Verify with lint-imports and the full test session.

## 2. Formatting

- [x] 2.1 Run black on `src`, `tests`, and `docs`, and verify that `black --check` passes.

## 3. Findings

- [x] 3.1 Fix every mypy error, pydoclint finding, and suppression comment in `skcomplex.models`, split by field across agents, with a typed view class for every agent group.
- [x] 3.2 Do the same for `skcomplex.engine`.
- [x] 3.3 Do the same for `skcomplex.core`, `analysis`, `experiments`, the root modules, the tests, and the docs.
- [x] 3.4 Add the tests that no source holds `TYPE_CHECKING`, `noqa`, `type: ignore`, or `nosec`. Verify that the checks session, the full test session, and the user guide pass.
