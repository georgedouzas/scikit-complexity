## 1. Structure

- [ ] 1.1 Make `BaseConfig` generic in its context and input. Type `_get_laws` on `BasePart` as taking the namespace and returning a sequence of parts. Remove the `TYPE_CHECKING` blocks of the core.
- [ ] 1.2 Move every spec that names a class of a package into that package's `specs` package, and rewrite the imports. Remove every `TYPE_CHECKING` block and `exclude_type_checking_imports`. Verify with lint-imports and the full test session.

## 2. Formatting

- [ ] 2.1 Run black on `src`, `tests`, and `docs`, and verify that `black --check` passes.

## 3. Findings

- [ ] 3.1 Fix every mypy error, pydoclint finding, and suppression comment in `skcomplex.models`, split by field across agents.
- [ ] 3.2 Do the same for `skcomplex.engine`.
- [ ] 3.3 Do the same for `skcomplex.core`, `analysis`, `experiments`, the root modules, the tests, and the docs.
- [ ] 3.4 Add the tests that no source holds `TYPE_CHECKING`, `noqa`, `type: ignore`, or `nosec`. Verify that the checks session, the full test session, and the user guide pass.
