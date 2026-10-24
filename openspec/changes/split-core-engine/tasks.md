# Tasks

## 1. Imports

- [ ] 1.1 Rewrite every import of `src/skcomplex`, `tests`, `docs` (examples, documents, generators, gallery tools),
  and `README.md` from the move map. Verify that `import skcomplex.core`, `skcomplex.engine`, `skcomplex.formalisms`,
  and `skcomplex.models` succeed, that the tests collect, and that a grep finds no old path.
- [ ] 1.2 Remove every import from `skcomplex.core` into `skcomplex.engine` or `skcomplex.formalisms` by moving the
  helper to the package that owns its concept. Verify with `lint-imports`.

## 2. Tests and contracts

- [ ] 2.1 Move the tests of the engine packages to `tests/engine` and the formalism tests to `tests/formalisms`, so
  they mirror the source. Verify that every test module names a source module at its mirrored path.
- [ ] 2.2 Rewrite the import contracts for the four packages, and verify that `lint-imports` passes and that a planted
  import from `core` into `engine` fails it.

## 3. Profile and verification

- [ ] 3.1 Update the constitution's Project Profile: the layout, the extension points, and the surface. Bump the minor
  version.
- [ ] 3.2 Compare the compiled problems and trajectories of every model with the latest records, and verify that none
  differs.
- [ ] 3.3 Run formatting, checks, the docs build, and tests through nox, and verify that all of them pass.
