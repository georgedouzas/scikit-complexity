# Proposal

## Why

The core mixed two things: what a system is, and the machinery that works on a system. A modeller who defines a
system needs only the first, while compiling, solving, running, viewing, experimenting, and analysing are clients of
it. The maintainer moved the packages to make that split. This change records the new layout and makes the code,
tests, documentation, and checks follow it.

## What Changes

- **BREAKING:** the package now has four subpackages, and the import paths of every moved name change:
  - `skcomplex.core` holds what a system is: `base`, `space`, `laws`, `components`, `interactions`, and `system`.
  - `skcomplex.engine` holds what works on a system: `compile`, `solvers`, `backends`, `runtime`, `viz`,
    `experiments`, and `analysis`. These moved from `skcomplex.core`.
  - `skcomplex.formalisms` holds the formalisms. They moved from `skcomplex.core.formalisms`.
  - `skcomplex.models` is unchanged.
- **Dependencies:** they run one way:
  - `core` imports no other subpackage;
  - `engine` imports `core`;
  - a formalism imports `core`, `engine`, and the formalisms it declares;
  - models import all three.
- **Imports:** every import of the source, the models, the tests, the gallery examples, and the documentation follows
  the new paths.
- **Tests:** the tests of the moved packages move to `tests/engine` and `tests/formalisms`, mirroring the source.
- **Import contracts:** the contracts follow the four packages.
- **Constitution:** the Project Profile's layout, extension points, and surface follow.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `package-layout`: the four subpackages, their dependency direction, and the public surface the models import from.
- `formalism-plugins`: formalisms live in `skcomplex.formalisms`, and neither the core nor the engine imports one.
- `system-modeling`: the package placement of primitives and backends.
- `rewriting-systems` and `model-gallery`: import paths in their requirements.

## Impact

- **Code:** every module of `src/skcomplex`, the models, the tests, `docs/examples`, the user guide, the README, and
  the docs generators.
- **Configuration:** `pyproject.toml`, for the import contracts and any package paths.
- **Constitution:** the Project Profile.
- **Behaviour:** none changes. The move is mechanical, and the compiled problems and trajectories of every model stay
  the same.
