# Design

## Context

The maintainer moved 283 files, all byte-identical:
- `compile`, `solvers`, `backends`, `runtime`, `viz`, `experiments`, and `analysis` went from `src/skcomplex/core` to
  `src/skcomplex/engine`;
- `formalisms` went from `src/skcomplex/core` to `src/skcomplex/formalisms`.

Nothing else changed, so every import that crosses the new boundaries is stale:
- the relative imports `..compile` inside `core`, `..components` inside `engine`, and `...base` inside a formalism;
- the absolute paths such as `skcomplex.core.compile` in the models, tests, examples, and documents.

## Goals / Non-Goals

**Goals:**
- Every import resolves under the new layout.
- Behaviour is unchanged.

**Non-Goals:**
- Changes to any concept, module, or name other than its package path.

## Decisions

### Rewrite imports from the move map

A script in the scratchpad rewrites imports from one map of the moved packages:
- `core.<p>` becomes `engine.<p>` for the seven engine packages;
- `core.formalisms` becomes `formalisms`.

It rewrites absolute imports by prefix. It recomputes relative imports from each module's new location to the target's
new location. Relative imports inside one moved package keep working as they are. The script stays out of the
repository.

### `core` must not import `engine`

Before the move, `core` packages imported the engine in a few places, for example components reaching `compile` for
probes. Any such import is a violation of the new direction. Each one is resolved by moving the helper to the package
that owns its concept, usually the engine side, as task 5.2 of `redesign-core-concepts` did. The import contracts make
each case visible.

### Tests mirror the moved packages

`tests/core/<p>` moves to `tests/engine/<p>` for the engine packages, and `tests/core/formalisms` moves to
`tests/formalisms`.

### Import contracts

The contracts become:
- a `layers` contract over `skcomplex.formalisms`, `skcomplex.engine`, and `skcomplex.core`, with `models` free above
  them;
- a `layers` contract inside `skcomplex.core`, ordered `base`, `space`, `laws`, `components`, `interactions`, `system`;
- a `layers` contract inside `skcomplex.engine`, ordered `compile`, `solvers`, `backends`, `runtime`, `viz`,
  `experiments`, `analysis`;
- the formalism independence contract;
- the model contracts, with their public-module allowances updated to the three packages.

## Risks / Trade-offs

- [Hidden reverse imports from core to engine] → `lint-imports` names each one, and each is fixed by concept.
- [Stale paths in prose] → A grep for the old paths over `src`, `tests`, `docs`, and the README confirms none remain.

## Migration Plan

1. Rewrite the imports.
2. Fix the reverse imports.
3. Move the tests.
4. Update the contracts, the specs, and the Profile.
5. Compare problems and trajectories with the baseline, and run the gate.
