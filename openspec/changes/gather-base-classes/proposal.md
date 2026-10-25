# Proposal

## Why

`skcomplex.core.base` holds none of the base classes of the core. `Component`, `Law`, and `Space` live in
`components`, `laws`, and `space`. Meanwhile `base` holds a mix of terms, declarations, and domains, some of which
only the engine or a single formalism uses. A reader cannot tell what `base` is for.

## What Changes

- **BREAKING** `skcomplex.core.base` holds the base classes of the core: `BaseComponent`, `BaseLaw`, `BaseSpace`,
  and `BaseDomain`. They replace `Component`, `Law`, and `Space`, which move out of `components`, `laws`, and
  `space`. Nothing keeps an alias of the old names.
- `base` also keeps the helpers these classes are declared with:
  - the errors;
  - `state`, `parameter`, and `role`;
  - `symbol` and `vector`;
  - the numeric, categorical, and tensor domains;
  - the law, space, and component kinds;
  - `Registry`.
- **BREAKING** The engine's vocabulary leaves `base` for `skcomplex.engine.compile`: `Form`, `Supports`,
  `OrderKind`, `NoiseKind`, `NUMERIC_DOMAINS`, and `vocabulary`.
- **BREAKING** The types and errors of one formalism leave `base` for that formalism:
  - `HilbertSpace`, `HilbertFamily`, and `HilbertSpaceError` go to `quantum`;
  - `Structure`, `RewritingError`, and `WordProblemError` go to `rewriting`;
  - `HorizonError` goes to `control`.
- Every domain derives from `BaseDomain` and names its kind. Validation then reads a domain's kind instead of
  listing the domain classes, so a formalism adds a domain without the engine naming it.
- **BREAKING** The solver and backend bases are renamed `BaseSolver` and `BaseBackend`:
  - `BaseSolver` moves from `_stepper.py` to `engine/solvers/_base.py`;
  - `BaseBackend` stays in `engine/compile`.

  Both stay in the engine because they are defined over the compiled problem.
- The constitution names the new base classes and what `base` holds.

## Capabilities

### New Capabilities

### Modified Capabilities
- `package-layout`: adds where the base classes live and what `core.base` holds.

## Impact

- **Code:**
  - every module that imports `Component`, `Law`, `Space`, `Stepper`, or `Backend`, about 1,100 references across
    `src`, `tests`, and `docs`;
  - the modules that import the moved engine and formalism names;
  - `engine/compile/_validation.py`, which today lists the domain classes.
- **Tests:** the tests of the moved modules move with them:
  - `tests/core/components/test_component.py` and `tests/core/laws/test_law.py` go to `tests/core/base`;
  - the tests of the moved domains go to their formalisms.
- **Constitution:** the project profile names the base classes, which bumps it to 1.9.0.
- **Other changes:** the active changes that name the old classes need their text updated when they are applied.
