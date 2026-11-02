# Proposal

## Why

- **`core.base` holds none of the bases.** `Component`, `Law`, and `Space` live in other packages, and `base`
  instead mixes terms, declarations, and domains, some used by only one formalism.
- **The bases relate wrongly.** `Interaction` and `System` derive from `Component`, yet a system uses only its label.
  `ExternalBackend` is a solver.
- **Names do not say what a class is.** `Rate` is a law, `RK4` is a solver, and `Mesh` is a space, and nothing in the
  names says so.

## What Changes

- **BREAKING** `skcomplex.core.base` holds the bases. No base derives from another, and each relates to the others by
  having them.
  - `BaseComponent`, `BaseInteraction`, `BaseSpace`, `BaseTime`, `BaseSolver`, and `BaseBackend` derive from
    `sklearn.base.BaseEstimator`.
  - `BaseLaw`, `BaseDomain`, and `BaseProblem` derive from nothing.
- **BREAKING** `System` and the interactions no longer derive from a component. A system has components,
  interactions, subsystems, a space, and a time. `System` stays the one concrete system class.
- **BREAKING** The framework keeps the fewest classes:
  - users subclass `BaseInteraction` directly, so `Interaction` and `core/interactions` are removed;
  - there is no `BaseSystem`;
  - the backend base is generic instead of adding protocol classes.
- **BREAKING** Every class whose ancestry reaches a base ends with that concept's name, in the framework, the models,
  the tests, and the documents. Examples:
  - `RateLaw`, `RK4Solver`, `MeshSpace`, and `NumericDomain`;
  - `ElementComponent` and `PairPotentialInteraction`;
  - `QuantumProblem`;
  - in the models, `MinSkeletonComponent`.

  Names that say the wrong thing are corrected:
  - `ExternalBackend` becomes `ExternalSolver`;
  - `GroupSpace` becomes `CayleySpace`.
- **BREAKING** The engine's bases take the `Base` prefix and stay in the engine: `BaseObserver`, `BaseVisual`,
  `BaseClock`, `BaseDesign`, `BaseSurrogate`, and `BasePass`. `SystemPass` derives from `BasePass`.
- **`base` modules by concept.** `core/base` is grouped into fundamental modules and auxiliary ones:
  - fundamental: components and interactions, laws, domains, space and time, problems, solvers and backends,
    variables, and symbols;
  - auxiliary: kinds, errors, and utils.
- **BREAKING** The types and errors of one formalism leave `base` for that formalism: the Hilbert domains for
  `quantum`, the structure domain for `rewriting`, `WordProblemError` for `groups`, and `HorizonError` for
  `fractional`.
- Validation reads a domain's kind from the domain.
- The constitution states the naming rule, the bases, their parents, and what `base` holds.
- The classes that the change `fix-class-relations` reworks keep their names until that change.

## Capabilities

### New Capabilities

### Modified Capabilities
- `package-layout`: adds the bases, their relations, the naming rule, and what `core.base` holds.
- `system-modeling`: a system holds nested systems without being a component.

## Impact

- **Code:** about 2,500 class names across `src`, `tests`, and `docs`, and the imports of every moved name.
- **Packages:** `core/interactions` is removed, and `core/base` is regrouped.
- **Constitution:** 1.9.0.
- **Other changes:** the active changes that name the old classes need their text updated when they are applied.
