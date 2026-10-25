# Design

## Context

- **Where the base classes are today:**

  | Base class | Module | Derived classes |
  |---|---|---|
  | `Component` | `core/components/_component.py` | `Element`, `Group`, `Interaction`, `System` |
  | `Law` | `core/laws/_law.py`, with `register_laws`, `law_classes`, `is_registered`, `has_trait` | every law |
  | `Space` | `core/space/_spaces.py` | `SiteSpace`, `ContinuousSpace`, the lattices, the networks |
  | `Stepper` | `engine/solvers/_stepper.py` | every solver, and `ExternalBackend` |
  | `Backend` | `engine/compile/_backends.py` | the numpy and numba backends |

- **What their modules import:** each of the three core modules imports only from `core.base`.
- **What `core.base` holds:**
  - errors;
  - terms and kinds;
  - variable declarations;
  - symbols;
  - domains;
  - a registry;
  - `VisualSpec`.
- **Names that the rest of core never uses:**
  - `Form`, `Supports`, `OrderKind`, `NoiseKind`, `NUMERIC_DOMAINS`, `Categorical`, and `Tensor` (only the engine
    uses them);
  - `HilbertSpace` and `HilbertFamily` (only quantum);
  - `Structure` (only rewriting).
- **Domain checks:** `engine/compile/_validation.py` maps the domain classes to their kinds by `isinstance`.
- **What the engine bases are defined over:** `Stepper` and `Backend` take `Problem`, `Vector`, `Record`, and
  `Kernel`, which `engine.compile` defines. `Problem` imports the layout, hybrid, quantity, and reduction modules of
  the compiler.

## Goals / Non-Goals

**Goals:**
- `core.base` is the base classes of the core and the helpers they are declared with, and nothing else.
- Every class of the core derives from a base of `core.base`.

**Non-Goals:**
- Changing what any class does. This change moves and renames; it does not redesign behaviour.
- Moving the engine bases into `core`.
- Renaming `ExternalBackend`. It is a solver that hands a problem to another library, and it is named in a later
  change.

## Decisions

- **What `base` holds.** `core/base` holds one module per base class, and the helper modules those classes are
  declared with:

  | Module | Holds |
  |---|---|
  | `_errors.py` | the error hierarchy |
  | `_registry.py` | `Registry` |
  | `_kinds.py` | `Term`, `LawKind`, `LawClass`, `TimeKind`, `Trait`, `SpaceKind`, `ComponentKind`, `DomainKind`, `Binding`, `Requires` |
  | `_domain.py` | `BaseDomain`, `Domain`, `Categorical`, `Tensor`, `wrap_angle` |
  | `_symbolic.py` | `symbol`, `vector`, `Expression`, `SymbolicVector`, and the symbol helpers |
  | `_variables.py` | `state`, `parameter`, `role`, `Variable`, `VariableKind`, and the declaration readers |
  | `_visuals.py` | `VisualSpec` |
  | `_space.py` | `BaseSpace` |
  | `_law.py` | `BaseLaw`, `register_laws`, `law_classes`, `is_registered`, `has_trait` |
  | `_component.py` | `BaseComponent` |

  Modules import downward in the order of the table. `core.base` imports no other package of `skcomplex`.
  `Categorical` and `Tensor` stay because a component declares them as the domain of a state, like a numeric
  domain.

- **Base names carry a `Base` prefix.** The names are `BaseComponent`, `BaseLaw`, `BaseSpace`, `BaseDomain`,
  `BaseSolver`, and `BaseBackend`, after `sklearn.base.BaseEstimator`. Nothing keeps an alias of the old names. A
  codemod in the scratchpad rewrites the imports and uses; it never enters the repository.
  - *Alternative:* keep `Law` and the other old names, and move them only. Rejected because the maintainer asked for
    base names that say they are bases.

- **A domain is a base class.** `BaseDomain` declares four things:
  - `kind`, a `DomainKind`;
  - `contains(value)`;
  - `encode(value)`;
  - `decode(values)`.

  The numeric `Domain` enumeration, `Categorical`, `Tensor`, `HilbertSpace`, `HilbertFamily`, and `Structure`
  derive from it. `AnyDomain` is removed, and annotations name `BaseDomain`. Validation reads `domain.kind`, so the
  engine no longer imports the formalism domains.

- **The engine's vocabulary moves to `engine.compile`.** `Form`, `Supports`, `OrderKind`, `NoiseKind`,
  `NUMERIC_DOMAINS`, and `vocabulary` describe compiled problems, and what solvers and backends support. Validation
  compares them, so they go to `engine/compile/_support.py`. `Requires` stays in core, because laws declare it.
  `Form` stays an open vocabulary, and formalisms extend it from `engine.compile`, which they are allowed to import.

- **Errors stay in one hierarchy, except those of one formalism.** The shared errors stay in `core/base/_errors.py`,
  so a user catches any library error from one module. These include the ones only the engine raises, such as
  `SolverError`. An error that one formalism raises and no other package does moves to that formalism, next to its
  types.

- **The engine bases stay in the engine.** `BaseSolver` and `BaseBackend` are defined over `Problem`. `Problem`
  carries the compiler's layout, events, quantities, and reductions, so `core` cannot hold it without holding the
  compiler.
  - `BaseSolver` moves to `engine/solvers/_base.py`.
  - `BaseBackend` stays in `engine/compile/_backends.py`, because compilation and validation call it.
  - *Alternative:* a new `engine/base` package before `compile`. Rejected because the bases would then import
    `compile` while `compile` imports them, a cycle.

- **Tests follow their modules.** The test tree mirrors the source tree:
  - `tests/core/components/test_component.py` goes to `tests/core/base/test_component.py`;
  - `tests/core/laws/test_law.py` goes to `tests/core/base/test_law.py`;
  - the tests of the `Space` base go to `tests/core/base/test_space.py`;
  - the tests of the moved domains go to the formalism that now owns them.

## Risks / Trade-offs

- [The rename touches about 1,100 references.] → One codemod renames every reference. Then `ruff`, the import
  contracts, and the tests of the touched packages check the result.
- [Active changes name the old classes.] → Each one updates its text when it is applied, as after the earlier
  split.
- [Pickled systems name the old classes.] → None are stored in the repository, and the package makes no promise
  about pickles across versions.
