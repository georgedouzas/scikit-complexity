# Proposal

## Why

The bases name their families but do not define them.
- **The bases do not define what their children do.** `BaseSpace` declares no `size`, although five of its six
  spaces have one. `BaseProblem` defines none of the twenty methods the runtime calls on problems, so the runtime
  reaches for them with `getattr`. Learning one space, law, or problem does not teach the next one.
- **Behaviour hides in magic methods.**
  - 41 `__post_init__` validate at construction, far from the system that holds the value.
  - 31 `__repr__` print each class its own way.
  - `__call__`, `__getattr__`, `__eq__`, `__hash__`, `__reduce__`, and `__new__` turn ordinary calls into hidden
    ones.

## What Changes

- **A base defines the whole public API of its family.** Each public method or property of the base is written once
  and calls a private hook, `_name`, that a child overrides, as scikit-learn's `fit` calls `_fit`.
  - A family concept that does not apply to a child gets a trivial answer from the base, such as `size` returning
    `None` for a continuous space.
  - A child's public API is the base's API plus its own constructors, such as `NetworkSpace.erdos_renyi`.
  - What a child alone needs becomes private, or moves to the code that uses it.
- **No hand-written magic methods** apart from these:
  - `__init__`;
  - arithmetic on algebraic values the modeller writes as mathematics, the quantum operators;
  - the collection protocol of collections the modeller indexes or iterates, such as `trajectory['a.n']`;
  - the attribute namespace of laws, `s.x`.

  Specifically:
  - `__repr__` comes from scikit-learn's parameter printing.
  - Validation moves from `__post_init__` into a public `check` that calls private hooks, and validation runs it with
    the owner's path in every message.
  - Callable objects get named methods, such as `evaluate`.
  - Combining designs uses `product` and `concat` instead of `*` and `+`.
- **Values and behaviour are told apart.**
  - Plain value records stay frozen dataclasses without hand-written methods: terms, domains, laws, compiled entries,
    and problems.
  - Configurable objects with behaviour derive from their base on `BaseEstimator` with an explicit `__init__`:
    components, interactions, spaces, times, solvers, backends, visuals, observers, designs, and surrogates.
- **Terms become value records.** `LawKind`, `Form`, `SpaceKind`, and `Trait` are compared with `==` and need no
  interning hooks. `is` comparisons of terms become `==`.
- **Surrogates become estimators.** `BaseSurrogate(BaseEstimator)` takes its settings in `__init__` and is trained
  with `fit(series)`. It forecasts with `forecast`, which the base defines by simulating the fitted system. It holds
  the fitted `system_` and errors.
  - The DMD, SINDy, and reservoir surrogates derive from it.
  - `fit_dmd`, `fit_sindy`, and `fit_reservoir` are removed.
- **BREAKING:**
  - the surrogate functions;
  - the design operators;
  - the callable objects;
  - the repr of the classes that printed themselves;
  - construction-time validation of laws and records.

## Capabilities

### New Capabilities
- `base-interfaces`: what a base defines for its family, how children extend it, and which magic methods the code
  allows.

### Modified Capabilities

## Impact

- **Families:**

  | Family | Classes |
  |---|---|
  | spaces | 6 |
  | laws | 34 |
  | domains | 6 |
  | problems | 4 |
  | solvers | 38 |
  | visuals | 12 |
  | observers | 5 |
  | components | 9 |
  | interactions | 4 |
  | designs | 6 |
  | surrogates | 3 |

- **Magic methods:** about 90 hand-written ones in `core`, `engine`, and `formalisms`.
- **Callers:** the runtime and the visuals, which lose their `getattr` fallbacks, and every `is` comparison of a term.
- **Depends on:** `gather-base-classes`. The change `fix-class-relations` reworks the quantity classes, so this change
  leaves `Auxiliary`, `Invariant`, `Objective`, and `Wiener` to it.
