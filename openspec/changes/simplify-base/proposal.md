# Proposal

## Why

A modeller defines a system with three things: the elements that exist, the interactions between them, and the laws
by which their states change. `core/base` should hold the bases of exactly those three, plus shared types, errors,
and helpers. Today it holds thirteen modules with more than thirty classes:
- bases of parts a system does not need to be defined, such as spaces, times, and neighbourhoods;
- compiler records, such as `Entry` and `Layout`;
- a fourth concept, `BaseQuantity`, with five quantities that are not part of a system's definition:
  - auxiliaries are names for expressions;
  - invariants and observables are things a run checks and records;
  - objectives define a control problem;
  - aggregates are reductions that a co-simulation passes between partitions.

## What Changes

- **Base holds the three bases and shared modules only.** `core/base` has six modules:
  - `_elements.py` holds `BaseElement`;
  - `_interactions.py` holds `BaseInteraction`;
  - `_laws.py` holds `BaseLaw`;
  - `_types.py` holds the terms, the domains, the variable declarations (`state`, `parameter`, `role`), and
    `VisualSpec`;
  - `_errors.py` holds the named errors;
  - `_utils.py` holds `BaseEstimator`, `Registry`, and the symbol helpers.
- **Other bases move to their packages.**
  - `BaseSpace`, `BaseTime`, and `BaseNeighborhood` move to `core/space`.
  - `Entry` and `Layout` move to `core/problems`.
- **BREAKING: elements are named elements.**
  - `BaseComponent` becomes `BaseElement`.
  - `ElementComponent` becomes `Element`, `GroupComponent` becomes `GroupElement`, and `FieldComponent` becomes
    `FieldElement`.
  - Every model and test class ending in `Component` ends in `Element`, such as `PopulationElement`.
- **BREAKING: quantities are removed.** `BaseQuantity`, the `quantities(s)` method, and the five quantity classes
  go. Each use moves to where it belongs:
  - an auxiliary becomes a plain expression in `laws`;
  - an observable is a record that `Simulation(observables=...)` takes, as today;
  - an invariant is named by `Simulation(invariants={name: tolerance})`, which checks an observable of that name at
    every frame;
  - an objective is an argument of the optimal control solvers;
  - an aggregate becomes a reduction, such as `count(s.farmers)`, that a law of another partition reads and the
    co-simulation computes at every synchronisation point.

## Capabilities

### New Capabilities

### Modified Capabilities
- `package-layout`: the base holds three bases and shared modules, and the other bases live in their packages.
- `system-modeling`: elements are named elements, quantities are removed, and invariants become a run setting.
- `control`: objectives are an argument of the optimal control solvers.
- `co-simulation`: aggregates are reductions that laws read across partitions.
- `rerun-visualization`: auxiliaries are no longer logged on their own, since they are expressions.

## Impact

- **Code:**
  - `core/base`, `core/space`, `core/problems`, `core/components`, `core/laws`, and `core/system`;
  - the compiler's census and flows passes, the conservation and invariant observers, the control solvers, and the
    co-simulation solver;
  - 329 element classes, 47 auxiliary, 3 invariant, and 5 objective declarations in the models;
  - the tests, the user guide, the gallery, and `docs/overview/inventory.toml`.
- **Constitution:** the layout rule says a base package holds the bases of a system's mandatory parts, and other
  bases live in the package of their concept.
- **The `import-system-dynamics` change:** system dynamics auxiliaries map to expressions and observables instead of
  `Auxiliary`.
