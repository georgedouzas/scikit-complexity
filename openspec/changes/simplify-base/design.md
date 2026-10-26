# Design

## Context

- `core/base` today has these modules:
  - `_components.py`, `_laws.py`, and `_variables.py` (666 lines);
  - `_domains.py`, `_kinds.py` (13 term types), and `_spaces.py` (`BaseNeighborhood`, `BaseSpace`, `BaseTime`);
  - `_problems.py` (`Entry`, `Layout`) and `_quantities.py` (`BaseQuantity`);
  - `_estimator.py`, `_symbolic.py`, `_utils.py` (`Registry`, `VisualSpec`), and `_errors.py`.
- The quantities are used as follows:
  - `AuxiliaryQuantity` is substituted into laws and recorded under its name;
  - `InvariantQuantity` is checked by the invariant observer;
  - `ObjectiveQuantity` is read by the Pontryagin and collocation solvers;
  - `ObservableQuantity` is passed to `Simulation(observables=...)`;
  - `AggregateQuantity` is held in a private element of the co-simulation and bound through roles.

## Goals / Non-Goals

**Goals:**
- A modeller meets three base concepts, element, interaction, and law, and every other name in `core/base` is a
  shared type, error, or helper.
- Everything that is not part of a system's definition lives with the run, the solver, or the engine that uses it.

**Non-Goals:**
- Cutting the term types of `_kinds.py`. They move to `_types.py` unchanged.
- Changing what any law computes.

## Decisions

- **Six modules in `core/base`.**
  - **One base per module:** `_elements.py`, `_interactions.py`, and `_laws.py`.
  - **`_types.py`:** what values and declarations are, so the terms of `_kinds.py`, the domains of `_domains.py`,
    the variable declarations of `_variables.py`, and `VisualSpec`.
  - **`_errors.py`.**
  - **`_utils.py`:** `BaseEstimator`, `Registry`, and the symbol helpers of `_symbolic.py`.

  `BaseElement` and `BaseInteraction` share the declaration helpers in `_types.py`, so neither imports the other.
- **Bases of other concepts live with their concept.**
  - `BaseSpace`, `BaseTime`, and `BaseNeighborhood` move to `core/space/_bases.py`.
  - `Entry` and `Layout` move to `core/problems`, next to `BaseProblem`.
  - The constitution's layout rule changes to say that `core/base` holds the bases of a system's mandatory parts,
    and the base of any other concept lives in its package.
- **Element names.**
  - `BaseComponent` becomes `BaseElement`.
  - `ElementComponent` becomes `Element`, the class a modeller subclasses for a type of element:
    `class PopulationElement(Element)`.
  - `GroupComponent` becomes `GroupElement`, and `FieldComponent` becomes `FieldElement`.
  - Every class ending in `Component` ends in `Element`, in the library, models, tests, and documents, and in the
    names given to `define`.
  - `ComponentKind` becomes `ElementKind`.
  - A system's `components` argument keeps its name, since it holds elements and nested systems.
  - *Alternative:* keep `BaseComponent`. Rejected by the maintainer, so that the base names match the model.
- **Quantities are removed.**
  - **Auxiliaries.** A law reads an expression written in Python, `leverage = s.debt / s.equity`. To record it, the
    modeller passes an observable of the same expression. The stock-flow conservation check and loop analysis read
    the laws, so they lose nothing.
  - **Observables.** `ObservableQuantity` becomes `Observable`, a frozen record of a label and an expression or a
    group reduction. It is no subclass of a base, and `Simulation(observables=...)` takes it as today.
  - **Invariants.** `Simulation(invariants={name: tolerance})` names observables that must stay constant. The
    invariant observer checks them at every frame and raises `ConservationError` with the name and the first frame
    that breaks it.
  - **Objectives.** `PontryaginSolver` and `CollocationSolver` take `objective` (a running cost over dotted names),
    `terminal`, and `sense`. Validation reports a missing objective as the solver's own error.
  - **Aggregates.** A law of another partition reads a reduction of a group, `count(s.farmers)` or
    `total(s.farmers.wealth)`. The co-simulation computes every reduction that crosses partitions at each
    synchronisation point, and holds the value fixed until the next one.

## Risks / Trade-offs

- [Renaming 329 element classes and every `*Component` reference.] → A codemod in the scratchpad renames classes by
  their resolved base. mypy, the tests, the doctests, and the gallery check that every caller moved.
- [Auxiliaries were recorded without an observable, and the Rerun observer logged them.] → The models that log an
  auxiliary pass an observable instead, and the stock-flow models gain observables for the auxiliaries their
  galleries plot.
- [Cross-partition reductions are new to the co-simulation planner.] → The co-simulation tests that used aggregates
  are kept with reductions in their place, and must give the same values.
