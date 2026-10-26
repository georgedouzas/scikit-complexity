# Tasks

## 1. Base modules

- [x] 1.1 Split `core/base` into `_elements`, `_interactions`, `_laws`, `_types`, `_errors`, and `_utils`. Move the
  terms, domains, variable declarations, symbols, and `VisualSpec` to `_types`, and `BaseEstimator` and `Registry`
  to `_utils`. Verify with the scenario "Three bases".
- [x] 1.2 Move `BaseSpace`, `BaseTime`, and `BaseNeighborhood` to `core/space/_bases.py`, and `Entry` and `Layout` to
  `core/problems`. Update the import contracts. Verify with the scenario "Other bases".

## 2. Element names

- [x] 2.1 Rename `BaseComponent` to `BaseElement`, `ElementComponent` to `Element`, `GroupComponent` to
  `GroupElement`, `FieldComponent` to `FieldElement`, `ComponentKind` to `ElementKind`, and every class ending in
  `Component` to end in `Element`. Use a codemod kept in the scratchpad, and cover `src`, `tests`, `docs`, and
  `inventory.toml`. Verify with the scenarios "Hierarchy", "Model names", and "Derived classes".

## 3. Quantities

- [x] 3.1 Remove `BaseQuantity`, `quantities(s)`, and `AuxiliaryQuantity`. Write every auxiliary as an expression in
  its laws, and give the models whose galleries plot an auxiliary an observable for it. Verify with the scenarios
  "Auxiliary as an expression" and "Hilsa fishery".
- [x] 3.2 Rename `ObservableQuantity` to `Observable`, and replace `InvariantQuantity` with
  `Simulation(invariants=...)`. Verify with the scenario "Violated invariant".
- [x] 3.3 Replace `ObjectiveQuantity` with the `objective`, `terminal`, and `sense` arguments of the optimal control
  solvers. Verify with the scenario "Objective under discrete time".
- [x] 3.4 Replace `AggregateQuantity` and a system's `aggregates` with reductions read across partitions. Verify with
  the scenario "Count of agents".

## 4. Parts

- [x] 4.1 Add `BasePart` in `core/base/_parts.py` with `label`, `check`, `prepare`, and `run` and their hooks, and
  derive `BaseElement`, `BaseInteraction`, `BaseLaw`, and `System` from it, with its hooks abstract. Make a system's
  `run(s)` return the laws of its children. Verify with the scenarios "Parts share three
  verbs" and "System yields laws".
- [x] 4.2 Give laws an optional `label`, a `prepare(namespace)` that returns a copy with `targets_`, `reads_`, and
  `owner_`, and a `run(values)` that evaluates their contribution, and make the census read the prepared laws. Verify
  with the scenarios "Law run alone" and "Invalid law".

## 5. Rules and gate

- [x] 5.1 Amend the constitution's layout rule. Update the `import-system-dynamics` change so that auxiliaries map to
  expressions and observables. Update the user guide and the gallery.
- [ ] 5.2 Run formatting, checks, the docs build, and tests through nox, and verify that all of them pass.
