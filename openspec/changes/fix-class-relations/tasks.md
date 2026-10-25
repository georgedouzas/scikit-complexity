# Tasks

## 1. Quantities

- [ ] 1.1 Add `BaseQuantity` to `core/base/_quantity.py`, and derive `ObservableQuantity`, `AuxiliaryQuantity`, `InvariantQuantity`,
  `ObjectiveQuantity`, and `AggregateQuantity` from it. Verify with the scenario "No quantity is a law".
- [ ] 1.2 Add `quantities` next to `laws` on `BaseComponent` and `BaseInteraction`, reject quantities from `laws`
  and laws from `quantities`, and move every declaration in `src`, `tests`, and `docs` with a codemod kept in the
  scratchpad. Verify with the scenarios "Quantities of a component" and "Quantity among the laws".

## 2. Noise, rewriting, groups, and experiments

- [ ] 2.1 Detach `WienerProcess` from `ElementComponent`, and bind it through a noise role. Verify with the scenario "Correlated
  noise".
- [ ] 2.2 Derive `RewritingLaw` and `RewriteLaw` from `BaseLaw` with the law kind `rewrite`, and move them to the laws of
  the rewritten components. Verify with the scenario "Rule of a word".
- [ ] 2.3 Rename the five group solvers after the word problem.
- [ ] 2.4 Make the schedule of an experiment a member of its task, without registering an observer.

## 3. Gate

- [ ] 3.1 Run formatting, checks, the docs build, and tests through nox once, and verify that all of them pass.
