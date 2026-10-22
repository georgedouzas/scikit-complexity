# Tasks

## 1. Canonical and variational interactions

- [ ] 1.1 Add `Canonical` and `Variational` in `core.interactions` with symbolic derivation, the numeric mass-matrix
  fallback, and their `Requires` declarations and validation issues, and verify the pendulum, coordinate without a
  momentum, and coordinate of order 1 scenarios
- [ ] 1.2 Add the `Yoshida4` solver with its `Supports` declaration and the `relative_drift` observable, and verify the
  double pendulum and non-separable function scenarios, and that the declaration test passes

## 2. Conservation and symmetry

- [ ] 2.1 Add `poisson_bracket` and `is_conserved` with the numeric fallback, and verify the energy and angular momentum
  scenario
- [ ] 2.2 Add `Symmetry`, `is_invariant`, and `noether_charge` in `core.analysis`, and verify the rotation invariance
  scenario

## 3. Library

- [ ] 3.1 Add the mechanics models as registered factories whose catalog entries pass the drift observable, and verify
  the Kepler scenario and the conformance suite

## 4. Documentation and gate

- [ ] 4.1 Document canonical and variational interactions, conservation, and symmetries in the user guide, naming the
  Hamiltonian and Lagrangian they take, and verify the documentation test passes
- [ ] 4.2 Run formatting, checks, the docs build, and tests through nox, and verify all of them pass
