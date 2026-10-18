# Tasks

## 1. Variational interactions

- [ ] 1.1 Add `Hamiltonian` and `Lagrangian` with symbolic derivation and the numeric mass-matrix fallback, and verify
  the pendulum scenario
- [ ] 1.2 Add the `Yoshida4` stepper and the energy monitor, and verify the double pendulum scenario

## 2. Conservation and symmetry

- [ ] 2.1 Add `poisson_bracket` and `is_conserved` with the numeric fallback, and verify the energy and angular momentum
  scenario
- [ ] 2.2 Add `Symmetry` and `noether_charge`, and verify the rotation invariance scenario

## 3. Library

- [ ] 3.1 Add the mechanics models as registered factories, and verify the Kepler scenario and the conformance suite

## 4. Documentation and gate

- [ ] 4.1 Document Hamiltonians, Lagrangians, conservation, and symmetries in the user guide, and verify the
  documentation test passes
- [ ] 4.2 Run formatting, checks, the docs build, and tests through nox, and verify all of them pass
