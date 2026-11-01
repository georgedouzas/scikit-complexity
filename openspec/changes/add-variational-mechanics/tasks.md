# Tasks

## 1. Hamiltonian interaction

- [x] 1.1 Add `HamiltonianInteraction` in `core/interactions/_hamiltonian.py`, exported from
  `skcomplex.core.interactions`, with the `_get_hamiltonian` hook, vector and angle states, and the errors of unpaired,
  mismatched, and higher-order states, with tests in `tests/core/interactions/test_hamiltonian.py`; verify the
  oscillator, coupling, unpaired coordinate, and order 2 scenarios

## 2. Solver

- [x] 2.1 Add `Yoshida4Solver` in `engine/solvers/_yoshida4.py` with its `SupportSpec`, the block split from
  `ProblemSpec.jacobian`, and its Numba stepping as in `VelocityVerletSolver`, with tests in
  `tests/engine/solvers/test_yoshida4.py`; verify the fourth-order and non-separable scenarios

## 3. Conservation and symmetry

- [ ] 3.1 Add `SymmetrySpec` in `skcomplex.specs` and `InvariantAnalysis` in `analysis/_invariants.py` with the kinds
  `'brackets'`, `'conservation'`, and `'symmetry'` and the numeric fallback, with tests in
  `tests/analysis/test_invariants.py`; verify the energy and angular momentum, Jacobi integral, brackets without
  canonical pairs, rotation invariance, and broken symmetry scenarios

## 4. Models

- [ ] 4.1 Rebuild `two_body_orbit` and `restricted_three_body` in `models/physics/celestial_mechanics/_orbits.py` on
  `HamiltonianInteraction`, with their Hamiltonian observables and `invariants` defaults, keeping `n_body` and the
  Sitnikov problem as they are; verify the Kepler scenario and the model tests
- [ ] 4.2 Rebuild `henon_heiles` on `HamiltonianInteraction`, with `HenonHeilesElement` declaring `x`, `y`, `px`, and
  `py` of order 1 and no laws, and update its gallery example; verify the Hénon-Heiles scenario
- [ ] 4.3 Add the `variational-mechanics` primitive to the three entries of `docs/overview/inventory.toml`; verify the
  inventory check passes

## 5. Documentation and gate

- [ ] 5.1 Document `HamiltonianInteraction`, `InvariantAnalysis`, and `Yoshida4Solver` in the user guide next to
  `EulerLagrangeInteraction`, and add the changelog entry; verify the documentation test passes
- [ ] 5.2 Run formatting, checks, the docs build, and tests through nox; verify all of them pass
