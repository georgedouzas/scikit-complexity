# Proposal

## Why

Mechanics and much of mathematical physics describe a system by one scalar function rather than by its forces:
- **Hamiltonians and Lagrangians:** the equations follow from Hamilton's equations or from Euler-Lagrange equations
  through the Legendre transform. Conserved quantities are the observables whose Poisson bracket with the Hamiltonian
  vanishes, and Hamiltonian flows preserve phase-space volume (*The Structures of Mathematical Physics* §5.3.2, §7.6).
- **Hamiltonian chaos:** Dilão (2023 ch. 2, 8) and Datseris and Parlitz (2022 ch. 8) study it through the standard map,
  Hénon-Heiles, and celestial mechanics.
- **Rigid mechanisms:** Ryzhov et al. (2021) model pendulums and double pendulums.

The Sage layer already differentiates symbolically, so declaring a Hamiltonian or a Lagrangian and deriving everything
else costs little. It also gives every field a principled way to declare conserved quantities and symmetries.

## What Changes

- Add Hamiltonian and Lagrangian interactions over generalised coordinates and momenta, with Hamilton's and
  Euler-Lagrange equations derived symbolically, including angle coordinates.
- Add Poisson brackets, conservation checks of observables, and symmetry declarations whose invariance is checked
  symbolically, together with the conserved quantity that Noether's theorem associates with a continuous symmetry of a
  Lagrangian.
- Add symplectic steppers of higher order (Yoshida fourth order) next to velocity Verlet, and an energy-drift monitor.
- Add library models: simple, double, and coupled pendulums, the Kepler problem, the restricted three-body problem,
  the Chirikov standard map, and Hénon-Heiles as a Hamiltonian.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `system-modeling`: Hamiltonian and Lagrangian interactions and symmetry declarations.
- `symbolic-analysis`: Poisson brackets, conservation checks, and Noether charges.
- `model-library`: mechanics models.

## Impact

- `skcomplex.model` gains `Hamiltonian`, `Lagrangian`, and `Symmetry`, `skcomplex.analysis` gains `poisson_bracket`,
  `is_conserved`, and `noether_charge`, and `skcomplex.solvers` gains `Yoshida4`.
- Depends on `add-general-model-interface` for angle domains and observables, on `add-higher-order-equations` for
  coordinates of order 2 and equation laws, and on `add-analysis-toolkit` for the divergence check.
