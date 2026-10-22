# Proposal

## Why

Mechanics and much of mathematical physics describe a system by one scalar function rather than by its forces:
- **Hamiltonians and Lagrangians:** the equations follow from Hamilton's equations or from Euler-Lagrange equations
  through the Legendre transform. Conserved quantities are the observables whose Poisson bracket with the Hamiltonian
  vanishes, and Hamiltonian flows preserve phase-space volume (*The Structures of Mathematical Physics* §5.3.2, §7.6).
- **Hamiltonian chaos:** Dilão (2023 ch. 2, 8) and Datseris and Parlitz (2022 ch. 8) study it through the standard map,
  Hénon-Heiles, and celestial mechanics.
- **Rigid mechanisms:** Ryzhov et al. (2021) model pendulums and double pendulums.

The Sage layer already differentiates symbolically, so declaring one scalar function and deriving everything else costs
little. It also gives every field a principled way to check conserved quantities and symmetries.

**Rebased on `redesign-system-model`.** The redesign renamed the core's quantum `Hamiltonian` law to `Generator` and
holds core names to mathematical or modelling terms, so the interactions this change adds are named for their
mathematics: `Canonical` derives Hamilton's equations from a function of canonical pairs, and `Variational` derives
Euler-Lagrange equations from a function of coordinates and velocities. A system is now only components, interactions,
space, and time, so symmetries are no longer declared on a system: they are arguments of the analysis functions that
check them. Observables go to simulations, so the energy-drift monitor is an observable the catalog entries pass. The
rest of the change keeps its scope.

## What Changes

- Add the interactions `Canonical` and `Variational` in `core.interactions`, over generalised coordinates and momenta or
  velocities, with Hamilton's and Euler-Lagrange equations derived symbolically as `Rate` and `Equation` laws, including
  angle coordinates. Both declare their `Requires`, and `validate()` reports their misuse.
- Add Poisson brackets and conservation checks of expressions in `core.analysis`, and symmetries given to the analysis as
  infinitesimal transformations, whose invariance is checked symbolically, together with the conserved quantity that
  Noether's theorem associates with a continuous symmetry.
- Add the symplectic solver `Yoshida4` of fourth order in `core.solvers` next to velocity Verlet, with its `Supports`
  declaration, and a relative-drift observable of a conserved expression.
- Add library models: simple, double, and coupled pendulums, the Kepler problem, the restricted three-body problem,
  the Chirikov standard map, and Hénon-Heiles.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `system-modeling`: canonical and variational interactions.
- `symbolic-analysis`: Poisson brackets, conservation checks, symmetries, and Noether charges.
- `model-library`: mechanics models.

## Impact

- `skcomplex.core.interactions` gains `Canonical` and `Variational`; `skcomplex.core.analysis` gains `Symmetry`,
  `poisson_bracket`, `is_conserved`, `is_invariant`, `noether_charge`, and `relative_drift`; and
  `skcomplex.core.solvers` gains `Yoshida4`.
- Builds on the archived `add-general-model-interface` for angle domains, `add-higher-order-equations` for coordinates
  of order 2 and equation laws, and `add-analysis-toolkit` for the divergence check, and on `redesign-system-model` for
  the declarations and validation.
