# Proposal

## Why

Mechanics and much of mathematical physics describe a system by one scalar function rather than by its forces:
- **Hamiltonians:** the equations follow from Hamilton's equations over canonical coordinates and momenta. Conserved
  quantities are the expressions whose Poisson bracket with the Hamiltonian vanishes, and Hamiltonian flows preserve
  phase-space volume (*The Structures of Mathematical Physics* §5.3.2, §7.6).
- **Hamiltonian chaos:** Dilão (2023 ch. 2, 8) and Datseris and Parlitz (2022 ch. 8) study it through Hénon-Heiles and
  celestial mechanics.
- **Symmetries:** Noether's theorem gives the conserved quantity of every continuous symmetry of a Lagrangian.

The core already derives equations from a Lagrangian: `EulerLagrangeInteraction` returns one `EquationLaw` per
coordinate, sums the Lagrangians of separate interactions, takes dissipation and forcing, and the pendulums of
`physics/mechanics` and the restricted three-body problem are already built on it, so this change adds no variational
interaction and no pendulum. What is missing is the canonical form, the checks of conservation and symmetry, and a
symplectic solver of higher order than `VelocityVerletSolver`.

## What Changes

- Add `HamiltonianInteraction` in `skcomplex.core.interactions`, which derives Hamilton's equations as `RateLaw`s from a
  function of canonical coordinates and momenta, states of order 1 that may be angles. The Hamiltonians of separate
  interactions on one state sum, because their rate laws add, so a coupling is an interaction of its own.
- Add `InvariantAnalysis` in `skcomplex.analysis`, which computes Poisson brackets over the canonical pairs of a system,
  decides whether expressions given as `ObservableSpec`s are conserved, symbolically or numerically, and checks a
  symmetry given as a `SymmetrySpec` of `skcomplex.specs` against the system's Hamiltonian or Lagrangian, with the
  Noether charge of the symmetry.
- Add `Yoshida4Solver` in `skcomplex.engine.solvers`, the symplectic integrator of fourth order that composes three
  velocity Verlet steps, for systems whose states split into two blocks whose rates each read only the other block.
- Rebuild the Kepler problem (`two_body_orbit`), the restricted three-body problem, and Hénon-Heiles on
  `HamiltonianInteraction`, with the drift of their Hamiltonian checked by the existing `InvariantObserver` through the
  `invariants` default of their catalog entries. The Chirikov standard map stays a map of update laws.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `system-modeling`: Hamiltonian interactions.
- `symbolic-analysis`: Poisson brackets, conservation checks, symmetries, and Noether charges.
- `simulation`: the symplectic solver of fourth order.
- `model-library`: Hamiltonian mechanics models.

## Impact

- `skcomplex.core.interactions` gains `HamiltonianInteraction`, `skcomplex.analysis` gains `InvariantAnalysis`,
  `skcomplex.specs` gains `SymmetrySpec`, and `skcomplex.engine.solvers` gains `Yoshida4Solver`.
- `physics/celestial_mechanics` rebuilds `two_body_orbit` and `restricted_three_body`, and
  `physics/nonlinear_oscillators` rebuilds `henon_heiles`, whose element loses its own laws. `n_body` keeps the pair
  gravity of `physics/common/particles`.
- Builds on `EulerLagrangeInteraction`, `RateLaw`, `ObservableSpec`, `InvariantObserver`, `VelocityVerletSolver`, the
  Jacobian sparsity of `ProblemSpec.jacobian`, and the divergence of `StabilityAnalysis`.
