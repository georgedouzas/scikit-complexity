# Design

## Context

Potentials already derive forces from energies by symbolic gradients, which is the Hamiltonian idea restricted to
particles with Cartesian positions. Sage differentiates, solves linear systems, and simplifies.

## Goals / Non-Goals

**Goals:**

- Mechanical models written as one scalar function in any coordinates.
- Conservation and symmetry as checks that users can run on any system with a Hamiltonian, Lagrangian, or observable.

**Non-Goals:**

- Field theories and Lagrangian densities. Fields from `add-spatial-fields` can adopt a variational form later.
- Constrained systems with Lagrange multipliers beyond holonomic coordinates, and quantum mechanics.

## Decisions

### Hamiltonian and Lagrangian interactions

`Hamiltonian(coordinates, momenta, H)` derives `Rate(q, ∂H/∂p)` and `Rate(p, -∂H/∂q)` for every pair.
`Lagrangian(coordinates, velocities, L)` derives the Euler-Lagrange equations. It solves the mass matrix `∂²L/∂v∂v`
symbolically for the accelerations, or numerically per step when the symbolic inverse is too large. Coordinates and
momenta are states of the elements the interaction binds, so these interactions compose with forces and dissipation
written as ordinary laws.

### Brackets, conservation, and symmetry

`poisson_bracket(f, g, system)` sums over the system's canonical pairs. `is_conserved(f, system)` checks that
`df/dt = ∂f/∂t + {f, H}` simplifies to zero for Hamiltonian systems, and that the Lie derivative along the vector field
vanishes for others, returning the simplified remainder when it does not. `Symmetry(generator)` declares an
infinitesimal transformation of coordinates. The check confirms that the Lagrangian or Hamiltonian is invariant, and
`noether_charge(symmetry, system)` returns the conserved quantity `Σ ∂L/∂v · δq`.

### Steppers

`Yoshida4(dt)` composes three velocity Verlet steps with the Yoshida coefficients for separable Hamiltonians. An energy
monitor records the relative drift of the Hamiltonian as an observable when one is declared.

## Risks / Trade-offs

- [Symbolic mass-matrix inversion grows fast] → The numeric per-step solve is the fallback, chosen automatically above a
  size threshold.
- [Simplification may not prove zero] → `is_conserved` falls back to a numeric check at random states, and reports
  which method decided.
