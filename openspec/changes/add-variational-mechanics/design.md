# Design

## Context

Potentials already derive forces from energies by symbolic gradients, which is the canonical idea restricted to
particles with Cartesian positions. Sage differentiates, solves linear systems, and simplifies.

After `redesign-system-model`, a system is `System(label, components=, interactions=, space=, time=)`, interactions are
stateless and return laws, states of order 2 are declared with `state(order=2)`, observables are given to
`Simulation(observables=...)` and catalog entries, and every interaction and solver declares `Requires` or `Supports`
for `validate()`. Core names are mathematical or modelling terms, and the redesign renamed the quantum `Hamiltonian`
law to `Generator` for that reason.

## Goals / Non-Goals

**Goals:**

- Mechanical models written as one scalar function in any coordinates.
- Conservation and symmetry as checks that users can run on any system with a canonical or variational interaction, or
  on any expression of its states.

**Non-Goals:**

- Field theories and Lagrangian densities. Site groups on lattices can adopt a variational form later.
- Constrained systems with Lagrange multipliers beyond holonomic coordinates, and quantum mechanics, which the
  redesign's `Generator` covers.

## Decisions

### Canonical and variational interactions

`Canonical(coordinates, momenta, function)` in `core.interactions` derives `Rate(q, ∂H/∂p)` and `Rate(p, -∂H/∂q)` for
every pair, with `H` the function. `Variational(coordinates, function)` derives the Euler-Lagrange equations over
coordinates of order 2, `state(order=2)`, whose companion `<q>_d1` states are the velocities. It returns them as
`Equation` laws, so the joint solve for the highest derivatives inverts the mass matrix `∂²L/∂v∂v`, with `L` the
function, symbolically or numerically per step. Coordinates and momenta are states of the components the interaction
binds with `over = 'members'`, so these interactions compose with forces and dissipation written as ordinary laws.

The names follow the redesign's application-agnostic rule: the redesign rejected keeping `Hamiltonian` as a core name,
so the interactions are named for the structure they derive, canonical pairs and the variational principle. The user
guide gives the physics names of the functions.

Declarations and validation:

| Interaction | `Requires` | Issue when misused |
|---|---|---|
| `Canonical` | `time={CONTINUOUS}`, `domains={REAL, ANGLE}`, `bindings={MEMBERS}` | a coordinate without a momentum, or a coordinate of another domain, naming the state |
| `Variational` | `time={CONTINUOUS}`, `domains={REAL, ANGLE}`, `bindings={MEMBERS}` | a coordinate not declared of order 2, with the fix `state(order=2)` |

### Brackets, conservation, and symmetry

`poisson_bracket(f, g, system)` in `core.analysis` sums over the canonical pairs of the system's `Canonical`
interactions. `is_conserved(f, system)` checks that `df/dt = ∂f/∂t + {f, H}` simplifies to zero for canonical systems,
and that the Lie derivative along the vector field vanishes for others, returning the simplified remainder when it does
not.

`Symmetry(generator)` in `core.analysis` is an infinitesimal transformation of coordinates. It is not part of the system,
which holds only components, interactions, space, and time. `is_invariant(symmetry, system)` confirms that the function
of the system's canonical or variational interaction is invariant, and `noether_charge(symmetry, system)` returns the
conserved quantity `Σ ∂L/∂v · δq`.

### Solver and drift

`Yoshida4(dt)` in `core.solvers` composes three velocity Verlet steps with the Yoshida coefficients for separable
functions. It declares `Supports(time={CONTINUOUS}, laws={RATE}, domains={REAL, ANGLE, VECTOR})`, and `validate()`
reports a system whose canonical function is not separable, or which holds noise or jump laws, with the fix of a
non-symplectic solver.

`relative_drift(expression)` in `core.analysis` is an observable of the relative drift of an expression from its first
frame. The mechanics models' catalog entries pass the drift of their canonical function to the simulation.

## Risks / Trade-offs

- [Symbolic mass-matrix inversion grows fast] → The numeric per-step solve is the fallback, chosen automatically above a
  size threshold.
- [Simplification may not prove zero] → `is_conserved` falls back to a numeric check at random states, and reports
  which method decided.
- [Generic names are less familiar to physicists] → The user guide and the docstrings name the Hamiltonian and the
  Lagrangian as the functions the interactions take.

## Open Questions

- How the Chirikov standard map is rebuilt. It is a discrete-time map, while `Canonical` and `Variational` require
  continuous time under the redesign's law classes. Either it is written as a symplectic map of `Update` laws under
  `DiscreteTime`, which this design assumes, or the change adds a discrete canonical form (a generating function),
  which would be a new law and is not settled here.
