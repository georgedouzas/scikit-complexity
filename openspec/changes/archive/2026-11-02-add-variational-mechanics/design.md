# Design

## Context

`EulerLagrangeInteraction(target, states, lagrangian, dissipation=None, forcing=None)` in `skcomplex.core.interactions`
derives the Euler-Lagrange equation of every coordinate of order 2 as an `EquationLaw` with a `target`, and the compiler
adds the equations of one coordinate from every interaction before it solves for the highest derivatives. A subclass
overrides `_get_lagrangian`. The pendulums of `physics/mechanics` (`DoublePendulumInteraction`,
`PendulumSpringInteraction`, `CartPendulumInteraction`, `ParametricPendulumInteraction`) and the rotating frame of
`restricted_three_body` are built on it.

`VelocityVerletSolver` integrates states of order 2 symplectically at second order and declares its `SupportSpec`.
`StabilityAnalysis('fixed-points')` derives the divergence of a vector field. `InvariantObserver` checks at every frame
that observables stay within a tolerance of their first value, and a catalog entry passes it through
`defaults={'invariants': ...}` with the `observables` of `register_model`. A compiler reports the errors a part raises
while it declares its laws as `IssueSpec`s of its `report_`.

## Goals / Non-Goals

**Goals:**

- Mechanical models written as one scalar function of canonical coordinates and momenta.
- Conservation and symmetry as checks a user runs on any system with a Hamiltonian or an Euler-Lagrange interaction, or
  on any expression of its states.

**Non-Goals:**

- A second variational interaction, since `EulerLagrangeInteraction` is the Lagrangian form.
- Field theories and Hamiltonian densities, constrained systems with Lagrange multipliers, and quantum mechanics, which
  `GeneratorLaw` covers.
- Discrete canonical maps from generating functions. The Chirikov standard map stays a map of update laws.

## Decisions

### Build on the framework, not on a mechanics library

A library audit on 2026-11-01 found no Python library to take this capability from, so the plan stays in house:
- REBOUND 5.2.2 ships symplectic integrators such as WHFast and the Wisdom-Holman family, but it is GPL and integrates
  only gravitational N-body problems of its own particles. It cannot step a compiled vector field of arbitrary states,
  and `n_body` and `two_body_orbit` already run on the framework's solvers.
- No Python library composes symplectic integrators, such as Yoshida's fourth-order scheme, over an arbitrary compiled
  field, so `Yoshida4Solver` composes the existing `VelocityVerletSolver` step.
- Poisson brackets, Lie derivatives, and the first-order variation of a Hamiltonian are a few lines of Sage
  differentiation over the canonical pairs. No library offers them as a reusable component over a system's namespace,
  so `InvariantAnalysis` holds only that glue on top of Sage, the symbolic layer every law is written in.

### Hamilton's equations as rate laws

`HamiltonianInteraction(label, *, target, coordinates='q', momenta='p', hamiltonian=None)` derives
`RateLaw(q, dH/dp)` and `RateLaw(p, -dH/dq)` for every canonical pair, with `H = hamiltonian(ns)`. The coordinates and
the momenta are states of order 1 of every role that binds one element or group, `target` by default, paired in
order, and every component of a vector state is a coordinate or a momentum, as in `EulerLagrangeInteraction`. A
subclass overrides `_get_hamiltonian(namespace)` to derive the function from its parameters. Coordinates may be angle
states, which wrap as every angle does.

The rate laws of one state add, so the Hamiltonians of separate interactions sum: a coupling between two oscillators is
an interaction whose Hamiltonian holds only the coupling energy, and dissipation or forcing is a `RateInteraction` on
the momenta. The class takes no `form`, since agent models do not use it, and no `divisor`, since the Hamiltonian holds
the masses.

The name follows `EulerLagrangeInteraction`: both name the mathematical construction that derives the equations, not
a quantity of a field. It does not collide with the quantum laws, whose generator is `GeneratorLaw`, and
`SymbolicAnalysis('hamiltonian')` keeps its quantum meaning.

Misuse:
- Coordinates and momenta of different counts raise `ValidationError` in `_check`, naming the unpaired state.
- A coordinate and its momentum of different sizes, or a state of order other than 1, raise `ValidationError` from
  `_get_laws`, naming the state, with the fix `StateSpec(order=1)` or `EulerLagrangeInteraction` for a coordinate of
  order 2. The compiler reports the error as an issue of the interaction's path.

### Conservation and symmetry as an analysis

`InvariantAnalysis(kind='conservation', *, expressions=(), symmetry=None, samples=16)` in `skcomplex.analysis` reads a
system, as `SymbolicAnalysis` and `StabilityAnalysis` do, and takes the expressions as `ObservableSpec`s, the existing
spec of a named expression of a system's namespace. The kinds:

- `'brackets'`: `brackets_` maps every pair of labels to the Poisson bracket `{f, g}`, summed over the canonical pairs
  of the system's `HamiltonianInteraction`s. A system without one raises `AnalysisError`.
- `'conservation'`: `rates_` maps every label to `df/dt`, which is `diff(f, t) + {f, H}` for a system whose dynamics all
  come from Hamiltonian interactions, and the Lie derivative along the vector field of the equations of
  `SymbolicAnalysis('equations')` otherwise. `conserved_` maps every label to whether its rate simplifies to zero, and
  `methods_` to `'symbolic'`, or `'numeric'` when simplification does not decide and the rate vanishes within a
  tolerance at `samples` random states drawn by `random_state`.
- `'symmetry'`: `symmetry` is a `SymmetrySpec(label, generator)`, whose `generator` is a function of the namespace that
  maps the path of every coordinate it moves to its infinitesimal variation `dq`. `invariant_` says whether the
  Hamiltonian, or the summed Lagrangian of the system's `EulerLagrangeInteraction`s, is unchanged to first order, with
  the `remainder_` when it is not, and `charge_` is the Noether charge, `sum p dq` for a Hamiltonian and
  `sum dL/dq' dq` for a Lagrangian.

The symmetry is an argument of the analysis, not a part of the system, since a system is defined by elements,
interactions, and laws. `SymmetrySpec` lives in `skcomplex.specs` because it names no core type. The analysis is one
class with kinds, as `SymbolicAnalysis` is, and no public function stands outside it.

### The Yoshida solver

`Yoshida4Solver(dt=1e-3, *, random_state=None, until=inf)` in `skcomplex.engine.solvers` composes three velocity Verlet
steps with the coefficients `w1 = 1 / (2 - 2^(1/3))` and `w0 = 1 - 2 w1`, which gives fourth order and keeps the
symplectic structure. At `prepare` it splits the states into two blocks from the sparsity of `ProblemSpec.jacobian`:
a state of order 2 splits into its position and velocity, as in `VelocityVerletSolver`, and states of order 1 split so
that the rates of each block read only the other block, as the coordinates and momenta of a separable Hamiltonian do.
A system that admits no such split raises `SolverError` naming the state whose rate reads its own block, with the fix of
a solver that does not need a split, such as `RK4Solver`. Its `SupportSpec` declares continuous time, the laws
`RateLaw`, `EquationLaw`, and `BoundaryLaw`, and integer orders, so noise and jumps are refused by the existing
`solver-unsupported` issue.

### Drift through the invariant observer

No drift observable is added. A model passes its Hamiltonian as an `ObservableSpec` through the `observables` of
`register_model` and its tolerance through `defaults={'invariants': {'energy': ...}}`, and the existing
`InvariantObserver` raises `ConservationError` when the drift exceeds it.

### Models

- `two_body_orbit` in `physics/celestial_mechanics`: a `HamiltonianInteraction` of the relative coordinate with
  `H = norm(p)^2 / 2 - G M / norm(q)`. It is separable and runs on `Yoshida4Solver`.
- `restricted_three_body` in `physics/celestial_mechanics`: a `HamiltonianInteraction` of the rotating frame, whose
  Coriolis terms couple coordinates and momenta. It is not separable, runs on `RK4Solver`, and its Hamiltonian is the
  Jacobi integral.
- `henon_heiles` in `physics/nonlinear_oscillators`: a `HamiltonianInteraction` with
  `H = (px^2 + py^2) / 2 + V(x, y)`. `HenonHeilesElement` declares the states `x`, `y`, `px`, and `py` of order 1 in
  place of `x` and `y` of order 2, and loses its own laws.

`n_body` keeps `GravityInteraction` of `physics/common/particles`, so the pair gravity stays an example after
`two_body_orbit` moves to the canonical form. The Sitnikov problem stays on `GradientInteraction`. The standard map
stays a map of the update laws of `StandardMapElement`, since it is a discrete symplectic map that no continuous
Hamiltonian generates without kicks.

## Risks / Trade-offs

- [Simplification may not prove zero] → `InvariantAnalysis` falls back to a numeric check at random states and reports
  which method decided.
- [Renamed states of Hénon-Heiles] → The momenta `x_d1` and `y_d1` become `px` and `py`, so the trajectory variables
  change name. The model's tests and gallery example change with it.
- [`two_body_orbit` loses the star as a body] → The Kepler problem in the relative coordinate is the reduced two-body
  problem, and `n_body` keeps the bodies of the pair form.

## Open Questions

- None.
