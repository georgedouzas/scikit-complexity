# Design

## Context

All four issues reproduce on `development`. On the current code `lyapunov(KuramotoSivashinskySystem(), kind='spectrum')`
first stops at the limit of 64 members of symbolic expansion, and with 32 modes it raises `KeyError: 'line.u[0]'`.
A 40 by 40 Schnakenberg run takes about 3.6 s per unit of time with `RK4Solver(dt=0.005)`.

## Goals / Non-Goals

Goals: fix each issue with a test, and document the behaviour that changes.

Non-goals: new solvers beyond what the profile of #48 needs, and new analysis kinds.

## Decisions

1. **Spectra from the backend Jacobian.** The tangent dynamics need the Jacobian of the drift. Below the symbolic
   member limit it stays the Sage Jacobian. Above it, the spectrum uses the Jacobian the backend gives (JAX when
   installed) or a finite difference Jacobian of the compiled drift. The initial state is read as the state vector of
   the problem, not by component names. The test checks the largest exponent of Kuramoto Sivashinsky at `L = 22` near
   0.04 with a loose tolerance and a short run.
2. **Derivatives in dispersion.** A `partial` of order `k` along an axis becomes a multiplication by `(i q)^k` in the
   linearisation, as the Laplacian becomes `-|q|^2`. Mixed operators combine.
3. **Antisymmetric wrapped differences.** For every edge the difference is wrapped once, `w(theta_j - theta_i)` in
   `[-pi, pi)`, and the reverse orientation takes its negative. The total charge is then exactly 0 on a periodic
   lattice.
4. **Profile first for #48.** Find where the time goes, the step count, the per step cost in Python, or the solver
   choice. The fix may be a compiled kernel for the stepper, or an automatic choice of the existing `ETDRK4Solver` or an
   IMEX step for systems whose stiff part is linear diffusion. The target is a 40 by 40 Schnakenberg run to time 40 in
   under 30 s on a laptop.

## Risks / Trade-offs

- A finite difference Jacobian is slower and less exact than a symbolic one. It applies only above the symbolic limit.
- A new default solver for reaction diffusion changes recorded frames within the tolerance.

## Open Questions

None.
