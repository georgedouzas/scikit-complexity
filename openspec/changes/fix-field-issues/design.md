# Design

## Context

All four issues reproduce on `development`. On the current code `lyapunov(KuramotoSivashinskySystem(), kind='spectrum')`
first stops at the limit of 64 members of symbolic expansion, and with 32 modes it raises `KeyError: 'line.u[0]'`.
A 40 by 40 Schnakenberg run takes about 3.6 s per unit of time with `RK4Solver(dt=0.005)`.

## Goals / Non-Goals

Goals: fix each issue with a test, and document the behaviour that changes.

Non-goals: new solvers beyond what the profile of #48 needs, and new analysis kinds.

## Decisions

1. **Spectra from the backend Jacobian.** The tangent dynamics need the Jacobian of the drift. For small systems whose
   laws apply no field operator it stays the Sage Jacobian. For larger groups, and for fields at any size, whose
   expanded equations apply an operator to one site and so give a wrong Sage Jacobian, the spectrum uses JAX
   derivatives when the system compiles with `JaxBackend` and central differences of the compiled drift otherwise. The initial state is read as the state vector of
   the problem, not by component names. The test checks the largest exponent of Kuramoto Sivashinsky at `L = 22` near
   0.04 with a loose tolerance and a short run.
2. **Derivatives in dispersion.** A `partial` of order `k` along an axis becomes a multiplication by `(i q)^k` in the
   linearisation, as the Laplacian becomes `-|q|^2`. Mixed operators combine.
3. **Antisymmetric wrapped differences.** For every edge the difference is wrapped once, `w(theta_j - theta_i)` in
   `[-pi, pi)`, and the reverse orientation takes its negative. The total charge is then exactly 0 on a periodic
   lattice.
4. **Serial kernels for small fields (#48).** The profile showed the time inside the compiled RK4 loop: groups of
   1000 members or more printed every member kernel as a parallel Numba loop, so one step of a 40 by 40 field started
   24 parallel regions of a few operations per site each. Member kernels now run in parallel only from 200000 members,
   where threads win on an 8 core laptop, and pair kernels keep the threshold of 1000. No solver changes.

## Risks / Trade-offs

- A finite difference Jacobian is slower and less exact than a symbolic one. It applies only above the symbolic limit.
- Sums over every member of a group of 1000 to 200000 members may change in the last digits, from the summation order.

## Open Questions

None.
