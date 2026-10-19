# Design

## Context

`ScipyODE.advance` calls `solve_ivp` with the drift alone, and `_advance_hybrid` builds `METHODS[method](...)` the same
way at every restart. SciPy's BDF and Radau accept `jac` as a callable that returns a sparse matrix and `jac_sparsity`
as a pattern. With a pattern they group structurally independent columns, estimate the Jacobian with one drift
evaluation per group, and factor `I - h γ J` by sparse LU. LSODA accepts a dense `jac` and the half bandwidths `lband`
and `uband` only.

The compiled problem already holds what the pattern needs. Every `Kernel` maps each symbol to a `Reference` whose
`Source` says how it is read:
- `MEMBER`: the same member;
- `SINGLE`: one component shared by every member;
- `NEIGHBOR`: the neighbour table of a field;
- `DERIVED`: an operator applied to an operand;
- `FIRST` and `SECOND`: the two members of a pair found at run time.

`SplitOperator.sparse` assembles the linear terms of finite-difference and finite element fields as a CSR matrix, the
finite element Laplacian is the sparse `-M⁻¹K` with lumped mass from scikit-fem, finite volumes reconstruct over a
stencil of two cells per side, and `problem.constrained` lists the fixed rows.

## Goals / Non-Goals

**Goals:**

- Make SciPy's BDF and Radau as fast on finite-difference and finite element fields as the split steppers, with the
  adaptive error control the split steppers lack.
- Derive the structure once per compiled problem, from the laws, with no input from the user.

**Non-Goals:**

- Replacing ETDRK4, the IMEX steppers, Crank-Nicolson, or `SplitOperator`. They win on spectral fields.
- Sparse Jacobians for SDE, jump, or map solvers.
- `diffrax` or another JAX integrator. It is the later option for gradients through the integrator and GPUs.
- Implementing column grouping or sparse LU. SciPy does both.

## Decisions

### Sparsity from the kernel references

`jacobian_pattern(problem)` returns a boolean CSR matrix over the state vector. For every ODE kernel and every output
with a state target, each state symbol the output's expression uses adds entries from the target's rows by its source:
- `MEMBER`: member `i` reads member `i`;
- `SINGLE`: every member reads the one shared component;
- `NEIGHBOR`: member `i` reads member `table[i]`, skipping the ghost index of fixed edges;
- `DERIVED` with a sparse operator: member `i` reads every `j` with a nonzero `A[i, j]`, for each operand symbol;
- `DERIVED` without one, spectral multipliers and inverse Laplacians: the dense block of the field;
- `FIRST` and `SECOND`: the dense block of the group, because pairs change with the state;
- callable laws and `Call` functions: every state the callable can read.

Finite-volume conservation laws add the band of their reconstruction stencil per axis, with wrap-around on periodic
edges. The diagonal is always included. Rows in `problem.constrained` are emptied. The result is the union with the
pattern of `SplitOperator.sparse`, which the tests use as a cross-check.

Delayed reads do not add entries. With delays, the hybrid path evaluates the history as a value fixed during one step,
which is the Jacobian SciPy's step needs.

Alternatives considered:
- Probing the drift with random perturbations. It misses entries that vanish at the probed states, and it costs one
  drift evaluation per state, which is what the change removes.
- Requiring the user to give a pattern. The laws already state it.

A new `Source` must state its pattern. Where it cannot, its block is dense, so the pattern stays a superset.

### Exact sparse Jacobian from Sage

`jacobian_function(problem)` differentiates every output expression by Sage's `diff` with respect to each state symbol
it uses, neighbour symbols included, so each partial derivative lands on the entry the pattern assigned to it.
Derived reads with a sparse operator use the chain rule `∂f/∂D · A · diag(∂u/∂s)`. The partials of a kernel compile to
one function with the problem's backend. They are evaluated over members and scattered into the data array of the
pattern's CSR matrix through a precomputed index map that sums duplicate entries. The parameters come from the vector
passed to `advance`, so parameter changes apply at the next call.

The exact Jacobian is unavailable for callable laws, `Call` functions, pair kernels, and spectral derived reads.
`jacobian='exact'` raises a named error that names the law and suggests `'auto'`.

### The `jacobian` parameter

`ScipyODE(method='RK45', rtol=1e-6, atol=1e-9, jacobian='auto')`. `jacobian` is stored unmodified:
- `'auto'`: BDF and Radau get `jac_sparsity`. LSODA gets `lband` and `uband` when both half bandwidths are at most a
  tenth of the size, and nothing otherwise. When the pattern's density exceeds one half, nothing is passed.
- `'exact'`: BDF and Radau get the sparse `jac`, and LSODA gets it as a dense array.
- `'dense'`: nothing is passed, as today.

RK45, RK23, and DOP853 never build the structure, so explicit runs are unchanged and cost nothing more. The pattern and
the compiled Jacobian are built in `initialize` for implicit methods and kept on the stepper as `jacobian_pattern_`.

Alternative considered: making BDF the default ODE method. Non-stiff models outnumber stiff ones in the library, and
RK45 is cheaper per step on them, so the default stays.

### The hybrid path

`_advance_hybrid` passes the same keyword arguments to every `METHODS[method](...)` it builds. Each restart at a
breakpoint or event re-estimates the Jacobian, which costs one evaluation per column group rather than one per state.

### Recommending the stiff path

The stability warning of explicit fixed-step steppers names `ScipyODE(method='BDF')` and `ScipyODE(method='Radau')`
for finite-difference and finite element fields, and the split steppers for spectral fields. An adaptive explicit
method warns once on the first `advance` whose span exceeds 1000 times `problem.stable_step`. The user guide gives the
audit's comparison.

### Libraries surveyed

- SciPy `solve_ivp`: adopted. It is already a dependency, and its BDF and Radau take sparse Jacobians.
- `diffrax`: Kvaerno5 took 4.3 s on Kuramoto-Sivashinsky against BDF's 0.44 s on CPU, and it needs a `jax.numpy`
  printer. Kept as a later option.
- SUNDIALS CVODE through `scikits.odes` or `scikit-sundae`: a new compiled dependency for a speed SciPy already reaches
  on the audit's problem. Rejected.
- py-pde and Dedalus: compared in the audit and rejected for fields there.

## Risks / Trade-offs

- [A missing entry in the pattern makes Newton converge slowly or fail] → The pattern is built as a superset, unknown
  couplings are dense, and a conformance test checks the pattern against finite differences on every library model
  with an ODE form.
- [Dense blocks for pairs or spectral reads remove the gain] → Above a density of one half, `'auto'` passes nothing,
  and the split steppers remain for spectral fields.
- [Symbolic differentiation is slow on large laws] → It runs per kernel, not per member, and is cached with the kernels.
- [Restarts in the hybrid path re-estimate the Jacobian] → Each estimate costs a few drift evaluations with the pattern.

## Migration Plan

`'auto'` changes only the speed and the step sequence of implicit methods, within their tolerance. `'dense'` restores
the previous path.
