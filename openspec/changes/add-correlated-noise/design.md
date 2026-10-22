# Design

## Context

This design is rebased on `redesign-system-model`: the noise convention is an argument of each noise law,
`Noise(convention=...)`, and every solver and backend declares its support, which validation compares with the system
before compilation.

`Noise(target, value, convention='ito')` gives every component of its target, and every member of a group, one fresh Wiener process. The
compiler lowers the noise laws to one diffusion function that returns one amplitude per process, and to `noise_rows`,
the state each process drives. `EulerMaruyama`, `Milstein`, `Heun`, and the hybrid SDE loop draw one normal increment
per amplitude and add `amplitude * increment` at its row. The Numba Euler-Maruyama loop does the same in compiled code.
The diffusion matrix is therefore diagonal per process: a process never drives two states, and two processes are never
correlated.

The convention correction adds `alpha g dg/dx` per noise law, with `alpha` 0, 1/2, or 1 for the law's convention, Itô,
Stratonovich, or anti-Itô. It raises an ambiguity error when several noise laws drive one state.

## Goals / Non-Goals

**Goals:**

- Write a reaction's noise as one Wiener process that drives every species the reaction changes.
- Correlate Wiener processes across components with a constant covariance.
- Keep the Itô, Stratonovich, and anti-Itô conventions exact for a general diffusion matrix.
- Keep every existing seeded SDE run bit for bit the same.

**Non-Goals:**

- State-dependent or time-dependent covariance. A law can scale its amplitude by any expression instead.
- Strong order 1 for non-diagonal noise. That needs Lévy areas or a commutativity check, which a later change can add
  to `Milstein`.
- Wiener groups, one correlated block per member. Member laws build them from local processes, as shown below.
- Coloured or fractional noise, which `add-anomalous-processes` covers.

## Decisions

### Library survey

The constitution asks to take sampling and SDE integration from a mature library. The survey:

| Library | What it offers | Verdict |
|---|---|---|
| `sdeint` | Itô and Stratonovich Euler and SRI schemes for a dense diffusion matrix `G(x, t)` | Rejected: small and no longer developed, calls a Python function per step, needs a dense `n × m` matrix that groups of 10⁵ members cannot hold, and has no events, delays, or Numba |
| `torchsde` | General, diagonal, scalar, and additive noise, Brownian intervals, adjoints | Rejected: needs PyTorch, has its own random stream, and cannot run the NumPy and Numba kernels the compiler emits |
| `diffrax` | `ControlTerm` with a virtual Brownian tree for any noise shape | Rejected: needs JAX, whose keyed random numbers cannot follow the simulation's NumPy `Generator`, and the numerics audit keeps diffrax for gradients and GPUs only |
| SciPy | `multivariate_normal` sampling, no SDE integrator | Rejected for sampling: a Python call per step and no Numba support |
| NumPy | `Generator.standard_normal`, `Generator.multivariate_normal`, `linalg.cholesky`, `linalg.eigh` | Adopted: the steppers already draw from the run's `Generator`, Numba compiles `Generator.normal`, and `linalg` factors the covariance once at compile time |

`Generator.multivariate_normal` is not used inside the steps, because Numba does not compile it and it refactors the
matrix on every call. The framework draws standard normal increments and multiplies them by a factor that NumPy
computes once. The code kept in house is the adapter: the mapping of outputs to processes and the scaled scatter.

### Naming processes

`Noise(target, value, order=None, convention='ito', process=None)` in `skcomplex.core.laws` gains `process`:
- `None` keeps today's behaviour, one independent process per component and member;
- a string is a label local to the component instance, and to the member for member laws. Every noise law of the
  component with that label, and every component of a vector target, shares one process;
- a tuple of strings gives one label per component of a vector target;
- `s.<role>.dW` or `s.<role>.dW[k]` names process `k` of a `Wiener` element bound by a role, which every law of every
  component can share. Group laws that name it give all members one common process.

A reaction channel is then `Noise(s.first.n, -sqrt(a), process='channel')`, `Noise(s.second.n, -sqrt(a),
process='channel')`, and `Noise(s.product.n, sqrt(a), process='channel')`. All three get one increment, so
`first - second` has no noise.

Two correlated processes per member come from two local labels:
`x: σ₁ dW_a` and `y: σ₂ (ρ dW_a + sqrt(1 - ρ²) dW_b)`.

### Wiener elements

`Wiener(label, dimension=1, covariance=None)` in `skcomplex.core.components` is an `Element` with no states. It declares `dimension` processes whose
increments over a step `h` have covariance `h C`, with `C` the identity by default. `C` is a constructor argument
fixed at compile time, not a run-time parameter.

`Wiener` declares `Requires(time={TimeKind.CONTINUOUS})`, since Wiener increments exist only in continuous time.
`validate()` reports:
- `wiener-time`: a `Wiener` element in a system whose time is not continuous, with the fix of continuous time;
- `wiener-covariance`: a covariance that is not square of size `dimension`, symmetric, and positive semidefinite within
  `1e-12` of its largest eigenvalue, naming the element and the offending eigenvalue or entry;
- `wiener-handle`: an expression outside `Noise.process` that uses `s.<role>.dW`, naming the law, with the fix of
  naming the handle as the law's process.

`compile_system` and `Simulation` raise these in one `ValidationError`, as for every issue.

The factor `L` with `L Lᵀ = C` is the Cholesky factor from `numpy.linalg.cholesky`. A singular `C`, such as two
processes with correlation 1, falls back to `V sqrt(max(Λ, 0))` from `numpy.linalg.eigh`.

### Lowering

The diffusion function keeps returning one amplitude per noise output. The problem gains:
- `noise_columns`, the process column of every output, beside `noise_rows`;
- `n_processes`, the number of standard normal draws per step;
- `noise_blocks`, the start, size, and factor of every correlated block, empty when no `Wiener` has a covariance.

Columns are numbered in order of first use in evaluation order. A `Wiener` element takes a contiguous block of
`dimension` columns at its first use. For a system that names no process, `noise_columns` is `arange(n_outputs)` and
`noise_blocks` is empty, so the steppers draw the same normals in the same order as today.

### Declared noise support

The vocabulary of `skcomplex.core.base` gains `NoiseKind` with the terms `'diagonal'` (every process drives one state),
`'shared'` (a process drives several states), and `'correlated'` (a `Wiener` element with a covariance), and `Supports`
gains a `noise` field over them. A system's noise structure is the set of terms its noise laws and `Wiener` elements
use, computed from the system's definition without compiling it. The declarations are:

| Solver or backend | `noise` |
|---|---|
| `EulerMaruyama`, `Heun`, the hybrid SDE loop | diagonal, shared, correlated |
| `Milstein` | diagonal |
| NumPy and Numba backends | diagonal, shared, correlated |
| LAMMPS backend | diagonal |

`validate()` reports `noise-unsupported` when the chosen or `'auto'` solver or backend lacks a term of the system's
noise structure. The issue names the first law or `Wiener` element that needs the term, and its fix lists the solvers
or backends that support it. The solvers and backends no longer raise for unsupported noise themselves.

### Steppers

Each step draws `z = rng.normal(0, sqrt(h), n_processes)` from the stepper's `rng_`, which the simulation seeds,
multiplies each block of `z` by its factor, and scatters `amplitude[o] * dW[columns[o]]` into `rows[o]`. When the
problem has no blocks and `columns` is the identity, the stepper takes today's path unchanged.

- `EulerMaruyama` and the hybrid SDE loop apply the scaled scatter. The Numba loop gains the `columns`, block offsets,
  and flattened factors as arrays and applies the same loop, so it stays in compiled code.
- `Heun` applies the scaled scatter in its predictor and corrector, and stays an Itô scheme for any noise.
- `Milstein` and the LAMMPS backend are never given shared or correlated noise, since validation rejects that
  combination through their declared support.

### Convention corrections

With amplitudes `G_ij` of state `i` on process `j` and covariance `C`, the drift correction of state `i` is
`alpha Σ_k Σ_j Σ_l C_jl G_kl ∂G_ij/∂x_k`. Independent processes have `C` equal to the identity, so a process that
drives one state gives back today's `alpha g dg/dx`. The sum is well defined when several laws drive one state, so the
ambiguity error is removed. With conventions on the laws, `alpha` belongs to the process: every law that names one
process must declare the same convention, and `validate()` reports `noise-convention-mixed`, naming the laws and the
process, with the fix of one convention for all of them. Laws with independent processes keep their own conventions.

The compiler builds each term in the scope of the law whose amplitude it differentiates. A term needs `G_kl` of
another state `x_k`. When `x_k` gets its noise from a law of another component through a shared or correlated
process, that amplitude is not in the scope. `validate()` then reports `noise-correction-scope` for a Stratonovich or
anti-Itô law in that case, naming both laws, with the fix of the Itô convention or one component for both laws. Additive noise and
amplitudes that depend only on their own state, such as correlated geometric Brownian motions, never reach this case.

### Chemical Langevin mode

`Binding` and `Dissociation` gain `langevin: bool = False`. In Langevin mode a channel with propensity `a` adds
`change * a` to the drift of every species it changes and one noise law per species with amplitude
`change * sqrt(max(a, 0))` on the local process `'channel'`. Setting both `stochastic` and `langevin` raises a
`ValidationError` that names the interaction when the interaction is built. Both interactions are library realizations
in `skcomplex.models.chemistry.kinetics`, built on the core's `Rate`, `Jump`, and `Noise` laws. The flag keeps the library's `stochastic` convention, where a model
switches between rate and jump descriptions.

`chemical_langevin_network` is the chemical Langevin equation of the enzyme network `E + S <-> C -> E + P` with
Itô noise, with the same defaults as `michaelis_menten`. Under noise it conserves the enzyme total `E + C` and the
substrate total `S + C + P`. `correlated_geometric_brownian_motion` gives two assets with drifts, volatilities, and a
correlation `rho` through a `Wiener` element of dimension 2. `chemical_langevin_equation` stays the Schlögl model,
since each of its channels changes one species.

## Risks / Trade-offs

- [The covariance is fixed at compile time] → Changing it needs a recompile. A run-time correlation is written into
  the amplitudes with two local processes.
- [Chemical Langevin amplitudes need non-negative propensities] → The amplitude clips the propensity at zero, as the
  Schlögl model does. Species can still turn slightly negative near zero, which is a known limit of the equation.
- [Milstein rejects non-diagonal noise] → Validation names the alternatives, and a later change can add the commutative
  case by widening its declaration.
- [A different draw order would change seeded runs] → Systems that name no process keep the identity path, and a
  regression test compares a seeded run against a stored trajectory.
