# Design

## Context

See proposal.md for the motivation. The core has `Noise` laws, which add independent Wiener increments and compile to
the SDE form, and `Jump` laws, which fire at exponential waiting times and compile to the jump form run by `Gillespie`.
Both are Markovian. The hybrid layer of `add-hybrid-dynamics` already stops fixed-step and adaptive steppers at
scheduled breakpoints, applies `Event` resets, and records each event in the trajectory with the states before and
after it. Laws are Sage expressions that the compiler prints to NumPy and Numba code. The analysis layer already has
`mean_squared_displacement` and Hurst exponents by detrended fluctuation analysis, which the acceptance scenarios use.

## Goals / Non-Goals

**Goals:**

- Write a continuous-time random walk, a Lévy walk, annealed transient time motion, and fractional Brownian motion as
  symbolic laws, as Argun et al. (2021 ch. 6) describe them, with no model-specific code in the core.
- Keep every run reproducible from the simulation's random state, including runs of large groups on Numba.
- Take sampling and transforms from SciPy, and keep in the core only the adapter that lowers laws to them.

**Non-Goals:**

- Fractional derivatives in rate laws, which `add-fractional-dynamics` adds. Fractional noise here is a driving
  process, not a memory term in the drift.
- Multiplicative fractional noise with `H ≠ 1/2`. Its pathwise integral needs rough-path calculus, and no inventory
  model needs it.
- Estimating anomalous exponents or classifying trajectories, as the AnDi challenge does. The existing analysis layer
  measures the exponents the scenarios check.

## Decisions

### Library survey

The constitution requires a feature to be built on a popular, mature, or performant library where one fits. The survey
covered the candidates of `organize-model-library` and the libraries that implement the same models:

| Library | What it provides | State in October 2026 | Verdict |
|---|---|---|---|
| SciPy `stats` (BSD-3) | Every common distribution, including `pareto`, `lomax`, `levy_stable` by Chambers-Mallows-Stuck, and `rvs` that takes a NumPy `Generator` | Already a runtime dependency, wheels for Linux and macOS on Python 3.12 and 3.13 | Wrapped for every draw and waiting time |
| SciPy `fft` (BSD-3) | Real and complex FFTs, the only costly step of the Davies-Harte circulant embedding | Already a runtime dependency | Wrapped for fractional Gaussian noise |
| `stochastic` 0.7.0 (MIT) | Fractional Gaussian noise and Brownian motion by Davies-Harte, Hosking, and Cholesky, and renewal-free point processes | Last release July 2022, requires Python below 3.11 and NumPy below 2 | Rejected: it does not install on the supported Python versions or with NumPy 2.1 |
| `fbm` 0.3.0 (MIT) | Fractional Brownian motion by Davies-Harte, Hosking, and Cholesky | Last release May 2019, pure Python, samples from the global NumPy random state | Rejected: unmaintained, and it cannot draw from the simulation's `Generator`, which breaks reproducibility |
| `andi-datasets` 2.1 | The five anomalous diffusion models of the AnDi challenge, including ATTM, CTRW, Lévy walks, and fBm | Released October 2025, pins NumPy at or below 1.26.4, depends on `stochastic`, pandas, h5py, and matplotlib | Rejected as a dependency: it conflicts with NumPy 2.1 and generates whole trajectories outside the symbolic laws. Its published exponents serve as references for the acceptance tests |
| `colorednoise` 2.2 (MIT) | Gaussian `1/f^β` noise by spectral synthesis | Released 2023, NumPy only | Rejected: the spectral method gives a periodic approximation, not the exact covariance of fractional Gaussian noise |
| `fathon`, `nolds`, `hurst` | Estimation of Hurst exponents and fluctuation analysis | Maintained to varying degrees | Rejected: they estimate exponents and do not generate processes |

No maintained library generates exact fractional Gaussian noise with NumPy 2 and a caller's `Generator`. The core
therefore holds the Davies-Harte embedding as a short adapter over `scipy.fft`: it builds the first row of the
circulant covariance, takes its FFT, checks that the eigenvalues are nonnegative, and transforms complex Gaussian
draws from the simulation's stream. The fallback when an eigenvalue is negative is the exact Cholesky factorisation of
`scipy.linalg.cholesky`, used only for short paths. Every draw and waiting time is a call to a frozen SciPy
distribution's `rvs` with the simulation's `Generator` as `random_state`, so no sampling algorithm lives in the core.

### Random draws

`add-stochastic-maps` adds named draws, `uniform`, `normal`, `bernoulli`, and `categorical`, that update laws use and
that Numba samples inside kernels with a NumPy `Generator`. This change reuses that vocabulary and its rule that one
name reads one value per firing, and adds one function to it: `draw(name, distribution, **parameters)`, a named draw
from any `scipy.stats` distribution, such as `draw('wait', 'pareto', b=alpha, scale=t0)` or
`draw('jump', 'levy_stable', alpha=a, beta=0)`. The compiler checks the distribution name against `scipy.stats` and the
keywords against its shape parameters, `loc`, and `scale`, and raises the existing `ValidationError` otherwise.

The rule of `add-stochastic-maps` that draws belong to update laws is widened to accept draws in renewal laws. When
both changes are archived, that requirement of `stochastic-maps` names renewal laws too, so the rule lives in one place.

Draws in renewal laws are evaluated outside kernels, once per renewal generation, with one vectorised `rvs` call per
draw name over the firing members, the parameters evaluated at their states, and the simulation's `Generator` as
`random_state`. This keeps the stream order fixed and needs no SciPy code inside Numba, which cannot compile SciPy
samplers such as `levy_stable`. A SciPy draw in an update law, which `add-stochastic-maps` compiles into the kernel,
uses the same path: the kernel reads it from a buffer that the stepper refills before each step.

Alternatives considered: printing NumPy `Generator` calls for every distribution, which Numba supports only for a few
and not for `levy_stable` or `lomax`; and Python callables in laws, which already exist but cannot be checked.

### Renewal laws

`Renewal(waiting, reset=None, label=None)` holds a waiting-time expression, normally a single draw, and a reset mapping
like `Event.reset`, whose values may contain draws. The compiler checks that the waiting-time distribution has
nonnegative support from `scipy.stats` `support()`. Waiting times may depend on states, as in annealed transient time
motion, where the reset draws a diffusion coefficient and the next waiting time depends on it. The waiting time is drawn
after the reset, from the reset state.

A renewal is a scheduled event: the compiled problem holds, per renewal law, an array of next renewal times with one
entry per group member. In the hybrid layer this array is a breakpoint source, so steppers that already stop at
breakpoints stop at the earliest pending renewal, apply the reset to the members whose time has come, draw their next
waiting times, and continue. Each firing pass of a law becomes one `RenewalRecord` among the trajectory's events, with arrays
of the times and members of its renewals and of the members' states before and after them, since one record object per
renewal would not fit runs of 10⁴ walkers. The count of every member lives in a hidden state `<group>.<label>~renewals`,
so frames carry it at their own time, partitions slice it with their owner, and the observable `<label>.renewals`
sums it.

Renewals do not change the form of a system that also has rate or noise laws, in the same way `Event` does not. A system
or partition whose only laws are renewals compiles to the new `Form.RENEWAL`. Its `RenewalStepper` keeps the next
times in a vectorised loop: it fires every member whose next time is at most `until`, in time order per
member, until none remains. It needs no time grid, so a continuous-time random walk of 10⁴ walkers with power-law waits
costs one batch of draws per renewal generation.

Fixed-step steppers on groups whose members interact cannot stop every member at its own time without splitting the
step for all of them. For such groups the stepper splits the step at every distinct renewal time inside it. Groups whose
members do not interact, which the compiler detects from the absence of group sums and pair interactions, are advanced
with a per-member step array instead, so no global split is needed. `RK4` and `EulerMaruyama` take both paths; other
steppers reject renewal laws with an error that names them. Renewal laws read the states and parameters of the renewing
member, not reductions, and they do not combine with delays or events in one system. A component with renewal laws
alone beside components with other laws makes the system run as partitions, so renewal laws fold into the form of
rate and noise laws only within the same component.

Alternatives considered: extending `Jump` with a waiting-time distribution, which would make `Gillespie` non-Markovian
and break its exactness argument; and writing the renewal loop as a Mesa rule, which would move the models to the agent
form and lose compilation.

### Fractional noise laws

`FractionalNoise(target, value, hurst, order=None)` mirrors `Noise`. It compiles to the SDE form, and the problem
records for each fractional noise row its Hurst index. `hurst` is a number or a parameter. A Hurst index of exactly 1/2
lowers to an ordinary noise row, so every SDE stepper accepts it. For other values the compiler checks with Sage that
the amplitude does not depend on any state, since the Euler scheme `X += f dt + g ΔB_H` converges only for an additive
amplitude, which is all the inventory models need.

`FractionalEulerMaruyama(dt=1e-3, horizon=None)` is the stepper for the SDE form with fractional rows. On
initialisation it draws, per fractional row and member, the `ceil(horizon / dt)` increments of fractional Gaussian
noise with the Davies-Harte adapter, scaled by `dt^H`, and stores them in a buffer. Each step adds the drift, ordinary
Wiener increments for the ordinary noise rows, and the next stored fractional increment times the amplitude.
`Simulation.run` passes its end time as the horizon when the stepper has none, through the new no-op hook
`Stepper.plan(until)`. Advancing past the horizon raises
`HorizonError`, a `SolverError` subclass that names the horizon and the requested time. Other SDE steppers reject fractional rows with a
`SolverError` that names `FractionalEulerMaruyama`.

Alternatives considered: Hosking's sequential method, which extends a path without a horizon but costs `O(n²)` per
member; and a block-wise Davies-Harte with conditioning between blocks, which is exact but complex and only matters for
open-ended interactive runs, where a large horizon serves the same purpose.

### Compile pipeline, Numba, and runtime

- Sage layer: `draw` is a Sage symbolic function like the other named draws, `Renewal` and `FractionalNoise` are law
  dataclasses next to `Event` and `Noise`, and no SymPy is involved.
- Lowering: SciPy draws in kernels become buffer reads, renewal resets compile like event resets, and fractional
  rows add `FractionalRows`, the positions of the rows and their Hurst indices, to the problem beside `noise_rows`. The
fractional stepper loops in NumPy over the backend's compiled drift and diffusion with the stored increments, instead of
a Numba loop with an increments
  argument, which the per-step Python loop does not need. A SciPy draw in an update law is sampled into its buffer
  before every step in Python, so a map with one runs its compiled update step by step instead of the compiled loop.
- Runtime: `Simulation` seeds one `Generator` and hands it to the stepper, which owns the buffers, so two runs with the
  same random state draw the same values in the same order. Frame-by-frame runs, live parameter changes, and
  co-simulation work unchanged, except that a parameter change to a fractional amplitude takes effect at the next step,
  while a change of `hurst` after initialisation raises a named error, since the drawn path depends on it.
- Rerun: states stream as for any group. The renewal count is an observable, so the existing observable logging streams
  it as a scalar series, and renewal records stream as text log entries on the law's entity path with the member index.
- Experiments: ensembles over random states give the ensemble mean squared displacements that the acceptance scenarios
  check.

### Library models

The four models live in `skcomplex.models.physics.soft_matter`, a new module `_anomalous.py`, register under
`physics/soft_matter`, and use only the public names of `skcomplex.core`:
- continuous-time random walk: a group with a position, and a renewal law with Pareto waits that adds a normal draw;
- Lévy walk: a position with `Rate(x, v)` and a renewal law with Pareto flight times that redraws the velocity as
  `v0` times a random sign or unit vector;
- annealed transient time motion: a position with `Noise(x, sqrt(2 D))` and a renewal law that redraws `D` from a
  power-law density and waits a time whose mean scales as `D^(-γ)`;
- fractional Brownian motion: a position with `FractionalNoise(x, sqrt(2 D), hurst=H)`.

## Risks / Trade-offs

- [Pareto waits with tail exponent below 1 have no mean, so a few members renew rarely and others very often] → The
  renewal stepper processes members in vectorised generations, and the scenarios bound the run by time, not by the
  number of renewals.
- [Davies-Harte memory grows with the horizon times the group size] → The buffer holds `float64` increments, 8 MB per
  10⁶ member-steps, and the stepper raises a named error with the expected size when it exceeds a configurable limit.
- [Negative circulant eigenvalues for some Hurst indices and short paths] → The adapter falls back to the exact
  Cholesky method.
- [Renewal generations that fire many members at once draw in a different order than one member at a time] → The
  stepper always fires members in generations ordered by time, so the stream order depends only on the random state.
- [Archive order with `add-stochastic-maps`] → This change depends on it. If it is archived first, the widening of
  its draw rule becomes a modified requirement of `stochastic-maps` when this change is archived.

## Migration Plan

The change only adds names. Existing models and steppers keep their behaviour, and the inventory entries move from
`blocked add-anomalous-processes` to `implemented`.
