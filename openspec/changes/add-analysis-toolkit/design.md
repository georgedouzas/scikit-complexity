# Design

## Context

Compiled problems keep symbolic equations as Sage expressions, so derivatives are free. Trajectories hold states,
agents, and observables per frame, and experiments, after `add-experiments`, provide sweeps and ensembles.

## Goals / Non-Goals

**Goals:**

- Every measure takes plain arrays or trajectories, so it applies to simulated and measured data alike, as Datseris and
  Parlitz (2022 ch. 6) separate observation from state.
- Known values from the references act as acceptance tests.

**Non-Goals:**

- Numerical continuation of bifurcations with arclength methods. Orbit diagrams over sweeps cover the references'
  examples, and continuation can follow.
- Delay embedding and recurrence quantification (Datseris and Parlitz 2022 ch. 6, 8). They are recorded for a later
  change.

## Decisions

### Symbolic dynamics

`fixed_points(system, parameters=None)` solves `f(x) = 0`, or `f(x) = x` for maps, with Sage's `solve`, and falls back
to numeric root finding from a grid of starts when Sage returns no closed form. `jacobian(system)` returns the Sage
matrix of the drift, `stability(system, point)` classifies the point by its eigenvalues, `nullclines` returns the zero
sets per component, and `divergence` returns the trace of the Jacobian, which is zero for volume-preserving flows.

### Lyapunov exponents

`lyapunov_spectrum(system, n=None, until, dt, transient)` integrates the tangent dynamics with the compiled Jacobian
and re-orthonormalises with QR every step, as in Datseris and Parlitz (2022 ch. 3). Maps use the same method with the
Jacobian of the map. The largest exponent of Lorenz-63 with the standard parameters is about 0.906 and serves as the
acceptance test.

### Bifurcation and basins

`orbit_diagram(source, parameter, values, observable, transient, keep)` and `basin_stability(source, initial_bounds,
n, classifier)` run experiments and reduce them, so they inherit parallelism and seeds.

### Statistics

Power-law fits follow Clauset, Shalizi, and Newman: the exponent by maximum likelihood, `xmin` by minimising the
Kolmogorov-Smirnov distance, and a bootstrap p-value, with likelihood-ratio comparisons against exponential and
log-normal fits, as Charbonneau (2017 §4.5) and Caiani et al. (2016 ch. 4) stress.

### Information

Discrete measures take symbol sequences or lattices, and `symbolize(series, method='bins' | 'quantiles' | 'ordinal')`
produces them. Block entropies follow Lindgren (2024 ch. 3): `S_m`, the entropy rate as the limit of `S_{m+1} - S_m`,
the correlation information `k_m`, and the excess entropy, in one and two dimensions. Transfer entropy uses plug-in
estimates on symbols. Continuous estimators beyond binning are left to a later change.

### Spatial and network measures

Cluster labelling uses `scipy.ndimage.label` on grids with periodic merging, and network measures wrap NetworkX on
snapshots of network spaces, so dynamic networks from later changes are measured the same way.

## Risks / Trade-offs

- [Breadth] → Each measure is small and tested against a known value from the references, and the toolkit grows by
  submodule.
- [Plug-in entropy bias] → The functions document their bias, and block entropies warn when blocks exceed the data.

## Implementation Notes

- The measures live in private modules of `skcomplex.analysis` and are re-exported from the package, as the
  constitution requires, rather than in public submodules.
- `lyapunov_spectrum` prints the vector field and its Jacobian, with parameter values substituted, through the scalar
  dialect and compiles them with Numba, so the Benettin integration runs compiled. It reproduces Lorenz's 0.906, ln 2
  for the logistic map at 4, and the Hénon map's 0.419 and -1.62.
- Stability classification evaluates eigenvalues at the parameter values, so symbolic eigenvalues such as
  `±i sqrt(r d)` classify reliably as a centre.
- Power-law cutoffs are chosen among sample values, so discrete fits use integer cutoffs with the Hurwitz zeta
  normalisation.
- Basin stability through experiments inherits the cost of the frame-by-frame ODE runtime, about a second per long
  run with SciPy at small steps, so examples use coarse frames. A faster stepper path would help long ensembles.
- The `register_model` docstring example registers under the field `examples`, so it never changes the library's
  fields in tests.
