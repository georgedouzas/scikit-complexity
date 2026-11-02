# Design

## Context

See proposal.md for the models. The current core has:
- `FieldElement(label, element=, space=, boundary=, discretization=, symmetry=)` in `skcomplex.core.elements`, whose
  laws are written for one site. A state takes `'periodic'`, `'zero-flux'`, or `('fixed', value)` at the edges.
- `StateSpec(vector=True)` in `skcomplex.specs`, a state with one component per dimension of the lattice, which laws
  read as a Sage vector.
- The law utilities of `skcomplex.core.laws.utils`: `partial`, `gradient(expression, n_dim)`, `divergence`,
  `advection`, `laplacian`, `upwind`, and the symbolic `inverse_laplacian`. The compiler lowers them to stencils on
  lattices, to Fourier multipliers on spectral fields of periodic lattices, and to finite volumes and elements.
- `GradientInteraction(target, states, potential, variational=True)`, the gradient flow of a free energy on a field,
  and `RateInteraction`, a rate on a state, in `skcomplex.core.interactions`. `physics/common/fields` holds
  `LandauEnergyInteraction`, a thin subclass for a scalar order parameter.
- Validation through `PassCompiler.prepare(system).report_`, a `ValidationReportSpec` of `IssueSpec`s with a code, a
  path, a message, and a fix. A `ValidationError` raised while laws are collected becomes an issue.
- `AlignmentAnalysis` with the kinds `'mean-vector'`, `'mean-rotation'`, `'axial'`, and `'mean-phase'`, and
  `ImageVisual` and `RerunObserver` in `skcomplex.engine.viz`.
- `DomainKind.TENSOR` in `skcomplex.types`, which no state uses yet.

## Goals / Non-Goals

**Goals:**
- Write the equations of 1207.2929 §II.2 and §III with tensors as they appear in the reference, not component by
  component.
- Keep every tensor law a Sage expression of the existing operators, so the stencils, the spectral multipliers, the
  stiff splitting solvers, and the Numba backend apply to it unchanged.
- Keep the core free of liquid-crystal names: the Landau-de Gennes energy and the active stresses live in the model
  package.

**Non-Goals:**
- Tensors of rank 3 or more, and curvilinear grids.
- Incompressible flow between walls that are not a parity of a periodic lattice. The screened Poisson inversion on
  bounded lattices belongs to `add-nonlocal-operators`, and `leray_projection` uses its `inverse_laplacian` there once
  it lands.
- New solvers or backend support. Tensor components are real states.

## Decisions

- **Decided on 2026-11-02.** Anchoring: a Laplacian relaxes each component linearly, so perpendicular walls would hold
  the axis fixed and then jump it by 90 degrees, and a boundary gives every edge one value; the scenario relaxes to one
  fixed matrix, and per-edge tensor boundaries wait until a model needs them. Turbulence: `active_gels` at `K=1`,
  `zeta=1`, `xi=0` on 32 by 32 sites bursts and realigns without defects, so its turbulence defaults come from the
  source's parameters, never from a sweep. Field operators: identical applications are computed once per rate evaluation,
  since a 32 by 32 gel otherwise makes 623 Fourier applications from 37 reads. The vector `corotational` is
  `u.grad(p) - Omega p - xi E p` with `Omega = antisymmetric_part(vector_gradient(u))`, the sign that rotates `p` with
  the flow and agrees with the tensor form.

### Library survey

| Library | Fit | Decision |
|---|---|---|
| `py-pde` 0.58 (MIT) | tensor fields and operators on Cartesian grids, Poisson solves on bounded grids | rejected: the existing stencils, spectral multipliers, and per-state boundaries already give every operator once a tensor is expanded into components, and a second operator implementation would duplicate them |
| Dedalus 3 (GPL-3.0) | spectral tensor calculus | rejected: no wheels, needs MPI and FFTW, copyleft |
| `shenfun` 4.3 (BSD) | spectral Galerkin tensor spaces | rejected: source only, needs `mpi4py-fft` |
| Sage tensors on manifolds | symbolic tensors in general coordinates | rejected: the operators here are Cartesian, and the chart machinery slows the expansion of laws. Sage matrices, `matrix(SR, ...)`, carry the algebra |

The change adds no dependency. The tensor algebra uses Sage matrices, and the numerics are the existing lowerings.

### Tensor states

`StateSpec(tensor='traceless')` declares a rank-2 tensor state, and `StateSpec(vector=True)` stays the rank-1 case. The
field `tensor` holds the symmetry: `'full'` with `d²` components, `'symmetric'` with `d (d + 1) / 2`, `'antisymmetric'`
with `d (d - 1) / 2`, and `'traceless'` with `d (d + 1) / 2 - 1`, for `d` the dimension of the lattice. `EntrySpec`
expands the state into independent component entries named by index, `Q_xx` and `Q_xy` in two dimensions, as it
expands a vector state, and reports their kind as `DomainKind.TENSOR`. The namespace gives `s.Q` as a Sage matrix
whose dependent entries are expressions of the independent ones, such as `Q_yy = -Q_xx`. A `RateLaw` on a tensor state
gives one rate per independent component, read from the matrix entries of its value. A state of `vector=True` and a
`tensor` symmetry together raise a `ValidationError` in `StateSpec.check`.

`FieldElement` takes a matrix as the value of `('fixed', value)` for a tensor state and expands it into the fixed value
of every component. A matrix that does not have the declared symmetry raises a `ValidationError` naming the state.

### The symmetry check

`PassCompiler` checks every rate law on a tensor state: it simplifies the dependent entries of the rate against the
independent ones, such as the trace for `'traceless'` and `R - Rᵀ` for `'symmetric'`, and reports a mismatch as an
`IssueSpec` with the code `tensor-symmetry`, the path of the law, the state, and the entry or trace that breaks it.
When Sage cannot simplify an entry to zero, the check compares the entries at random component values and names the
method that decided in the message.

### Tensor operations

The new utilities live in a module `_tensors` of `skcomplex.core.laws.utils` beside `_operators`:
- `vector_gradient(v)` is the matrix `∂_j v_i`, built from `partial`.
- `tensor_divergence(T)` is the vector `∂_j T_ij`, built from `partial`.
- `symmetric_part(M)` and `antisymmetric_part(M)` are `(M + Mᵀ) / 2` and `(M - Mᵀ) / 2`, so the strain rate of a
  velocity is `symmetric_part(vector_gradient(u))` and its vorticity tensor `antisymmetric_part(vector_gradient(u))`.
- `laplacian` of a matrix or a vector applies entry by entry, which Sage's `apply_map` gives with no new function.

Each returns an expression of `partial` and `laplacian` calls, so the stiff splitting solvers still find the linear
elastic term `K ∇²Q` and the Numba backend still fuses the kernel. A tensor utility given an expression of the wrong
shape, such as `tensor_divergence` of a scalar, raises a `ValidationError` that names the utility and the shape, which
the compiler reports with the existing code `law-domain`.

### Objective derivatives

`corotational(T, u, xi=0)` returns `u·∇T - S`, the Beris-Edwards transport of a symmetric tensor, with
`S = (xi E + Ω)(T + I/d) + (T + I/d)(xi E - Ω) - 2 xi (T + I/d) tr(T ∇u)`, the strain rate `E`, and the vorticity
tensor `Ω`. For a vector `p` it returns `u·∇p + Ω p - xi E p`. `upper_convected(T, u)` returns
`u·∇T - (∇u)ᵀ T - T ∇u`. Both are expressions, so a law reads
`RateLaw(s.Q, -corotational(s.Q, u, s.xi) + H / s.gamma)`.

### Leray projection

`leray_projection(v)` returns `v - gradient(inverse_laplacian(divergence(v)))`. On a spectral field of a periodic
lattice the compiler already lowers `inverse_laplacian` to the multiplier `-1/k²`, so the projection is the multiplier
`I - k kᵀ / k²`, exact in two and three dimensions. With the `symmetry` setting of `FieldElement`, a velocity odd about
the middle of an axis gives no-slip walls there, as the convection models give free-slip walls today. The overdamped
Stokes velocity of a force density `f` with viscosity `η` is `leray_projection(inverse_laplacian(-f / η))`, which the
models write themselves, so the core names no flow.

### Analysis and visualization

- `AlignmentAnalysis(kind='principal-axis')` reads a field of a symmetric tensor state from a trajectory, or an array
  of components, and sets `axes_`, the eigenvector of the largest eigenvalue at every site, and `order_`, the scalar
  order, by NumPy's batched `eigh`.
- `DefectAnalysis(period=np.pi)` reads a field of axis angles and finds point defects by the winding of the angle around
  every plaquette, `period=np.pi` for axes without head and tail and `2 * np.pi` for vectors. It sets `positions_` and
  `charges_` per frame. No library detects defects on lattice axis fields, and the computation is a few array
  operations.
- `AxisVisual(state, scalar=None, every=4)` logs Rerun `LineStrips2D` along the principal axis of a tensor state, or
  `Arrows2D` for a vector state, at every `every`-th site, over the image of `scalar` that `ImageVisual` would log.

### Models

Both factories live in `skcomplex.models.physics.active_matter` and use only public names:
- `LandauDeGennesEnergyInteraction(GradientInteraction)` in that package gives the molecular field
  `H = -a Q - c tr(Q²) Q + K ∇²Q` as the variational derivative of the Landau-de Gennes energy, projected onto the
  symmetric traceless part. A uniform two-dimensional state relaxes to the scalar order `sqrt(-2 a / c)`.
- `dry_active_nematics` is a `FieldElement` with a density and `Q`, the density conserved by a `ConservationInteraction`
  whose flux holds the active current along `tensor_divergence(Q)`.
- `active_gels(order='nematic' | 'polar')` transports `Q` or `p` by `corotational` in the overdamped Stokes velocity of
  the elastic and active stresses, `-ζ Q` for the nematic and `-ζ p p` for the polar gel, written with
  `leray_projection`. 1207.2929 states no numbers for its 2D turbulence, so the nematic defaults are those of its
  citation Giomi et al. 2012 (arXiv 1110.4338), whose Landau-de Gennes energy has the same form: side 10, `c* = 3π/2`,
  `c0 = 2 c*`, `a = (c* - c0) / 2`, `c = c0`, `zeta = 3 c0²`, `xi = 0.1`, `K = gamma = eta = 1`, with a uniform density.

## Risks / Trade-offs

- [The symmetry check cannot simplify an entry] → It falls back to a numeric comparison and names the method.
- [Active turbulence is chaotic, so trajectories differ across platforms] → The scenarios check conserved quantities,
  charges, and statistics, not trajectories.
- [Walls only as parities of periodic lattices] → It covers the film geometries of the models, and bounded inversion
  comes with `add-nonlocal-operators`.
