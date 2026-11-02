# Design

## Context

A field is `FieldElement(label, element=..., space=LatticeSpace(...), discretization=...)`, a group with one member
per site, whose laws are written for one site with `laplacian`, `gradient`, `divergence`, `advection`, and
`inverse_laplacian` reading the neighbouring sites. The compiler lowers them by finite differences, spectrally, or by
finite volumes through `FluxLaw`s and `FiniteVolumeKernel`. `LatticeSpace(shape, spacing=..., boundary=...)` has one
boundary for every axis, and every axis is a position.

The kinetic models of `add-kinetic-models`, now archived, use three general pieces this change reuses:
- `DensityElement`, whose state `f` is the value of a density at one class, held by a group on the sites of a lattice
  of classes,
- `EncounterInteraction(candidate, field, rate, kernel, outcome, state='f')`, which lowers to `EncounterLaw`,
  `DestructionLaw`, or `ProliferationLaw`, gain-loss rates over the classes with a dense rate `eta[h, k]` and kernel
  `kernel[h, k, i]`,
- `moment(expression, order, coordinates, over=None, normalized=False, central=False)`, a reduction over every member
  of a group with one coordinate per member.

`AdvectionInteraction(target, state, drift, source, limiter, riemann)` carries a state by a drift in conservative form,
lowering to a `FluxLaw` on finite-volume fields or to `-divergence(a u)` with `riemann=None` on spectral and
finite-difference ones. `RelaxationInteraction(target, state, goal, time)` relaxes a state towards a goal.
`ImageVisual` logs lattice states as images.

What is missing for kinetic equations:
- axes that are classes rather than positions, with their own coordinates and boundaries,
- coefficients that vary along an axis with its coordinate, such as the streaming speed `v`,
- reductions over some axes of a field into a field over the others,
- encounters at every position over the class axes, and collision integrals over continuous velocity grids.

## Goals / Non-Goals

**Goals:**

- Write a kinetic equation as Bellomo et al. (2017 §2.2–2.3) write it, transport plus collisions, from the general
  interactions, and get a discretized, split, compiled model without writing the discretization by hand.
- Read macroscopic quantities as moments, and couple them back as mean fields.
- Keep the core free of field terms: the axes are named by the user, and no core class names a velocity, an
  orientation, or a model.

**Non-Goals:**

- Direct simulation Monte Carlo in phase space. Particles with position and velocity are agents.
- The nonlocal Enskog collision, where the partners sit one diameter apart. It needs shifted reads, which
  `add-nonlocal-operators` adds. This change gives the local Enskog equation, a density factor on the collision
  frequency.
- Class axes on meshes and networks, and adaptive or non-uniform class grids.
- Three position dimensions with three velocity dimensions at useful resolution. The six-dimensional grid is beyond a
  single machine's memory, and the reference models need at most three axes in total plus two class axes.

## Decisions

### Library survey

The constitution requires the feature to wrap a mature library unless none fits. Surveyed in October 2026:

| Library | Version, licence | Wheels | Fit |
|---|---|---|---|
| `py-pde` | 0.60.0, MIT, active (October 2026) | pure Python, Python 3.12–3.15 | Rejected as the substrate |
| `pylbm` | 0.11.0, BSD-3-Clause (May 2026) | pure Python, compiles at run time | Rejected |
| `lbmpy` | 2.1.1, GPL-3.0-or-later (September 2026) | pure Python | Rejected |
| Dedalus | 3.0.5, GPL-3 (January 2026) | source only, builds against MPI and FFTW | Rejected |
| Struphy | 3.4.0, MIT-style (October 2026) | pure Python | Rejected |
| Gkeyll | C and Lua | not on PyPI | Rejected |
| SciPy `fft` and `ndimage` | SciPy, BSD-3-Clause | Linux and macOS, Python 3.12–3.13 | Wrapped |

Why each is not wrapped:
- `py-pde` solves PDEs on Cartesian, polar, spherical, and cylindrical grids with Numba-compiled operators. Its
  Numba Laplacian, gradient, and divergence raise `NotImplementedError` above three dimensions, so a 2D-2V phase space
  does not fit. It has no class axes, no moments over a subset of axes, and no collision operator, and its operators
  are built from SymPy expressions, which the project bans.
- `pylbm` and `lbmpy` implement lattice Boltzmann schemes with a few discrete velocities. They do not solve kinetic
  equations over a continuous velocity grid. `pylbm` also requires `mpi4py` and Cython and compiles code at run time,
  and `lbmpy` is GPL.
- Dedalus is GPL and ships no wheels.
- Struphy solves Vlasov-Maxwell plasma models by particle-in-cell and finite-element exterior calculus. It pins
  `numpy<=2.5`, `scipy<=1.18`, and run-time dependencies such as `pytest`, `ruff`, `vtk`, and `pyvista`, and it is
  specific to plasmas.
- No maintained Python library ships the fast spectral Boltzmann collision operator of Mouhot and Pareschi.

Decision: phase-space fields reuse the core's own field lowering, which already handles lattices of any dimension.
The new numerics wrap SciPy: `scipy.fft` for spectral transport and the fast spectral collision operator, and
`scipy.ndimage.map_coordinates` and `scipy.ndimage.shift` for semi-Lagrangian interpolation. The core holds only the
adapters, plus the assembly of the fast spectral collision weights, which no library provides.

### Phase spaces

`PhaseSpace(position, axes, *, limit=2**26, label='space')` in `skcomplex.core.spaces` takes a `LatticeSpace` and a
sequence of `AxisSpec(name, size, low, high, periodic=False)`, a frozen spec of `skcomplex.specs`. It is a lattice
space and behaves as the product lattice, with the position axes first, so `FieldElement` and the existing lowering
accept it unchanged. A class axis has the spacing `(high - low) / size` and its own boundary: zero flux for an interval
and periodic for `periodic=True`, such as angles on `[0, 2π)`. The position axes keep the boundary of the position
lattice. `prepare` derives `coordinates_`, one array per class axis at the cell centres, `position_axes_`, and
`class_axes_`, a mapping from axis name to its index. Sites stay in row-major order, so a slice at fixed position is
contiguous.

No velocity or orientation class exists, since the core names no quantity: the user names the axes, such as `'v'` or
`'theta'`.

### Coordinates

`coordinate(name)` in `skcomplex.core.laws.utils` returns a symbol that the lowering replaces by the coordinate array
of the named class axis, broadcast over the field. A drift is then written with the general classes alone:
- streaming is `AdvectionInteraction(target='gas', state='f', drift=lambda s: [coordinate('v'), 0])`,
- acceleration by a force `F` is the drift `[0, F]`, and both together are one drift `[coordinate('v'), F]`.

The finite-difference lowering of `upwind` already takes an expression as speed and reads it per site. The
finite-volume lowering reads the coordinate per cell, as `FiniteVolumeKernel` already reads parameters and states that
no flux conserves. The spectral lowering multiplies in physical space before the transform.

### Moments over class axes

`moment` gains `axes`, a sequence of axis names. With `axes` it reduces over those class axes at every position, with
the coordinates of the phase space and the cell widths as weights, and returns a field over the position axes that a
law reads broadcast back along the reduced axes. Without `axes` it keeps its current behaviour, so the kinetic models
of activity classes are unchanged. The weights make the discrete density, momentum, and energy exactly the quantities
that discrete transport and the maximum-entropy equilibrium conserve. On a periodic axis the first and second moments
of `cos` and `sin` of the coordinate give the polarization and the nematic order, written with `coordinate` in the
expression. A moment over all class axes and then a reduction over positions is a scalar observable.

### Encounters at every position

`EncounterInteraction` keeps its signature. When its candidate and field are on a `PhaseSpace`, the classes `h`, `k`,
and `i` are the sites of the class axes, and encounters pair classes at the same position. The rate and the kernel may
read `coordinate`. The kernels of `EncounterLaw`, `DestructionLaw`, and `ProliferationLaw` batch over positions. The
role `field` becomes optional, `RoleSpec(optional=True)`, and without it the encounter is linear: a candidate at class
`h` moves to class `i` at the rate `eta[h]` with the probability `kernel[h, i]`. This is how the angular diffusion of
Bertin, Droz, and Grégoire runs, while their binary alignment is a conservative encounter of the density with itself
whose kernel puts the pair at its mean angle plus noise. Both are thin subclasses in `physics/active_matter`,
`AngularAlignmentInteraction` and `AngularDiffusionInteraction`. An orientation grid of tens of angles keeps the dense
kernel small.

### The binary collision integral

On a continuous velocity grid of `N` points the dense kernel of an encounter has `N³` entries, a billion for a 32 by
32 grid, and the outcome of a collision is fixed by its geometry, not drawn from a table. So the Boltzmann integral is
its own interaction:
- `CollisionInteraction(label, *, target, state='f', cross_section, factor=None)` in `skcomplex.core.interactions`
  takes the cross-section as a function of the relative speed and the cosine of the deflection angle, and the factor
  as a function of the namespace, such as a function of `moment(f, 0, axes=...)` for the local Enskog equation.
- It lowers to `CollisionLaw(target, cross_section, factor)` in `skcomplex.core.laws`, a nonlinear rate of the law
  kind `rate` with `RequirementSpec(time=(ContinuousTime,), space=(PhaseSpace,))`.
- The compiler assembles the weights of the fast spectral method of Mouhot and Pareschi once per cross-section and
  grid, padded by the factor they give, and each evaluation is a set of FFT products on `scipy.fft`.

The collision integral is a fundamental paradigm of kinetic theory and applies to any density over continuous
classes, so it enters the core.

### Maximum-entropy equilibrium

`maximum_entropy(expression, axes, order=2)` in `skcomplex.core.laws.utils` returns the density
`exp(a + b · c + d |c|²)` on the named class axes, with `c` their coordinates, whose discrete moments up to the order
equal those of the expression at every position. The lowering finds its parameters by a few Newton steps per position,
after Mieussens' discrete-velocity construction, so a relaxation towards it conserves the moments to round-off, which a
plain continuous Maxwellian on a truncated grid does not. The BGK model is then
`RelaxationInteraction(target='gas', state='f', goal=lambda s: maximum_entropy(s.target.f, axes=('v',)), time=...)`,
written as the thin subclass `BhatnagarGrossKrookInteraction` in `physics/fluids`, which names its authors in the model
library and not in the core.

### Mean-field coupling

A drift may read a moment of its own field, so the Vlasov-Poisson acceleration is the drift component
`-gradient(inverse_laplacian(1 - moment(f, 0, axes=('v',))))` along the velocity axis, where the operators of a field
over position act along the position axes. A drift may also read another field over the position lattice through the
new `sources` roles of `ConservationInteraction` and `AdvectionInteraction`, variadic `RoleSpec`s as on
`RateInteraction`, broadcast along the class axes. A law over the position lattice reads a moment of a phase-space
field through the `sources` of `RateInteraction`.

### Solvers

- `SemiLagrangianSolver(dt=0.01, random_state=None, until=inf)` advances fields whose fluxes are linear drifts, `a u`
  with `a` free of the state and constant along its own axis, by shifting each one-dimensional slice along its
  characteristic in the directional splitting of Cheng and Knorr. Constant shifts on periodic axes use the Fourier
  shift theorem through `scipy.fft`, and other shifts use cubic-spline interpolation through
  `scipy.ndimage.map_coordinates`. It is stable at any Courant number and conserves the total on periodic axes. A
  compile pass records the drift of every axis of such a field on the problem.
- `StrangSplittingSolver(transport, collision)` composes a half transport step, a full collision step, and a half
  transport step. The transport solver advances the flux differences and the collision solver the source rates, which
  `ProblemSpec` keeps apart from the flux differences, a split this change adds.

Both live in `skcomplex.engine.solvers` and declare:

```text
SemiLagrangianSolver:  SupportSpec(time={ContinuousTime}, space={PhaseSpace, LatticeSpace}, laws={FluxLaw, RateLaw})
StrangSplittingSolver: SupportSpec(time={ContinuousTime}, space={PhaseSpace, LatticeSpace},
                                   laws={FluxLaw, RateLaw, EncounterLaw, CollisionLaw})
```

`StrangSplittingSolver` checks its two inner solvers against the laws each advances, and `SemiLagrangianSolver`
rejects in `_prepare` a flux that is not a linear drift, naming the field.

### Compile report

The compiler reports:
- a phase space with more sites than its `limit`, as `space-size`, with the size it would take,
- a `coordinate` or a `moment` with `axes` in a law whose group is not on a `PhaseSpace`, or that names an axis the
  space does not hold, as `law-space`, with the axes it holds,
- a `CollisionInteraction` on a phase space with a periodic or a missing class axis, as `law-space`, with the fix of
  declaring an interval `AxisSpec`.

### Visualization

`ImageVisual` gains `slices`, a mapping from axis to a fixed index for every axis but two, so a phase-space field logs
its position-velocity plane, and `moments`, the orders of the moments over the class axes it logs as series over a
one-dimensional position lattice or as images over a two-dimensional one.

## Risks / Trade-offs

- [Memory grows with the product of all axes] → The reference models use at most four axes. The compiler refuses a
  phase space larger than its limit with the size it would take.
- [Fast spectral collisions alias on a truncated velocity grid] → The grid is padded by the factor Mouhot and Pareschi
  give, and the BKW scenario checks accuracy.
- [Semi-Lagrangian interpolation adds diffusion and small negative values] → Cubic splines keep the diffusion small,
  and a filamentation scenario in the tests bounds it. Positivity is not enforced, as in the references, and the
  domain of `DensityElement` is checked only by checked runs.
- [The fast spectral weights are in-house code] → They are the one piece no surveyed library provides. They are tested
  against the BKW exact solution and against direct quadrature of the collision integral on a coarse grid.

## Open Questions

- None. Angular collisions are encounters in the model library, and BGK is a relaxation named in the model library.
