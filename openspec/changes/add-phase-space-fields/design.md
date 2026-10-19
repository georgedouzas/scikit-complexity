# Design

## Context

A `Field` is a group over a `LatticeSpace`, a `NetworkSpace`, or a `Mesh`. Its laws are Sage expressions written for
one site, with `laplacian`, `gradient`, `divergence`, and `advection` reading neighbouring sites. The compiler lowers
them to NumPy and Numba by finite differences, spectrally, by finite volumes with `Flux` laws, or by finite elements.
The field lowering already handles lattices of any dimension, the stiff steppers split linear spectral parts from
nonlinear remainders, and `ImageVisual` streams two-dimensional states to Rerun.

What is missing for kinetic equations (see proposal.md, Why):
- axes that are velocities or orientations rather than positions;
- coefficients that vary along an axis with the coordinate, such as the streaming speed `v`;
- reductions over some axes of a field into a field over the others;
- collision operators, which are integrals over the velocity axes at a fixed position.

### Boundary with `add-kinetic-models`

`add-kinetic-models` holds distributions over a discrete activity variable, without position or on network nodes,
changed by encounter laws that compile to gain-loss ODEs, and lists spatial kinetic models with velocity and position
as a non-goal. This change holds fields over a continuous position lattice times velocity or orientation axes,
changed by transport and collision laws. The two do not overlap:
- an activity grid is not a velocity axis, because nothing streams along it, and encounter tables are not collision
  integrals, because they conserve no momentum or energy;
- this change adds no encounter law, and `add-kinetic-models` adds no transport or collision law;
- the names differ. `add-kinetic-models` owns `Distribution`, `Encounter`, `Proliferation`, `Destruction`,
  `Relaxation`, `ExternalAction`, and `ParticleKinetics`, and this change uses none of them;
- the moment observables of both are quadratures. Whichever change lands second reuses the first one's quadrature
  helper, with unit weights for an activity grid, so the fact lives in one place.

The discrete-velocity Boltzmann model (Bellomo et al. 2017 §2.3.2), with a handful of velocities, stays an ordinary
system of fields, as the inventory already records.

## Goals / Non-Goals

**Goals:**

- Write a kinetic equation as Bellomo et al. (2017 §2.2–2.3) write it, transport plus collisions, and get a
  discretized, split, compiled model without writing the discretization by hand.
- Read macroscopic quantities as moments, and couple them back as mean fields.
- Keep the core domain-agnostic: velocity and orientation are axis kinds, and no operator names a model.

**Non-Goals:**

- Direct simulation Monte Carlo in phase space. Particles with position and velocity are agents, and
  `add-kinetic-models` owns particle sampling of distributions.
- The nonlocal Enskog collision, where the partners sit one diameter apart. It needs shifted reads, which
  `add-nonlocal-operators` adds. This change gives the local Enskog equation, a density factor on the Boltzmann
  collision frequency.
- Velocity axes on meshes and networks, and adaptive or non-uniform velocity grids.
- Three position dimensions with three velocity dimensions at useful resolution. The six-dimensional grid is beyond a
  single machine's memory, and the reference models need at most three axes in total plus two velocity axes.

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
  does not fit. It has no notion of a velocity axis, no moments over a subset of axes, and no collision operator, so
  the adapter would have to supply everything kinetic. Its operators are built from SymPy expressions, which the
  project bans, while the laws are Sage. Whether `py-pde` should replace the core's finite-difference and spectral
  lowering in general is the audit that `organize-model-library` records, and this change inherits its outcome.
- `pylbm` and `lbmpy` implement lattice Boltzmann schemes, which use a few discrete velocities to recover the Navier-
  Stokes equations. They do not solve kinetic equations over a continuous velocity grid. `pylbm` also requires
  `mpi4py` and Cython and generates and compiles code at run time, and `lbmpy` is GPL, which an MIT package cannot
  depend on.
- Dedalus is GPL and ships no wheels.
- Struphy solves Vlasov-Maxwell plasma models by particle-in-cell and finite-element exterior calculus. It pins
  `numpy<=2.5`, `scipy<=1.18`, and run-time dependencies such as `pytest`, `ruff`, `vtk`, and `pyvista`, which would
  constrain every user of the package, and it is specific to plasmas.
- No maintained Python library ships the fast spectral Boltzmann collision operator of Mouhot and Pareschi or the
  angular collision operators of Bertin, Droz, and Grégoire.

Decision: phase-space fields reuse the core's own field lowering, which already handles lattices of any dimension.
The new numerics wrap SciPy: `scipy.fft` for spectral transport, the fast spectral collision operator, and angular
convolutions, and `scipy.ndimage.map_coordinates` and `scipy.ndimage.shift` for semi-Lagrangian interpolation. The
core holds only the adapters, plus the assembly of the fast spectral collision weights, which no library provides.
No new dependency is declared.

### Phase spaces

`PhaseSpace(position, axes)` takes a `LatticeSpace` and a sequence of `VelocityAxis(bounds, size)` and
`OrientationAxis(size)`. It behaves as the product lattice, with the position axes first, so `Field` and the existing
lowering accept it unchanged: a velocity axis is a lattice axis with spacing `(high - low) / size` and zero-flux
edges, an orientation axis a periodic one with spacing `2 π / size`. The phase space records which axes are which,
and `coordinates` returns one array per axis. Sites stay in row-major order, so a slice at fixed position is
contiguous.

### Phase coordinates and transport

`s.v`, `s.v[i]`, and `s.theta` are symbols that the lowering replaces by the coordinate array of their axis,
broadcast over the field. `streaming(f)` expands to `advection(f, velocity)` along the position axes, with the
velocity coordinates as speeds, and to `advection(f, (v0 cos θ, v0 sin θ))` on orientation spaces with a declared
speed `v0`. `acceleration(f, force)` expands to `advection` along the velocity axes. Since both expand into existing
operators, finite-difference, finite-volume, and spectral discretizations work without new stencils. The only change
in the stencil lowering is that a speed may now be an array along another axis, which the upwind kernel reads per
site.

### Moments

`moment(f, order)` is a symbolic reduction over the velocity or orientation axes. It lowers to a weighted sum over the
trailing axes in NumPy and a loop in Numba. The weights are the cell widths, which make the discrete density,
momentum, and energy exactly the quantities that discrete transport and BGK conserve. On an orientation axis, the
first and second moments are the polarization `∫ e^{iθ} f` and the nematic order `∫ e^{2iθ} f`, returned as real
components. A moment is also an `Observable`, recorded in trajectories under the field's path.

### Collision operators

- `BGK(target, rate)` adds `rate (M[f] - f)`. `M[f]` is the discrete Maxwellian whose density, momentum, and energy
  equal those of `f` exactly, found by a few Newton steps on its three parameters per site, after Mieussens'
  discrete-velocity construction. A plain continuous Maxwellian would leak mass at every step on a truncated grid.
- `Collision(target, kernel, enskog=None)` adds the Boltzmann collision integral by the fast spectral method of
  Mouhot and Pareschi. The kernel is a symbolic function of the relative speed and angle. The compiler assembles the
  weights once per kernel and grid, and each evaluation is a set of FFT products on `scipy.fft`. The optional `enskog`
  expression of the density moment multiplies the result.
- `AngularCollision(target, alignment, diffusion, noise)` adds the self-diffusion and binary alignment operators of
  Bertin, Droz, and Grégoire. In angular Fourier modes they are products of the noise distribution's characteristic
  function and convolutions of mode pairs, so they lower to `scipy.fft` along the orientation axis.

Collision laws are nonlinear rate terms, so the method-of-lines steppers integrate them like any rate. They are kept
out of the linear part of the stiff splitting.

### Mean-field coupling

A law over a phase space may read a state of a field over its position lattice. The lowering broadcasts that state
along the velocity axes. A law over the position lattice may read `moment(...)` of a phase-space field in the same
system. Vlasov-Poisson is then a phase-space field whose acceleration reads an electric field `-gradient(phi)`, with
`phi = inverse_laplacian(1 - moment(f, 0))` on a spectral position field. No new solver is needed for the Poisson
step.

### Steppers

- `SemiLagrangian` advances transport by shifting each one-dimensional slice along its characteristic, in the
  directional splitting of Cheng and Knorr. Constant shifts on periodic axes use the Fourier shift theorem through
  `scipy.fft`, and other shifts use cubic-spline interpolation through `scipy.ndimage.map_coordinates`. It is stable
  at any Courant number and conserves the total on periodic axes.
- `StrangSplitting(transport, collision)` composes a half transport step, a full collision step, and a half transport
  step, each with its own stepper. Collision steps run per site, so the stepper vectorises them over positions.

Both register in the solver registry, so they are chosen by name like the existing steppers.

### Compile pipeline, runtime, and visualization

- Compile: phase coordinates, `streaming`, `acceleration`, and `moment` are Sage functions, like `laplacian`, so the
  symbolic layer, the stability check, and the printers see ordinary expressions. Collision operators are law objects,
  like `Flux`, that the lowering turns into calls to precomputed kernels.
- Runtime: a phase-space field is one state vector, so `Simulation`, trajectories, observers, experiments, and
  co-simulation handle it without change. Moments are observables and so reach every analysis tool.
- Visualization: `ImageVisual` gains a `slice` argument naming two axes and fixed indices for the rest, so a phase-space
  field streams as the position-velocity image. Moments over a one-dimensional lattice stream as series, and over a
  two-dimensional one as images.

## Risks / Trade-offs

- [Memory grows with the product of all axes] → The reference models use at most four axes. The documentation states
  the size of a phase space before it is built, and the compiler refuses one larger than a declared limit with the
  size it would take.
- [Fast spectral collisions alias on a truncated velocity grid] → The grid is padded by the factor Mouhot and Pareschi
  give, and the BKW scenario checks accuracy.
- [Semi-Lagrangian interpolation adds diffusion and small negative values] → Cubic splines keep the diffusion small,
  and a filamentation scenario in the tests bounds it. Positivity is not enforced, as in the references.
- [The fast spectral weights are in-house code] → They are the one piece no surveyed library provides. They are tested
  against the BKW exact solution and against direct quadrature of the collision integral on a coarse grid.
