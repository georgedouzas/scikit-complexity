# Design

## Context

After `redesign-system-model`, a site group, `Group(on='sites')` on a `LatticeSpace`, holds one value per site for every
state of its element, as the former `Field` did. Laws are Sage expressions written for one site, and the operators
`partial`, `laplacian`, `upwind`, and `inverse_laplacian` stay symbolic until `skcomplex.core.compile` lowers them. The
redesign's inventory gate added the state domain `Tensor(rank, symmetry)`, of kind `DomainKind.TENSOR`, which
validation reads but which nothing stores or lowers yet. Finite differences become stencils on lattices and the graph Laplacian on networks. Spectral fields on
periodic lattices become Fourier multipliers, and a global operator such as `inverse_laplacian` becomes a derived read
that the kernel evaluates once per rate evaluation. The stiff steppers split the linear part of a law from the
remainder, and the Numba backend fuses the remainder into one kernel.

A state declared with `vector=True` already has one component per dimension. The two-dimensional fluid models build
their velocity from a streamfunction with `inverse_laplacian`, and walls come from odd or even extensions of a periodic
lattice. See proposal.md for the models this change unblocks.

## Goals / Non-Goals

**Goals:**

- Write the equations of 1207.2929 §II.2 and §III with tensors as they appear in the reference, not component by
  component.
- Keep every tensor law a Sage expression, so the compile pipeline, the stiff split, the Numba backend, and the
  symbolic analysis apply to it unchanged.
- Build the numerics of bounded domains on a maintained library instead of new in-house solvers.

**Non-Goals:**

- Curvilinear grids, such as polar or spherical coordinates, and tensors on curved surfaces. `py-pde` has these grids,
  so a later change can expose them.
- Tensors of rank 3 or above, and tensor states on networks or unstructured meshes.
- Compressible flow and the full Navier-Stokes equations in three dimensions with walls. The Leray projection covers
  periodic inertial flow.
- Replacing the in-house finite-difference and spectral operators of scalar fields. The audit in
  `organize-model-library` owns that comparison.

## Decisions

### Library survey

| Library | Version | License | Wheels for Linux and macOS, Python 3.12 and 3.13 | Fit |
|---|---|---|---|---|
| `py-pde` | 0.58 (July 2026), monthly releases | MIT | pure Python `py3-none-any`, Python 3.12 to 3.15 | chosen |
| Dedalus | 3.0.5 (January 2026) | GPL-3.0 | none, source only, needs MPI and FFTW at build time | rejected |
| `shenfun` | 4.3.0 (June 2026) | BSD | none, source only, needs `mpi4py-fft` | rejected |
| FEniCSx | current | LGPL-3.0 | distributed through conda and its own installers, bound to MPI | rejected |
| Sage tensors on manifolds | in `passagemath` | GPL-2.0 or later | already installed | rejected for components |

- `py-pde` provides scalar, vector, and rank-2 tensor fields on Cartesian and curvilinear grids, and compiles the
  operators `gradient`, `divergence`, `vector_gradient`, `vector_laplace`, and `tensor_divergence` with Numba through
  `grid.make_operator(name, bc)`. It handles boundary conditions for every component, including tensor values at
  Dirichlet edges, and solves Poisson equations on bounded grids with SciPy sparse solvers. Its dependencies, NumPy 2,
  SciPy, Numba, SymPy, matplotlib, and `tqdm`, are already in the environment. It imports SymPy internally, which the
  `ruff` ban on SymPy in `skcomplex` does not reach.
- Dedalus has the best spectral tensor calculus and IMEX schemes, but it ships no wheels, needs an MPI and FFTW tool
  chain at install time, and its GPL licence would bind the MIT package. It also replaces the symbolic layer with its
  own equation parser, so it would fit only as an optional external backend in a later change, like the LAMMPS backend.
- `shenfun` has spectral Galerkin tensor spaces, but ships only a source distribution with MPI dependencies.
- FEniCSx solves Stokes and tensor problems on unstructured meshes, but it installs through conda with MPI, and
  `scikit-fem` already covers meshes in the core.
- Sage's manifold tensors handle general coordinates symbolically, but every operator here is on Cartesian components,
  and their chart machinery slows the expansion of every law. The tensor algebra uses Sage symbolic matrices and
  vectors, `matrix(SR, ...)` and `vector(SR, ...)`, which the symbolic layer already loads.

The change therefore wraps `py-pde` for the operators and solves on bounded finite-difference lattices, and keeps the
existing NumPy FFT path for spectral fields, where the divergence of a tensor and the Leray projection are Fourier
multipliers that the spectral discretization already applies. The core holds only the adapter that lowers symbolic
operator calls to `py-pde` grids, fields, and compiled operators. The adapter serves the NumPy and Numba backends alike,
so it is a private module of `core.backends` that both import, and `core.compile` stays free of backends.

### Tensor states

`state(domain=Tensor(2, 'traceless'))` declares a symmetric traceless tensor state with the redesign's existing domain,
and a rank-1 tensor is the existing vector state, `state(vector=True)`. The declaration expands into independent
component states named by index, such as `Q_xx` and `Q_xy` in two dimensions, the way `order` expands into
`<name>_d1` companions. The symbols namespace returns `s.Q` as a Sage matrix whose dependent entries are expressions of
the independent ones, such as `Q_yy = -Q_xx`. A rate law on a tensor state gives one law per independent component.

Validation checks that the right-hand side keeps the declared symmetry by simplifying the dependent entries against the
independent ones, and reports a mismatch with the code `tensor-symmetry`, naming the law, the state, and the offending
entry or trace. It also reports a tensor state outside a site group of a lattice, with the existing code `law-space`,
since tensors on networks and meshes are a non-goal.

### Declarations

- The ODE solvers, the stiff splitting solvers, and the spectral discretization add `DomainKind.TENSOR` to their
  declared domains, so `validate()` accepts rate laws on tensor states with them and lists them when another solver is
  chosen, with `solver-unsupported`.
- The NumPy and Numba backends add `DomainKind.TENSOR`, and the `py-pde` adapter declares that it serves bounded
  finite-difference lattices only.
- The tensor operators, objective derivatives, and incompressible velocities declare
  `Requires(space={SpaceKind.LATTICE}, domains={DomainKind.VECTOR, DomainKind.TENSOR})`, so a call on a scalar state or
  outside a lattice is reported with `law-domain` or `law-space`.

### Tensor operators

`vector_gradient`, `tensor_divergence`, `strain_rate`, `vorticity_tensor`, in `core.laws` beside the existing
differential operators, and the existing `laplacian` applied entry by entry expand into calls of the existing `partial` and `laplacian` functions on components. This keeps one symbolic
form for scalar and tensor laws, so the stiff split still finds the linear elastic term `K ∇²Q`, and the Numba backend
still fuses the kernel. On a bounded finite-difference lattice with tensor boundary conditions, the compiler groups the
component calls of one operator on one state into a single derived read, evaluated by the `py-pde` operator of the
same name on a `CartesianGrid` with the site group's boundary conditions translated: `'periodic'` to periodic axes,
`'zero-flux'` to zero derivative, and `('fixed', value)` to a Dirichlet value per component.

### Objective derivatives

`corotational(T, u, xi=0)` returns the advection of `T` minus the Beris-Edwards term `S(∇u, T)`, which rotates `T`
with the vorticity tensor and aligns it with the strain rate with the parameter `xi`. For a vector it returns
`u·∇p + Ω·p - xi E·p`. `upper_convected(T, u)` returns `u·∇T - (∇u)ᵀ·T - T·∇u`. Both are plain functions returning
tensor expressions, so a law reads `Rate(s.Q, -corotational(s.Q, u, s.xi) + H / s.gamma)`.

### Incompressible velocities

`stokes_velocity(force, viscosity, friction=0)` returns a velocity vector whose components are derived reads. On a
spectral periodic field it applies the Fourier multiplier `(I - k kᵀ / k²) / (viscosity k² + friction)` to the force,
which is the Stokes solve and the projection in one step, in two and three dimensions. On a bounded finite-difference
lattice in two dimensions it solves `∇²ψ = -ω` and `viscosity ∇²ω - friction ω = -curl f` with `py-pde`'s Poisson
solver and no-slip or free-slip walls, and returns the velocity from `ψ`. `project(rate)` applies the Leray multiplier
alone, for inertial flow with velocity states on periodic lattices. Free-slip walls on spectral fields keep using the
existing `symmetry` extensions.

### Analysis and visualization

`principal_axis(field, state)` returns the largest eigenvalue and its eigenvector of a symmetric tensor at every site,
with NumPy's batched `eigh`. `axis_defects(angle, period)` finds point defects by the winding of the axis angle around
every plaquette, with `period=π` for nematic axes and `2π` for polar vectors. No library provides defect detection on
lattice axis fields, and the computation is a few array operations. `AxisVisual` in `core.viz` logs Rerun
`LineStrips2D` for headless axes or `Arrows2D` for vectors at a sampled subset of sites, over the existing image visual
of a scalar. It registers under the kind `'axis'`. The name describes what it draws, rather than the liquid-crystal
term director, to keep the core application agnostic.

### Runtime

The new states are ordinary component states of site groups, so trajectories, observers, checkpoints, experiments, and
the catalog need no change. Derived reads from `py-pde` run in the rate evaluation of the NumPy and Numba backends, and the explicit
stability check gains the elastic diffusivity of tensor laws.

### Models

The three factories live in `skcomplex.models.physics.active_matter` and use only public core names. Each returns a
`System` with site groups on a `LatticeSpace` and `ContinuousTime()`. The Landau-de Gennes molecular field is a
realization of the gel models, so it lives in their package, not in the core. The gel models use the molecular field `H = -a Q - c tr(Q²) Q + K ∇²Q`, so a uniform two-dimensional state relaxes to
the scalar order `sqrt(-2 a / c)`. The active stress is `-ζ Q` for the nematic and `-ζ p p` for the polar gel.

## Risks / Trade-offs

- [Derived reads from `py-pde` break the fused Numba kernel into several calls] → Each operator is one compiled call
  per rate evaluation on whole arrays, and only bounded lattices with tensor boundaries use it. Spectral and periodic
  fields keep the fused kernel.
- [The symmetry check cannot simplify a law to the expected form] → The check falls back to a numeric comparison at
  random component values and reports which method decided, as the conservation checks of `add-variational-mechanics`
  do.
- [A new dependency on a single-group project] → `py-pde` is MIT, released monthly, and published in JOSS, and the
  adapter is one module, so a replacement touches the adapter alone.
- [Active turbulence is chaotic, so trajectories are not comparable across platforms] → The model scenarios check
  conserved quantities and statistics, not trajectories.
