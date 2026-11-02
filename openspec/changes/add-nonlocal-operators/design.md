# Design

## Context

- `FieldElement(label, element, space, discretization, ...)` in `skcomplex.core.elements` is a group on the sites of a
  lattice, the nodes of a network, or a mesh, whose laws are written for one site. Its discretizations are
  `'finite-difference'`, `'spectral'`, `'finite-volume'`, and `'finite-element'`, and every state takes the boundary
  `'periodic'`, `'zero-flux'`, or `('fixed', value)`.
- The operators `laplacian`, `gradient`, `divergence`, `advection`, `partial`, `upwind`, and `inverse_laplacian` are
  Sage functions of `skcomplex.core.laws.utils`.
- `FieldLowering` in `skcomplex.engine.compile` lowers them. `_shift` turns stencil reads into neighbour tables, and
  `_spectral` and `_finite_element` turn global operators into a new symbol whose value is a `FieldOperatorKernel` of
  `skcomplex.engine.problems`: a Fourier multiplier or a sparse matrix applied to the operand at every site.
  `inverse_laplacian` is a spectral multiplier only. A stencil cannot read a `FieldOperatorKernel` at a neighbour, so on
  finite differences an inverted streamfunction cannot be differentiated. The Numba backend compiles a kernel only when
  it holds no global operator, and otherwise the kernel runs on NumPy.
- `CouplingInteraction(target, source, read, write, function, reduce, mode, ...)` is a `NeighborInteraction` that runs
  `function(ns)` once per edge of a lattice or network group. On a `FieldElement` it does not lower, so `legion` and
  `nanotube_gap_junction_exchange` keep their coupling inside `TermanWangElement` and `ExchangingCellElement`, written
  as `laplacian(active) + 4 * active` and `(g + xi) laplacian(x) + xi k x`.
- The pass compiler reports a law that its space does not support with the issue code `law-space`, and running a
  compiler with an error in its `report_` raises one error that lists every issue.

## Goals / Non-Goals

**Goals:**

- Write convolutions, shifted reads, and screened Poisson inversions symbolically in field laws, as the references
  write them, and lower them through `FieldOperatorKernel` and the neighbour tables.
- Delegate the numerics, FFTs, interpolation, fast transforms, and sparse factorizations, to SciPy and, on meshes, to
  scikit-fem, which are already runtime dependencies.
- Lift the spectral-only restriction of `inverse_laplacian`.
- Lower couplings over the neighbours of a field, so field models write their coupling as an interaction.

**Non-Goals:**

- Convolutions on networks or meshes. A kernel of graph distance or of mesh geometry is a different operator, and no
  inventory model needs it.
- Kernels that depend on the states or on time. Kernels depend on the displacement and on parameters.
- General elliptic inversion with variable coefficients, and multigrid. The basins of the inventory are rectangles, and
  sparse direct factorization handles the sizes the library runs.
- Compiling kernels with global operators to Numba. They run on NumPy, as spectral kernels do today.

## Decisions

- **LEGION is compared by its rates** (decided on 2026-11-02): it amplifies rounding so much that perturbing its initial
  state by 1e-14 moves the trajectory by 2e-3, so the rebuild is checked by its rates at many random states and by its
  trajectory over a short window.

- **Mixed boundaries need `boundary=`** (decided on 2026-11-01): an inversion whose operand mixes states of different
  boundary conditions has no single boundary to solve with, so it raises an error asking for `boundary=`, and with
  `boundary=` it runs on the fast transforms. Per-axis boundaries (a channel periodic along one axis and walled along
  the other) would need a sparse path; they are not provided until a model needs them.

- **Couplings on fields already lower** (found while applying 5.1): a per-edge `CouplingInteraction` on a lattice or
  network `FieldElement` compiles through the edge kernels on NumPy and Numba and matches a direct sum. Task 5.1 adds
  only the `laplacian` lowering of a constant times `(other.x - member.x)`, and couplings on fields read a fixed boundary
  value as `laplacian` does, so a coupling and its lowering agree everywhere. Coupling functions read `s.member` and
  `s.other`. Zero-flux edges interpolate with the `reflect` mode and pad with `symmetric`, matching the half-sample
  reflection of the lattice's reflecting boundary; fixed edges use `grid-constant`. Decided on 2026-11-01.

### Library survey

- SciPy (`scipy.signal.fftconvolve` and `oaconvolve`, `scipy.ndimage.convolve` and `map_coordinates`, `scipy.fft.rfftn`,
  `dstn`, and `dctn`, `scipy.sparse.linalg.factorized`) covers convolution, interpolated reads with wrap, mirror, and
  constant modes, fast sine and cosine transforms for Dirichlet and Neumann Poisson problems, and sparse LU. It is
  BSD-3, a NumFOCUS project released twice a year, with wheels for Linux and macOS, and already a runtime dependency.
  Wrapped.
- scikit-fem gives the stiffness and mass matrices on meshes. It is BSD-3 and already a runtime dependency for
  finite-element fields. Wrapped, for inversion on meshes.
- `py-pde` solves Poisson equations on its own grids, with its own field types and boundary language that would
  duplicate the core's, and offers no screened variant, convolution, or look-ahead reads. Not adopted.
- Dedalus solves elliptic problems with spectral tau methods, but is GPL-3, needs MPI and FFTW, and installs through
  conda. Not adopted.
- `pyFFTW` and `mkl_fft` are optional accelerators that `scipy.fft` can take as a backend, not needed at the library's
  sizes. Not adopted.

The core holds only the adapter: sampling kernels, choosing the transform from the boundary, and mapping boundary
conditions to SciPy modes.

### Convolution

`convolve(expression, kernel, radius=None)` returns a symbolic call. The kernel is a Sage expression of the
displacement symbols `displacement(axis)` and of parameters. At compile time the lowering substitutes the current
parameter values, prints the kernel to a NumPy function with the existing printer, and samples it at the lattice
displacements times the spacings, up to `radius` or the lattice size, times the cell volume. The
`FieldOperatorKernel` applies:
- on periodic lattices, the product of the real FFTs of the operand and of the wrapped kernel, through `scipy.fft`. On
  spectral fields this is one more Fourier multiplier and composes with the others;
- at fixed edges, `scipy.signal.oaconvolve` of the operand padded with the fixed value, cropped to the lattice;
- at zero-flux edges, the same with mirror padding;
- `scipy.ndimage.convolve` when the kernel has at most 7 sites per axis, where it is faster than an FFT.

Kernels are resampled when the problem is compiled with new parameters, which experiments already do for every member.
A symbolic kernel is chosen over a NumPy array so the law stays printable, analysable, and parameterised.

### Shifted reads

`shifted(expression, displacement)` takes one physical displacement per axis. When every component is a number that is
a whole multiple of the spacing, the lowering reuses `FieldLowering._shift`, so it becomes a neighbour table and stays
compilable to Numba. Otherwise the lowering creates a `FieldOperatorKernel` whose function evaluates the operand and the
displacement at every site and calls `scipy.ndimage.map_coordinates` at the site coordinates plus the displacement over
the spacing, with `order=1` by default and `order=3` on request. Field boundaries map to its modes: periodic to
`grid-wrap`, zero-flux to `mirror`, and fixed to `constant` with `cval` the fixed value. `FieldOperatorKernel` gains an
optional `displacement` operand for this, since its `apply` takes only the operand's values.

`shifted` differs from `shift_member`, which reads a member of a group at an offset of its index.

### Modified-Helmholtz inversion

`inverse_helmholtz(expression, screening, boundary=None)` returns a symbolic call, and `inverse_laplacian(u)` becomes
`inverse_helmholtz(u, 0)`. The solver is chosen by the discretization and the boundary:
- spectral fields multiply by `-1 / (k² + screening²)`, with the zero mode set to zero when the screening is zero;
- finite-difference lattices with fixed edges use `scipy.fft.dstn` of type 1 on the interior, with the fixed value
  lifted into the right-hand side, and zero-flux edges use the `dctn` type that matches the mirror convention of the
  lattice, both diagonalising the second-order stencil exactly, so the solution inverts the stencil `laplacian` uses;
- periodic finite-difference lattices use `rfftn` with the eigenvalues of the discrete stencil, not `-k²`;
- networks, mixed boundaries, and meshes factorise `L - screening² M` once with `scipy.sparse.linalg.factorized`, where
  `L` is the graph Laplacian or the scikit-fem stiffness matrix and `M` the identity or the mass matrix, and pin the
  mean when the screening is zero.

Factorisations are cached per field, boundary, and screening value. The fast transforms are chosen over a sparse solve
on lattices because they cost `N log N` per step and need no factorisation.

### Composition

A stencil read of the symbol of a global operator needs its value at a neighbour. `FieldLowering._shift` learns to
shift such a symbol by indexing the values of its `FieldOperatorKernel` with the neighbour table, rather than the member
columns. On spectral fields the existing chaining of multipliers covers convolution and inversion.

### Couplings on fields

`CouplingInteraction` with a `FieldElement` target in the mode `'edges'` lowers in `FieldLowering` instead of the edge
kernel of `NeighborInteraction`:
- on a lattice, the reduction over the neighbours becomes the sum, or the mean, over the offsets of the field's
  neighbourhood of `function(ns)` with `ns.source` read by whole-site `shifted` reads, which follow the boundaries;
- on a network, it becomes a gather over the neighbour table of the graph with a sum per node;
- when `function` is a constant times `source.x - target.x`, it becomes that constant times `laplacian(x)`, so the
  spectral and finite-element discretizations keep their own operator.

So `TermanWangElement` loses its excitation term, which becomes
`CouplingInteraction(target='cells', read='x', write='x', function=lambda s: s.W * s.target.m * s.source.m *
logistic(s.source.x - s.theta_x, 0, s.kappa))`, and `ExchangingCellElement` loses its exchange, which becomes
`function=lambda s: s.g * (s.source.x - s.target.x) + s.xi * s.source.x`, with no degree term.

### Issues

The operators are Sage functions, not law kinds, so they add no `BaseLaw` subclass. The pass compiler reports, in its
`report_`, with the law's path and a fix:
- `law-space`: a convolution or a shifted read in a law of a field on a network or a mesh, with the fix of a lattice
  space, and a spectral convolution or inversion on a lattice without periodic edges;
- `kernel-state`: a kernel that reads a state, naming the state;
- `screening`: a screening whose square can be negative on the domains of its parameters, or that reads a state, naming
  the screening;
- `operator-spaces`: an operand that reads fields of more than one space.

### Fit with the pipeline, runtime, and visualization

- Symbolic layer: the three operators are Sage functions like `laplacian`, so laws print, substitute, and analyse as
  before. `StabilityAnalysis('dispersion')` treats a convolution and an inversion as linear operators of their operand.
- Compile: kernels with global operators run on the NumPy backend, and the rest of a system compiles to Numba as before.
- Stiff splitting: on spectral fields a convolution with a parameter kernel is linear and joins the stiff part of
  `ETDRK4Solver` and the IMEX solvers as a multiplier. Elsewhere nonlocal terms stay in the explicit part.
- Runtime: the sparsity pattern of `ProblemSpec.jacobian` marks a convolution as coupling the sites within its radius,
  and an inversion as dense, so implicit solvers stay correct.
- Visualization: the operators add no state. A user streams a streamfunction or a synaptic input as an image through an
  `ObservableSpec` whose expression uses the operator.

### Models

- `amari_neural_field` in `biology/neuroscience`: a `RateInteraction` on a `FieldElement` with the rate
  `(-u + convolve(heaviside(u), w) + h) / tau`.
- `double_gyre` in `earth/oceanography`: an `AdvectionInteraction` of the potential vorticity by the velocity of the
  streamfunction `inverse_helmholtz(q - beta y, 1 / Rd)`, with the wind forcing as a `RateInteraction` and the viscosity
  as a `DiffusionInteraction`.
- `gkt_traffic` in `engineering/traffic`: `ConservationInteraction`s of density and momentum whose relaxation source
  reads the density and the velocity by `shifted` at the look-ahead distance.
- `active_viscoelastic_gel` in `physics/active_matter`: the gel velocity `inverse_helmholtz` of the active stress
  gradient with the screening `sqrt(gamma / eta)`.
- `legion` and `nanotube_gap_junction_exchange`: rebuilt with `CouplingInteraction` on their fields, compared against
  their previous trajectories.

## Risks / Trade-offs

- [Dense Jacobians from inversions make implicit solvers slow] → The inventory models use explicit or exponential
  solvers, and the user guide says so.
- [Linear interpolation smooths look-ahead reads] → Cubic interpolation is available, and the traffic model's
  acceptance test runs at two resolutions.
- [Large kernels on non-periodic lattices cost an FFT of a padded array per evaluation] → `oaconvolve` limits the
  padding to the kernel radius, and a declared radius bounds it further.
- [Heaviside firing rates make the Amari field discontinuous] → The bump test checks the width only after the
  transient.
- [The rebuilt couplings change the order of floating-point sums] → `legion` and `nanotube_gap_junction_exchange` are
  compared with their previous trajectories within 1e-10.
