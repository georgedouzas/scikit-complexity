# Design

## Context

After `redesign-system-model`, a field is a group on the sites of a lattice, network, or mesh, `Group(on='sites')`,
and its differential operators are expressions in `core.laws`. Field laws are written for one site. `FieldLowering` in `skcomplex.core.compile` lowers `laplacian`, `partial`, and
`upwind` to stencils, whose neighbour reads are `NeighborRead` tables of member indices, and lowers spectral operators
and the finite-element Laplacian to `DerivedRead`s: a new symbol whose value at every site is a global function, a
Fourier multiplier or a sparse matrix, applied to the operand evaluated at every site. `inverse_laplacian` is a
spectral multiplier, and on any other discretization it raises "needs a spectral field". A stencil cannot read a
`DerivedRead` at a neighbour, so on finite differences an inverted streamfunction cannot be differentiated. The Numba
backend compiles a kernel only when it has no `DerivedRead`, and otherwise the kernel runs on NumPy. Rerun streams the
states of lattice site groups as images. Validation reads the `Requires` declarations of primitives and reports every
invalid combination before compilation.

The constitution requires a feature that a mature library implements to be built on it, with the core holding only
the adapter, and requires a new runtime dependency to be justified.

## Goals / Non-Goals

**Goals:**

- Write convolutions, shifted reads, and screened Poisson inversions symbolically in field laws, as the references
  write them, and lower them through the existing `DerivedRead` path.
- Delegate the numerics, FFTs, interpolation, fast transforms, and sparse factorizations, to SciPy and, on meshes, to
  scikit-fem, which the project already depends on.
- Lift the spectral-only restriction of `inverse_laplacian`.

**Non-Goals:**

- Convolutions on networks or meshes. A kernel of graph distance or of mesh geometry is a different operator, and no
  inventory model needs it.
- Kernels that depend on the states, such as adaptive smoothing, or on time. Kernels depend on the displacement and on
  parameters.
- General elliptic inversion with variable coefficients, and multigrid. The basins of the inventory are rectangles, and
  sparse direct factorization handles the sizes the library runs.
- Compiling kernels with nonlocal reads to Numba. They run on NumPy, as spectral kernels do today. Evaluating derived
  values outside the Numba kernel is a follow-up shared with spectral fields.

## Decisions

### Library survey

| Library | Covers | Maturity and fit | Verdict |
|---|---|---|---|
| SciPy (`scipy.signal.fftconvolve` and `oaconvolve`, `scipy.ndimage.convolve` and `map_coordinates`, `scipy.fft.rfftn`, `dstn`, and `dctn`, `scipy.sparse.linalg.factorized`) | Convolution, interpolated reads with wrap, mirror, and constant modes, fast sine and cosine transforms for Dirichlet and Neumann Poisson problems, sparse LU | BSD-3, NumFOCUS, releases twice a year (1.18 in 2026), wheels for Linux and macOS on Python 3.12 and 3.13, already a runtime dependency (`scipy>=1.13`), pocketfft and SuperLU in compiled code | Wrap |
| scikit-fem | Stiffness and mass matrices on meshes | BSD-3, already a runtime dependency for finite-element fields | Wrap, for inversion on meshes |
| `py-pde` | `solve_poisson_equation` on its own grids, Numba-compiled stencils | MIT, maintained, pure Python wheels. Its Poisson solver builds a SciPy sparse matrix itself, it brings its own grid and field types and boundary language that would duplicate the core's, it offers no screened variant, convolution, or look-ahead reads, and the audit in `organize-model-library` evaluates it separately as a replacement for the finite-difference fields | Not adopted |
| Dedalus | Spectral tau methods with Chebyshev bases, any elliptic problem in a bounded box | GPL-3, needs MPI and FFTW, and installs through conda rather than wheels. Too heavy for three operators, and its license does not match | Not adopted |
| `pyFFTW`, `mkl_fft` | Faster FFTs | Optional accelerators that `scipy.fft` can take as a backend. No need at the library's sizes | Not adopted |
| `scipy.ndimage.convolve` alone | Direct convolution | Quadratic in the kernel size, but exact for small compact kernels | Used below a kernel size threshold |

SciPy covers all three operators, is already a dependency, and meets every criterion, so the change adds no runtime
dependency. The core holds only the adapter: sampling kernels, choosing the transform from the boundary, and mapping
boundary conditions to SciPy modes.

### Convolution

`convolve(expression, kernel, radius=None)` returns a symbolic call. The kernel is a Sage expression of the
displacement symbols `displacement(axis)`, which the module exports, and of parameters. At compile time the adapter
substitutes the current parameter values, prints the kernel to a NumPy function with the existing printer, and samples
it at the lattice displacements times the spacings, up to `radius` or the lattice size, times the cell volume. The
`DerivedRead` applies:
- on periodic lattices, the product of the real FFTs of the operand and of the wrapped kernel, through `scipy.fft`. On
  spectral fields this is one more Fourier multiplier and composes with the others;
- at fixed edges, `scipy.signal.oaconvolve` of the operand padded with the fixed value, cropped to the lattice;
- at zero-flux edges, the same with mirror padding;
- `scipy.ndimage.convolve` when the kernel has at most 7 sites per axis, where it is faster than an FFT.

Kernels are resampled when the compiled problem is rebuilt with new parameters, which experiments already do for every
member. A symbolic kernel is chosen over a NumPy array so the law stays printable, analysable, and parameterised.

### Shifted reads

`shifted(expression, displacement)` takes one physical displacement per axis. When every component is a number that
is a whole multiple of the spacing, the lowering reuses `FieldLowering.shift`, so it becomes a `NeighborRead` and stays
compilable. Otherwise the lowering creates a `DerivedRead` whose function evaluates the operand and the displacement at
every site and calls `scipy.ndimage.map_coordinates` at the site coordinates plus the displacement over the spacing,
with `order=1` by default and `order=3` on request. Field boundaries map to its modes: periodic to `grid-wrap`,
zero-flux to `mirror`, and fixed to `constant` with `cval` the fixed value. A `DerivedRead` gains an optional
`displacement` operand for this, since `apply` today takes only the operand's values.

### Modified-Helmholtz inversion

`inverse_helmholtz(expression, screening, boundary=None)` returns a symbolic call, and `inverse_laplacian(u)` becomes
`inverse_helmholtz(u, 0)`. Validation checks that the screening's free symbols are parameters and that its square is
non-negative on their domains. The solver is chosen by the discretization and the boundary:
- spectral fields multiply by `-1 / (k² + screening²)`, with the zero mode set to zero when the screening is zero;
- finite-difference lattices with fixed edges use `scipy.fft.dstn` of type 1 on the interior, with the fixed value
  lifted into the right-hand side, and zero-flux edges use `dctn` of the type that matches the mirror convention of
  the lattice's reflecting edges, both diagonalising the second-order stencil exactly,
  so the solution is the inverse of the same stencil `laplacian` uses;
- periodic finite-difference lattices use `rfftn` with the eigenvalues of the discrete stencil, not `-k²`;
- networks, mixed boundaries, and meshes factorise `L - screening² M` once with `scipy.sparse.linalg.factorized`,
  where `L` is the graph Laplacian or the scikit-fem stiffness matrix and `M` the identity or the mass matrix, and pin
  the mean when the screening is zero.

Factorisations are cached per field, boundary, and screening value. The fast transforms are chosen over a sparse
solve on lattices because they cost `N log N` per step and need no factorisation, which matters at every right-hand
side evaluation.

### Composition

A stencil read of a derived symbol needs its value at a neighbour. `FieldLowering.shift` learns to shift a derived
symbol by wrapping its `DerivedRead` so the neighbour table indexes the derived values, rather than the member
columns. On spectral fields the existing chaining of multipliers covers convolution and inversion.

### Declarations and validation

The operators live in `core.laws` beside `laplacian` and `advection`. Each is an expression rather than a law kind, so
it adds no `LawKind`, but it declares what it requires of the space, which validation reads wherever a law uses it:

```text
convolve:          Requires(space={lattice})
shifted:           Requires(space={lattice})
inverse_helmholtz: Requires(space={lattice, network, mesh})
```

Validation reports, each as an `Issue` with the law's path and a fix:
- a convolution or a shifted read in a law of a group on a network or a mesh, with the fix of a lattice space;
- a kernel that reads a state, naming the state;
- a screening whose square can be negative or that depends on a state, naming the screening;
- a spectral convolution or inversion on a lattice without periodic edges;
- an operator whose operand reads groups of more than one space.

These checks were compiler raises before the redesign, and they move to validation so a user sees all of them at once.

### Fit with the pipeline, runtime, and visualization

- Symbolic layer: the three operators are Sage `BuiltinFunction`s like `laplacian`, so laws print, substitute, and
  analyse as before. Linear-stability analysis treats a convolution and an inversion as linear operators of their
  operand.
- Compile: the lowering adds `DerivedRead`s. Kernels with them run on the NumPy backend, and the rest of a system
  compiles to Numba as before.
- Stiff splitting: on spectral fields a convolution with a parameter kernel is linear and can join the stiff part as a
  multiplier. Elsewhere nonlocal terms stay in the explicit part.
- Runtime: operators are evaluated at every right-hand side call. The sparsity pattern of the Jacobian marks a
  convolution as coupling sites within its radius, and an inversion as dense, so implicit steppers stay correct.
- Visualization: the operators add no state. A user streams a streamfunction or a synaptic input as an image by
  passing an observable whose expression uses the operator to the simulation or the catalog entry.

## Risks / Trade-offs

- [Dense Jacobians from inversions make implicit steppers slow] → The inventory models use explicit or exponential
  steppers, and the user guide says so.
- [Linear interpolation smooths look-ahead reads] → Cubic interpolation is available, and the traffic model's
  acceptance test runs at two resolutions.
- [Large kernels on non-periodic lattices cost an FFT of a padded array per evaluation] → `oaconvolve` limits the
  padding to the kernel radius, and a declared radius bounds it further.
- [Heaviside firing rates make the Amari field discontinuous] → The model uses `where`, and the bump test checks the
  width only after the transient.
