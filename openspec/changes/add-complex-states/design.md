# Design

## Context

Every variable is real from declaration to display:
- `Domain` in `skcomplex.core.base` has only real, integer, boolean, angle, and categorical members. `Domain.encode`
  casts to `float`, and `symbol` declares every Sage symbol real or integer.
- `skcomplex.core.compile` lays out states and parameters in flat `float64` vectors through `Entry` objects. It prints
  Sage expressions to NumPy through `NUMPY_DIALECT`, which already prints complex literals, and to Numba scalar code
  through `SCALAR_DIALECT`, which maps functions to `math` and so fails on complex arguments.
- Solvers in `skcomplex.core.solvers` receive real vectors: the default ODE solver wraps SciPy `solve_ivp`, and the
  fixed-step, stiff, and exponential steppers are in house. Spectral fields already couple two real fields per Fourier
  mode, which is how `complex_ginzburg_landau` runs today.
- Trajectories return `float64` arrays, and `ImageVisual` colors one real image per state.

The proposal explains why the Mandelbrot and Julia sets need complex values. Sage already treats an undeclared
symbol as complex, with `real_part`, `imag_part`, `conjugate`, `abs`, and `arg` as symbolic functions, so the gap is
in the domain, the layout, the printers, the runtime arrays, and the visuals.

## Goals / Non-Goals

**Goals:**

- Write a complex law as it appears in the reference, such as `z² + c` or `i/2 ∇²ψ`, without splitting it by hand.
- Keep every solver, stepper, discretization, experiment, and backend working on complex systems without
  complex-specific code paths in them.
- Keep real systems bit-for-bit unchanged: same layout, same kernels, same trajectories.

**Non-Goals:**

- Arbitrary-precision arithmetic for deep fractal zooms. `mpmath` and `gmpy2` provide it, and a future change could
  wrap them.
- Holomorphic Jacobians and complex-step differentiation. The stiff solvers keep differentiating the real form.
- Complex values on the LAMMPS and Mesa backends. Particles and agents stay real, and compiling a complex state for
  them raises a named error.
- Rewriting the complex Ginzburg-Landau model or adding the Schrödinger equation with complex fields. Both become
  possible, and section *Optional model rewrites* says how, but neither is required here.

## Decisions

### Library survey

The constitution requires the change to build on mature libraries. Complex support is not a separate library but a
feature of the numeric stack the framework already depends on:

| Library | Role here | Maturity and fit | Decision |
|---|---|---|---|
| NumPy (BSD-3) | `complex128` arrays, ufuncs `np.exp`, `np.abs`, `np.angle`, `np.conj`, `np.real`, `np.imag`, zero-copy `view(np.complex128)` of interleaved float pairs | Core dependency since 1.0, wheels for Linux and macOS on Python 3.12–3.13 | Wrapped. It is the reference backend's arithmetic. |
| Numba (BSD-2) | `complex128` scalars and arrays in nopython mode, `.real`, `.imag`, `.conjugate()`, the `cmath` module (`exp`, `log`, `sqrt`, `phase`, trigonometric and hyperbolic functions), and NumPy complex ufuncs | Core dependency, documented complex support, wheels for Linux and macOS on Python 3.12–3.13 | Wrapped. The Numba dialect prints complex subexpressions to `cmath` and keeps `math` for real ones. |
| passagemath-symbolics (GPL-2+, Sage) | Complex symbols, `I`, `real_part`, `imag_part`, `conjugate`, `abs`, `arg`, and simplification over the complex plane | Core dependency, the symbolic layer every law is written in | Wrapped. Complex symbols are left undeclared, which Sage treats as complex. |
| SciPy `solve_ivp` (BSD-3) | Accepts a complex `y0` for RK23, RK45, DOP853, and BDF, but not for Radau or LSODA, and BDF needs a holomorphic right side | Core dependency | Not used with complex vectors. Paired storage keeps every method available, including Radau and LSODA, and laws with `abs` or `conjugate` that are not holomorphic. |
| Rerun (MIT/Apache-2.0) | Images, tensors, and scalar series | Core dependency. Its tensor and image types take integer and float element types only, with no complex type | Wrapped through an adapter that logs the modulus and the phase. |
| JAX | Complex arrays, automatic differentiation | Mature, but a heavy new dependency and a third backend | Rejected. It adds nothing NumPy and Numba lack for this feature. |
| `py-pde` | Complex scalar fields with spectral and finite-difference operators | Mature, but it would replace the field discretizations rather than add complex values to them | Rejected for this change. The audit in `organize-model-library` evaluates `py-pde` for the field discretizations as a whole. |

The change adds no dependency. The core holds only the adapter code: the domain, the layout of pairs, the dialect
entries, the runtime views, and the visual.

### A complex domain

`Domain.COMPLEX` joins the enumeration with Sage domain `complex`. `symbol` declares real and integer symbols as now
and leaves complex symbols undeclared. `encode` returns a `complex128` array and `contains` accepts finite complex
values. For real domains, `contains` accepts a complex value whose imaginary part is zero and rejects any other, so
the error in the spec's real-domain scenario names the complex domain. Companion states of a complex state, its
derivatives and lags, are complex too.

Alternative considered: a `complex=True` flag on `state` and `parameter`, beside the domain. Rejected, because the
domain already carries the set of values a variable takes, and a flag would allow contradictions such as a complex
boolean.

### Paired storage in the flat vectors

A complex variable occupies two consecutive `float64` slots per component, the real part then the imaginary part.
`Entry` gains `complex: bool` and its `size` doubles, so slices and member strides stay correct. The layout keeps
each complex variable's pairs contiguous, which makes `vector[entry.indices].view(np.complex128)` a zero-copy complex
view. Numba supports the same view on contiguous arrays.

Every solver keeps seeing a real vector of length `2n`, which is the realification of the complex system:
- explicit, implicit, stiff, and exponential steppers and `solve_ivp` methods need no change;
- finite-difference Jacobians are real and remain valid for laws that are not holomorphic, such as `|z|² z`;
- events, boundaries, ensembles, calibration, and co-simulation read and write the same real vectors.

Alternative considered: a `complex128` state vector for any system with a complex variable. Rejected: it excludes
Radau and LSODA, requires holomorphic right sides for BDF, forces complex arithmetic on every real state of a mixed
system, and adds a dtype branch to every in-house stepper and to the hybrid and co-simulation solvers.

### Lowering complex laws

The compiler infers a type for every node of a lowered expression: complex when it contains a complex symbol or a
non-real number, and real when it is a real symbol or number, or the result of `real_part`, `imag_part`, `abs`, or
`arg`. The rules that use this type:
- A law's value must be real for a real target, or the compiler raises `ValidationError` with the law and the state.
- A relation, `min`, `max`, `floor`, `ceil`, `mod`, `sgn`, and `heaviside` must take real operands. This covers the
  ordering scenario.
- A complex law for a complex target writes its value to the complex view.

The dialects gain entries:
- NumPy: `real_part` → `np.real`, `imag_part` → `np.imag`, `conjugate` → `np.conj`, `arg` → `np.angle`. The existing
  ufunc entries already accept complex arrays.
- Numba scalar: a complex dialect beside `SCALAR_DIALECT`, chosen per node by the inferred type. It maps elementary
  functions to `cmath`, `abs` to `abs`, `arg` to `cmath.phase`, `real_part` to `.real`, `imag_part` to `.imag`, and
  `conjugate` to `.conjugate()`. Real nodes keep `math`, so real kernels print exactly as today.

The Numba kernels read complex variables through the complex view of the state and parameter vectors and write
results through the view of the derivative or update vector. Kernel cache keys include the inferred types, so a
system that differs only in a variable's domain does not reuse a stale kernel.

### Fields and spectral operators

`laplacian`, `partial`, and `gradient` act on a complex field component-wise, which the paired layout gives with no
new stencil code. On spectral fields, a complex linear term such as `(1 + i b) ∇²A` lowers to the 2 × 2 real block per
Fourier mode that the spectral discretization already builds for the two-field Ginzburg-Landau equation. The
exponential steppers therefore see the same operator as they do for the split form, which the spec's agreement
scenario checks.

Alternative considered: diagonal complex linear operators in the spectral discretization. It would be faster for
large complex fields, but it would add a complex path to the exponential steppers. A later change can add it as an
optimisation that the agreement scenario would keep honest.

### Runtime arrays and analysis

`Trajectory.__getitem__`, `Trajectory.lattice`, `Frame` reads, and observables return `complex128` views for complex
variables. The NumPy array then provides `.real`, `.imag`, `np.abs`, and `np.angle` for the real views, so the
framework adds no accessor of its own. Analysis functions that cast to `float` gain a guard that raises
`ValidationError` naming the variable and those real views, instead of NumPy's `ComplexWarning` that drops the
imaginary part silently.

### Visualization

`ImageVisual` and `MeshVisual` log a complex state under `<path>/<state>/modulus` with the existing palette and range,
and `<path>/<state>/phase` with a cyclic palette over `(-π, π]`. `ScalarsVisual` logs `<path>/<state>/real` and
`<path>/<state>/imag`. Real states log exactly as today. The cyclic palette is a fixed color table in the visuals
module beside the existing palette.

### Mandelbrot and Julia models

`skcomplex.models.mathematics.fractals` defines one element with a complex state `z`, an escape-time state `n` in the
nonnegative integers, a complex parameter `c`, and an escape radius `radius`. Its laws are two updates on a
`LatticeSpace` field:
- `z ← where(abs(z) <= radius, z**2 + c, z)`;
- `n ← n + where(abs(z) <= radius, 1, 0)`.

The escaped sites freeze, so values never overflow. `mandelbrot(shape, window, steps)` sets `c` per site from the
window and `z = 0`. `julia(c, shape, window, steps)` sets `z` per site from the window and `c` uniform. Both declare
an `ImageVisual` on `n`, and the documentation page also streams `z` as modulus and phase. They register under
`mathematics/fractals` with the source Dilão 2023 §5.5, and the inventory rows move from
`blocked add-complex-states` to `implemented`.

### Optional model rewrites

Two models could use complex fields. Neither is required by this change:
- The complex Ginzburg-Landau model could become `Rate(A, A + (1 + i b) laplacian(A) - (1 + i c) abs(A)**2 * A)`.
  The test for the spec's agreement scenario builds this form, so promoting it to the library factory is a one-line
  swap once the scenario passes.
- The Schrödinger equation, listed in the inventory as real and imaginary fields (Schwichtenberg ch. 4), could be
  written as `Rate(psi, i * (laplacian(psi) / 2 - V * psi))`.

## Risks / Trade-offs

- [Paired storage hides holomorphic structure from the stiff solvers, so their Jacobians are twice the size of a
  complex Jacobian] → The fractal models are maps and need no Jacobian, and the stiff complex fields in the library
  are spectral and use exponential steppers. A complex-Jacobian path can be added later if a benchmark asks for it.
- [Users take `float(...)` of a complex trajectory and lose the imaginary part] → Analysis functions raise a named
  error, and the user guide shows the real views.
- [Numba type unification fails when a `where` mixes a complex and a real branch] → The printer promotes the real
  branch to `complex(...)` when the inferred type of the `where` is complex. A backend agreement test covers it.
- [Cache collisions between real and complex variants of one system] → The kernel cache key includes the inferred
  types.
- [Rerun has no complex type, so complex fields need two entities] → Modulus and phase are the conventional domain
  coloring split, and both are logged under the state's own path.
