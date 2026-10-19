# Proposal

## Why

Every state and parameter in the framework is real. Dilão (2023 §5.5) builds the Mandelbrot and Julia sets from the
complex quadratic map `z → z² + c` iterated over a grid of complex points. The map, its escape test `|z| > 2`, and its
parameter `c` are complex, so neither set can be written today. The inventory of `organize-model-library` records both
as blocked on this change.

Complex amplitudes are also common in continuum models. The complex Ginzburg-Landau equation and the Schrödinger
equation are written today as two coupled real fields. That works, but the law is longer than the equation it encodes.

## What Changes

- Add a complex domain for states and parameters, on single elements, groups, and fields. Laws over complex
  symbols use the imaginary unit and the complex functions: real part, imaginary part, conjugate, modulus, argument,
  and the elementary functions over the complex plane.
- Lower complex laws to NumPy `complex128` arithmetic and to Numba nopython kernels over `complex128` scalars, with
  both backends giving the same results.
- Store every complex component as a contiguous pair of floats in the flat state and parameter vectors, so every
  existing solver, stepper, and spectral discretization runs complex systems unchanged. The compiled kernels view the
  pairs as complex numbers without copying.
- Return complex arrays from trajectories, frames, and observables. Analysis functions that need real input reject
  complex series with a named error that names the real views to use.
- Stream complex states to Rerun as their modulus and phase. Lattice fields log two images, one colored by modulus
  and one colored by phase with a cyclic map. Scalar states log their real and imaginary parts as series.
- Add library models: the Mandelbrot set and the filled Julia sets of the quadratic map, as escape-time fields over
  a lattice of complex points.
- Optionally rewrite the complex Ginzburg-Landau model with one complex field, and add the Schrödinger equation as
  one. The existing two-field forms stay valid, and this change does not require the rewrite.

## Capabilities

### New Capabilities

- `complex-states`: the complex domain for states and parameters, complex symbolic laws, their lowering to the
  NumPy and Numba backends, solver compatibility through paired storage, complex trajectories, and the visualization
  of complex values.

### Modified Capabilities

- `model-library`: the Mandelbrot and Julia sets under `mathematics/fractals`.

## Impact

- `skcomplex.core.base` gains `Domain.COMPLEX`, and `symbol` stops declaring complex symbols real.
- `skcomplex.core.compile` gains complex-aware layout entries, NumPy and Numba dialect entries for the complex
  functions, and type inference that chooses complex or real temporaries per expression.
- `skcomplex.core.runtime` returns complex arrays for complex variables, and `skcomplex.core.viz` logs modulus and
  phase.
- `skcomplex.models.mathematics.fractals` is a new subpackage with `mandelbrot` and `julia`.
- No new dependency: NumPy, Numba, SciPy, passagemath, and Rerun already ship everything needed.
- No breaking change: real states keep their layout, kernels, and trajectories.
