# Tasks

## 1. Complex domain and storage

- [x] 1.1 Add `Domain.COMPLEX` with encoding, containment, and the real-domain rejection of nonzero imaginary parts,
  leave complex symbols undeclared in `symbol`, and verify the complex initial value and complex value in a real
  domain scenarios and the doctests
- [x] 1.2 Store complex variables as contiguous real and imaginary pairs in the state and parameter layouts, with
  complex companion states, and verify that the complex views are zero-copy and that the layout of real systems is
  unchanged

## 2. Lowering

- [x] 2.1 Split complex laws into the real laws of their real and imaginary parts with Sage, and reject complex values
  for real states and complex operands of relations and ordering functions with named errors, and verify the complex
  value for a real state and ordering scenarios
- [x] 2.2 Lower the split laws with the existing NumPy and Numba dialects, and verify the harmonic phase and real
  readout scenarios on both backends
- [x] 2.3 Run the split laws through the existing NumPy and Numba kernels for elements, groups, and fields, and verify
  the backends agree on a complex map and caching scenarios
- [x] 2.4 Raise a named error when a complex state is compiled for the LAMMPS or Mesa backends, and verify it names
  the state and the backend

## 3. Solvers, fields, and runtime

- [x] 3.1 Lower complex linear terms of spectral fields to the per-mode real blocks, and verify the complex and split
  Ginzburg-Landau forms agree and the unitary evolution scenario
- [x] 3.2 Run a complex system through every registered ODE solver, fixed-step stepper, and stiff stepper, and verify
  each matches the split real form within 1e-8
- [x] 3.3 Return complex arrays from trajectories, frames, and observables, and guard the real-valued analysis
  functions with a named error, and verify the complex trajectory and real analysis scenarios
- [x] 3.4 Document complex states, laws, and real views in the user guide, and verify the documentation test passes

## 4. Visualization

- [x] 4.1 Log complex lattice and mesh states as modulus and phase images, with a cyclic phase palette, and complex
  scalars as real and imaginary series, and verify the two viewer scenarios on a recorded stream
- [x] 4.2 Document the complex visuals in the visualization guide, and verify the documentation test passes

## 5. Library

- [x] 5.1 Add `skcomplex.models.mathematics.fractals` with registered `mandelbrot` and `julia` factories under
  `mathematics/fractals`, citing Dilão 2023 §5.5, and verify the Mandelbrot escape times, Julia set, and symmetry
  scenarios, the conformance suite, and the layout test
- [x] 5.2 Move the Mandelbrot and Julia inventory rows to implemented and add the fractals field page, and verify the
  inventory check and the documentation build pass
- [ ] 5.3 Optionally switch `complex_ginzburg_landau` to one complex field and add the Schrödinger equation under
  `physics/waves`, and verify the existing Ginzburg-Landau scenario and the unitary evolution scenario still pass.
  Left for a later change: the switch renames the model's states `medium.u` and `medium.v` to `medium.A`, which breaks
  its users, and the inventory already lists the Schrödinger equation as implemented with split fields. The unitary
  evolution and agreement scenarios are verified by the core tests with one complex field.

## 6. Gate

- [ ] 6.1 Run formatting, checks, the docs build, and tests through nox, and verify all of them pass
