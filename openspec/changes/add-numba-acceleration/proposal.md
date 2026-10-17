# Proposal

## Why

After element groups, the remaining per-step cost is NumPy temporaries for every operation of a law and Python
overhead in the stepper loops, one call per internal step. A prototype showed that the same SymPy law compiled with
Numba runs 5.7 times faster over a million elements than NumPy broadcasting, and compiles in 0.5 seconds independent of
the group size.

## What Changes

- **BREAKING** Make `numba` a runtime dependency.
- Lower symbolic laws to Numba kernels: one fused, compiled function per form function, looping over members without
  temporaries. Keep the NumPy lowering as a reference backend selected with `backend='numpy'`.
- Compile the fixed-step steppers, a new `RK4`, `EulerMaruyama`, `Gillespie`, and `Iteration`, so all internal steps
  of a frame run in compiled code.
- Replace the KD-tree neighbour search of pair interactions with compiled cell lists.
- Accept Numba-compiled callables as law values inside compiled kernels, and keep plain Python callables working
  through a slower Python path.
- Cache compiled kernels on disk, so a second run of the same system skips compilation.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `model-compilation`: compiled backends, backend equivalence, compiled callables, and kernel caching.
- `simulation`: compiled steppers and their throughput.

## Impact

- `skcomplex.compile` gains a backend layer. `skcomplex.solvers` gains `RK4` and compiled versions of the fixed-step
  steppers. `ScipyODE` keeps calling the compiled drift.
- Depends on `add-element-groups`. Numba constrains NumPy to versions it supports, currently below 2.6.
