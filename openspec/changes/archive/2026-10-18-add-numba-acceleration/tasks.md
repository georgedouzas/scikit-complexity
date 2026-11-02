# Tasks

## 1. Setup

- [x] 1.1 Add `numba` as a dependency, relock, and verify `pdm install` succeeds on Python 3.12 and 3.13

## 2. Backends

- [x] 2.1 Add the backend registry and a `backend` parameter to `compile_system` and `Simulation`, with the current
  lowering as `numpy`, and verify the unknown-backend test passes
- [x] 2.2 Generate and compile fused Numba kernels from law expressions, and verify the backend agreement test passes
- [x] 2.3 Call Numba-compiled law values inside kernels and route Python callables to the Python path, and verify the
  callable law tests pass
- [x] 2.4 Add compiled cell lists for pair interactions, and verify the pair force and momentum tests pass on both
  backends
- [x] 2.5 Cache generated kernels on disk, and verify the two-process compilation test passes

## 3. Steppers

- [x] 3.1 Add `RK4` and compiled loops for `EulerMaruyama`, `Gillespie`, and `Iteration`, and verify the accuracy and
  reproducibility tests pass
- [x] 3.2 Add `VelocityVerlet` with a compiled loop, and verify the energy conservation and position pair tests pass

## 4. Performance and documentation

- [x] 4.1 Add benchmark tests for the million-member speedup and the 100000-member real-time gas, and verify they pass
- [x] 4.2 Document backends and compiled callables in the user guide, and verify the documentation test passes
- [x] 4.3 Run formatting, checks, the docs build, and tests through nox, and verify all of them pass

## Workflow follow-up

- Archive the change after review.
