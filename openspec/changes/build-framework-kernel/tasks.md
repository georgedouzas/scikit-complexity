# Tasks

## 1. Setup

- [x] 1.1 Declare `numpy`, `scipy`, `sympy`, `rerun-sdk`, and `scikit-learn` as dependencies, move `lammps` to the
  `lammps` extra, add the `sage` extra, and verify `pdm install -G :all` succeeds
- [x] 1.2 Remove `base`, `component`, and `system`, create the layered package skeleton, and verify
  `python -c "import skcomplex"` succeeds
- [x] 1.3 Fill the constitution's Project Profile and verify it names the ecosystem contract and dependencies

## 2. Core and model

- [x] 2.1 Add the errors module, domains, variable declarations, and registry in `core`, and verify their tests pass
- [x] 2.2 Add `Component`, `Element`, `Interaction`, `System`, `EuclideanSpace`, the law kinds, and `Symbols`, and
  verify tests for stored arguments, roles, nesting, and labels pass

## 3. Compilation

- [x] 3.1 Add `compile_system` with flattening, validation, layout, form inference, and lowering, and verify tests for
  every form, dotted names, parameter arguments, callable laws, and errors pass

## 4. Solvers and runtime

- [x] 4.1 Add the ODE, SDE, jump, and discrete steppers with the registry, and verify accuracy and reproducibility
  tests pass
- [x] 4.2 Add `Simulation`, clocks, observers, `Recorder`, and `Trajectory`, and verify tests for frames, pacing, live
  parameters, background runs, and solver errors pass

## 5. Visualization

- [x] 5.1 Add `ScalarsVisual`, `PointsVisual`, and `RerunObserver`, and verify a test that saves a recording and
  checks entity paths, interval thinning, and parameter logging passes

## 6. Analysis

- [x] 6.1 Add `equations_of` and `solve_symbolically`, and verify tests for the free particle, the oscillator, and the
  non-ODE error pass
- [x] 6.2 Add a test that compiles a Sage expression, skipped when the extra is missing, and verify it passes with the
  extra installed

## 7. Model library

- [x] 7.1 Add the physics module, and verify tests for constant acceleration, energy conservation, and mass validation
  pass
- [x] 7.2 Add the ecology, epidemiology, economics, and finance modules, and verify the tests for their spec scenarios
  pass

## 8. Documentation and gate

- [x] 8.1 Rewrite the README usage section and the package docstrings with runnable examples, and verify the doctests
  pass
- [x] 8.2 Run formatting, checks, and tests through the nox sessions, and verify all of them pass

## Workflow follow-up

- Archive the change after review.
