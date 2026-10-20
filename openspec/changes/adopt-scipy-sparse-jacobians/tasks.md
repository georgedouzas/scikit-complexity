# Tasks

## 1. Jacobian structure

- [x] 1.1 Keep the sparse operator matrix on `DerivedRead` for finite element and finite-difference operators, and
  verify the existing field tests still pass
- [x] 1.2 Add `jacobian_pattern` to `skcomplex.core.compile`, deriving entries from every reference source, the
  finite-volume stencils, the diagonal, and the fixed rows, and verify the stencil, mesh, and run-time coupling
  scenarios and its agreement with `SplitOperator.sparse` on linear fields
- [x] 1.3 Add a conformance test that checks the pattern contains the finite-difference Jacobian of every library model
  with an ODE form, and verify the superset scenario
- [x] 1.4 Add `jacobian_function` with Sage partial derivatives, the chain rule for sparse derived reads, compilation
  with the problem's backend, and the scatter into the pattern's data, and verify the exact Jacobian, parameter change,
  and callable law scenarios

## 2. SciPy integrators

- [x] 2.1 Add the `jacobian` parameter to `ScipyODE`, build the structure in `initialize` for implicit methods only, and
  pass it in `advance` to BDF, Radau, and LSODA, and verify the speed, Radau and exact, banded LSODA, and explicit
  method scenarios
- [x] 2.2 Pass the same structure in `_advance_hybrid`, and verify the event and delayed stiff field scenarios
- [x] 2.3 Name BDF and Radau in the stability warning and warn once for adaptive explicit methods on stiff fields, and
  verify the spatial-fields scenarios

## 3. Benchmarks and documentation

- [x] 3.1 Add Fisher-KPP on 4000 points with the dense, pattern, exact, and Crank-Nicolson paths to the benchmark
  suite, and verify it runs and records the ratios
- [x] 3.2 Document the `jacobian` option and the stiff path for finite-difference, finite element, and spectral fields
  in the user guide and the changelog, and verify the documentation test passes (the changelog entry is left to the
  coordinator of the merge, who owns `CHANGELOG.md`)
- [x] 3.3 Run formatting, checks, the docs build, and tests through nox, and verify all of them pass
