# Tasks

Each task writes its regression test first, checks that the test fails on the current code, and then fixes the
defect. Every test follows the naming and public-API rules of the constitution.

## 1. Drivers and reductions

- [x] 1.1 Build `series` as a balanced binary search of conditions, and verify the long held and long linear series
  scenarios on NumPy and Numba, the `series` docstring example, and the existing driver tests
- [x] 1.2 Find the symbols outside reductions by walking the expression tree without entering reduction calls, and
  verify the share-of-a-group-total and weighted-denominator scenarios
- [x] 1.3 Lower the reductions of observable member expressions through the reduction lowering of laws, raising the
  named error of laws for a reduction over another group, and verify the local field energy and plain reduction
  scenarios and the existing observable tests

## 2. Roles and jumps

- [x] 2.1 Resolve a member reference bound to a plain role as one single-element view of that member, document it on
  `role`, and verify the hub-reads-one-site and member-past-the-end scenarios
- [x] 2.2 Accept offset reads as targets of group jumps, filling one channel per member in the Gillespie target table
  with the empty target past a fixed end, document it on `Jump`, and verify the ring, open end, and
  same-law-as-per-link scenarios on NumPy and Numba
- [x] 2.3 Keep running offsets in the variable declarations, index blocks by path once per compilation, and visit each
  distinct subexpression once in `breakpoints_of`, and verify the many one-link interactions scenario
- [x] 2.4 Rebuild `zero_range_process` and `asep` on one group law with offset jumps, keeping their trajectory paths
  and their reservoirs, and verify their existing scenarios, the conformance suite, their gallery examples, and the
  many-sites scenario

## 3. Visuals

- [x] 3.1 Give lattice groups the visual their element type declares, and an image only when it declares none, and
  verify the two lattice visual scenarios and the existing lattice blueprint tests
- [x] 3.2 Give every time series view of the default blueprint the whole-run time axis, and verify the
  one-dimensional points and other-states scenarios and the blueprint docstring examples

## 4. Default ODE solver

- [x] 4.1 Make `ScipyODE(method='LSODA')` the default ODE stepper, document it on `Simulation` and in the user guide,
  and leave its breaking changelog entry to the release. Verify the stiff-model-built-directly and non-stiff accuracy
  scenarios, and move tests that pin values of default `RK45` runs to tolerances
- [ ] 4.2 Run the registered ODE models of the catalog with the new default, and verify that their tests, the
  documentation test, and the gallery examples pass

## 5. Benchmarks

- [ ] 5.1 Rerun the real-time benchmarks of the 100 000-sphere interacting gas and the 1000 × 1000 Game of Life on an
  idle machine with `pytest -m benchmark`, and record their timings in the change's design.md. A failure on an idle
  machine is a defect for a later change, not a reason to relax the benchmark

## 6. Gate

- [ ] 6.1 Run formatting, checks, the docs build, and tests through nox, and verify all of them pass
