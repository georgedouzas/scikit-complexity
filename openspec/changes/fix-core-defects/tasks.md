# Tasks

Each task writes its regression test first, checks that the test fails on the current code, and then fixes the
defect. Every test follows the naming and public-API rules of the constitution.

## 1. Boundaries after every solver step

- [ ] 1.1 Apply the boundary after every internal step of the NumPy and fused Numba loops of the fixed-step, SSP, split,
  iteration, and Gillespie steppers, and after every accepted step of `ScipyODE`, restarting it when the state changes.
  Stop applying it per frame in `Simulation` and co-simulation, and state in the user guide that boundaries act after
  every solver step. Verify the long-frame reflecting box scenario with `RK4` and `ScipyODE`, the existing box and gas
  tests, the real-time gas benchmarks, and the documentation test

## 2. Mesa clock and agent groups

- [ ] 2.1 Run each native agent step through `model_.step()` on a `mesa.Model` subclass, and verify the scenario of
  rules reading the model time, the reproducible-run test, and the wrapped-model tests
- [ ] 2.2 Read per-member values and locations of one-member groups from one-entry sequences, and verify the
  one-agent-at-a-cell scenario
- [ ] 2.3 Reject agent state names that Mesa reads, `position`, `pos`, `cell`, `unique_id`, and `model`, with a named
  error when the type is defined, and verify the state named `position` scenario

## 3. Conditions and equations

- [ ] 3.1 Simplify `where` with equal branches to that branch, and verify the derivative-of-a-condition scenario and
  the driver and condition scenarios of equations
- [ ] 3.2 Evaluate both branches of `where` on NumPy inside a NumPy error state that ignores division, invalid, and
  overflow warnings, and verify the branch-not-chosen scenario with warnings turned into errors

## 4. Values given to laws and observables

- [ ] 4.1 Convert NumPy scalar law values to Sage numbers and reject NumPy array law values with a named error, and
  verify the two NumPy scenarios of Sage expressions
- [ ] 4.2 Reject `Jump` increments that hold symbols with a named error, and verify the parameter-as-increment scenario
- [ ] 4.3 Check at construction that an observable's expression and callable reduction pickle, and move the lambda
  observables of docstrings, tests, and the user guide to module-level functions or symbolic expressions. Add the
  breaking changelog entries for NumPy array law values and lambda observables. Verify the lambda and module-level
  observable scenarios and the documentation test

## 5. Runs and frames

- [ ] 5.1 Keep the parameters of the previous run in `_prepare`, document it in `run` and the user guide, and add its
  breaking changelog entry. Verify the change-between-runs scenarios for a logistic population and an agent group
- [ ] 5.2 Add `Simulation.extend(until)` with its docstring example and a user guide section, and verify the
  run-in-two-parts and extend-backwards scenarios and the documentation test
- [ ] 5.3 Choose the default frame duration of continuous forms from `until` in 1, 2, 5 steps of at most 1000 frames,
  document the rule on `dt`, and verify the short and long default frame scenarios and the catalog tests that use
  default frames

## 6. Solver defects

- [ ] 6.1 Keep one SciPy integrator across frames of `ScipyODE`, restarting it on parameter, state, breakpoint, event,
  and boundary changes, and verify the many-frames and parameter-change scenarios and the existing delay, event, and
  driver tests
- [ ] 6.2 Limit the first step of `ScipyODE` to half the smallest delay, and verify the very-short-delay scenario
- [ ] 6.3 Restart the `IMEX2` history when a call does not continue the previous one, and verify the
  second-call scenario and the implicit-explicit agreement test
- [ ] 6.4 Record the diffusive limit of Laplacian rates on finite-volume fields as an expression of the parameters, and
  bound the `SSPRK` step by it, and verify the viscous Burgers scenario and the Burgers shock test

## 7. Compiled code

- [ ] 7.1 Compile the kernels called per event by the fused Gillespie, iteration, and lattice loops with
  `parallel=False`, and keep parallel drift and pair kernels for groups of at least 1000 members. Verify the compiled
  Gillespie scenario and the million-member gas benchmark
- [ ] 7.2 Store the sizes, slices, and shapes of `Entry` once, and resolve kernel references to fixed slices and shapes
  at build time. Verify the small-system scenario and the backend agreement tests

## 8. Gate

- [ ] 8.1 Run formatting, checks, the docs build, and tests through nox, and verify all of them pass
