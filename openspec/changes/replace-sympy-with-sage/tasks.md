# Tasks

## 1. Baseline and dependencies

- [x] 1.1 Record the trajectories of every library model and user guide example with fixed random states, and verify
  the recorded file loads in a comparison test that passes on the current code
- [x] 1.2 Make `passagemath-symbolics`, `passagemath-maxima`, and `cysignals` runtime dependencies, remove `sympy` and
  the `sage` extra, raise NumPy's floor to 2.1, and remove the `LAMMPS on Windows` workflow, and verify `pdm install`
  and `deptry` pass
- [x] 1.3 Add the private Sage import module in `core` and the `ruff` banned-API rule for `sympy` in `src`, and verify
  `ruff` reports every current `sympy` import

## 2. Symbols and laws

- [x] 2.1 Create state, parameter, and time symbols with `SR.var` and their domains, and vector states as Sage vectors,
  and verify the symbol tests pass
- [x] 2.2 Add `where`, `either`, `both`, and `mod`, and verify they stay unevaluated on symbols and evaluate on numbers
- [x] 2.3 Validate law values, reject SymPy with the named error, and verify the SymPy rate test passes

## 3. Lowering

- [x] 3.1 Add the expression printer with NumPy, Numba, and Lepton targets and common subexpression elimination, and
  verify one printer test per operator and target passes
- [x] 3.2 Lower kernels, the Numba backend, the steppers' symbol matching, and the Lepton potentials through the
  printer, and verify the compile, solver, and backend tests pass
- [x] 3.3 Port the physics models, boxes, potentials, and thermostat to Sage expressions and the new functions, and
  verify the reflection, periodic, and LAMMPS condition tests pass

## 4. Analysis

- [x] 4.1 Return equations as Sage relations and solve with `desolve` and `desolve_system`, and verify the logistic
  equation, free particle, harmonic oscillator, and non-ODE tests pass

## 5. Equivalence, documentation, and gate

- [x] 5.1 Compare every recorded trajectory with the new results, and verify all of them match within a relative
  tolerance of 1e-10
- [x] 5.2 Port the docstring examples, the README, and the user guide to Sage, add the migration table, and verify the
  documentation tests pass
- [x] 5.3 Update the Project Profile of the constitution for the dependencies and platforms with a MINOR version, add
  the breaking changelog entry, and verify `grep -rn sympy src docs README.md` finds no import or example
- [x] 5.4 Run formatting, checks, the docs build, the tests through nox, and the benchmarks with `--benchmarks`, and
  verify all of them pass, with the 100k soft-sphere benchmark still meeting real time

## Workflow follow-up

- Archive the change after review, then update the Purpose of the `symbolic-analysis` main spec, which still names
  SymPy, and delete the recorded trajectories.
