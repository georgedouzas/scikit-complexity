# Tasks

## 1. Runtime axes

- [ ] 1.1 Add the internal `Form.STATIC` and `Form.PROCEDURE`, the law kind `PROCEDURE`, and the `Evaluation` and
  `ProcedureStepper` solvers with their `Supports` declarations, and verify the procedure frames and unsupported solver
  scenarios
- [ ] 1.2 Add `Sweep` as a run setting of `Simulation` and the catalog, and `Trajectory.axis`, and verify the dose
  sweep scenario
- [ ] 1.3 Log the sweep axis as a Rerun timeline, and verify the survival curve scenario

## 2. Algebraic models

- [ ] 2.1 Add `Define` with the law kind `DEFINE` in the constraint class, dependency ordering, and the cycle issue of
  validation, and verify the survival and cyclic definition scenarios
- [ ] 2.2 Add `unknowns` and `guess` to algebraic `Equation`, with Sage's exact solve and SciPy's `root` as the
  fallback, `SolveError`, and `best_response`, and verify the exact, numerical, no-solution, wrong unknown, and Cournot
  scenarios

## 3. Random ensembles

- [ ] 3.1 Add `Quenched` parameters drawn in `_prepare`, and verify the same-seed and disorder average scenarios
- [ ] 3.2 Add the `log_sum_exp` reduction on NumPy and Numba, and `log_partition` and `boltzmann_mean` to
  `skcomplex.models.physics.realizations`, and verify the entropy and large energy scenarios
- [ ] 3.3 Add `PointProcess` with its space requirement and the `clusters` analysis and observables, and verify the
  nested sweep, outside a continuous space, and spanning scenarios

## 4. Games, constraints, and search

- [ ] 4.1 Add `nashpy` as a runtime dependency, `Game` with its time requirement, `equilibria`, and the learning
  dynamics in `skcomplex.models.economics.realizations`, and verify the payoff, matching pennies, fictitious play, and
  dynamic under static time scenarios
- [ ] 4.2 Add `python-sat` as a runtime dependency, `Satisfiability` with incremental reveal and enumeration, and
  verify the pattern constraint, small perceptron, and capacity scenarios
- [ ] 4.3 Add `Learner` over scikit-learn `partial_fit` with its time requirement, and verify the separable patterns
  scenario
- [ ] 4.4 Add `pymoo` as a runtime dependency, `Search` over an `Objective` with the pymoo algorithms and annealing on
  the Metropolis acceptance of `Sample`, and verify the unknown algorithm, search under static time, Rastrigin, and
  cooling scenarios

## 5. Library

- [ ] 5.1 Add `lq_cell_survival`, `random_energy_model`, `oligopoly_equilibria`, `perceptron_satisfiability`,
  `evolutionary_optimisers`, `continuum_percolation`, and `von_thunen` as registered factories with default sweeps or
  iteration counts and catalog observables, and verify the fractionation, mixed price equilibrium, von Thünen rings,
  and glass transition scenarios and the conformance suite
- [ ] 5.2 Add the inventory entries for `random_energy_model` and `perceptron_satisfiability` and mark the six models
  implemented, and verify the inventory check passes
- [ ] 5.3 Add a gallery example per model with a Rerun thumbnail and the analysis plot of the design, and verify the
  gallery coverage, budget, and screenshot checks pass

## 6. Documentation and gate

- [ ] 6.1 Document the static form, sweeps, procedures, ensembles, games, constraint problems, learners, and searches
  in the user guide, add the dependencies to the Project Profile and the changelog, and verify the documentation test
  passes
- [ ] 6.2 Run formatting, checks, the docs build, and tests through nox, and verify all of them pass
