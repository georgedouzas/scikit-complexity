# Tasks

## 1. Static form

- [x] 1.1 Add `FormKind.STATIC` to `skcomplex.types` and the compilation of static systems in `PassCompiler`, with the
  exact solve through Sage and the summed equations of one target per element and per group member, and verify the
  survival at a dose, static system without equations, exact linear solution, and cyclic equations scenarios
- [x] 1.2 Add `RootSolver` to `skcomplex.engine.solvers` over `scipy.optimize.root` with the symbolic Jacobian and its
  `SupportSpec`, and verify the numerical solution and no solution scenarios
- [x] 1.3 Add the issue `equation-target` to `PassCompiler`, and verify the target that is not a state and rate law
  under static time scenarios
- [x] 1.4 Document the static form and sweeps through `Experiment` in the user guide with a runnable example, and verify
  the documentation test passes

## 2. Random ensembles

- [ ] 2.1 Add `log_sum_exp` to `skcomplex.core.laws.utils` with its NumPy and Numba lowering, and verify the large
  values and random energy model entropy scenarios
- [x] 2.2 Add `radius` to `ClusterAnalysis` with the contact graph of points, and verify the spanning probability and
  isolated points scenarios

## 3. Satisfiability and search

- [ ] 3.1 Keep the existing `python-sat>=1.8.dev16` specifier, add `pymoo>=0.6.2` to the runtime dependencies, name
  both in the Project Profile, and verify `pdm install`, `deptry`, and `pip-audit` pass
- [ ] 3.2 Add `ClauseSpec` and `CardinalitySpec` to `skcomplex.specs` and `SatisfiabilityInteraction` to
  `skcomplex.core.interactions`, and verify the perceptron pattern, real state, small perceptron, and capacity scenarios
- [ ] 3.3 Add `DifferentialEvolutionSpec`, `ParticleSwarmSpec`, `EvolutionStrategySpec`, and
  `CovarianceAdaptationSpec` to `skcomplex.specs` and `OptimizationInteraction` to `skcomplex.core.interactions`, and
  verify the differential evolution and algorithm that is not a spec scenarios
- [ ] 3.4 Add `cooling` to `SamplingInteraction`, and verify the cooling scenario

## 4. Library

- [ ] 4.1 Add `lq_cell_survival`, `random_energy_model`, `continuum_percolation`, and `von_thunen`, and verify the
  fractionation, glass transition, percolation threshold, and von Thünen rings scenarios and the conformance suite
- [ ] 4.2 Add `oligopoly_equilibria` with its static and fictitious-play dynamics, and verify the mixed price
  equilibrium and fictitious play scenarios and the conformance suite
- [ ] 4.3 Add `perceptron_satisfiability` and `evolutionary_optimisers`, and verify the satisfiable fraction and
  algorithms compared scenarios and the conformance suite
- [ ] 4.4 Add `gravity_urban_scaling` and the package `skcomplex.models.social.language` with
  `mandelbrot_least_effort`, and verify the urban scaling exponent and least-effort frequencies scenarios and the
  conformance suite
- [ ] 4.5 Mark the nine entries implemented in `docs/overview/inventory.toml`, add a gallery example per model with an
  analysis plot from an `Experiment`, and verify the inventory check and the gallery coverage, budget, and screenshot
  checks pass

## 5. Documentation and gate

- [ ] 5.1 Document `log_sum_exp`, contact clusters, `SatisfiabilityInteraction`, `OptimizationInteraction`, and
  annealing in the user guide, add the changelog entry, and verify the documentation test and the changelog session pass
- [ ] 5.2 Run `nox -s formatting`, `nox -s checks`, `nox -s docs`, and `nox -s tests` in that order, and verify all of
  them pass
