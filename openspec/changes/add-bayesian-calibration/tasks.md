## 1. Dependencies

- [x] 1.1 Add `optuna>=5.0.0`, `pyabc>=0.13.1`, and `numpyro>=0.22.0` to the runtime dependencies in `pyproject.toml`, and verify `pdm install` resolves and the three import
- [x] 1.2 Name the three libraries and their uses in the runtime dependencies of the Project Profile in `openspec/constitution.md`, and verify the dependency freshness check passes

## 2. Objective and priors

- [x] 2.1 Move the closures of `calibrate` into the private class `_CalibrationObjective` in `_calibration.py`, and verify the existing calibration tests and doctest pass unchanged
- [x] 2.2 Accept SciPy distributions as bounds through `_check_bounds`, derive the search box of point fits, and test a normal prior and a rejected string bound
- [x] 2.3 Add `method` with the table of allowed method and loss pairs, and test the errors for an unknown method and for a loss that does not fit
- [x] 2.4 Add `samples`, `weights`, `diagnostics`, and `predictive` to `CalibrationResult`, and test that a `'local'` fit holds `None` in the posterior fields

## 3. Point fits

- [x] 3.1 Add `method='global'` with a seeded Optuna TPE study over `budget` trials, and test reproducible estimates on `BrockHommesSystem`
- [x] 3.2 Document `'local'` and `'global'` in the docstring of `calibrate` with a runnable example, and verify the doctest passes

## 4. Recorded draws and replay

- [x] 4.1 Add `draw(name, distribution, **arguments)` to the system view protocol and its implementations, and test that a rule's draw samples from `rng`
- [x] 4.2 Add `record_draws` to `simulate` and the draw recorder of the NumPy backend, store draws by name and step in `Trajectory`, and test the recorded `kick` draws of a noisy map
- [x] 4.3 Record the time and law of every firing in `GillespieSolver`, and test that the states rebuilt from the firings match the recorded states
- [x] 4.4 Add `replay` to `simulate` with forced draws and `log_probability`, and test replay at the recording parameters and the error for draws that do not match
- [x] 4.5 Add the Gillespie path log-likelihood to replay, and test it against the closed form on a recorded SIR run
- [x] 4.6 Document recording and replay in the user guide page on simulation, and verify the strict docs build runs its examples

## 5. Event likelihood

- [x] 5.1 Add `steepness` to `DeffuantSystem` with the outcome drawn through the view's named draw, and test that steepness 1e4 matches the deterministic run
- [x] 5.2 Add `loss='events'` that takes a recorded trajectory, and test the recovery of `confidence` of Deffuant within 0.02 and of `beta` and `gamma` of stochastic SIR within 10%

## 6. Posterior methods

- [x] 6.1 Map the supported SciPy prior families to numpyro distributions, and test that an unsupported family raises `ValidationError` for `'mcmc'`
- [x] 6.2 Add `method='abc_smc'` on pyabc with summaries, a temporary database, and restored global random state, and test the stochastic SIR interval scenario and the per-generation diagnostics
- [x] 6.3 Move the unroll of `GradientDescentSolver` to a method of `JaxBackend` that both callers use, and verify the learning-dynamics tests pass unchanged
- [x] 6.4 Add `method='mcmc'` with NUTS on differentiable systems and AIES with a host callback otherwise, and test R-hat and effective sample size on the logistic posterior
- [x] 6.5 Add `method='variational'` with numpyro SVI and the tag check, and test the error on a system without `differentiable`
- [x] 6.6 Add posterior predictive quantiles and the share inside the band to every posterior method, and test the well-specified logistic case
- [x] 6.7 Add the user guide section on Bayesian calibration in `docs/overview/user_guide/run_experiments.md`, covering methods, priors, summaries, events, and diagnostics, and verify its examples run

## 7. Coverage

- [x] 7.1 Add `posterior_coverage` and `CoverageResult` in `src/skcomplex/core/experiments/_coverage.py`, export them, and test the coverage of the logistic posterior at levels 0.5 and 0.9
- [x] 7.2 Document `posterior_coverage` with a runnable docstring example and a paragraph in the user guide, and verify both run

## 8. Relaxed draws and Delaunay neighbourhoods

- [x] 8.1 Add `temperature` and `hard` to `bernoulli`, `categorical`, and `DrawSpec`, and test that NumPy runs draw exact values with them
- [ ] 8.2 Add the symbolic `stop_gradient`, lowered by the JAX printer to `jax.lax.stop_gradient` and by the others to the identity, and test both lowerings
- [ ] 8.3 Lower relaxed draws on JAX to the concrete relaxation with straight-through hard values, draw the uniform values of the differentiable unroll from a JAX key, and test the gradient against finite differences and the low-temperature limit
- [ ] 8.4 Report a `not-differentiable` issue for a Bernoulli or categorical draw without a temperature in the JAX lowering, and test its message
- [x] 8.5 Add `DelaunayNeighborhood` in `spaces/_delaunay.py` with the search in `_metric_search.py`, periodic images, and per-step recomputation, and test the known point set, the periodic boundary, and moving members

## 9. Probe systems

- [ ] 9.1 Add `OffLatticeTumourGrowthSystem` in its own module of `biology/oncology`, built on `DelaunayNeighborhood`, with tests of biphasic growth and spring spacing
- [x] 9.2 Add `CellCycleInvasionSystem` in its own module of `biology/oncology`, with tests of the scratch assay and the full lattice
- [ ] 9.3 Add `DifferentiableCovidAbmSystem` in its own module of `biology/epidemiology`, with discrete agents and relaxed draws, with tests of the gradient against finite differences and of zero transmission
- [ ] 9.4 Test the variational recovery of the three betas of `DifferentiableCovidAbmSystem` within 0.15
- [ ] 9.5 Add inventory entries for `off_lattice_tumour_growth`, `cell_cycle_invasion`, and `differentiable_covid_abm` with sources 2409.19675 and 2305.15340, and verify the inventory check passes
- [ ] 9.6 Add gallery examples for the three systems, each with a calibration (ABC-SMC for the two oncology systems, variational for the epidemic), and one for the event likelihood of `DeffuantSystem`, and verify strict gallery coverage passes

## 10. Integration

- [ ] 10.1 Add a changelog entry for the new methods and the breaking fields of `CalibrationResult`, and verify the docs build includes it
- [ ] 10.2 Run `nox` checks, tests, and the strict docs build, and fix every finding
