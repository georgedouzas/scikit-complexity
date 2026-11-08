## 1. Dependencies

- [x] 1.1 Add `optuna>=5.0.0`, `pyabc>=0.13.1`, and `numpyro>=0.22.0` to the runtime dependencies in `pyproject.toml`, and verify `pdm install` resolves and the three import
- [x] 1.2 Name the three libraries and their uses in the runtime dependencies of the Project Profile in `openspec/constitution.md`, and verify the dependency freshness check passes

## 2. Objective and priors

- [ ] 2.1 Move the closures of `calibrate` into the private class `_CalibrationObjective` in `_calibration.py`, and verify the existing calibration tests and doctest pass unchanged
- [ ] 2.2 Accept SciPy distributions as bounds through `_check_bounds`, derive the search box of point fits, and test a normal prior and a rejected string bound
- [ ] 2.3 Add `method` with the table of allowed method and loss pairs, and test the errors for an unknown method and for a loss that does not fit
- [ ] 2.4 Add `samples`, `weights`, `diagnostics`, and `predictive` to `CalibrationResult`, and test that a `'local'` fit holds `None` in the posterior fields

## 3. Point fits

- [ ] 3.1 Add `method='global'` with a seeded Optuna TPE study over `budget` trials, and test reproducible estimates on `BrockHommesSystem`
- [ ] 3.2 Document `'local'` and `'global'` in the docstring of `calibrate` with a runnable example, and verify the doctest passes

## 4. Recorded draws and replay

- [ ] 4.1 Add `draw(name, distribution, **arguments)` to the system view protocol and its implementations, and test that a rule's draw samples from `rng`
- [ ] 4.2 Add `record_draws` to `simulate` and the draw recorder of the NumPy backend, store draws by name and step in `Trajectory`, and test the recorded `kick` draws of a noisy map
- [ ] 4.3 Record the time and law of every firing in `GillespieSolver`, and test that the states rebuilt from the firings match the recorded states
- [ ] 4.4 Add `replay` to `simulate` with forced draws and `log_probability`, and test replay at the recording parameters and the error for draws that do not match
- [ ] 4.5 Add the Gillespie path log-likelihood to replay, and test it against the closed form on a recorded SIR run
- [ ] 4.6 Document recording and replay in the user guide page on simulation, and verify the strict docs build runs its examples

## 5. Event likelihood

- [ ] 5.1 Add `steepness` to `DeffuantSystem` with the outcome drawn through the view's named draw, and test that steepness 1e4 matches the deterministic run
- [ ] 5.2 Add `loss='events'` that takes a recorded trajectory, and test the recovery of `confidence` of Deffuant within 0.02 and of `beta` and `gamma` of stochastic SIR within 10%

## 6. Posterior methods

- [ ] 6.1 Map the supported SciPy prior families to numpyro distributions, and test that an unsupported family raises `ValidationError` for `'mcmc'`
- [ ] 6.2 Add `method='abc_smc'` on pyabc with summaries, a temporary database, and restored global random state, and test the stochastic SIR interval scenario and the per-generation diagnostics
- [ ] 6.3 Move the unroll of `GradientDescentSolver` to a method of `JaxBackend` that both callers use, and verify the learning-dynamics tests pass unchanged
- [ ] 6.4 Add `method='mcmc'` with NUTS on differentiable systems and AIES with a host callback otherwise, and test R-hat and effective sample size on the logistic posterior
- [ ] 6.5 Add `method='variational'` with numpyro SVI and the tag check, and test the error on a system without `differentiable`
- [ ] 6.6 Add posterior predictive quantiles and the share inside the band to every posterior method, and test the well-specified logistic case
- [ ] 6.7 Add the user guide section on Bayesian calibration in `docs/overview/user_guide/run_experiments.md`, covering methods, priors, summaries, events, and diagnostics, and verify its examples run

## 7. Coverage

- [ ] 7.1 Add `posterior_coverage` and `CoverageResult` in `src/skcomplex/core/experiments/_coverage.py`, export them, and test the coverage of the logistic posterior at levels 0.5 and 0.9
- [ ] 7.2 Document `posterior_coverage` with a runnable docstring example and a paragraph in the user guide, and verify both run

## 8. Probe systems

- [ ] 8.1 Add `OffLatticeTumourGrowthSystem` in its own module of `biology/oncology`, with tests of biphasic growth and spring spacing
- [ ] 8.2 Add `CellCycleInvasionSystem` in its own module of `biology/oncology`, with tests of the scratch assay and the full lattice
- [ ] 8.3 Add `DifferentiableCovidAbmSystem` in its own module of `biology/epidemiology`, with tests of the gradient against finite differences and of zero transmission
- [ ] 8.4 Test the variational recovery of the three betas of `DifferentiableCovidAbmSystem` within 0.15
- [ ] 8.5 Add inventory entries for `off_lattice_tumour_growth`, `cell_cycle_invasion`, and `differentiable_covid_abm` with sources 2409.19675 and 2305.15340, and verify the inventory check passes
- [ ] 8.6 Add gallery examples for the three systems, each with a calibration (ABC-SMC for the two oncology systems, variational for the epidemic), and one for the event likelihood of `DeffuantSystem`, and verify strict gallery coverage passes

## 9. Integration

- [ ] 9.1 Add a changelog entry for the new methods and the breaking fields of `CalibrationResult`, and verify the docs build includes it
- [ ] 9.2 Run `nox` checks, tests, and the strict docs build, and fix every finding
