## MODIFIED Requirements

### Requirement: Calibration fits parameters to data
`calibrate(system, data, bounds, ...)` of `skcomplex.core.experiments` SHALL fit the model parameters of a system within
bounds to observed time series, and SHALL return a frozen `CalibrationResult` that holds the fitted system as `system`,
the estimates as `estimates`, and the goodness of fit as `r2`, `rmse`, `mae`, and `mape`. Its `method` SHALL default to
`'local'`, the local SciPy fit, so a call without `method` keeps its former estimates.

#### Scenario: Recover logistic parameters
- **WHEN** a user calibrates the logistic growth system on data generated with `r = 0.8` and `K = 500` plus 1% noise
- **THEN** `estimates` lies within 5% of the true values and `r2` exceeds 0.99

#### Scenario: Default method
- **WHEN** a user calibrates the logistic growth system without `method` and again with `method='local'`
- **THEN** both results hold the same estimates

### Requirement: Experiments are functions
Parameter sweeps, ensembles, scenarios, sensitivity, basin stability, calibration, and posterior coverage SHALL be
public functions of a system or of a factory of systems, in the role of `validation_curve`, `cross_validate` with
repeats, and the search functions of scikit-learn, and SHALL return tidy results. Every experiment SHALL check the tags
it needs, such as `stochastic` for an ensemble over seeds or `differentiable` for gradient calibration.

#### Scenario: A sweep
- **WHEN** a user calls a sweep of a system over one parameter with a bound analysis
- **THEN** it returns a table with one row per value and the analysis's result per row

#### Scenario: Calibration fits a system
- **WHEN** a user calls `calibrate` on a system and observed data
- **THEN** it returns a `CalibrationResult` whose `system` holds the system with the fitted parameters

#### Scenario: Variational calibration needs a differentiable system
- **WHEN** a user calls `calibrate` with `method='variational'` on a system whose tags do not hold `differentiable`
- **THEN** it raises `ValidationError`, whose message names the tag `differentiable` and the methods that do not need it

## ADDED Requirements

### Requirement: Calibration methods are named
`calibrate` SHALL take `method`, one of `'local'`, `'global'`, `'abc_smc'`, `'mcmc'`, and `'variational'`, and SHALL
raise `ValidationError` naming the allowed methods for any other value, and naming the allowed losses when the loss
does not fit the method.

#### Scenario: Unknown method
- **WHEN** a user calls `calibrate` with `method='nested'`
- **THEN** it raises `ValidationError`, whose message lists the five methods

#### Scenario: Loss that does not fit the method
- **WHEN** a user calls `calibrate` with `method='abc_smc'` and `loss='events'`
- **THEN** it raises `ValidationError`, whose message names the losses that ABC-SMC accepts

### Requirement: Bounds are priors
Every bound of `calibrate` SHALL be a pair `(low, high)`, read as a uniform prior, or a SciPy distribution, read as the
prior of that parameter. Point fits SHALL search within the pair, or within the central 99.9% of the distribution.

#### Scenario: Normal prior
- **WHEN** a user calibrates the logistic growth system by MCMC with the prior `scipy.stats.norm(0.8, 0.05)` on `r`
- **THEN** the standard deviation of the samples of `r` is below 0.05 and their mean lies within 0.1 of 0.8

#### Scenario: Bound that is neither
- **WHEN** a user passes the string `'wide'` as the bound of `r`
- **THEN** `calibrate` raises `ValidationError`, whose message asks for a pair or a SciPy distribution

### Requirement: Global point fits search the bounds
`calibrate` with `method='global'` SHALL search the bounds with a seeded tree-structured Parzen estimator for `budget`
evaluations of the loss, and SHALL return the best evaluated parameters as the estimates.

#### Scenario: Rough loss of a stochastic model
- **WHEN** a user calibrates the two belief parameters of the Brock-Hommes system to its own simulated prices with
  `method='global'`, a budget of 200, and seed 0, twice
- **THEN** both calls return the same estimates, and the loss at them is below the loss at the midpoints of the bounds

### Requirement: ABC-SMC samples an approximate posterior
`calibrate` with `method='abc_smc'` SHALL sample an approximate posterior by sequential Monte Carlo approximate
Bayesian computation, comparing the summaries of simulated and observed data with an adaptive distance, and SHALL stop
at a final tolerance or after `budget` simulations.

#### Scenario: Stochastic SIR
- **WHEN** a user calibrates `beta` and `gamma` of the stochastic SIR system by ABC-SMC to the infected counts of one
  run at `beta = 0.3` and `gamma = 0.1`
- **THEN** the central 90% interval of the weighted samples of each parameter holds its true value

#### Scenario: Summary statistics
- **WHEN** a user passes `summary=` a function that maps the series to the final counts of three cell phases
- **THEN** ABC-SMC compares the three counts of every simulation with the three observed counts

### Requirement: MCMC samples the posterior of a likelihood
`calibrate` with `method='mcmc'` SHALL sample the posterior of the Gaussian likelihood or of the event likelihood with
`n_samples` draws per chain. It SHALL use a gradient-based sampler on a differentiable system and a gradient-free
ensemble sampler otherwise.

#### Scenario: Logistic posterior
- **WHEN** a user calibrates the logistic growth system by MCMC on data with 2% noise, with four chains of 500 samples
- **THEN** the diagnostics hold a split R-hat below 1.05 and an effective sample size above 200 for every parameter

### Requirement: Variational calibration runs through differentiable runs
`calibrate` with `method='variational'` SHALL fit a variational posterior by stochastic gradients of the expected loss
plus the divergence from the prior, differentiating the simulated series through the run. With `loss='likelihood'` the
loss SHALL be the negative Gaussian log-likelihood, and with `loss='squares'` the sum of squares.

#### Scenario: Differentiable epidemic agents
- **WHEN** a user calibrates the three location betas of the differentiable COVID agent system to the log daily
  infections of a run at `(0.9, 0.6, 0.3)` with `method='variational'`
- **THEN** the posterior median of every beta lies within 0.15 of its true value

### Requirement: The event likelihood scores the model's own draws
`calibrate` with `loss='events'` SHALL take as data a trajectory recorded with its draws, and SHALL score candidate
parameters by the log-probability of those draws when the run is replayed under the candidate parameters.

#### Scenario: Bounded confidence from interactions
- **WHEN** a user calibrates `confidence` of the stochastic Deffuant system with steepness 50 by `loss='events'` on a
  recorded run of 100 agents over 50 steps at `confidence = 0.25`
- **THEN** the estimate lies within 0.02 of 0.25

#### Scenario: Gillespie firings
- **WHEN** a user calibrates `beta` and `gamma` of the stochastic SIR system by `loss='events'` on one recorded
  Gillespie run of 1000 people
- **THEN** both estimates lie within 10% of the true values

### Requirement: Posterior results hold samples and diagnostics
A `CalibrationResult` of a posterior method SHALL hold the samples by parameter as `samples`, their normalised weights
as `weights`, the diagnostics of the method as `diagnostics`, and the posterior median as `estimates`. A point fit SHALL
hold `None` in `samples`, `weights`, and `predictive`.

#### Scenario: Fields of a point fit
- **WHEN** a user calibrates with `method='local'`
- **THEN** `samples`, `weights`, and `predictive` are `None`, and `covariance` holds the Jacobian covariance

#### Scenario: Diagnostics of ABC-SMC
- **WHEN** a user calibrates with `method='abc_smc'`
- **THEN** `diagnostics` holds the tolerance and the acceptance rate of every generation

### Requirement: Posterior predictive checks are part of the result
A `CalibrationResult` of a posterior method SHALL hold in `predictive` the 5%, 50%, and 95% quantiles of the simulated
series of posterior draws at the data times, by path, and the share of observed points inside the central 90% band.

#### Scenario: Well-specified model
- **WHEN** a user calibrates the logistic growth system by MCMC on data from the same system with 5% Gaussian noise
- **THEN** the share of observed points inside the 90% band lies between 0.75 and 1

### Requirement: Posterior coverage is an experiment
`posterior_coverage(system, bounds, times, ...)` SHALL draw parameters from the priors, simulate data at them, calibrate
each data set with a posterior method, and return a frozen `CoverageResult` with the empirical coverage of the central
credible intervals at each requested level, by parameter, and the rank of every true value among its samples.

#### Scenario: Calibrated posterior
- **WHEN** a user calls `posterior_coverage` on the logistic growth system with MCMC, 40 truths, and the levels 0.5 and
  0.9
- **THEN** the empirical coverage at each level lies within 0.2 of the level for every parameter
