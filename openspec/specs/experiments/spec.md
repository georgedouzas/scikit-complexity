# experiments Specification

## Purpose

Lets a user study a system through many runs: replicates, parameter designs, scenarios, sensitivity analysis, and
calibration, with reproducible results.

## Requirements

### Requirement: Ensembles run replicates with independent streams
An experiment SHALL run replicates of a system, each with its own random stream derived from the experiment's seed, and
the results SHALL not depend on the number of worker processes.

#### Scenario: Reproducible ensemble
- **WHEN** an ensemble of 20 stochastic SIR runs with seed 7 executes serially and again on four processes
- **THEN** both executions return identical results

#### Scenario: Independent replicates
- **WHEN** an ensemble of 20 stochastic runs executes
- **THEN** no two replicates have identical trajectories

### Requirement: Designs sweep parameters and initial states
A design SHALL be the `design=` argument of `sweep`, one of `'grid'`, `'random'`, `'latin_hypercube'`, and `'sobol'`,
which gives the rows of model parameters and dotted initial states or part parameters that the system runs over.

#### Scenario: Orbit diagram sweep
- **WHEN** a user sweeps the growth rate of the logistic map over 200 values and keeps its last 100 values per run
- **THEN** the result holds 200 rows with the swept rate and the kept values

#### Scenario: Design without an experiment
- **WHEN** a user calls `sweep(system, {'r': (0.5, 1.5), 'K': (100, 1000)}, design='sobol', n=64)`
- **THEN** the result holds 64 runs, each with a value of `r` and of `K` inside its bounds

### Requirement: Scenarios compare policies
An experiment SHALL run named scenarios that override parameters and initial states before a run and change parameters
at scheduled times during it.

#### Scenario: Policy at a time
- **WHEN** a scenario halves the harvesting effort at time 50 and the baseline keeps it
- **THEN** both scenarios run with the same seeds, and their results differ only after time 50

### Requirement: Results are tidy and summarised
Experiment results SHALL hold one row per run with its design values, scenario, replicate, and seed, SHALL summarise
outputs across replicates after discarding a declared warm-up, and SHALL return the ensemble of a named series as one
array with one row per run aligned on common times.

#### Scenario: Summary across replicates
- **WHEN** a user summarises an ensemble of 50 runs per design point
- **THEN** the summary gives the mean, standard deviation, and quantiles of every output per design point and scenario

#### Scenario: Ensemble of a series
- **WHEN** a user reads the ensemble of `x` from an experiment of 200 replicates of an Ornstein-Uhlenbeck process
- **THEN** the result has one row per replicate and one column per common time, and the rows carry their seeds

### Requirement: Sensitivity analysis ranks parameters
The experiments layer SHALL hold `sensitivity`, a function that, called on a system with bounds,
computes one-at-a-time effects, Morris elementary effects, or Sobol indices of an output with respect to those
bounds, and returns them as fields of its result.

#### Scenario: Sobol indices of a known function
- **WHEN** a user calls `sensitivity` on the Ishigami function wrapped as a system with `method='sobol'`
- **THEN** the first-order indices in `first_order` of its result match the known values within 0.05

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

### Requirement: Dead worker processes raise a named error
An experiment run on processes SHALL raise a named error when its worker processes die, and the message SHALL name the
main guard that a script needs and the executors that avoid processes.

#### Scenario: Script without a main guard
- **WHEN** a script without `if __name__ == '__main__':` builds an orbit diagram with `executor='process'`
- **THEN** it raises `ExperimentError`, whose message names the main guard and `executor='thread'`

### Requirement: Basin stability is an experiment
The experiments layer SHALL estimate the basin stability of the attractors of a system by running it from initial
states sampled within bounds and classifying each final state, as the experiment function `basin_stability` whose
result holds the fraction of runs per attractor.

#### Scenario: Bistable system
- **WHEN** a user runs a basin stability experiment of a symmetric double-well system over symmetric bounds of the
  initial position
- **THEN** each of the two attractors gets a fraction within 0.05 of one half

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

### Requirement: Identifiability is an experiment function
The experiments layer SHALL hold `identifiability(system, data, bounds, *, method, ...)`, with `method` one of
`'profile'` and `'refits'`, which returns a frozen `IdentifiabilityResult` with the best-fit estimates and a verdict
per parameter, and SHALL raise `ValidationError` naming both methods for any other value.

#### Scenario: Unknown method
- **WHEN** a user calls `identifiability` with `method='fisher'`
- **THEN** it raises `ValidationError`, whose message names `'profile'` and `'refits'`

### Requirement: Profile likelihood gives intervals and verdicts
With `method='profile'`, `identifiability` SHALL profile the negative Gaussian log-likelihood over each parameter
within its bounds, re-optimising the others, and SHALL return each profile, the likelihood-ratio interval at `level`,
and the verdict `'identifiable'` when both ends of the interval lie inside the bounds, else `'not identifiable'`.

#### Scenario: Logistic growth through saturation
- **WHEN** a user profiles `r` and `K` of the logistic growth system on 30 points that reach saturation with 2% noise
- **THEN** both verdicts are `'identifiable'`, and each 95% interval holds the true value

#### Scenario: Data before saturation
- **WHEN** a user profiles `r` and `K` on points that end while the population is below a tenth of `K`
- **THEN** the verdict of `K` is `'not identifiable'`, and its interval reaches the upper bound

### Requirement: Refits under noise give relative errors
With `method='refits'`, `identifiability` SHALL simulate the best fit at the data times, add Gaussian noise of each
relative level in `noise` to `n` copies, refit every copy, and return the average relative error of every parameter
per level, with the verdict `'strong'` at most the level, `'weak'` at most ten times it, and `'none'` above.

#### Scenario: Saturating logistic data
- **WHEN** a user runs refits of the logistic growth system with 100 copies at the noise level 0.01 on data that reach
  saturation
- **THEN** the verdicts of `r` and `K` are `'strong'` or `'weak'`

#### Scenario: Reproducible refits
- **WHEN** a user runs the same refits twice with seed 3
- **THEN** both results hold the same relative errors

### Requirement: Virtual populations keep plausible parameters
The experiments layer SHALL hold `virtual_population(system, bounds, outputs, constraints, *, n, ...)`, which draws `n`
parameter rows from the priors, runs each, computes the named outputs of each trajectory, and returns a frozen
`VirtualPopulationResult` with the plausible rows, whose outputs lie inside the constraints, and the acceptance rate.

#### Scenario: Plausible patients
- **WHEN** a user samples 500 patients of the immuno-oncology QSP system with the constraint that the baseline tumour
  diameter lies between 1 and 10 centimetres
- **THEN** every plausible row has a baseline diameter in that range, and the acceptance rate equals their share of 500

#### Scenario: No plausible patient
- **WHEN** no row satisfies the constraints
- **THEN** `virtual_population` raises `ExperimentError`, whose message names the constraint that rejected most rows

### Requirement: Virtual populations match observed values by inclusion
Given `target`, observed values of one output, `virtual_population` SHALL select among the plausible rows with the
probability of inclusion proportional to the density of the observed values over the density of the plausible
population at each row's output, and SHALL hold the selected rows and their probabilities in the result.

#### Scenario: Match an observed distribution
- **WHEN** a user samples 2000 patients of the QSP system with a target of 100 observed log ratios of CD8 to
  regulatory T cells drawn from a normal distribution
- **THEN** the two-sample Kolmogorov-Smirnov statistic between the selected outputs and the target is below 0.15

### Requirement: Individuals are calibrated one by one
The experiments layer SHALL hold `calibrate_individuals(system, data, bounds, *, known=None, ...)`, which calls
`calibrate` on the data of every individual, with that individual's known model parameters set first, and returns a
frozen `IndividualCalibrationResult` with the calibration of every individual and a table of their estimates.

#### Scenario: Digital twins of virtual patients
- **WHEN** a user calibrates the tumour growth rate and the T cell killing rate of five virtual patients of the QSP
  system, each with its own dose, from their simulated tumour diameters with 2% noise
- **THEN** every estimate lies within 10% of that patient's true value, and the table holds one row per patient

#### Scenario: Known parameter that is also fitted
- **WHEN** a user names a parameter both in `known` for an individual and in `bounds`
- **THEN** `calibrate_individuals` raises `ValidationError`, whose message names the parameter and the individual

### Requirement: Basin stability starts count states from integers

Basin stability SHALL draw each start of a state whose domain holds only integers from the integers within the given
bounds, and each start of a real state from the real interval.

#### Scenario: Basin stability of a counting process

- **WHEN** basin stability is run on a system whose state counts events, with bounds from 0 to 5
- **THEN** every start of that state is an integer from 0 to 5 and the runs finish without a domain error

#### Scenario: Real states are unchanged

- **WHEN** basin stability is run on a double well with real states and a fixed seed
- **THEN** the starts and the fractions equal those drawn before this change
