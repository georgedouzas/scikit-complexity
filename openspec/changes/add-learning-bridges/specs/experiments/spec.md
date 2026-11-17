# Spec Delta

## ADDED Requirements

### Requirement: Identification and calibration select their settings by cross-validation
`identify` and `calibrate` SHALL accept a scikit-learn splitter or a number of folds as `cv`, with `groups` for group
splitters, a scorer as `scoring`, and a parameter grid as `param_grid`. They SHALL fit every setting on the training
runs of every fold, score it on the test runs, fit all the runs again with the best setting, and return a frozen result
that reports `cv_results`, `best_params`, and `best_score`.

#### Scenario: Folds over runs
- **WHEN** `identify` runs with `cv=KFold(5)` on 20 observed runs of a Lotka-Volterra system with a learned term
- **THEN** `cv_results` holds 5 test scores, and every fold fits on 16 runs and scores on 4

#### Scenario: Groups kept whole
- **WHEN** `calibrate` runs with `cv=GroupKFold(3)` and `groups` naming a condition shared by the runs of each of 6
  patients
- **THEN** no fold has runs of one patient in both its training and its test runs

#### Scenario: Grid of learning rates
- **WHEN** `identify` searches 3 learning rates with 3 folds and `scoring='r2'`
- **THEN** `best_params` holds the learning rate of the highest mean test score, `best_score` equals that score, and
  the `system` of the result is a system that `simulate` runs

#### Scenario: Same seed, same search
- **WHEN** the search runs twice with the same `random_state`
- **THEN** both results hold equal `cv_results` scores and equal estimates

#### Scenario: No selection
- **WHEN** `identify` runs without `cv` and `param_grid`
- **THEN** `cv_results`, `best_params`, and `best_score` are `None`

### Requirement: Selection splits the times of a run
`identify` and `calibrate` SHALL take `split='time'`, with which `cv` splits the observation times of every run in time
order and every test part is scored by running the fitted system from the start of its run. They SHALL refuse a
splitter that shuffles times.

#### Scenario: Selection on one run
- **WHEN** `identify` runs with `split='time'` and `cv=TimeSeriesSplit(3)` on the one run of the pelt records
- **THEN** `cv_results` holds 3 test scores, and every test part holds only years after those of its training part

#### Scenario: Shuffled times
- **WHEN** `split='time'` is given with `KFold(5, shuffle=True)`
- **THEN** it raises `ValidationError` that says a split by time keeps the order of times and names `TimeSeriesSplit`

### Requirement: Fits report information criteria
`IdentificationResult` and `CalibrationResult` SHALL hold the maximised log-likelihood, `aic`, and `bic`, counting every
fitted value, learned weights, initial states, and noise scales included, and the observed values that are not missing.

#### Scenario: Criteria of a Gaussian fit
- **WHEN** a model of 3 fitted values is calibrated on 50 observed values with Gaussian noise
- **THEN** `aic` equals `6 - 2 log L` and `bic` equals `3 log 50 - 2 log L` within 1e-10

#### Scenario: Rival models of the pelts
- **WHEN** the mechanistic and the hybrid forms of `UniversalLotkaVolterraSystem` are identified on the pelt records of
  1900 to 1915 and ranked by `aic` and by the error on 1916 to 1920 held out with `split='time'`
- **THEN** both rankings are reported, and the result of each fit holds its criteria and its held-out error

### Requirement: Bayesian fits report predictive scores
`calibrate(method='mcmc')` SHALL report `loo`, the Pareto smoothed importance sampling leave-one-out score, and `waic`,
each with its expected log pointwise predictive density, standard error, and effective number of parameters, from the
pointwise log-likelihoods of the posterior draws. Other methods SHALL report `None`.

#### Scenario: Missing term
- **WHEN** a logistic growth model and an exponential growth model are calibrated with `'mcmc'` on 40 noisy points of
  a logistic run
- **THEN** the logistic model has the higher `loo` and `waic` scores, each by more than twice the standard error of
  the difference, and the largest Pareto shape of the logistic fit is below 0.7

### Requirement: Fits report their residuals
`IdentificationResult` and `CalibrationResult` SHALL hold `residuals`, the observed minus the fitted values at the
observation times as observed runs, the one step prediction errors when the fit used `horizon=1` and the output errors
otherwise.

#### Scenario: Residuals of a run
- **WHEN** a fit reads 2 runs of 30 observation times
- **THEN** `residuals` holds 2 runs of 30 times whose values equal the data minus the run of the fitted system

### Requirement: Identification fits replicates
`identify` SHALL take `replicates`, fit that many times from seeds drawn from `random_state`, and report the fitted
systems and the standard deviation of every estimate over them, keeping the fit of the first seed as `system`.

#### Scenario: Spread of a network
- **WHEN** a universal Lotka-Volterra system is identified with `replicates=5`
- **THEN** the result holds 5 systems that `simulate` runs, `spread` holds one value per estimate, and `system` equals
  the result of a call with `replicates=1` and the same `random_state`
