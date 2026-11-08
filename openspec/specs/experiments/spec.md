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
The experiments layer SHALL hold a calibration experiment that fits system parameters within bounds to observed time
series and keeps the estimates with R², RMSE, MAE, and MAPE as fitted attributes.

#### Scenario: Recover logistic parameters
- **WHEN** a user runs a calibration experiment of the logistic growth system on data generated with `r = 0.8` and
  `K = 500` plus 1% noise
- **THEN** `estimates_` lies within 5% of the true values and `r2_` exceeds 0.99

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

### Requirement: Experiments are functions, except calibration
Parameter sweeps, ensembles, scenarios, sensitivity, and basin stability SHALL be public functions of a system or of a
factory of systems, in the role of `validation_curve` and `cross_validate` with repeats, and SHALL return tidy
results. Calibration SHALL stay an estimator, in the role of `GridSearchCV`, whose `run` returns the system with fitted
parameters. Every experiment SHALL check the tags it needs, such as `stochastic` for an ensemble over seeds or
`differentiable` for gradient calibration.

#### Scenario: A sweep
- **WHEN** a user calls a sweep of a system over one parameter with a bound analysis
- **THEN** it returns a table with one row per value and the analysis's result per row

#### Scenario: Calibration fits a system
- **WHEN** calibration runs on a system and observed data
- **THEN** its fitted attribute holds the system with the fitted parameters
