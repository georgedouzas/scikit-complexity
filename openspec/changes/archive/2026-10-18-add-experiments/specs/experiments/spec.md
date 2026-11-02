# Spec Delta

## Purpose

Lets a user study a model through many runs: replicates, parameter designs, scenarios, sensitivity analysis, and
calibration, with reproducible results.

## ADDED Requirements

### Requirement: Ensembles run replicates with independent streams
An experiment SHALL run replicates of a model, each with its own random stream derived from the experiment's seed, and
the results SHALL not depend on the number of worker processes.

#### Scenario: Reproducible ensemble
- **WHEN** an ensemble of 20 stochastic SIR runs with seed 7 executes serially and again on four processes
- **THEN** both executions return identical results

#### Scenario: Independent replicates
- **WHEN** an ensemble of 20 stochastic runs executes
- **THEN** no two replicates have identical trajectories

### Requirement: Designs sweep parameters and initial states
An experiment SHALL run a model over grids, random samples, Latin hypercube samples, and Sobol samples of parameters and
initial states, addressed by factory argument or by dotted path.

#### Scenario: Orbit diagram sweep
- **WHEN** a user sweeps the growth rate of the logistic map over 200 values and keeps its last 100 values per run
- **THEN** the result holds 200 rows with the swept rate and the kept values

### Requirement: Scenarios compare policies
An experiment SHALL run named scenarios that override parameters and initial states before a run and change parameters
at scheduled times during it.

#### Scenario: Policy at a time
- **WHEN** a scenario halves the harvesting effort at time 50 and the baseline keeps it
- **THEN** both scenarios run with the same seeds, and their results differ only after time 50

### Requirement: Results are tidy and summarised
Experiment results SHALL hold one row per run with its design values, scenario, replicate, and seed, and SHALL
summarise outputs across replicates, after discarding a declared warm-up.

#### Scenario: Summary across replicates
- **WHEN** a user summarises an ensemble of 50 runs per design point
- **THEN** the summary gives the mean, standard deviation, and quantiles of every output per design point and scenario

### Requirement: Sensitivity analysis ranks parameters
The experiments layer SHALL compute one-at-a-time effects, Morris elementary effects, and Sobol indices of an output
with respect to declared parameter bounds.

#### Scenario: Sobol indices of a known function
- **WHEN** a user computes Sobol indices of the Ishigami function wrapped as a model
- **THEN** the first-order indices match the known values within 0.05

### Requirement: Calibration fits parameters to data
The experiments layer SHALL fit model parameters to observed time series and report the estimates with R², RMSE, MAE,
and MAPE.

#### Scenario: Recover logistic parameters
- **WHEN** a user calibrates the logistic growth model to data generated with `r = 0.8` and `K = 500` plus 1% noise
- **THEN** the estimates lie within 5% of the true values and R² exceeds 0.99
