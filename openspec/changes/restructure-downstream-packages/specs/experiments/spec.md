## MODIFIED Requirements

### Requirement: Designs sweep parameters and initial states
An experiment SHALL run a model over grids, random samples, Latin hypercube samples, and Sobol samples of parameters and
initial states, addressed by factory argument or by dotted path. Every design SHALL be an estimator that checks its
settings, prepares against the bounds of the values it samples, and runs to rows of values.

#### Scenario: Orbit diagram sweep
- **WHEN** a user sweeps the growth rate of the logistic map over 200 values and keeps its last 100 values per run
- **THEN** the result holds 200 rows with the swept rate and the kept values

#### Scenario: Design without an experiment
- **WHEN** a user runs `SobolDesign(n=64).check().prepare({'r': (0.5, 1.5), 'K': (100, 1000)}).run()`
- **THEN** it returns 64 rows, each with a value of `r` and of `K` inside its bounds

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
The experiments layer SHALL hold a sensitivity analysis that, prepared on the result of an experiment over a design,
computes one-at-a-time effects, Morris elementary effects, or Sobol indices of an output with respect to the design's
bounds, and keeps them as fitted attributes.

#### Scenario: Sobol indices of a known function
- **WHEN** a user runs an experiment of the Ishigami function wrapped as a model over a Sobol design and prepares a
  sensitivity analysis with `method='sobol'` on its result
- **THEN** the first-order indices in `first_order_` match the known values within 0.05

### Requirement: Calibration fits parameters to data
The experiments layer SHALL hold a calibration experiment that fits model parameters within bounds to observed time
series and keeps the estimates with R², RMSE, MAE, and MAPE as fitted attributes.

#### Scenario: Recover logistic parameters
- **WHEN** a user runs a calibration experiment of the logistic growth model on data generated with `r = 0.8` and
  `K = 500` plus 1% noise
- **THEN** `estimates_` lies within 5% of the true values and `r2_` exceeds 0.99

## ADDED Requirements

### Requirement: Experiments are estimators with three verbs
Every experiment SHALL take a system or a factory of systems as its source, never a model name, since experiments
cannot import the model library. Every experiment SHALL store its source, design, scenarios, replicates, and analyses in its constructor, validate them in
`check()`, resolve the source and the runs in `prepare()`, and execute the runs in `run()`, keeping the result in
`result_`.

#### Scenario: Experiment with analyses
- **WHEN** a user runs an experiment of 20 stochastic SIR replicates with a peak analysis
- **THEN** `result_` holds one row per run with the fitted attributes of the analysis of that run

#### Scenario: Invalid replicates
- **WHEN** a user builds an experiment with zero replicates and checks it
- **THEN** a named error names `replicates` and the value

### Requirement: Basin stability is an experiment
The experiments layer SHALL estimate the basin stability of the attractors of a system by running it from initial
states sampled within bounds and classifying each final state, as an experiment whose fitted attributes hold the
fraction of runs per attractor.

#### Scenario: Bistable system
- **WHEN** a user runs a basin stability experiment of a symmetric double-well system over symmetric bounds of the
  initial position
- **THEN** each of the two attractors gets a fraction within 0.05 of one half

### Requirement: The experiments surface is estimators only
The public surface of the experiments package SHALL hold only designs, experiments, the sensitivity analysis, scenarios,
the experiment result, and the bases a user subclasses, and SHALL hold no function.

#### Scenario: Surface listing
- **WHEN** a user lists the public names of the experiments package
- **THEN** every name is a class, and none of `calibrate`, `sensitivity`, `run_task`, or `resolve_source` exists

#### Scenario: Model of the library as a source
- **WHEN** a user runs an experiment of a library model
- **THEN** they pass the model's factory or a system created by `skcomplex.models`, and a model name raises a named error
  that says to pass a system or a factory

