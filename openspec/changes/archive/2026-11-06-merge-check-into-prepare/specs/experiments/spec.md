## RENAMED Requirements

- FROM: `### Requirement: Experiments are estimators with three verbs`
- TO: `### Requirement: Experiments are estimators with two verbs`

## MODIFIED Requirements

### Requirement: Designs sweep parameters and initial states
An experiment SHALL run a model over grids, random samples, Latin hypercube samples, and Sobol samples of parameters and
initial states, addressed by factory argument or by dotted path. Every design SHALL be an estimator that validates its
settings and prepares against the bounds of the values it samples, and runs to rows of values.

#### Scenario: Orbit diagram sweep
- **WHEN** a user sweeps the growth rate of the logistic map over 200 values and keeps its last 100 values per run
- **THEN** the result holds 200 rows with the swept rate and the kept values

#### Scenario: Design without an experiment
- **WHEN** a user runs `SobolDesign(n=64).prepare({'r': (0.5, 1.5), 'K': (100, 1000)}).run()`
- **THEN** it returns 64 rows, each with a value of `r` and of `K` inside its bounds

### Requirement: Experiments are estimators with two verbs
Every experiment SHALL take a system or a factory of systems as its source, never a model name, since experiments
cannot import the model library. Every experiment SHALL store its source, design, scenarios, replicates, and analyses in its constructor, validate them and
resolve the source and the runs in `prepare()`, and execute the runs in `run()`, keeping the result in
`result_`.

#### Scenario: Experiment with analyses
- **WHEN** a user runs an experiment of 20 stochastic SIR replicates with a peak analysis
- **THEN** `result_` holds one row per run with the fitted attributes of the analysis of that run

#### Scenario: Invalid replicates
- **WHEN** a user builds an experiment with zero replicates and prepares it
- **THEN** a named error names `replicates` and the value
