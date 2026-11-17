## MODIFIED Requirements

### Requirement: Designs sweep parameters and initial states
A design SHALL be the `design=` argument of `sweep`, one of `'grid'`, `'random'`, `'latin_hypercube'`, and `'sobol'`,
which gives the rows of model parameters and dotted initial states or component parameters that the system runs over.

#### Scenario: Orbit diagram sweep
- **WHEN** a user sweeps the growth rate of the logistic map over 200 values and keeps its last 100 values per run
- **THEN** the result holds 200 rows with the swept rate and the kept values

#### Scenario: Design without an experiment
- **WHEN** a user calls `sweep(system, {'r': (0.5, 1.5), 'K': (100, 1000)}, design='sobol', n=64)`
- **THEN** the result holds 64 runs, each with a value of `r` and of `K` inside its bounds
