# Spec Delta

## ADDED Requirements

### Requirement: Surrogate analyses read observed runs
`sparse_identification`, `mode_decomposition`, and `reservoir` SHALL accept an `ObservedRunsSpec` as source, pool its
runs in one fit, take the inputs of every run as its control inputs where the method has them, and refuse runs whose
times are not evenly spaced where the method needs them evenly spaced, naming the run.

#### Scenario: Runs from a reader
- **WHEN** `sparse_identification` reads three noise-free Lorenz runs from a data frame through `read_runs`
- **THEN** it returns the same coefficients as on the three trajectories within 1e-10

#### Scenario: Uneven times
- **WHEN** `mode_decomposition` reads a run whose times are not evenly spaced
- **THEN** it raises `AnalysisError` that names the run and says the method needs evenly spaced times
