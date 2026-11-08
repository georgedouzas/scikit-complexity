## RENAMED Requirements

- FROM: `### Requirement: Surrogates are estimators`
- TO: `### Requirement: Surrogates are analysis functions`

## MODIFIED Requirements

### Requirement: Surrogates are analysis functions
A surrogate SHALL be an analysis function of `skcomplex.core.analysis`, `mode_decomposition`, `sparse_identification`,
or `reservoir`, which takes its settings as keyword arguments and returns a frozen `ModeDecompositionResult`,
`SparseIdentificationResult`, or `ReservoirResult`. The result SHALL hold the fitted `system`, its `training_error` and
`validation_error`, and a `forecast` of `horizon` samples.

#### Scenario: Prepare and run
- **WHEN** a user calls `sparse_identification` on the series of a Lotka-Volterra run with `horizon=50`
- **THEN** its result's `system` is a system, `training_error` is a number, and `forecast` has 50 rows
