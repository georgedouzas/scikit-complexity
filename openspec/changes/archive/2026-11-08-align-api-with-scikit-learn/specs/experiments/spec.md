## ADDED Requirements

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

## REMOVED Requirements

### Requirement: Experiments are estimators with two verbs
**Reason**: Experiments explore a system, so they become functions, as model selection is in scikit-learn. Only
calibration produces a fitted system and stays an estimator.
**Migration**: Call the function of the experiment with the settings that the constructor took.

### Requirement: The experiments surface is estimators only
**Reason**: The experiments package becomes a package of functions plus the calibration estimator.
**Migration**: Import the function of the experiment.
