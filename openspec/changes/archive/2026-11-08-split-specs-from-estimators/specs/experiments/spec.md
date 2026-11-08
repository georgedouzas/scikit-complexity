## RENAMED Requirements

- FROM: `### Requirement: Experiments are functions, except calibration`
- TO: `### Requirement: Experiments are functions`

## MODIFIED Requirements

### Requirement: Calibration fits parameters to data
`calibrate(system, data, bounds, ...)` of `skcomplex.core.experiments` SHALL fit the model parameters of a system within
bounds to observed time series, and SHALL return a frozen `CalibrationResult` that holds the fitted system as `system`,
the estimates as `estimates`, and the goodness of fit as `r2`, `rmse`, `mae`, and `mape`.

#### Scenario: Recover logistic parameters
- **WHEN** a user calibrates the logistic growth system on data generated with `r = 0.8` and `K = 500` plus 1% noise
- **THEN** `estimates` lies within 5% of the true values and `r2` exceeds 0.99

### Requirement: Experiments are functions
Parameter sweeps, ensembles, scenarios, sensitivity, basin stability, and calibration SHALL be public functions of a
system or of a factory of systems, in the role of `validation_curve`, `cross_validate` with repeats, and the search
functions of scikit-learn, and SHALL return tidy results. Every experiment SHALL check the tags it needs, such as
`stochastic` for an ensemble over seeds or `differentiable` for gradient calibration.

#### Scenario: A sweep
- **WHEN** a user calls a sweep of a system over one parameter with a bound analysis
- **THEN** it returns a table with one row per value and the analysis's result per row

#### Scenario: Calibration fits a system
- **WHEN** a user calls `calibrate` on a system and observed data
- **THEN** it returns a `CalibrationResult` whose `system` holds the system with the fitted parameters
