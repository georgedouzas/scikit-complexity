## MODIFIED Requirements

### Requirement: Calibration fits parameters to data
`Calibration` of `skcomplex.core.experiments` SHALL fit the model parameters of a system within bounds to observed time
series, and its `run` SHALL keep the fitted system in `system_` and the estimates with R², RMSE, MAE, and MAPE in the
fitted attributes `estimates_`, `r2_`, `rmse_`, `mae_`, and `mape_`.

#### Scenario: Recover logistic parameters
- **WHEN** a user runs a calibration experiment of the logistic growth system on data generated with `r = 0.8` and
  `K = 500` plus 1% noise
- **THEN** `estimates_` lies within 5% of the true values and `r2_` exceeds 0.99

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
- **THEN** `run` returns the calibration, and its `system_` holds the system with the fitted parameters
