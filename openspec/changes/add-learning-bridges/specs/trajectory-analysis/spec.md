# Spec Delta

## ADDED Requirements

### Requirement: Residuals are tested for whiteness and independence
`whiteness(source, inputs, lags, level)` SHALL return a frozen result with the Ljung-Box statistic and p-value of every
residual path up to `lags`, pooled over runs, the cross-correlations of every residual with every input with their
bounds, and the flags `white` and `independent` at the level given, after the model validation of Schoukens and Ljung
(1902.00683).

#### Scenario: White noise
- **WHEN** the source is 2000 draws of Gaussian white noise with a fixed seed
- **THEN** the Ljung-Box p-value at 20 lags is above 0.05 and `white` is true

#### Scenario: Unmodelled dynamics
- **WHEN** the residuals of a first-order model fitted to a second-order linear system driven by a random input are
  tested with that input
- **THEN** `white` is false, `independent` is false, and the largest cross-correlation lies outside its bounds

#### Scenario: Runs pooled
- **WHEN** the source holds 10 runs of 200 white residuals each
- **THEN** the pooled statistic equals the sum of the statistics of the runs, and no lag joins two runs
