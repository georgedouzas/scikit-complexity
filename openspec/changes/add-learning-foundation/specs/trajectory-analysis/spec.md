# Spec Delta

## ADDED Requirements

### Requirement: Forecasts are scored by lead time
`forecast_skill(forecast, truth, ...)` SHALL compare a forecast with the true continuation, as trajectories or arrays,
and SHALL return a frozen result with the error of every variable at every lead time, by the symmetric mean absolute
percentage error or the normalised root mean squared error, the valid prediction time, the first lead at which the
error exceeds `threshold`, and the skill against a persistence forecast.

#### Scenario: Perfect forecast
- **WHEN** the forecast equals the truth
- **THEN** every error is 0, the valid time is the whole horizon, and the skill is 1

#### Scenario: Persistence
- **WHEN** the forecast repeats the last observed value
- **THEN** the skill against persistence is 0 within 1e-12

#### Scenario: Lengths differ
- **WHEN** the forecast and the truth have different numbers of samples
- **THEN** it raises `AnalysisError` that names both lengths
