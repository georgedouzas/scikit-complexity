## ADDED Requirements

### Requirement: Pretrained models forecast series zero-shot
`pretrained_forecast(source, horizon, ...)` SHALL forecast every variable of a series `horizon` samples past its end
with a pretrained forecasting model, from a context of the last samples and without training, and SHALL return a frozen
result with the median forecast and the requested quantiles.

#### Scenario: Forecast of a Lorenz series
- **WHEN** a user forecasts 64 samples of a Lorenz trajectory from a context of 512 samples
- **THEN** the result holds a median forecast of 64 rows and three columns, and lower, median, and upper quantiles in
  order at every row

#### Scenario: Context longer than the series
- **WHEN** the context is longer than the series
- **THEN** it raises `ValidationError`, whose message names the context and the length of the series

### Requirement: Forecasts are compared by attractor statistics
Given the true continuation as `truth`, `pretrained_forecast` SHALL report the symmetric mean absolute percentage error
per step and the valid prediction time as the result of `forecast_skill` of `add-learning-foundation`, and the largest
Lyapunov exponent and the correlation dimension of the forecast and of the truth.

#### Scenario: Statistics of forecast and truth
- **WHEN** a user forecasts 1000 samples of a Lorenz trajectory with `truth` the next 1000 samples
- **THEN** the result holds both largest exponents and both correlation dimensions, and its skill equals
  `forecast_skill` of the same forecast and truth with the same threshold

### Requirement: Pretrained weights are fetched once
`pretrained_forecast` SHALL take `model`, the name of a published model or the path of a saved one. A named model SHALL
be downloaded on first use into the local cache and read from it after, and in offline mode a missing model SHALL
raise an error that names the model and the cache.

#### Scenario: Offline without the weights
- **WHEN** a user forecasts in offline mode with a named model that is not in the cache
- **THEN** it raises `AnalysisError`, whose message names the model and says how to download it

#### Scenario: Saved model
- **WHEN** a user forecasts with `model` the path of a saved model
- **THEN** no network access happens
