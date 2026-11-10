# Spec Delta

## ADDED Requirements

### Requirement: Identification creates a system from data
`identify(system, data, trained=...)` SHALL fit the adaptive parameters named by `trained`, learned weights included, to
observed runs, and SHALL return a frozen `IdentificationResult` whose `system` is the system with the fitted values set
through `set_params`, with the estimates, the training and validation losses of every epoch, the training and validation
errors, and the trajectory of the training system.

#### Scenario: Universal differential equation
- **WHEN** the predation terms of a Lotka-Volterra system are replaced by a network, and its weights are identified on
  noise-free prey and predator series of 50 points over 20 time units
- **THEN** the network is within 5% of the true terms in root mean square over the observed states, and the fitted
  system matches the series within 2%

#### Scenario: Rates fitted with the network
- **WHEN** `trained` names the weights and the two linear rates of that system, started 30% off
- **THEN** both rates end within 5% of their true values

#### Scenario: Reproducible fit
- **WHEN** it runs twice with the same `random_state`
- **THEN** the losses of every epoch are equal

#### Scenario: Fitted system is a system
- **WHEN** the `system` of the result is simulated on the NumPy backend
- **THEN** its run equals the run of the training system's inner system with the fitted parameters within 1e-10

### Requirement: Identification checks its parameters and system
`identify` SHALL raise `ValidationError` before any fit when a trained name is not an adaptive parameter of the system,
or when the method is `'gradient'` and the tags of the system are not differentiable, naming the parameter or the tag.

#### Scenario: Parameter not adaptive
- **WHEN** `trained` names a parameter not declared `adaptive=True`
- **THEN** it raises `ValidationError` whose message says to declare the parameter `ParameterSpec(adaptive=True)`

#### Scenario: Stochastic system
- **WHEN** the system holds jump laws and the method is `'gradient'`
- **THEN** it raises `ValidationError` that names the law kind `jump` and the tag `differentiable`

### Requirement: Regressors are identified from derivatives
`identify` with `method='derivatives'` SHALL estimate the rates of the observed states by smoothed differences, subtract
the known terms of every rate, and fit the scikit-learn regressors of the learned functions to the residuals.

#### Scenario: Regressor of a missing term
- **WHEN** the predation terms of a Lotka-Volterra system are a `KNeighborsRegressor` and the data are 20 noise-free
  runs from different initial states
- **THEN** the fitted system matches a held-out run within 5% over 10 time units

#### Scenario: Unobserved state
- **WHEN** a rate with a learned term belongs to a state that the data do not observe
- **THEN** it raises `ValidationError` that names the state and says the derivatives method needs it observed

### Requirement: Calibration reads observed runs
`calibrate` SHALL accept an `ObservedRunsSpec` as `data`, sum the loss of every run with its conditions set, and give the
same estimates as its former mapping for one run.

#### Scenario: One run
- **WHEN** the pelt records are calibrated once as a mapping and once as an `ObservedRunsSpec` of one run
- **THEN** both results hold the same estimates
