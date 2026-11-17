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
or when the method is `'gradient'` and the system holds stochastic laws other than SDE terms, naming the parameter
or the law kind.

#### Scenario: Parameter not adaptive
- **WHEN** `trained` names a parameter not declared `adaptive=True`
- **THEN** it raises `ValidationError` whose message says to declare the parameter `ParameterSpec(adaptive=True)`

#### Scenario: Jump laws
- **WHEN** the system holds jump laws and the method is `'gradient'`
- **THEN** it raises `ValidationError` that names the law kind `jump` and the tag `differentiable`, and says SDE terms
  are allowed

### Requirement: Identification fits initial states and noise scales
`identify` SHALL fit the initial values of the states named by `initial`, one value per run, and with `noise=True` the
scale of the observation model of every path by the likelihood loss, and SHALL report them as `initial_states` and
`noise_scales` of the result.

#### Scenario: Initial states of a run
- **WHEN** a Lotka-Volterra system started 20% off the true initial states is identified on one noise-free run of 30
  points with `initial=('prey.x', 'predators.x')`
- **THEN** both fitted initial states are within 1% of the true ones, and the rates within 2%

#### Scenario: Noise scale
- **WHEN** the same system is identified on one run of 200 points observed with log-normal noise of scale 0.2 and
  `noise=True`
- **THEN** the fitted scale is within 15% of 0.2

### Requirement: Identification scores short predictions or windows
`identify` SHALL, with `horizon=k`, score rollouts of `k` observation steps started from every time at which all states
are observed, and with `windows=m`, cut every run into `m` windows with trained start states and a penalty on the gaps
between consecutive windows. Without either it SHALL score the whole run.

#### Scenario: Chaotic run
- **WHEN** the three rates of a Lorenz system started 30% off are identified on a noise-free run of 20 time units
  sampled every 0.01, once with the whole run and once with `windows=40`
- **THEN** the fit with windows ends with every rate within 2% of its true value, and the fit of the whole run ends
  farther from the true rates

#### Scenario: One step prediction
- **WHEN** `horizon=1` is given for data whose states are all observed
- **THEN** the loss equals the sum of the squared errors of one step rollouts from every observed state, within 1e-10

#### Scenario: Horizon with unobserved states
- **WHEN** `horizon=1` is given and a state of the system is not observed
- **THEN** it raises `ValidationError` that names the state and says `windows` allows unobserved states

### Requirement: Identification fits SDE terms
`identify(method='gradient')` SHALL accept systems whose only stochastic laws are SDE terms, draw `paths` Brownian paths
per run and epoch from `random_state`, and fit the drift and diffusion by the mean and the variance of the paths at
the observation times.

#### Scenario: Ornstein-Uhlenbeck process
- **WHEN** the rate `theta` and the noise intensity `sigma` of `dx = -theta x dt + sigma dW`, started 50% off, are
  identified on 50 runs of 10 time units with `theta=1` and `sigma=0.5`
- **THEN** both end within 10% of their true values, and a second fit with the same `random_state` gives equal losses

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
