# Spec Delta

## Purpose

Lets a user put a learned model inside any law an interaction states, an Equinox module whose weights are one array
parameter of the interaction or a scikit-learn regressor or pipeline, so hybrid laws, learned interactions, and learned
controllers are ordinary interactions that every backend that lowers them runs and that the fits of the library train.

## ADDED Requirements

### Requirement: Laws call learned models
`learned_function(model, *inputs, weights=...)` SHALL return one expression per output of the model, usable in any law.
For an Equinox module, the weights SHALL be the expression of one array parameter of the interaction that states the
law, read at every evaluation, so `set_params` and the `schedule` of `simulate` change the outputs. `learned_weights`
SHALL return the flat vector of the arrays of a module, the initial value of that parameter.

#### Scenario: Network in a rate law
- **WHEN** an interaction states a rate law that reads the output of an Equinox network with two inputs, one hidden
  layer of 8 units, and one output, with `weights=learned_weights(model)` as its parameter
- **THEN** the compiled rate equals the network evaluated on the same inputs and weights within 1e-12 on the NumPy and
  JAX backends

#### Scenario: Weights changed
- **WHEN** the weights parameter is set to new values through `set_params` with its nested name and the system is
  prepared again
- **THEN** the rate equals the network evaluated with the new weights within 1e-12

#### Scenario: Weights recorded with their shape
- **WHEN** a system whose interaction holds the 97 weights of a network is simulated
- **THEN** `parameters_` lists the weights parameter, and the trajectory records it with shape `(97,)` at every frame

#### Scenario: Weights of the wrong length
- **WHEN** the weights parameter holds fewer values than the network has weights
- **THEN** the compiler's report holds an issue with the code `learned-weights` that names the parameter, the expected
  length, and the given length

#### Scenario: Wrong number of inputs
- **WHEN** a network of three inputs is called with two
- **THEN** the compiler's report holds an issue with the code `learned-inputs` that names the law and both numbers

### Requirement: Scikit-learn regressors are learned models
`learned_function` SHALL accept a scikit-learn regressor, or a pipeline that ends in one, called through `predict` once
per evaluation over every member, on the NumPy backend. A system holding one SHALL not have the tag `differentiable`,
and an unfitted regressor SHALL be refused at `prepare` unless `identify(method='derivatives')` fits it.

#### Scenario: Fitted regressor
- **WHEN** a rate law reads a fitted `KNeighborsRegressor` of one state
- **THEN** the compiled rate equals `predict` on the states of every member within 1e-12

#### Scenario: Fitted pipeline
- **WHEN** a rate law reads a fitted pipeline of `StandardScaler` and `Ridge` of two states
- **THEN** the compiled rate equals `predict` of the pipeline on the states of every member within 1e-12

#### Scenario: Not differentiable
- **WHEN** a system with a regressor in a law is prepared with the JAX backend
- **THEN** the compiler's report holds an issue with the code `not-differentiable` that names the law and the regressor

#### Scenario: Unfitted regressor
- **WHEN** a system whose law reads an unfitted `Ridge` is prepared
- **THEN** `prepare` raises `ValidationError` that names the interaction and says the regressor is not fitted

### Requirement: Backends without a learned form fall back
A backend that cannot lower a learned model SHALL evaluate the law holding it in Python and SHALL warn once, naming the
backend, the law, and the reason, as for any kernel that falls back.

#### Scenario: Numba backend
- **WHEN** a group of 2000 members with a learned rate law is prepared with the Numba backend
- **THEN** preparing warns that the learned law runs in Python, and the rates equal those of the NumPy backend within
  1e-12

### Requirement: Learned laws are differentiable
A system whose learned models are all Equinox modules SHALL have the tag `differentiable`, so `TrainingInteraction`,
`identify`, and `calibrate(method='variational')` differentiate its runs in the weights.

#### Scenario: Gradient of the weights
- **WHEN** the gradient of the squared error of a run of a system with a learned rate law is taken in its weights
- **THEN** it agrees with central finite differences of step 1e-6 within 1e-6 relative error

### Requirement: Hybrid laws and learned interactions are ordinary interactions
An interaction SHALL add a learned term to known terms of a law, and SHALL call a learned model on the states of its
roles, with no other parameter than the weights, so it prepares, runs, and is analysed like other interactions.

#### Scenario: Hybrid rate
- **WHEN** an interaction states the rate law `r * x - learned_function(model, x, y, weights=s.weights)[0]` and the last
  layer of the model is zero
- **THEN** the run equals the run of `r * x` within 1e-12

#### Scenario: Learned pair energy
- **WHEN** an interaction bound with `among` and `within=RadiusNeighborhood(2.5)` over a group of 50 particles gives its
  pair energy as `ObjectiveLaw` of a network of the distance
- **THEN** the compiled second derivative of the position of every particle equals minus the sum of the derivatives of
  the network over its pairs, divided by the mass, within 1e-10 on the NumPy and JAX backends
