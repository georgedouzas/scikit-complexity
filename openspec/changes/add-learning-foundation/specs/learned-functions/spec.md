# Spec Delta

## Purpose

Lets a user put a learned model inside any law, an Equinox module whose weights are one adaptive parameter or a
scikit-learn regressor or pipeline, so hybrid laws, learned interactions, and learned controllers are ordinary
components that every backend that lowers them runs and that the fits of the library train.

## ADDED Requirements

### Requirement: Laws call learned models
`learned_function(model, *inputs, weights=...)` SHALL return one expression per output of the model, usable in any law.
For an Equinox module, the weights SHALL be one adaptive vector parameter of the component, read at every evaluation, so
`set_params` and the `schedule` of `simulate` change the outputs.

#### Scenario: Network in a rate law
- **WHEN** a rate law reads the output of an Equinox network with two inputs, one hidden layer of 8 units, and one
  output
- **THEN** the compiled rate equals the network evaluated on the same inputs and weights within 1e-12 on the NumPy and
  JAX backends

#### Scenario: Weights changed
- **WHEN** the weights parameter is set to new values through `set_params` and the system is compiled again
- **THEN** the rate equals the network evaluated with the new weights within 1e-12

#### Scenario: Weights of the wrong length
- **WHEN** the weights parameter holds fewer values than the network has weights
- **THEN** the compiler's report holds an issue with the code `learned-weights` that names the parameter, the expected
  length, and the given length

#### Scenario: Wrong number of inputs
- **WHEN** a network of three inputs is called with two
- **THEN** the compiler's report holds an issue with the code `learned-inputs` that names the law and both numbers

### Requirement: Parameters take fixed shapes
`ParameterSpec(shape=...)` SHALL accept whole numbers as fixed sizes beside role names, so a parameter holds an array of
a given shape that is addressed by path, recorded with its shape, and changed or trained like any parameter.

#### Scenario: Weight vector
- **WHEN** a component declares `ParameterSpec(shape=(97,), adaptive=True)` and the system is simulated
- **THEN** the trajectory records the parameter with shape `(97,)` at every frame

#### Scenario: Shape not positive
- **WHEN** a component declares `ParameterSpec(shape=(0,))`
- **THEN** `prepare` raises `ValidationError` that names the shape and says sizes are positive whole numbers or role
  names

### Requirement: Scikit-learn regressors are learned models
`learned_function` SHALL accept a scikit-learn regressor, or a pipeline that ends in one, called through `predict` once
per evaluation over every member, on the NumPy backend. A system holding one SHALL not be differentiable, and an
unfitted regressor SHALL be refused at `prepare` unless `identify(method='derivatives')` fits it.

#### Scenario: Fitted regressor
- **WHEN** a rate law reads a fitted `KNeighborsRegressor` of one state
- **THEN** the compiled rate equals `predict` on the states of every member within 1e-12

#### Scenario: Fitted pipeline
- **WHEN** a rate law reads a fitted pipeline of `StandardScaler` and `Ridge` of two states
- **THEN** the compiled rate equals `predict` of the pipeline on the states of every member within 1e-12

#### Scenario: Not differentiable
- **WHEN** a system with a regressor in a law is compiled with the JAX backend
- **THEN** the compiler's report holds an issue with the code `not-differentiable` that names the law and the regressor

### Requirement: Backends without a learned form fall back
A backend that cannot lower a learned model SHALL evaluate the law holding it in Python and SHALL warn once, naming the
backend, the law, and the reason, as for any kernel that falls back.

#### Scenario: Numba backend
- **WHEN** a group of 2000 members with a learned rate law is compiled with the Numba backend
- **THEN** compilation warns that the learned law runs in Python, and the rates equal those of the NumPy backend within
  1e-12

### Requirement: Learned laws are differentiable
A system whose learned models are all Equinox modules SHALL validate as differentiable, so `GradientStepLaw`,
`identify`, and `calibrate(method='variational')` differentiate its runs in the weights.

#### Scenario: Gradient of the weights
- **WHEN** the gradient of the squared error of a run of a system with a learned rate law is taken in its weights
- **THEN** it agrees with central finite differences of step 1e-6 within 1e-6 relative error

### Requirement: Hybrid laws and learned interactions are ordinary components
A law SHALL add a learned term to known terms, and an interaction SHALL call a learned model on the states of its roles,
with no other declaration than the weights parameter, so they validate, run, and are analysed like other components.

#### Scenario: Hybrid rate
- **WHEN** a rate law is `r * x - learned_function(model, x, y, weights=theta)[0]` and the last layer of the model is
  zero
- **THEN** the run equals the run of `r * x` within 1e-12

#### Scenario: Learned pair potential
- **WHEN** a pair interaction of 50 particles takes as potential a network of the distance
- **THEN** the compiled acceleration of every particle equals minus the sum of the derivatives of the network over its
  pairs within 1e-10 on the NumPy and JAX backends
