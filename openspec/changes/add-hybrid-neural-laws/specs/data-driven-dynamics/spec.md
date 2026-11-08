# Spec Delta

## ADDED Requirements

### Requirement: Hybrid surrogates by neural identification
`neural_identification(system, data, trained=...)` SHALL fit the parameters named by `trained` of a differentiable
system to observed series of its states or observables, by gradients of the loss through the solution of the ODEs. It
SHALL return a frozen result whose `system` is the system with the fitted values set through `set_params`.

#### Scenario: Universal differential equation
- **WHEN** the interaction terms of a Lotka-Volterra system are replaced by a neural term, and the weights are fitted
  to noise-free prey and predator series of 50 points over 20 time units
- **THEN** the neural term is within 5% of the true terms in root mean square over the observed states, and the fitted
  system matches the series within 2%

#### Scenario: Known rates fitted with the network
- **WHEN** `trained` names the weights and the two linear rates of that system, started 30% off
- **THEN** both rates end within 5% of their true values

#### Scenario: Reproducible fit
- **WHEN** it runs twice with the same `random_state`
- **THEN** the losses of every epoch are equal

### Requirement: Fits read missing samples
The data SHALL map paths of states or observables to times and values, with one column per member for a group, and
SHALL read NaN values as missing samples that do not enter the loss, so members sampled at different times are fitted
together.

#### Scenario: Sparse samples
- **WHEN** each member of a group is observed at three of twelve times and the other values are NaN
- **THEN** the loss and its gradient do not depend on the predictions at the missing times, within 1e-12

#### Scenario: Unknown path
- **WHEN** the data names a variable the system does not have
- **THEN** it raises `AnalysisError` that names the variable and lists the states and observables of the system

### Requirement: Fits check the trained parameters and the system
`neural_identification` SHALL raise `AnalysisError` before any fit when a trained or individual name is not an adaptive
parameter of the system, or when the tags of the system are not differentiable, naming the parameter or the tag.

#### Scenario: Parameter not adaptive
- **WHEN** `trained` names a parameter not declared `adaptive=True`
- **THEN** it raises `AnalysisError` whose message says to declare the parameter `ParameterSpec(adaptive=True)`

#### Scenario: Stochastic system
- **WHEN** the system holds jump laws
- **THEN** it raises `AnalysisError` that names the law kind `jump` and the tag `differentiable`

### Requirement: Parameters mapped from covariates
A parameter of a law SHALL be given by a neural network of covariates, which are per-member parameters of a group, so
one fit learns the map from covariates to the parameter across members, and a new member with its covariates set
through `set_params` gets its parameter from the map.

#### Scenario: Bioavailability of new compounds
- **WHEN** the hybrid form of `oral_iv_pk_hybrid` is fitted on the oral and intravenous series of 200 compounds of the
  reference form
- **THEN** the predicted oral bioavailability of 50 held-out compounds has a coefficient of determination of at least
  0.8 against the reference

### Requirement: Mixed effects
`neural_identification` SHALL take `individual`, a mapping from per-member parameters to the standard deviation of
their Gaussian prior, fit one value per member with the penalty of that prior beside the shared parameters, and record
the effects of every member in the result.

#### Scenario: Population clearance
- **WHEN** the mechanistic form of `tacrolimus_population_pk` with random effects on clearance and central volume is
  fitted on 200 patients with the typical values trained and the effects individual
- **THEN** the typical clearance and central volume are within 10% of the values of the system, and the effects
  correlate with the drawn ones with a coefficient of at least 0.8

#### Scenario: Prior not positive
- **WHEN** an individual parameter is given a standard deviation of 0
- **THEN** it raises `AnalysisError` that names the parameter and says the deviation is positive

### Requirement: Effects of a new member from few samples
Called with no trained parameter and with individual parameters, `neural_identification` SHALL keep the shared values
of the system and fit only the effects of its members, the maximum a posteriori estimate from their samples.

#### Scenario: Exposure from three samples
- **WHEN** the fitted tacrolimus system is given 100 new patients, each with samples at 0, 1, and 3 hours after a dose
- **THEN** the median relative error of the area under the concentration curve over the dosing interval is below 10%

### Requirement: Latent neural ODEs with mixed effects
A system whose states are latent, whose rates are neural functions of the latent states, the doses, and per-member
effects, and whose observable is a readout of the latent states SHALL be fitted by `neural_identification` like any
hybrid system.

#### Scenario: Misspecified elimination
- **WHEN** the latent form of `tacrolimus_population_pk` and its mechanistic form with linear elimination are both
  fitted on patients of the saturable elimination form
- **THEN** the median relative error of the exposure of held-out patients is lower for the latent form
