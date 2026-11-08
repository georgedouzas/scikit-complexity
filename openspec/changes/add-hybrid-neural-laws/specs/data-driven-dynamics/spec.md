# Spec Delta

## ADDED Requirements

### Requirement: Hybrid surrogates by neural identification
`neural_identification(system, data, trained=...)` SHALL fit the parameters named by `trained` of a differentiable
system to observed series of its states or observables, by gradients of the loss through the solution of the ODEs. It
SHALL return a frozen result whose `system` is the system with the fitted values set through `set_params`.

#### Scenario: Universal differential equation
- **WHEN** the interaction terms of `lotka_volterra` are replaced by a neural term, and the weights are fitted to
  noise-free prey and predator series of 50 points over 20 time units
- **THEN** the neural term is within 5% of the true terms in root mean square over the observed states, and the fitted
  system matches the series within 2%

#### Scenario: Known rates fitted with the network
- **WHEN** `trained` names the weights and the two linear rates of that system, started 30% off
- **THEN** both rates end within 5% of their true values

#### Scenario: Reproducible fit
- **WHEN** it runs twice with the same `random_state`
- **THEN** the losses of every epoch are equal

### Requirement: Fits take several individuals with missing samples
The data SHALL be one mapping, or a list of mappings with one per individual, from paths of states or observables to
times and values. `settings` SHALL give the parameter values of each individual, such as its doses. NaN values SHALL be
missing samples that do not enter the loss.

#### Scenario: Sparse samples
- **WHEN** each of ten individuals is observed at three of twelve times and the other values are NaN
- **THEN** the loss and its gradient do not depend on the predictions at the missing times, within 1e-12

#### Scenario: Unknown path
- **WHEN** the data names a variable the system does not have
- **THEN** it raises `AnalysisError` that names the variable and lists the states and observables of the system

#### Scenario: Settings of the wrong length
- **WHEN** three mappings of data are given with two mappings of settings
- **THEN** it raises `AnalysisError` that names both lengths

### Requirement: Fits check the trained parameters and the system
`neural_identification` SHALL raise `AnalysisError` before any fit when a trained or individual name is not a
parameter of the system, or when the tags of the system are not differentiable, naming the parameter or the tag.

#### Scenario: Unknown parameter
- **WHEN** `trained` names a parameter the system does not have
- **THEN** it raises `AnalysisError` that names the parameter and lists the parameters of the system

#### Scenario: Stochastic system
- **WHEN** the system holds jump laws
- **THEN** it raises `AnalysisError` that names the law kind `jump` and the tag `differentiable`

### Requirement: Parameters mapped from covariates
`neural_identification` SHALL take `maps`, a mapping from parameters to neural networks, and `covariates`, one vector
per individual, and SHALL set each mapped parameter of an individual to the network of its covariates. The result SHALL
hold the trained networks, so the parameters of a new individual follow from its covariates.

#### Scenario: Elimination from a covariate
- **WHEN** the elimination rate `k10` of `two_compartment_pharmacokinetics` is mapped from two covariates and fitted on
  200 individuals whose rates follow a smooth function of their covariates
- **THEN** the mapped rates of 50 held-out individuals have a coefficient of determination of at least 0.9 against the
  true rates

#### Scenario: Covariates missing
- **WHEN** `maps` is given without `covariates`
- **THEN** it raises `AnalysisError` that says the maps need one covariate vector per individual

### Requirement: Mixed effects
`neural_identification` SHALL take `individual`, a mapping from parameters to the standard deviation of a Gaussian prior
on the log scale, and SHALL give each individual the value `typical * exp(eta)`, fitting one `eta` per individual with
the penalty of that prior beside the shared values. The result SHALL record every effect.

#### Scenario: Typical elimination rate
- **WHEN** `drug_toxicity` runs for 200 individuals with log-normal elimination rates of deviation 0.3, and the fit
  trains the typical rate with the rate individual
- **THEN** the typical rate is within 10% of the median of the drawn rates, and the effects correlate with the drawn
  ones with a coefficient of at least 0.8

#### Scenario: Prior not positive
- **WHEN** an individual parameter is given a standard deviation of 0
- **THEN** it raises `AnalysisError` that names the parameter and says the deviation is positive

### Requirement: Effects of a new individual from few samples
Called with no trained parameter and with individual parameters, `neural_identification` SHALL keep the shared values
of the system and fit only the effects of the given individuals, the maximum a posteriori estimate from their samples.

#### Scenario: Exposure from three samples
- **WHEN** the system fitted on `drug_toxicity` is given 100 new individuals, each with three samples in the first
  dosing week
- **THEN** the median relative error of the toxicity at the end of the schedule is below 10%

### Requirement: Latent neural ODEs with mixed effects
A system whose states are latent, whose rates are neural functions of the latent states, the doses, and individual
effects, and whose observable is a readout of the latent states SHALL be fitted by `neural_identification` like any
hybrid system.

#### Scenario: Misspecified mechanism
- **WHEN** a latent system with individual effects and a one-compartment system with a linear elimination rate are both
  fitted on individuals of `two_compartment_pharmacokinetics` with random rates
- **THEN** the median relative error of the area under the curve of held-out individuals is lower for the latent system
