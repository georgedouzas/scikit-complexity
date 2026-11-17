# Spec Delta

## ADDED Requirements

### Requirement: Mixed effects
`identify` SHALL take `individual`, a mapping from per-member parameters of a group to the standard deviation of their
Gaussian prior, fit one value per member with the penalty of that prior beside the shared parameters named by `trained`,
and record the effects of every member in the field `effects` of its result.

#### Scenario: Population clearance
- **WHEN** the mechanistic form of `tacrolimus_population_pk` with random effects on clearance and central volume is
  identified on 200 patients with the typical values trained and the effects individual
- **THEN** the typical clearance and central volume are within 10% of the values of the system, and the effects
  correlate with the drawn ones with a coefficient of at least 0.8

#### Scenario: Prior not positive
- **WHEN** an individual parameter is given a standard deviation of 0
- **THEN** it raises `ValidationError` that names the parameter and says the deviation is positive

#### Scenario: Members held out
- **WHEN** `identify` runs with `individual` set and `validation=0.25` on 40 patients
- **THEN** the shared values are fitted on 30 patients, and the validation error is that of the other 10 with only
  their effects fitted

### Requirement: Effects of a new member from few samples
Called with `trained=()` and with individual parameters, `identify` SHALL keep the shared values of the system and fit
only the effects of its members, the maximum a posteriori estimate from their samples.

#### Scenario: Exposure from three samples
- **WHEN** the identified tacrolimus system is given 100 new patients, each with samples at 0, 1, and 3 hours after a
  dose
- **THEN** the median relative error of the area under the concentration curve over the dosing interval is below 10%

### Requirement: Parameters mapped from covariates
A parameter of a law SHALL be given by a learned function of covariates, which are per-member parameters of a group, so
one `identify` call learns the map from covariates to the parameter across members, and a new member with its
covariates set through `set_params` gets its parameter from the map.

#### Scenario: Bioavailability of new compounds
- **WHEN** the learned form of `oral_iv_pk_hybrid` is identified on the oral and intravenous series of 200 compounds of
  the reference form
- **THEN** the predicted oral bioavailability of 50 held-out compounds has a coefficient of determination of at least
  0.8 against the reference

### Requirement: Latent neural ODEs with mixed effects
A system whose states are latent, whose rates are learned functions of the latent states, the doses, and per-member
effects, and whose observable is a readout of the latent states SHALL be identified with `individual` like any hybrid
system.

#### Scenario: Misspecified elimination
- **WHEN** the latent form of `tacrolimus_population_pk` and its mechanistic form with linear elimination are both
  identified on patients of the saturable elimination form
- **THEN** the median relative error of the exposure of held-out patients is lower for the latent form
