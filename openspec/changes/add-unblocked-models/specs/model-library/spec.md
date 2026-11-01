# Spec Delta

## ADDED Requirements

### Requirement: Models on radius reductions, pair couplings, and surrogates
The library SHALL provide `szabo_cells` and `active_passive_flocking` in `physics/active_matter`,
`koopman_model_predictive_control` in `engineering/control`, and `next_generation_reservoir` in
`computer_science/neural_networks` as registered factories built from the general classes of the core, each with a
curated gallery example and an inventory entry tagged with the system it builds.

#### Scenario: Cells order collectively
- **WHEN** `szabo_cells` runs at a density above its ordering transition
- **THEN** the mean normalized velocity of the cells grows to an order parameter above 0.8

#### Scenario: Active agents flock
- **WHEN** `active_passive_flocking` runs with every agent active and flocking on
- **THEN** the polarization of the velocities rises above 0.9

#### Scenario: Active agents gather
- **WHEN** `active_passive_flocking` runs with a minority of active agents and flocking off
- **THEN** the mean distance between active agents falls below the mean distance between passive agents

#### Scenario: Surrogate predictive control
- **WHEN** `koopman_model_predictive_control` steers each of its plants to a reference with its fitted surrogate
- **THEN** the controlled state tracks the reference with a smaller mean squared error than the uncontrolled plant

#### Scenario: Reservoir forecast
- **WHEN** `next_generation_reservoir` is fitted to a training window of the Lorenz flow
- **THEN** its forecast stays within 10% of the attractor size of the true trajectory for at least one Lyapunov time

### Requirement: The next-generation reservoir reuses the analysis readout
`next_generation_reservoir` SHALL take its readout from a fitted `ReservoirAnalysis(kind='next_generation')` and SHALL
NOT fit a readout of its own.

#### Scenario: Agreement with the reservoir analysis
- **WHEN** `next_generation_reservoir` is built from the readout of a `ReservoirAnalysis(kind='next_generation')`
  fitted to a series, and both forecast from the end of that series
- **THEN** the system's forecast over one Lyapunov time agrees with the analysis's `forecast_` within 1e-6 of the
  attractor size
