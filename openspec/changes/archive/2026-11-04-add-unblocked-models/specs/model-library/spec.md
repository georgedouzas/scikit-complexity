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
- **WHEN** `active_passive_flocking` runs with the settings of Charbonneau (2017, figure 10.7), 45 active agents among
  342 at the compactness 0.672 from uniform random positions, with flocking on, for 50 time units
- **THEN** the mean distance between active agents over the last 10 time units falls below the mean distance between
  passive agents by at least 0.015, and the snapshot shows flocks of active agents among still passive agents

#### Scenario: Active agents stay mixed without flocking
- **WHEN** `active_passive_flocking` runs with the same settings and flocking off
- **THEN** the two mean distances over the last 10 time units stay within 0.01 of each other

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
