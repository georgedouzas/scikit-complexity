# Spec Delta

## ADDED Requirements

### Requirement: Models on reductions and learned surrogates
The library SHALL provide `szabo_cells`, `active_passive_flocking`, `koopman_model_predictive_control`, and
`next_generation_reservoir` as registered factories built from core primitives, each with a curated gallery example.

#### Scenario: Cells order collectively
- **WHEN** `szabo_cells` runs at a density above its ordering transition
- **THEN** the mean normalized velocity of the cells grows to an order parameter near one

#### Scenario: Passive agents are carried
- **WHEN** `active_passive_flocking` runs with an ordered active majority
- **THEN** the passive agents move with the mean heading of the flock

#### Scenario: Surrogate model predictive control
- **WHEN** `koopman_model_predictive_control` steers the Duffing oscillator to a reference with its fitted surrogate
- **THEN** the controlled state tracks the reference closer than the uncontrolled one

#### Scenario: Reservoir forecast
- **WHEN** `next_generation_reservoir` is fitted to a training window of a chaotic flow
- **THEN** its forecast stays close to the true trajectory for a time of the order of a Lyapunov time
