# Spec Delta

## ADDED Requirements

### Requirement: Surveyed physics models
The library SHALL provide `swarmalators`, `ehrenfest_urn`, `directed_percolation_langevin`, `sk_game`,
`keller_segel`, `benard_marangoni`, `charged_schrodinger`, `cahn_hilliard`, and `allen_cahn` as
registered factories built from core primitives, each with a curated gallery example.

#### Scenario: Ehrenfest urn relaxes to the binomial
- **WHEN** `ehrenfest_urn` runs from all balls in one urn
- **THEN** the count in the first urn relaxes to a binomial distribution with mean half the balls

#### Scenario: Directed percolation threshold
- **WHEN** `directed_percolation_langevin` sweeps its control parameter across the absorbing transition
- **THEN** the stationary density vanishes below a threshold and grows above it

### Requirement: Surveyed biology models
The library SHALL provide `hormone_cascade_amplifier`, `min_protein_skeleton`, `nanotube_gap_junction_exchange`,
`closed_contour_cell_exchange`, `adaptive_response_costs`, `glucose_dual_hormone_control`,
`coagulation_positive_feedback`, and `couzin_leadership` as registered factories built from core primitives, each with
a curated gallery example.

#### Scenario: Informed minority leads the group
- **WHEN** `couzin_leadership` runs with a small informed minority sharing a preferred direction
- **THEN** the group's mean heading turns towards that direction, and the accuracy grows with the minority's size

### Requirement: Surveyed earth, economics, and mathematics models
The library SHALL provide `sutera_fraedrich_tipping`, `ramped_stochastic_fold`, `metaphoria`,
`wilhite_trade_networks`, `koopman_slow_manifold`, and `ou_driven_rossler` as registered factories built from core
primitives, each with a curated gallery example.

#### Scenario: Tipping lags a ramp
- **WHEN** `ramped_stochastic_fold` ramps its control parameter through the fold faster than the system relaxes
- **THEN** the state tips after the control passes the static fold point, later for faster ramps
