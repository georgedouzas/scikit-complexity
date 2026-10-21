# Spec Delta

## ADDED Requirements

### Requirement: Higher-order probe models
The library SHALL provide `hypergraph_kuramoto`, `simplicial_contagion`, `triadic_percolation`, and `higher_order_random_lotka_volterra`, each with a curated gallery example.

#### Scenario: Explosive synchronization
- **WHEN** `hypergraph_kuramoto` sweeps the three-body coupling up and down
- **THEN** the order parameter shows hysteresis, as Majhi et al. (2022) review

#### Scenario: Bistable contagion
- **WHEN** `simplicial_contagion` runs with a triangle infection rate above the critical value
- **THEN** the epidemic has two stable prevalences depending on the initial seed
