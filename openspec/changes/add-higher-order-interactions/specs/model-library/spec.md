# Spec Delta

## ADDED Requirements

### Requirement: Higher-order probe models
The library SHALL provide the probe models of higher-order interactions that the inventory records as pending or
blocked by this change, each with a curated gallery example, built on the classes and reductions this change adds.

#### Scenario: Inventory models registered
- **WHEN** the change is applied and the inventory check runs
- **THEN** these entries are registered under their names and field paths and marked implemented:
  - `hypergraph_kuramoto` in `physics/synchronization`
  - `hypergraph_consensus` in `social/opinion_dynamics`
  - `simplicial_social_dilemmas` and `hypergraph_public_goods` in `social/cooperation`
  - `simplicial_contagion` and `hypergraph_exposure_sis` in `social/spreading`
  - `triadic_percolation`, `hypergraph_random_walk`, and `growing_simplicial_complex` in `mathematics/networks`

#### Scenario: Explosive synchronization
- **WHEN** `hypergraph_kuramoto` sweeps the three-body coupling up and down
- **THEN** the order parameter shows hysteresis, as Majhi et al. (2022) review

#### Scenario: Bistable contagion
- **WHEN** `simplicial_contagion` runs with a triangle infection rate above the critical value
- **THEN** the epidemic has two stable prevalences depending on the initial seed

### Requirement: Group mechanisms run as symbolic groupings
`galam_majority` and `public_goods_game` SHALL be built on `GroupingInteraction(form='symbolic')`, and SHALL agree with
their previous trajectories over 20 seeds. Kolkata and the minority game SHALL keep their agent forms.

#### Scenario: Galam on the symbolic grouping
- **WHEN** `galam_majority` runs from a share of 0.6 for 20 steps over 20 seeds
- **THEN** the mean final share agrees with the previous agent form within the standard error of the 20 seeds
