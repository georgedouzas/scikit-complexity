# Spec Delta

## ADDED Requirements

### Requirement: Network models
The library SHALL provide Kuramoto on complete graphs and networks, the power-grid swing equation, SIS and SIR on
networks, adaptive SIS with rewiring, Barabási-Albert growth, voter and majority models on networks, random Boolean
networks, coupled logistic maps, Vicsek flocking, and DebtRank interbank contagion as registered factories.

#### Scenario: Random Boolean network phases
- **WHEN** random Boolean networks with in-degree 1, 2, and 3 run from random states
- **THEN** damage from a single flipped node dies out at in-degree 1, stays near constant at 2, and spreads at 3

### Requirement: Adaptive network models are recorded in the model inventory
Every model this change adds SHALL register under the field path of its subfield and SHALL have an entry in the model
inventory, and a model the inventory records as pending this change SHALL register under the name of its entry. A
model that the library already implements SHALL be rebuilt on the new formalism under the same name, not registered
twice.

#### Scenario: Inventory models registered
- **WHEN** the change is applied and the inventory check runs
- **THEN** these entries are registered under their field paths and marked implemented:
  - `kuramoto_on_networks` in `physics/synchronization`
  - `coupled_map_lattices_on_networks` in `physics/synchronization`
  - `vicsek` in `physics/active_matter`
  - `gregoire_chate` in `physics/active_matter`
  - `ginelli_nematics` in `physics/active_matter`
  - `sequentially_linear_ecosystem` in `biology/ecology`
  - `quenched_mean_field_sis` in `biology/epidemiology`
  - `adaptive_sis` in `biology/epidemiology`
  - `activity_driven_epidemics` in `biology/epidemiology`
  - `jain_krishna` in `biology/evolution`
  - `sole_manrubia` in `biology/evolution`
  - `hegselmann_krause` in `social/opinion_dynamics`
  - `coevolving_voter` in `social/opinion_dynamics`
  - `join_a_club` in `social/opinion_dynamics`
  - `barabasi_albert` in `mathematics/networks`
  - `holme_kim` in `mathematics/networks`
  - `vertex_copying` in `mathematics/networks`
  - `network_attack` in `mathematics/networks`
  - `random_graph_ensembles` in `mathematics/networks`

#### Scenario: New model recorded
- **WHEN** the change registers a model with no inventory entry
- **THEN** the inventory check fails until the entry is added with its sources and field path
