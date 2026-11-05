# Spec Delta

## ADDED Requirements

### Requirement: Lattice gauge model samples the Wilson action
The library SHALL provide `lattice_gauge_fields` in `physics/waves`, after Schwichtenberg (2020 §2.1, §4.4). With its
default Monte Carlo dynamics, it SHALL sample the Wilson action of U(1) or Z_N link angles on the cell complex of a
periodic lattice under discrete time, through `WilsonSamplingInteraction`, a subclass of `SamplingInteraction` in its
package, and its catalog entry SHALL declare the mean plaquette as an observable.

#### Scenario: Plaquette against the exact result
- **WHEN** `lattice_gauge_fields` runs with its defaults, U(1) on a periodic 32 × 32 lattice at β = 2, after its
  transient
- **THEN** the mean plaquette observable is within 0.01 of `I_1(2) / I_0(2) ≈ 0.698`

#### Scenario: Gauge-invariant run
- **WHEN** a random gauge transformation is applied to the initial links before a Monte Carlo run with a fixed seed
- **THEN** the mean plaquette series equals the series of the untransformed run within 1e-12

### Requirement: Lattice gauge model evolves Maxwell fields
With `dynamics='hamiltonian'`, `lattice_gauge_fields` SHALL return a system under continuous time that evolves a lattice
Maxwell potential and electric field on links from optional static charges on sites. Its catalog entry SHALL declare
the Gauss residual and the field energy as observables.

#### Scenario: Maxwell dynamics conserve the Gauss law
- **WHEN** `lattice_gauge_fields` runs with Hamiltonian dynamics and a static charge pair on a periodic 32 × 32 lattice
- **THEN** the Gauss residual stays below 1e-10 at every frame, and the field energy changes by less than 1e-3
  relative error

### Requirement: Gauge observables are measured in the gauge package
The package `skcomplex.models.physics.waves` SHALL hold `GaugeAnalysis`, a subclass of `BaseAnalysis` that measures
Wilson loops of given extents, Polyakov loops along the last axis, the topological charge of a 2D U(1) link state, and
the Gauss residual of an electric link state against a site charge. No public name of `skcomplex.core` SHALL contain
`wilson`, `gauge`, or `polyakov`.

#### Scenario: Invariance of loops
- **WHEN** a random gauge transformation is applied to a U(1) configuration
- **THEN** the mean plaquette and the 2 × 2 Wilson loop are unchanged within 1e-12

#### Scenario: Integer charge
- **WHEN** the topological charge of any U(1) configuration on a periodic 2D lattice is computed
- **THEN** it equals an integer within 1e-9

### Requirement: Topological Kuramoto model
The library SHALL provide `topological_kuramoto` in `physics/synchronization`, after Millán, Torres, and Bianconi
(2203.06601 III): phases on the nodes and links of a clique complex, the link phases driven through the codifferential
and the exterior derivative of the sines of the phases, with node and link dynamics coupled by their global order
parameters.

#### Scenario: Link synchronization grows with coupling
- **WHEN** `topological_kuramoto` runs on the clique complex of a random graph at zero coupling and at a coupling well
  above its transition
- **THEN** the order parameter of the link phases at the end of the strong-coupling run exceeds that of the run at zero
  coupling by at least 0.3

### Requirement: Gauge arbitrage market model
The library SHALL provide `gauge_arbitrage_market` in `economics/finance`, after Schwichtenberg (2020 §1.7, §2.1,
§2.2.1): exchange-rate potentials on the links of a lattice complex evolved by the discrete inhomogeneous Maxwell
equations, sourced by the trader current on links, and the good on sites moved along the links by its covariant
gradient.

#### Scenario: Conservation in the market
- **WHEN** `gauge_arbitrage_market` runs with its defaults
- **THEN** the total good stays constant within 1e-10 relative error, and the Gauss residual against the density of the
  good stays below 1e-8

### Requirement: Gauge network agents
The library SHALL provide `gauge_network_agents` in `social/opinion_dynamics`, after Mihaylov and Cacciatori (2025,
arXiv 2509.01581): agent phases on the nodes of a network and connections on its edges, the agents relaxing towards
their neighbours' transported phases through `TransportedCouplingInteraction`, a subclass of `CouplingInteraction` in
its package, so frustrated loops prevent global agreement.

#### Scenario: Frustration blocks agreement
- **WHEN** `gauge_network_agents` runs on a ring of 12 agents with a flat connection and with a total flux of pi around
  the ring
- **THEN** the flat ring reaches agreement with zero energy, and the frustrated ring keeps a residual energy and a twist

### Requirement: Link field models are recorded in the model inventory
Every model this change adds SHALL register under the field path of its subfield and SHALL be marked implemented in
the model inventory.

#### Scenario: Inventory models registered
- **WHEN** the change is applied and the inventory check runs
- **THEN** these entries are registered under their field paths and marked implemented:
  - `lattice_gauge_fields` in `physics/waves`
  - `topological_kuramoto` in `physics/synchronization`
  - `gauge_arbitrage_market` in `economics/finance`
  - `gauge_network_agents` in `social/opinion_dynamics`
