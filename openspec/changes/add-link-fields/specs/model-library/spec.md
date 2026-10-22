# Spec Delta

## ADDED Requirements

### Requirement: Lattice gauge model samples the Wilson action
The library SHALL provide `lattice_gauge_fields` in `physics/waves`, after Schwichtenberg (2020 §2.1, §4.4), as one
registered factory. With its default Monte Carlo dynamics, it SHALL return a system under discrete time that samples
the Wilson action of U(1), Z_N, or SU(2) link variables on the cell complex of a periodic lattice of two to four
dimensions, one sweep per unit of time. Its catalog entry SHALL declare the mean plaquette as a standard observable.

#### Scenario: Plaquette against the exact result
- **WHEN** `lattice_gauge_fields` runs with its defaults, U(1) on a periodic 32 × 32 lattice by heat-bath at β = 2,
  for 2000 sweeps after its transient
- **THEN** the mean plaquette observable is within 0.01 of `I_1(2) / I_0(2) ≈ 0.698`

#### Scenario: Gauge-invariant run
- **WHEN** a random gauge transformation is applied to the initial links before a Monte Carlo run with a fixed seed
- **THEN** the mean plaquette series equals the series of the untransformed run within 1e-12

### Requirement: Lattice gauge model evolves Maxwell fields
With Hamiltonian dynamics, `lattice_gauge_fields` SHALL return a system under continuous time that evolves a lattice
Maxwell potential and electric field on links in real time from optional static charges on sites, so the trajectory
holds the fields at physical times. Its catalog entry SHALL declare the Gauss law residual and the field energy as
standard observables.

#### Scenario: Maxwell dynamics conserve the Gauss law
- **WHEN** `lattice_gauge_fields` runs with Hamiltonian dynamics and a static charge pair on a periodic 32 × 32 lattice
- **THEN** the Gauss law residual stays below 1e-10 at every frame, and the total field energy changes by less than
  1e-3 relative error

### Requirement: Gauge realizations live in the lattice gauge package
The package `skcomplex.models.physics.waves` SHALL hold the Wilson action as an energy of the core's `Sample` law and
the gauge transformation, one group element per site, and none of them SHALL be a name of `skcomplex.core`.

#### Scenario: Wilson action outside the core
- **WHEN** the field-term test of the core runs after this change
- **THEN** no public name of `skcomplex.core` contains `wilson`, `gauge`, or `polyakov`

### Requirement: Gauge observables are measured from trajectories
The package `skcomplex.models.physics.waves` SHALL measure the mean plaquette, Wilson loops of given extents, Polyakov
loops along the time axis, the topological charge of a U(1) link field over any closed oriented surface of a complex,
and the Gauss law residual of an electric link state against a site charge.

#### Scenario: Invariance of loops
- **WHEN** a random gauge transformation is applied to an SU(2) configuration
- **THEN** the mean plaquette and the 2 × 2 Wilson loop are unchanged within 1e-12, and the energy of the Wilson action
  is unchanged

#### Scenario: Integer charge
- **WHEN** the topological charge of any U(1) configuration on a periodic 2D lattice is computed
- **THEN** it equals an integer within 1e-9

### Requirement: Gauge link field example in the gallery
The gallery SHALL hold the example `docs/examples/physics/waves/plot_lattice_gauge_fields.py`. Its Rerun thumbnail
SHALL show the plaquette angles of the last frame of the default run as an image with a cyclic color map. Its analysis
figure SHALL plot the mean plaquette measured from runs at several values of β against the exact curve
`I_1(β) / I_0(β)`.

#### Scenario: Gallery page
- **WHEN** a user opens the gallery page of `physics/waves`
- **THEN** the `lattice_gauge_fields` thumbnail is the Rerun image of its plaquette angles, and its page shows the
  measured mean plaquettes on the exact curve below the screenshot

### Requirement: Gauge network agents
The library SHALL provide `gauge_network_agents` in `social/opinion_dynamics`, after Mihaylov and Cacciatori (2025, arXiv
2509.01581): a site group of agents with internal U(1) phases on a network cell complex, and an edge group of
connections, the agents relaxing towards their neighbours' parallel-transported phases, so frustrated loops prevent
global agreement. The package SHALL define its own alignment energy and read the core's frustration measure.

#### Scenario: Frustration blocks agreement
- **WHEN** `gauge_network_agents` runs on a ring of 12 agents with a flat connection and with a total flux of pi around
  the ring
- **THEN** the flat ring reaches agreement with zero energy, and the frustrated ring keeps a residual energy and a twist

### Requirement: Link field models are recorded in the model inventory
Every model this change adds SHALL register under the field path of its subfield and SHALL have an entry in the model
inventory, tagged with its time kind, space kind, component kinds, bindings, law classes, and domains, and a model the
inventory records as pending this change SHALL register under the name of its entry.

#### Scenario: Inventory models registered
- **WHEN** the change is applied and the inventory check runs
- **THEN** these entries are registered under their field paths and marked implemented:
  - `lattice_gauge_fields` in `physics/waves`, sources Sc 2.1 and Sc 4.4
  - `gauge_network_agents` in `social/opinion_dynamics`, source arXiv 2509.01581

#### Scenario: New model recorded
- **WHEN** the change registers a model with no inventory entry
- **THEN** the inventory check fails until the entry is added with its sources and field path
