# Spec Delta

## ADDED Requirements

### Requirement: Lattice gauge model samples the Wilson action
The library SHALL provide `lattice_gauge_fields` in `physics/waves`, after Schwichtenberg (2020 §2.1, §4.4), as one
registered factory. With its default Monte Carlo dynamics, it SHALL sample the Wilson action of compact link variables
in U(1), Z_N, or SU(2) on a periodic lattice of two to four dimensions, with one sweep per unit of time, so `run`
returns the Markov chain and the trajectory holds one configuration per recorded sweep. It SHALL record the mean
plaquette as an observable.

#### Scenario: Plaquette against the exact result
- **WHEN** `lattice_gauge_fields` runs with its defaults, U(1) on a periodic 32 × 32 lattice by heat-bath at β = 2,
  for 2000 sweeps after its transient
- **THEN** the mean plaquette observable is within 0.01 of `I_1(2) / I_0(2) ≈ 0.698`

#### Scenario: Gauge-invariant run
- **WHEN** a random gauge transformation is applied to the initial links before a Monte Carlo run with a fixed seed
- **THEN** the mean plaquette series equals the series of the untransformed run within 1e-12

### Requirement: Lattice gauge model evolves Maxwell fields
With Hamiltonian dynamics, `lattice_gauge_fields` SHALL evolve a lattice Maxwell potential and electric field on links
in real time from optional static charges, so the trajectory holds the fields at physical times. It SHALL record the
Gauss law residual and the field energy as observables.

#### Scenario: Maxwell dynamics conserve the Gauss law
- **WHEN** `lattice_gauge_fields` runs with Hamiltonian dynamics and a static charge pair on a periodic 32 × 32 lattice
- **THEN** the Gauss law residual stays below 1e-10 at every frame, and the total field energy changes by less than
  1e-3 relative error

### Requirement: Gauge link field example in the gallery
The gallery SHALL hold the example `docs/examples/physics/waves/plot_lattice_gauge_fields.py`. Its Rerun thumbnail
SHALL show the plaquette angles of the last frame of the default run as an image with a cyclic color map. Its analysis
figure SHALL plot the mean plaquette measured from runs at several values of β against the exact curve
`I_1(β) / I_0(β)`.

#### Scenario: Gallery page
- **WHEN** a user opens the gallery page of `physics/waves`
- **THEN** the `lattice_gauge_fields` thumbnail is the Rerun image of its plaquette angles, and its page shows the
  measured mean plaquettes on the exact curve below the screenshot

### Requirement: Link field models are recorded in the model inventory
Every model this change adds SHALL register under the field path of its subfield and SHALL have an entry in the model
inventory, and a model the inventory records as pending this change SHALL register under the name of its entry.

#### Scenario: Inventory models registered
- **WHEN** the change is applied and the inventory check runs
- **THEN** these entries are registered under their field paths and marked implemented:
  - `lattice_gauge_fields` in `physics/waves`, sources Sc 2.1 and Sc 4.4

#### Scenario: New model recorded
- **WHEN** the change registers a model with no inventory entry
- **THEN** the inventory check fails until the entry is added with its sources and field path
