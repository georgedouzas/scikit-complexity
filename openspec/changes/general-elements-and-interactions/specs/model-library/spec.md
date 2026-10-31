# Spec Delta

## MODIFIED Requirements

### Requirement: Specific realizations live in the model library
The library SHALL hold every specific realization of a core primitive or a general analysis in its own private module
of the model package that uses it, next to the model definitions, exported from that package. A realization that
several packages of one field share SHALL live in `skcomplex.models.<field>.common`, grouped in family packages such as
`particles`, `oscillators`, `excitable`, or `forecasting`, with no loose modules beside the families, and only that
field's packages import it. A realization SHALL be a subclass of a general core class, even a pure renaming, and no field
SHALL define the same realization twice. A field whose packages share no code SHALL have no `common`
package. A model class whose mechanism the core provides SHALL use the core class and keep only what is its own.

#### Scenario: Order book from the library
- **WHEN** the finance models build their order book, and an engineering model pairs riders with cars
- **THEN** both use the core `MatchingInteraction`, the order book binds it to bids, asks, money, and goods with
  `ContinuousClearingSpec`, and the earlier of two same-limit sell orders trades first

#### Scenario: Potential from the library
- **WHEN** a user imports the Lennard-Jones potential from `skcomplex.models.physics.common.particles`
- **THEN** it is a realization built on the core `PairCouplingInteraction`, and the existing LAMMPS agreement tests pass
  with it

#### Scenario: Realization of one package
- **WHEN** only the climate models use `EarlyWarningAnalysis`
- **THEN** it lives in `skcomplex.models.earth.climate`, and the earth field has no `common` package

#### Scenario: No copy across fields
- **WHEN** the ecology meadow and the economics commons models are built
- **THEN** both build foraging and regrowth from the core `TransferInteraction`, `LifeCycleInteraction`, and
  `MapInteraction`

### Requirement: Pair potentials for particle groups
The physics module SHALL provide Lennard-Jones and soft-sphere pair interactions with a cutoff, built on the core
`PairCouplingInteraction` on positions of order 2, which run on the native solvers and on the LAMMPS backend.

#### Scenario: Lennard-Jones minimum
- **WHEN** two members interact through a Lennard-Jones potential with `sigma = 1`
- **THEN** the force between them is zero at the distance `2 ** (1 / 6)`

### Requirement: Langevin thermostat
The physics module SHALL provide a Langevin thermostat that adds drag and random forces to a particle group, built on
the core `RelaxationInteraction` acting on the first derivative of the position, which runs on the native SDE solvers and on the LAMMPS
backend.

#### Scenario: Equilibrium temperature
- **WHEN** an ideal gas starts at temperature 0.5 under a Langevin thermostat whose `temperature` is 2
- **THEN** its kinetic temperature approaches 2 within 5 percent after many relaxation times

### Requirement: Economic and social agent models
The library SHALL provide registered agent models that exercise the phases and the core's matching, feedback, choice,
and evolution interactions in distinct ways, and SHALL cover spatial cooperation with the existing lattice model of the
spatial prisoner's dilemma rather than registering it again.

#### Scenario: Spatial cooperation
- **WHEN** the spatial prisoner's dilemma runs with imitate-the-best updating and a temptation payoff of 1.6
- **THEN** cooperators survive in clusters for 200 steps instead of dying out
