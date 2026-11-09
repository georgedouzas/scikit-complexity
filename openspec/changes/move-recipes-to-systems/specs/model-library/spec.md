## MODIFIED Requirements

### Requirement: Specific realizations live in the system library
The library SHALL hold every specific realization of a core primitive or a general analysis, and every recipe, in its
own private module of the system package that uses it, next to the system definitions, exported from that package. A
realization that several packages of one field share SHALL live in `skcomplex.systems.<field>.common`, grouped in family
packages such as `particles`, `oscillators`, `excitable`, or `forecasting`, with no loose modules beside the families,
and only that field's packages import it. A realization that several fields share SHALL live in a concept subpackage of
`skcomplex.systems.general`. A realization SHALL be a subclass of a general core class, even a pure renaming, and no
field SHALL define the same realization twice. A field whose packages share no code SHALL have no `common` package. A
system class whose mechanism the core provides SHALL use the core class and keep only what is its own.

#### Scenario: Order book from the library
- **WHEN** the finance systems build their order book, and an engineering system pairs riders with cars
- **THEN** both use `MatchingInteraction` of `skcomplex.systems.general.matching`, the order book binds it to bids,
  asks, money, and goods with `ContinuousMatchingSpec`, and the earlier of two same-limit sell orders trades first

#### Scenario: Potential from the library
- **WHEN** a user imports the Lennard-Jones potential from `skcomplex.systems.physics.common.particles`
- **THEN** it is a realization built on the core `PairInteraction`, and the existing LAMMPS agreement tests pass with it

#### Scenario: Realization of one package
- **WHEN** only the climate systems use `early_warning`
- **THEN** it lives in `skcomplex.systems.physics.climate`

#### Scenario: No copy across fields
- **WHEN** the ecology meadow and the economics commons systems are built
- **THEN** both build foraging and regrowth from the core `TransferInteraction` and `MapInteraction` and from
  `BirthDeathInteraction` of `skcomplex.systems.general.agents`

### Requirement: Pair potentials for particle groups
The physics module SHALL provide Lennard-Jones and soft-sphere pair interactions with a cutoff, built on the core
`PairInteraction` on positions of order 2, which run on the native solvers and on the LAMMPS backend.

#### Scenario: Lennard-Jones minimum
- **WHEN** two members interact through a Lennard-Jones potential with `sigma = 1`
- **THEN** the force between them is zero at the distance `2 ** (1 / 6)`

### Requirement: Ready-made systems are classes
Every ready-made system SHALL be a class on `BaseSystem`, or on a general system of `skcomplex.systems.general` such as
`ReactionNetworkSystem`, in the module of its subfield. It SHALL take its model parameters as arguments of its
constructor and SHALL give its field path, topics, references, and tuned run settings as class attributes.
`all_systems()` SHALL find every such class, and SHALL list no general system of `skcomplex.systems.general`, which has
no field.

#### Scenario: Browse the catalog
- **WHEN** a user calls `all_systems(field='biology')` or `all_systems(field='biology/ecology')`
- **THEN** the result holds the name and class of every system of that field or subfield, and each class gives its
  field, topics, and references as class attributes

#### Scenario: Create by name
- **WHEN** a user builds `LotkaVolterraSystem(a=0.2)`, imported from `skcomplex.systems.biology.ecology`
- **THEN** the result is a system that `simulate` runs like the same system built from its components, to the tuned
  end time of its class

## ADDED Requirements

### Requirement: Recipes of one field live in that field
A recipe that leaves the core and serves one field SHALL live in that field with its arguments, checks, reports, and
tests: toppling on `CascadeLaw` in `physics.common.avalanches`, feedback on parameters in `economics.common.policy`,
population search in `computer_science.evolutionary_computation`, incremental satisfiability in
`computer_science.neural_networks`, zone flocking in `biology.collective_behaviour`, and the region and holonomy
scores in the cellular Potts and Wilson sampling modules.

#### Scenario: Sandpile and fault from one physics family
- **WHEN** the sandpile, criticality, and fault systems build their avalanches
- **THEN** all use `TopplingInteraction` of `skcomplex.systems.physics.common.avalanches`, and their avalanche
  statistics equal those before the move

#### Scenario: Zones of a flock
- **WHEN** the Couzin flock and the leadership systems run with repulsion, orientation, and attraction zones
- **THEN** both use `ZoneCouplingInteraction`, their trajectories equal those before the move, and the core
  `CouplingInteraction` takes no argument `radii` or `clamp`

#### Scenario: Search and satisfiability in computer science
- **WHEN** `EvolutionaryOptimisersSystem` and `perceptron_satisfiability` run with the same seeds as before the move
- **THEN** their reports are the same, and `skcomplex.core.specs` holds no algorithm or constraint spec of either
