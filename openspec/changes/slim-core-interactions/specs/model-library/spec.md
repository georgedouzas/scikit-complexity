## MODIFIED Requirements

### Requirement: Langevin thermostat
The physics module SHALL provide a Langevin thermostat that adds drag and random forces to a particle group, built on
its `RelaxationTimeInteraction`, a `RateInteraction` with the rate `(goal - x) / time` acting on the first derivative
of the position, which runs on the native SDE solvers and on the LAMMPS
backend.

#### Scenario: Equilibrium temperature
- **WHEN** an ideal gas starts at temperature 0.5 under a Langevin thermostat whose `temperature` is 2
- **THEN** its kinetic temperature approaches 2 within 5 percent after many relaxation times

### Requirement: Specific realizations live in the system library
The library SHALL hold every specific realization of a core primitive or a general analysis in its own private module
of the system package that uses it, next to the system definitions, exported from that package. A realization that
several packages of one field share SHALL live in `skcomplex.systems.<field>.common`, grouped in family packages such as
`particles`, `oscillators`, `excitable`, or `forecasting`, with no loose modules beside the families, and only that
field's packages import it. A realization SHALL be a subclass of a general core class, even a pure renaming, and no field
SHALL define the same realization twice. A field whose packages share no code SHALL have no `common`
package. A system class whose mechanism the core provides SHALL use the core class and keep only what is its own.

#### Scenario: Order book from the library
- **WHEN** the finance systems build their order book, and an engineering system pairs riders with cars
- **THEN** both use the core `MatchingInteraction`, the order book binds it to bids, asks, money, and goods with
  `ContinuousMatchingSpec`, and the earlier of two same-limit sell orders trades first

#### Scenario: Potential from the library
- **WHEN** a user imports the Lennard-Jones potential from `skcomplex.systems.physics.common.particles`
- **THEN** it is a realization built on the core `PairCouplingInteraction`, and the existing LAMMPS agreement tests pass
  with it

#### Scenario: Realization of one package
- **WHEN** only the climate systems use `early_warning`
- **THEN** it lives in `skcomplex.systems.physics.climate`

#### Scenario: No copy across fields
- **WHEN** the ecology meadow and the economics commons systems are built
- **THEN** both build foraging and regrowth from the core `TransferInteraction`, `BirthDeathInteraction`, and
  `MapInteraction`

### Requirement: Kinetic equations of gases and plasmas
The library SHALL provide the spatially inhomogeneous Boltzmann equation, its BGK system, the Vlasov-Poisson equation
with Landau damping and two-stream set-ups, and the local Enskog equation as system classes under
`physics/fluids` (Bellomo et al. 2017 §2.2–2.3), built on `PhaseSpace`, the physics `AdvectionInteraction` on
`ConservationInteraction`, `EncounterInteraction`, and the physics `RelaxationTimeInteraction` on `RateInteraction`. The inventory SHALL record them as implemented.

#### Scenario: Sod shock tube near the fluid limit
- **WHEN** the BGK system runs the Sod shock tube in one position and one velocity dimension at a Knudsen number of 1e-4
- **THEN** the density moment at time 0.2 matches the exact Euler solution within 2% in the L1 norm

#### Scenario: Landau damping system
- **WHEN** the Vlasov-Poisson system runs with its default Landau damping set-up
- **THEN** the measured damping rate of the electric energy lies within 5% of 0.1533, and the system passes the
  conformance suite

### Requirement: Dynamic equilibrium, satisfiability, and search systems
The library SHALL provide the fictitious play of `OligopolyEquilibriaSystem` with `dynamics='fictitious-play'`, built on
`ChoiceInteraction` and the economics `PayoffLearningInteraction` on `MapInteraction`, `perceptron_satisfiability` in `computer_science/neural_networks`,
built on `SatisfiabilityInteraction`, and `EvolutionaryOptimisersSystem` in `computer_science/evolutionary_computation`,
built on `OptimizationInteraction` and on `SamplingInteraction` with `cooling`, all under discrete time.

#### Scenario: Fictitious play of a discretised duopoly
- **WHEN** `OligopolyEquilibriaSystem` runs with `dynamics='fictitious-play'` for 2000 rounds
- **THEN** the empirical distribution of prices is within 0.2 of `F(P)` in the Kolmogorov distance

#### Scenario: Satisfiable fraction
- **WHEN** a `sweep` runs `perceptron_satisfiability` at `N = 15` and `N = 21` over 20 replicates
- **THEN** the share of satisfiable replicates against `p/N` falls from above 0.9 at 0.5 to below 0.1 at 1.2

#### Scenario: Algorithms compared
- **WHEN** `EvolutionaryOptimisersSystem` runs differential evolution, particle swarm, and annealing on Rastrigin for 100
  generations with the same seed
- **THEN** every run reports a best value that never increases, and differential evolution ends below 1e-3

## ADDED Requirements

### Requirement: Field realizations of merged interactions
A field that has its own term for a form of a core interaction SHALL hold it as a named subclass of that core class, in
the system module of its one user or in the field's `common` package when several packages share it. Fields without
such a term SHALL use the core class with its function. The realization SHALL keep the arguments, checks, and hooks
the form had in the core, and its tests SHALL move with it.

#### Scenario: Relaxations by field
- **WHEN** the physics Langevin thermostat and BGK, the engineering optimal velocity, the chemistry reactors, the
  neuroscience populations, and the food sector of economics relax a state towards a goal
- **THEN** they use `RelaxationTimeInteraction`, `SpeedAdaptationInteraction`, `MassExchangeInteraction`,
  `LeakyIntegrationInteraction`, and `PartialAdjustmentInteraction`, each a subclass of `RateInteraction`, and every
  run keeps its trajectory

#### Scenario: Transport by field
- **WHEN** the heat, fluid, ocean, and gel systems diffuse or advect a field, and the multilevel selection and forest
  size systems carry a density along a trait or a size
- **THEN** physics uses `DiffusivityInteraction` on `RateInteraction` and `AdvectionInteraction` on
  `ConservationInteraction`, social `SelectionDriftInteraction`, and biology `SizeGrowthInteraction`, both on
  `ConservationInteraction`

#### Scenario: Pulses, kicks, and fractions
- **WHEN** the circuit, magnetic levitation, and minimum-time control systems pulse a state, the impulsive oscillator
  kicks it, and radiotherapy gives dose fractions
- **THEN** they use `PulseInteraction`, `KickInteraction`, and `IrradiationInteraction`, each a subclass of
  `CrossingInteraction` whose guard is `periodic_guard`, and every event falls at the time it fell before

#### Scenario: Learning and price search in economics
- **WHEN** the oligopoly, Brock-Hommes, perishables, El Farol, and minority systems score options, and the shops search
  their prices
- **THEN** they use `PayoffLearningInteraction` of `skcomplex.systems.economics.common` and `PriceAdjustmentInteraction`
  of `skcomplex.systems.economics.markets`, both subclasses of `MapInteraction`

#### Scenario: Physics-only interactions
- **WHEN** the squirmers and the coupled traps move spheres through a Stokes fluid, and the metric cosmologies solve the
  field equations of an ansatz
- **THEN** they use `StokesianMobilityInteraction` of `skcomplex.systems.physics.common.particles` and
  `EinsteinFieldEquationsInteraction` of `skcomplex.systems.physics.cosmology`
