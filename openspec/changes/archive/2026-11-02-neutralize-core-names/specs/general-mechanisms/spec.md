## MODIFIED Requirements

### Requirement: Jumps, reactions, and transitions
`JumpInteraction` SHALL change states by signed amounts at a rate, with a probability per step, or after drawn waiting
times, after which it SHALL reset scalar or vector states. `ReactionInteraction` SHALL move units between species, given as roles and states, by mass action or a given
propensity, with real stoichiometry and modifiers, as rates, jumps, or Langevin noise. `TransitionInteraction` SHALL
move a discrete state between values by a rate, a probability per step, a condition, a transition matrix, or an
emission matrix, after an optional residence time, and `TransitionInteraction` SHALL spread an active value along edges through a rate that reads the neighbours.

#### Scenario: Deterministic and stochastic reactions agree
- **WHEN** `S + I -> 2 I` and `I -> R` run in rate mode and in jump mode with 100,000 units
- **THEN** the jump means stay within 2 percent of the rate solution, and `S + I + R` stays constant

#### Scenario: Epidemic threshold
- **WHEN** a contact process runs on a square lattice below and above its critical spreading rate
- **THEN** the active fraction dies out below it and stays positive above it

### Requirement: Encounters, crossings, impulses, and sampling
`EncounterInteraction` SHALL run conservative, destructive, or proliferative encounters over densities on activity
classes. `CrossingInteraction` SHALL reset states or end the run when a guard crosses zero. `ImpulseInteraction` SHALL
reset states at given or periodic times. `SamplingInteraction` SHALL run Monte Carlo sweeps by moves accepted from the
change of an objective at an intensity.

#### Scenario: Integrate and fire
- **WHEN** a state integrates a constant input and a crossing resets it at a threshold
- **THEN** it fires with the closed-form period

#### Scenario: Ising magnetization
- **WHEN** a `SamplingInteraction` samples a square-lattice Ising energy well below the critical temperature
- **THEN** the absolute magnetization exceeds 0.9

## ADDED Requirements

### Requirement: Core names carry no field vocabulary
Core classes, arguments, specs, and law utilities SHALL use the vocabulary of the operation, and a field's vocabulary SHALL
appear only in thin subclasses in `skcomplex.models`.

#### Scenario: Sampling an objective
- **WHEN** a user samples a function that is not an energy with `SamplingInteraction(objective=..., intensity=...)`
- **THEN** moves are accepted from the change of the objective, with no `temperature` or `energy` argument in the core

#### Scenario: Physics keeps its words
- **WHEN** a statistical mechanics model sweeps the `temperature` of its thin sampling subclass
- **THEN** the subclass gives the core an intensity of one over the temperature

### Requirement: One way to select partners
`PartnerSpec` SHALL be the only spec that selects partners by candidates, a choice, a score, a fallback, and a capacity;
matching algorithms over offers SHALL be named for the operation, such as batch, continuous, and iterative matching.

#### Scenario: Greedy matching by a partner spec
- **WHEN** a matching uses a partner spec with a score and a capacity
- **THEN** it pairs agents as `GreedyMatchingSpec` did, and `GreedyMatchingSpec` no longer exists

### Requirement: Bilinear laws are one law
One bilinear law SHALL express gains and losses of a density at the rate of the product of two densities.

#### Scenario: Encounter outcomes
- **WHEN** an encounter is destructive or proliferative
- **THEN** it lowers to the bilinear law with the sign of its outcome, with the trajectories it had before

### Requirement: Encounters take a collision integral
`EncounterInteraction` SHALL evaluate a binary integral over the class axes when given a `cross_section`.

#### Scenario: Binary integral over classes
- **WHEN** an encounter with a `cross_section` targets a density on two velocity axes
- **THEN** it gives the fast spectral integral `CollisionInteraction` gave, within 1e-12

### Requirement: Operator dynamics are named for their mathematics
The core and the engine SHALL name linear operator dynamics on tensor products by the operation, with no quantum vocabulary
in their class names, and physics SHALL keep its quantum vocabulary in `skcomplex.models.physics.common.quantum`.

#### Scenario: Quantum system
- **WHEN** a user runs a catalog quantum model
- **THEN** it compiles with the operator compiler and gives the trajectory it gave before

#### Scenario: Core names
- **WHEN** the class names of `skcomplex.core`, `skcomplex.engine`, and `skcomplex.analysis` are searched for `Quantum`
- **THEN** none is found
