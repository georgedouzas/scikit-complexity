## MODIFIED Requirements

### Requirement: Jumps, reactions, and transitions
`JumpInteraction` SHALL change states by signed amounts at a rate, with a probability per step, or after drawn waiting
times, after which it SHALL reset scalar or vector states. `ReactionInteraction` SHALL move units between species, given as roles and states, by a product propensity or a given
propensity, with real stoichiometry and modifiers, as rates, jumps, or Langevin noise. `TransitionInteraction` SHALL
move a discrete state between values by a rate, a probability per step, a condition, a transition matrix, or an
emission matrix, after an optional residence time, and `TransitionInteraction` SHALL spread an active value along edges through a rate that reads the neighbours.

#### Scenario: Deterministic and stochastic reactions agree
- **WHEN** `S + I -> 2 I` and `I -> R` run in rate mode and in jump mode with 100,000 units
- **THEN** the jump means stay within 2 percent of the rate solution, and `S + I + R` stays constant

#### Scenario: Epidemic threshold
- **WHEN** a contact process runs on a square lattice below and above its critical spreading rate
- **THEN** the active fraction dies out below it and stays positive above it

### Requirement: Per-member reactions and renamed roles
`ReactionInteraction` SHALL take species as state names of its `target` alone and react in every member when the
target is a group or a field, and a subclass SHALL be able to bind its species under roles of its own names.

#### Scenario: Reactions at every site of a field
- **WHEN** the Gray-Scott system runs its cubic autocatalysis, feed, and removal as reactions on the state names of its
  field beside its two diffusions
- **THEN** its trajectory equals the one of sites that carry the same rate laws of their own, up to round-off

#### Scenario: A reaction under renamed roles
- **WHEN** a subclass binds its species under the roles `susceptible` and `infected` and returns their pairs
- **THEN** it follows the product propensity of `S + I -> 2 I` at its own rate constant, as a rate equation or as jumps

### Requirement: General elements
The core SHALL provide `NonnegativeElement`, which holds one scalar state `x` in `[0, oo)`, and `PointElement`, a
member with a vector state of order 2 in a continuous space. A compartment, a density at one class, a species, or a
population SHALL be `NonnegativeElement` or a field realization that subclasses it and keeps the state `x`. The core
SHALL provide no second element for a nonnegative scalar under another name or another state name.
`ReactionNetworkSystem` SHALL build one `NonnegativeElement` per species and key its reactions by `(species, 'x')`, and
`EncounterInteraction` SHALL read the state `x` unless it is given another.

#### Scenario: Kinetic systems share one element
- **WHEN** the kinetic systems of wealth, opinion, crowds, immune competition, and gases declare their densities
- **THEN** every density is a `NonnegativeElement`, through the realization of its own field, and its state is `x`

#### Scenario: Mass action species
- **WHEN** a user builds a `ReactionNetworkSystem` with the reaction `S + I -> 2 I` and simulates it
- **THEN** the trajectory holds `S.x` and `I.x`, and the reaction reads the keys `('S', 'x')` and `('I', 'x')`

#### Scenario: Former elements are gone
- **WHEN** a user imports `CompartmentElement` or `DensityElement` from `skcomplex.core.components.elements`
- **THEN** the import fails with `ImportError`, since the project ships no alias for a removed class

## ADDED Requirements

### Requirement: Reactions follow a product propensity
A reaction without a given propensity SHALL have the propensity `k` times the product of its reactant and modifier
amounts, each raised to its stoichiometric coefficient, where `k` is its rate constant. The core SHALL describe this
rate law in mathematical words and SHALL name field terms only as examples, such as the law of mass action in
chemistry, homogeneous mixing in epidemiology, and the encounter term of Lotka-Volterra in ecology.

#### Scenario: Propensity of a bimolecular reaction
- **WHEN** a user builds the reaction `S + I -> 2 I` at the rate constant `beta` with no given propensity
- **THEN** its propensity is `beta S I`

#### Scenario: Propensity with a coefficient above one
- **WHEN** a user builds the reaction `2 A -> B` at the rate constant `k` with no given propensity
- **THEN** its propensity is `k A^2`

#### Scenario: Core wording is cross-domain
- **WHEN** a reviewer searches the core, the user guide, and the specs for "mass action" or "mass-action"
- **THEN** the term appears only as a named example of a field, never as the name of the rate law or of a class
