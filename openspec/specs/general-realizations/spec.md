# general-realizations Specification

## Purpose
Realizations that systems of several fields share live in the cross-domain package `skcomplex.systems.general`, one
subpackage per concept, built from the primitives of the core, so the core holds building blocks only and no field
copies another field's code.

## Requirements

### Requirement: Shared realizations live in one cross-domain package
A realization that systems of two or more fields use SHALL live in a concept subpackage of `skcomplex.systems.general`,
with no loose module at its root. The package SHALL import only the public modules of the core and no field
package. Every field MAY import it, and the core SHALL import nothing from it. A realization used by one field only
SHALL live in that field. A class of `skcomplex.systems.general` SHALL be named by its mathematics or by a neutral
term, never by the term of one field.

#### Scenario: Import contracts
- **WHEN** the checks session runs the import contracts
- **THEN** no module of `skcomplex.systems.general` imports a field package, and no module of the core imports
  `skcomplex.systems.general`

#### Scenario: Users of a shared module
- **WHEN** the catalog tests read the importers of every module of `skcomplex.systems.general`
- **THEN** each module has importers in at least two fields

#### Scenario: Core features return plain composites
- **WHEN** a data-driven function of the core, such as `sparse_identification`, returns a fitted system
- **THEN** it is a `CompositeSystem` of core components, and the core imports nothing from `skcomplex.systems.general`

#### Scenario: No engine support in the package
- **WHEN** a reviewer searches the engine for the classes of `skcomplex.systems.general`
- **THEN** no compile pass, kernel, solver, or backend names any of them

### Requirement: Agents move, give birth, and die
`skcomplex.systems.general.agents` SHALL provide `MovementInteraction`, which moves agents by a neighbour, Gaussian,
Lévy, best, or custom step, and `BirthDeathInteraction`, which creates and removes members. Both SHALL be agent rules
built on `RuleLaw`, with the arguments, checks, and reports they had in the core.

#### Scenario: Diffusion of walkers
- **WHEN** 10,000 agents step by Gaussian steps of variance `2 D dt` from the origin of a plane
- **THEN** their mean squared displacement after time `t` is `4 D t` within 5 percent

#### Scenario: Life cycle variants
- **WHEN** a birth-death interaction gives births into another group with values of the parent, keeps a share of a
  split state, replaces parents by their offspring, renews dying agents in place, removes kin of another group, or
  brings arrivals without a parent at placed locations
- **THEN** the new agents join the named group with the given values, the parent keeps that share, a parent that gave
  birth is gone, a renewed agent keeps its identity and the states the new values leave out, the first kin member is
  removed, and a Poisson number of arrivals of the given mean appear, with the counts reported as `births`, `deaths`,
  and `arrivals`

### Requirement: Members choose and imitate
`skcomplex.systems.general.agents` SHALL provide `ChoiceInteraction`, which draws an option from utilities by the
logit, Metropolis, Fermi, or best rule, and `ImitationInteraction`, which copies a neighbour's state at random, by
majority, from the best, from a random better neighbour, or by the Fermi rule, as rates, per-step probabilities, or
agent rules. Their symbolic forms SHALL lower to core update and jump laws and their agent forms to core rules.

#### Scenario: Glauber dynamics
- **WHEN** fully connected spins choose by the logit rule at intensity 2
- **THEN** the magnetization approaches the root of `m = tanh(2 m)`

#### Scenario: Voter consensus
- **WHEN** voters on a ring of 50 imitate a random neighbour
- **THEN** every run reaches consensus, and each opinion wins with its initial fraction

#### Scenario: Choice reading history
- **WHEN** an agent system's logit choice reads the payoffs the system recorded in earlier rounds
- **THEN** it uses `ChoiceInteraction` of `skcomplex.systems.general.agents` with a utility of the agent and the
  system view

### Requirement: Strategies evolve
`skcomplex.systems.general.agents` SHALL provide `EvolutionInteraction`, which selects by tournament, roulette,
proportional, or rank selection, replaces by generational, Moran, death-birth (parents among survivors or all),
truncation, or best-mutant replacement, mutates to another, a redrawn, a stepped, or a new label, runs once every
period, and reads the fitness from a state or evaluates it from the strategies.

#### Scenario: Virtual scores and truncation
- **WHEN** players of the minority game add the virtual payoff of each strategy by cumulative reinforcement, and every
  50 rounds the poorest tenth redraw their strategies by truncation
- **THEN** the volatility near `alpha = 2^m / N ~ 0.34` falls below that of random play

#### Scenario: Best-mutant climbing
- **WHEN** a population sharing one strategy takes the fittest mutant only when it is fitter or every member mutated
- **THEN** with few mutants its evaluated fitness never falls

### Requirement: Requesters are matched with providers
`skcomplex.systems.general.matching` SHALL provide `MatchingInteraction`, which pairs requesters with providers
through the core `MatchLaw` by a rule spec, reading the states that `reserve`, `quantity`, and `cap` name. It SHALL
provide the rule specs of random search, partners, partner switching, queues, iterative, continuous, batch, and stable
matching, and the record `MatchSpec`. `PartnerMatchingSpec` SHALL subclass the core `PartnerSpec` and add the clearing
`value`.

#### Scenario: Matching outside markets
- **WHEN** a ride-sharing system matches riders with idle cars by `StableMatchingSpec`
- **THEN** it uses `MatchingInteraction` of `skcomplex.systems.general.matching`, and no rider and car prefer each other
  to their matches

#### Scenario: Neutral matching names
- **WHEN** a matching binds `reserve=('bid', 'ask')` with `StableMatchingSpec(value='requester')`
- **THEN** every pair settles at the reserve of its requester, and no argument of the matching or its specs is named
  `limit`, `allowance`, or `clearing`

#### Scenario: Partners with a clearing value
- **WHEN** a commons system pairs cows with patches by `PartnerMatchingSpec('local', 'score', score=...,
           value='provider')`
- **THEN** every match settles at the reserve of its provider, as it did with the former `PartnerSpec` value

### Requirement: Densities meet in binary encounters
`skcomplex.systems.general.kinetics` SHALL provide `EncounterInteraction`, the binary kinetic collision operator of a
candidate density with a field density over class axes, with conservative, destructive, or proliferative outcomes. It
SHALL lower to the core `BilinearLaw` and `RateLaw`, and with a `cross_section` it SHALL evaluate the binary integral
over the class axes through the core `collision_integral`.

#### Scenario: Binary integral over classes
- **WHEN** an encounter with a `cross_section` targets a density on two velocity axes
- **THEN** it gives the fast spectral integral it gave in the core, within 1e-12

#### Scenario: Encounters as a collision operator
- **WHEN** a user reads the documentation of `EncounterInteraction`
- **THEN** it describes the binary kinetic collision operator over the class axes

### Requirement: Reactions and reaction networks
`skcomplex.systems.general.reactions` SHALL provide `ReactionInteraction`, a `JumpInteraction` that moves units
between species, as rates, jumps, Langevin noise, or agent rules, with real or parametric stoichiometry and modifiers,
and `ReactionNetworkSystem`, which builds one `NonnegativeElement` per species and one reaction per entry. Without a
given propensity, a reaction SHALL have `k` times the product of its reactant and modifier amounts, each raised to its
coefficient.

#### Scenario: Deterministic and stochastic reactions agree
- **WHEN** `S + I -> 2 I` and `I -> R` run in rate mode and in jump mode with 100,000 units
- **THEN** the jump means stay within 2 percent of the rate solution, and `S + I + R` stays constant

#### Scenario: Propensity with a coefficient above one
- **WHEN** a user builds the reaction `2 A -> B` at the rate constant `k` with no given propensity
- **THEN** its propensity is `k A^2`

#### Scenario: Parametric stoichiometry
- **WHEN** the Oregonator's reactions change `x` and `y` scaled by `1 / epsilon` and `1 / delta`
- **THEN** it uses `ReactionInteraction`, and its limit cycle equals the previous system's

#### Scenario: Reactions at every site of a field
- **WHEN** the Gray-Scott system runs its cubic autocatalysis, feed, and removal as reactions on the state names of its
  field beside its two diffusions
- **THEN** its trajectory equals the one of sites that carry the same rate laws of their own, up to round-off

#### Scenario: Reaction network species
- **WHEN** a user builds a `ReactionNetworkSystem` from `skcomplex.systems.general.reactions` with the reaction
  `S + I -> 2 I` and simulates it
- **THEN** the trajectory holds `S.x` and `I.x`, and the reaction reads the keys `('S', 'x')` and `('I', 'x')`

#### Scenario: Epidemics and chemistry share the network
- **WHEN** an epidemic model and a chemical kinetics model both build species and reactions
- **THEN** both subclass `ReactionNetworkSystem` of `skcomplex.systems.general.reactions`, and neither imports the
  other's field

### Requirement: Nonnegative amounts and points
`skcomplex.systems.general.elements` SHALL provide `NonnegativeElement`, which holds one scalar state `x` in `[0, oo)`,
and `skcomplex.systems.general.elements` SHALL provide `PointElement`, a member with a vector state of order 2 and the
role `'position'`. A compartment, a density at one class, a species, or a population SHALL be `NonnegativeElement` or a
field realization that subclasses it and keeps the state `x`.

#### Scenario: Kinetic systems share one element
- **WHEN** the kinetic systems of wealth, opinion, crowds, immune competition, and gases declare their densities
- **THEN** every density is a `NonnegativeElement` of `skcomplex.systems.general.elements`, through the realization of
  its own field, and its state is `x`

#### Scenario: Points on the native and LAMMPS solvers
- **WHEN** Lennard-Jones particles built on `PointElement` of `skcomplex.systems.general.elements` run on the velocity
  Verlet solver and on LAMMPS
- **THEN** both runs agree as they did when the element was in the core
