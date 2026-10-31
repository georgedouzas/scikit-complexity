# Spec Delta

## Purpose
Gives models of every field the operations they share as general core elements and interactions, configured by the
states they bind, by functions, and by named variants, so a model defines only what is its own.

## ADDED Requirements

### Requirement: Core classes name operations
The core SHALL provide one general element or interaction per mathematical operation that models use. Each SHALL take
the roles it binds, the names of the states it reads and changes, the functions that define it, and its variant as a
named argument. No core class, argument, or variant SHALL name a quantity of a field, such as a force, a position, a
velocity, a mass, a price, an infection, an energy, or a harvest.

#### Scenario: Field terms are rejected
- **WHEN** the field-term test checks the public names of `skcomplex.core`
- **THEN** none of them contains a field term

#### Scenario: A force is a rate
- **WHEN** a model adds a constant force `F` to a particle whose position `x` is a state of order 2 with mass `m`
- **THEN** it uses `RateInteraction(state='x', rate=F, divisor='m')`, and the second derivative of `x` is `F / m`

### Requirement: Field cases are subclasses
A field case of a general class SHALL be a subclass of it, even a pure renaming, that lives with its one model or in
the `common` package of its field when several of its packages use it. No field SHALL define the same case twice.

#### Scenario: Lennard-Jones from the core
- **WHEN** a physics model uses Lennard-Jones particles
- **THEN** `LennardJonesInteraction` of `physics/common/particles` is a `PairCouplingInteraction` with its pair
  potential, and its minimum lies at `2 ** (1 / 6)` times its length

### Requirement: Rates, relaxation, gradients, and Lagrangians
`RateInteraction` SHALL add a rate and an optional noise to the highest derivative of a state of any order, divided by
an optional divisor. `RelaxationInteraction` SHALL relax a state towards a goal expression over a time, with optional
noise. `GradientInteraction` SHALL move states down the gradient of a potential, as a gradient flow on states of order
1, as Newtonian motion on states of order 2, and variationally and conserved on fields. `EulerLagrangeInteraction` SHALL
give the equations of motion of a Lagrangian with dissipation and forcing.

#### Scenario: Ornstein-Uhlenbeck variance
- **WHEN** a `RelaxationInteraction` with goal 0, time `1 / theta`, and noise `sigma` runs long
- **THEN** the variance of its state is `sigma**2 / (2 theta)` within 5 percent

#### Scenario: Gradient flow and Newtonian motion
- **WHEN** one `GradientInteraction` with a harmonic potential acts on a state of order 1 and on a state of order 2
- **THEN** the first decays exponentially and the second oscillates with the harmonic period

### Requirement: Divisors, Hurst indices, and summed Lagrangians
A divisor SHALL be the name of a state or parameter of the moving member or an expression of them, and a gradient SHALL
take one divisor per moving state. The Hurst index of a fractional noise SHALL be a number or a parameter. The equations of one coordinate from several `EulerLagrangeInteraction`s SHALL add,
so that their Lagrangians sum.

#### Scenario: Divisors per state
- **WHEN** a `GradientInteraction` joins two coordinates of one element by a spring, each divided by its own mass
- **THEN** the momentum of the two coordinates stays constant

#### Scenario: Summed Lagrangians
- **WHEN** two pendulums each have an `EulerLagrangeInteraction` and a third one holds only the energy of a spring
  between them
- **THEN** the angles follow the equations of motion of the sum of the three Lagrangians

### Requirement: Conservation and diffusion on fields
`ConservationInteraction` SHALL change a conserved state of a field by the divergence of a flux and a source, by a
finite-volume numerical flux, or by the field's own divergence operator on spectral and finite-difference fields.
`AdvectionInteraction` SHALL use the flux of a drift field times the state. `DiffusionInteraction` SHALL use the flux
minus a coefficient times the gradient.

#### Scenario: Conserved total
- **WHEN** a state diffuses on a periodic field
- **THEN** its total stays constant and its profile matches the heat kernel

### Requirement: Coupled conservation laws
The conservation laws of several states of one field, from any interactions, SHALL form one system whose numerical flux
reads the wave speeds of all of them, and a flux SHALL read parameters and states that vary across the cells.

#### Scenario: Coupled conservation laws
- **WHEN** the depth and the discharge of shallow water are conserved by two `ConservationInteraction`s
- **THEN** the trajectory equals that of the two flux laws of one element, and the total depth stays constant

#### Scenario: Advection by a drift field
- **WHEN** an `AdvectionInteraction` carries a state on a periodic field
- **THEN** it gives the trajectory of the conservation law of the flux `a u`, and the total stays constant

### Requirement: Maps, local rules, and cascades
`MapInteraction` SHALL set the next value of a state to a function of the namespace, or add it, updated synchronously,
sequentially, or in random order. `LocalRuleInteraction` SHALL update cells from their ordered neighbourhoods by a
table, a totalistic rule, or a function. `CascadeInteraction` SHALL drive a state, then relax every member at or above a
threshold until all are below it, recording cascades as events.

#### Scenario: Logistic map
- **WHEN** a `MapInteraction` iterates `r x (1 - x)` at `r = 3.2`
- **THEN** the state settles on the closed-form 2-cycle

#### Scenario: Sandpile avalanches
- **WHEN** grains drop on a 50 by 50 lattice with redistribution to four neighbours
- **THEN** the avalanche sizes follow a power law over two decades

### Requirement: Coupling over structures
`CouplingInteraction` SHALL give every member a reduction over its neighbours, its group, every member of a separate
source group, or the next member of an ordered group, of a function that reads both members, the edge, the parameters,
and the time, by sum, mean, maximum, minimum, normalized mean, or circular mean, with weights given as an edge state, a
matrix, or a function of both members, lag, and delay, and SHALL add it to a rate or combine it, with a function that
reads the member, the parameters, and the time, into the next value. `PairCouplingInteraction` SHALL do the same for
every pair within a cutoff, from a function of one or several states or from minus the gradient of a pair potential.

#### Scenario: Kuramoto synchronization
- **WHEN** 500 phase oscillators with sine coupling above the critical strength run on a complete graph
- **THEN** the order parameter exceeds 0.8 after the transient

#### Scenario: Vicsek alignment
- **WHEN** members align their headings within a radius by the map form at low noise
- **THEN** the polar order parameter exceeds 0.9

#### Scenario: Coupling to a separate population
- **WHEN** a group of oscillators couples by the mean sine of the phase differences to every member of a second group
- **THEN** every rate equals the direct mean over the second group, and the second group does not move

#### Scenario: Bounded confidence
- **WHEN** members take the normalized mean of the states within a distance, themselves included, weighted by an
  indicator of that distance
- **THEN** every next value equals the direct mean of the members within the distance

#### Scenario: Map reading parameters and time
- **WHEN** the map form combines the member's state, a parameter, and the time with the mean of its neighbours
- **THEN** every next value equals the direct evaluation of the combination

### Requirement: Jumps, reactions, and transitions
`JumpInteraction` SHALL change states by signed amounts at a rate, with a probability per step, or after drawn waiting
times, after which it SHALL reset scalar or vector states. `ReactionInteraction` SHALL move units between species, given as roles and states, by mass action or a given
propensity, with real stoichiometry and modifiers, as rates, jumps, or Langevin noise. `TransitionInteraction` SHALL
move a discrete state between values by a rate, a probability per step, a condition, a transition matrix, or an
emission matrix, after an optional residence time, and `SpreadingInteraction` SHALL spread an active value along edges.

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
- **WHEN** the Gray-Scott model runs its cubic autocatalysis, feed, and removal as reactions on the state names of its
  field beside its two diffusions
- **THEN** its trajectory equals the one of sites that carry the same rate laws of their own, up to round-off

#### Scenario: A reaction under renamed roles
- **WHEN** a subclass binds its species under the roles `susceptible` and `infected` and returns their pairs
- **THEN** it follows the mass action of `S + I -> 2 I` at its own rate constant, as a rate equation or as jumps

### Requirement: Encounters, crossings, impulses, and sampling
`EncounterInteraction` SHALL run conservative, destructive, or proliferative encounters over densities on activity
classes. `CrossingInteraction` SHALL reset states or end the run when a guard crosses zero. `ImpulseInteraction` SHALL
reset states at given or periodic times. `SamplingInteraction` SHALL run Monte Carlo sweeps by moves accepted from the
change of an energy.

#### Scenario: Integrate and fire
- **WHEN** a state integrates a constant input and a crossing resets it at a threshold
- **THEN** it fires with the closed-form period

#### Scenario: Ising magnetization
- **WHEN** a `SamplingInteraction` samples a square-lattice Ising energy well below the critical temperature
- **THEN** the absolute magnetization exceeds 0.9

### Requirement: Choice, imitation, reinforcement, and evolution
`ChoiceInteraction` SHALL draw an option from utilities by the logit, Metropolis, Fermi, or best rule.
`ImitationInteraction` SHALL copy a neighbour's state at random, by majority, from the best, or by the Fermi rule.
`ReinforcementInteraction` SHALL move option scores towards realized payoffs. `EvolutionInteraction` SHALL replace
strategies by selection, replacement, crossover, and mutation, drawing from the model's random stream.

#### Scenario: Glauber dynamics
- **WHEN** fully connected spins choose by the logit rule at intensity 2
- **THEN** the magnetization approaches the root of `m = tanh(2 m)`

#### Scenario: Voter consensus
- **WHEN** voters on a ring of 50 imitate a random neighbour
- **THEN** every run reaches consensus, and each opinion wins with its initial fraction

### Requirement: Agents, groups, and exchange
`MovementInteraction` SHALL move agents by a step. `TransferInteraction` SHALL move amounts of states between members,
cells, sources, and sinks. `LifeCycleInteraction` SHALL create and remove members. `GroupingInteraction` SHALL draw pairs
or groups every step and act on each. `MatchingInteraction` SHALL pair requesters with providers by the algorithm its rule spec names.
`FeedbackInteraction` SHALL set parameters from observables once per round. `TopologyInteraction` SHALL add, remove, and
rewire nodes and edges.

#### Scenario: Diffusion of walkers
- **WHEN** 10,000 agents step by Gaussian steps of variance `2 D dt` from the origin of a plane
- **THEN** their mean squared displacement after time `t` is `4 D t` within 5 percent

#### Scenario: Matching outside markets
- **WHEN** a ride-sharing model matches riders with idle cars by `StableMatchingSpec`
- **THEN** it uses `MatchingInteraction`, and no rider and car prefer each other to their matches

#### Scenario: Galam groups
- **WHEN** the Galam discussion model draws groups of three every step
- **THEN** it uses `GroupingInteraction`, and the majority of every group wins as before the change

### Requirement: Transfer flows and pairings
`TransferInteraction` SHALL report every symbolic leg as a flow under its name, under the element when the transfer moves
a quantity within one element and under the interaction otherwise. Its agent form SHALL pair agents into disjoint pairs
of neighbours, and SHALL pair with a fallback group in a step in which no target member lives.

#### Scenario: Named flows keep their reporters
- **WHEN** the Bass diffusion model moves adopters by a `TransferInteraction` within the market with `flow='adoption'`
- **THEN** the trajectory records the flow as `market.adoption`, as when the market declared the transfer itself

#### Scenario: Disjoint neighbour pairs
- **WHEN** traders on a grid barter with `pairing='neighbor-pairs'`
- **THEN** every trader is in at most one pair a step, and the two traders of a pair stand in neighbouring cells

#### Scenario: Fallback group
- **WHEN** a transfer pays a random member of a group that has no living member and names a `fallback` group
- **THEN** a random member of the fallback group receives the amount

### Requirement: Reference states
A state declared `StateSpec(reference=group)` SHALL hold the id of a member of another agent group, given initially as
its position among the group's initial members, and agent rules SHALL read that member with `agent.referenced(state)`.
A leg end `'<reference>.<state>'` of an agent transfer SHALL move the state of the referenced member, and a variadic
role entry naming a whole group SHALL bind its agents.

#### Scenario: Payment mirrored at the banks
- **WHEN** a customer whose reference state `bank` names a bank pays another customer with the legs `bank.deposits`
  and `bank.reserves`
- **THEN** the deposits and reserves move from the payer's bank to the payee's bank by the same amount

### Requirement: Source sets and sign steps
`TransitionInteraction` SHALL move a state from a set of source values or from the values a condition on the value
accepts. `SignAdjustmentInteraction` SHALL step a state of discrete time by the sign of a signal, multiplicatively or
additively, within bounds, in both forms.

#### Scenario: Separation from any employer
- **WHEN** a transition moves the employer of a worker to `-1` from the source `lambda value: value >= 0`
- **THEN** every employed worker loses its job at the probability, whichever firm employs it

#### Scenario: Price search by trial and error
- **WHEN** a shop's price steps by `SignAdjustmentInteraction(signal=direction, step=adjustment, lower=unit_cost)`
- **THEN** the price moves by the relative step in the direction, never below the unit cost

### Requirement: Agent and symbolic lowering
A general class whose operation agent models use SHALL lower to agent rules in its agent form, with functions of the
agent and the model view, and to symbolic laws in its symbolic form, chosen by a `form` argument.

#### Scenario: Choice reading history
- **WHEN** an agent model's logit choice reads the payoffs the model recorded in earlier rounds
- **THEN** it uses `ChoiceInteraction` with a utility of the agent and the model view

### Requirement: General elements
The core SHALL provide `CompartmentElement`, which holds one named state as a count or a fraction, `DensityElement`,
which holds a density over activity classes, and `PointElement`, a member with a vector state of order 2 in a
continuous space.

#### Scenario: Kinetic models share one element
- **WHEN** the kinetic models of wealth, opinion, crowds, and immune competition declare their populations
- **THEN** they all use `DensityElement`

### Requirement: Neighbour interactions carry member laws, degrees, delays, and cascades
`NeighborInteraction` SHALL take laws of the members at either end of an edge that run once per member, SHALL offer
the degrees of both ends, SHALL let `delayed` read the past state of either end, and SHALL take cascade, drive, and
local-rule laws, recording cascades as events.

#### Scenario: Recovery beside spreading
- **WHEN** a spreading interaction on a network declares spreading per edge and recovery per member
- **THEN** recovery runs once per member, not once per edge, and the fraction of neighbours uses the degree

### Requirement: Reductions read the reading member and reduce to an argmax
A reduction over a group SHALL be able to read the member that reads it apart from the reduced member, and a reduction
SHALL return the state of the neighbour at which an expression is largest.

#### Scenario: Mean field of a general function
- **WHEN** a coupling reads the mean of `sin(other.x - x)` over its group
- **THEN** every member gets the mean of the sine of its own difference to every other member

### Requirement: Law utilities
The law namespace SHALL offer `hill`, `logistic`, `pulse_train` with a count, `clip`, `minimum`, `maximum`, `sign`, the
stock-and-flow delays `delay1`, `delay3`, and `smooth`, reductions keyed by a state, and a circular mean.

#### Scenario: Hill function
- **WHEN** a rate reads `hill(x, K, n)` at `x = K`
- **THEN** it evaluates to one half
