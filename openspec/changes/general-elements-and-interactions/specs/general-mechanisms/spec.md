# Spec Delta

## Purpose
Gives models of every field the mechanisms they share as ready core elements and interactions, configured by states,
parameters, functions, and named mathematical variants, so a model defines only what is its own.

## ADDED Requirements

### Requirement: Mechanisms are field-neutral core classes
The core SHALL provide every mechanism that models of more than one field use as a concrete element or interaction in
`skcomplex.core.elements` or `skcomplex.core.interactions`. Each SHALL take the states it reads and changes by name, its
parameters as declared parameters, and its variant as a named argument. Every class name and every variant name SHALL
be a mathematical term, never a field term, and a model SHALL use the class as it is or subclass it to add what is its
own.

#### Scenario: Same mechanism in two fields
- **WHEN** an epidemic model and a chemical kinetics model both declare a reaction `A + B -> 2B` with rate `k`
- **THEN** both use `ReactionInteraction`, and their trajectories agree with the mass-action equations

#### Scenario: Field terms are rejected
- **WHEN** the field-term test checks the public names of `skcomplex.core`
- **THEN** none of the new classes or variant names contains a field term

### Requirement: Agents move by random walks
`RandomWalkInteraction` SHALL move the members of a group at every step, or at a given rate, by one of the variants
`'neighbor'` on a lattice or network, `'gaussian'`, or `'levy'` in continuous space. It SHALL take a step probability and
an optional cost charged to a named state.

#### Scenario: Diffusion of walkers
- **WHEN** 10,000 walkers start at the origin of a continuous plane with Gaussian steps of variance `2 D dt`
- **THEN** their mean squared displacement after time `t` is `4 D t` within 5 percent

### Requirement: Reactions and compartment flows
`ReactionInteraction` SHALL move units between named states by mass action, from reactants to products with a rate
expression, in the mode `'rate'` (equations), `'jump'` (stochastic events), or `'langevin'` (chemical Langevin noise),
and SHALL conserve the total of states that every reaction preserves.

#### Scenario: Deterministic and stochastic modes agree
- **WHEN** an infection `S + I -> 2 I` and a recovery `I -> R` run in rate mode and in jump mode with 100,000 units
- **THEN** the jump means stay within 2 percent of the rate solution, and `S + I + R` stays constant in both

### Requirement: Spreading over neighbours
`SpreadingInteraction` SHALL change a member's discrete state to the spreading value at a rate proportional to the
number, or the fraction, of its neighbours that hold it, with optional recovery and loss of immunity, on lattices and
networks.

#### Scenario: Epidemic threshold
- **WHEN** a contact process runs on a square lattice below and above its critical spreading rate
- **THEN** the active fraction dies out below it and stays positive above it

### Requirement: Coupling of states
`CouplingInteraction` SHALL add to the rate of a state of a target the coupling strength times a function of the
source's and the target's states, the difference by default, with an optional phase lag, delay, or mean field over a
group.

#### Scenario: Kuramoto synchronization
- **WHEN** 500 phase oscillators with sine coupling above the critical strength run on a complete graph
- **THEN** the order parameter exceeds 0.8 after the transient

### Requirement: Thresholds and cascades
`ThresholdInteraction` SHALL fire a member whose state, or the weighted sum of its inputs, reaches a threshold, then
reset the state, or redistribute it to its neighbours, and SHALL record every cascade as an event with its size.

#### Scenario: Sandpile avalanches
- **WHEN** grains drop on a 50 by 50 sandpile with redistribution to four neighbours
- **THEN** the avalanche sizes follow a power law over two decades

### Requirement: Choices by a rule
`ChoiceInteraction` SHALL set a discrete state of every member to one of its options, drawn from the utilities of the
options by the rule `'logit'`, which is the heat-bath rule, `'metropolis'`, or `'fermi'`, with an intensity of choice.

#### Scenario: Glauber dynamics
- **WHEN** fully connected spins choose by the logit rule at inverse temperature 2
- **THEN** the magnetization approaches the mean-field solution of `m = tanh(2 m)`

### Requirement: Imitation of neighbours
`ImitationInteraction` SHALL copy a discrete state from neighbours by the rule `'random'` (voter), `'majority'`, or
`'best'` (highest payoff), at a rate.

#### Scenario: Voter consensus
- **WHEN** voters on a ring of 50 imitate a random neighbour
- **THEN** every run reaches consensus, and the probability of each opinion winning equals its initial fraction

### Requirement: Foraging and regrowth
`ForagingInteraction` SHALL let members take a resource from the cell they stand on into an energy state, pay a
metabolism, reproduce by splitting their energy above a threshold, and die at zero energy. `RegrowthInteraction` SHALL
let the resource of every cell regrow, by a delay or towards a capacity.

#### Scenario: Two fields, one mechanism
- **WHEN** the ecology meadow and the economics commons models run
- **THEN** both use `ForagingInteraction` and `RegrowthInteraction`, and their trajectories match the ones before the
  change

### Requirement: Ornstein-Uhlenbeck and geometric noise
`OrnsteinUhlenbeckInteraction` SHALL relax a state towards a mean at a rate with additive noise, or with noise
proportional to the state when it is multiplicative, which gives geometric Brownian motion.

#### Scenario: Stationary variance
- **WHEN** an Ornstein-Uhlenbeck state with rate `theta` and noise `sigma` runs long
- **THEN** its variance is `sigma**2 / (2 theta)` within 5 percent

### Requirement: Growth and decay
`GrowthInteraction` SHALL change a population at a rate by the form `'exponential'`, `'logistic'`, `'gompertz'`,
`'power'`, or `'bertalanffy'`, with an optional capacity, exponent, harvest, and delay.

#### Scenario: Logistic capacity
- **WHEN** a population grows logistically with capacity 100 from 1
- **THEN** it approaches 100, and its inflection is at 50

### Requirement: Alignment
`AlignmentInteraction` SHALL turn the heading or the velocity of every member towards the mean of the members within a
radius, polar or nematic, with optional noise, repulsion, and attraction zones.

#### Scenario: Vicsek ordering
- **WHEN** self-propelled members align within a radius at low noise
- **THEN** the polar order parameter exceeds 0.9

### Requirement: Evolution of strategies
`EvolutionInteraction` SHALL replace the strategies of a group every generation by selection on a fitness expression,
`'tournament'` or `'roulette'`, one-point crossover, and mutation, over an alphabet or with Gaussian noise for real
strategies, drawing from the model's random stream.

#### Scenario: Selection raises fitness
- **WHEN** bit-string strategies evolve on the count of ones for 50 generations
- **THEN** the mean fitness rises, and two runs with the same seed give the same strategies

### Requirement: Forces and potentials by state names
`PotentialInteraction` and `PairPotentialInteraction` SHALL take the names of the position, velocity, and mass states,
`x`, `v`, and `m` by default. `PairForceInteraction` SHALL apply the force of a potential of the distance given as a
function, within a cutoff. `DampingInteraction` SHALL damp a velocity linearly, and `ConstantForceInteraction` SHALL
add a constant force.

#### Scenario: Potential on renamed states
- **WHEN** a pair potential acts on members whose position state is `r` and velocity state is `u`
- **THEN** the force derives from the potential of `r` and changes `u`

### Requirement: Densities over classes
`DensityElement` SHALL hold a density over activity classes for encounter laws, and `CompartmentElement` SHALL hold
named compartments as counts or fractions.

#### Scenario: Kinetic model with one element
- **WHEN** the kinetic models of wealth, opinion, crowds, and immune competition declare their populations
- **THEN** they all use `DensityElement`

### Requirement: Matching of requesters with providers
`MatchingInteraction` SHALL pair the members of a `requesters` group with those of a `providers` group by a protocol,
read their `limit`, `quantity`, and `budget` states by the names the user gives, and settle every match by transfers
of named states. The protocols SHALL be `RandomSearchProtocol`, `PartnerSwitchProtocol`, `QueueProtocol`,
`PriceAdjustmentProtocol`, `PriceTimePriorityProtocol`, `UniformPriceProtocol`, and `StableMatchingProtocol`.

#### Scenario: Matching outside markets
- **WHEN** a ride-sharing model matches riders with idle cars by `StableMatchingProtocol`
- **THEN** it uses `MatchingInteraction`, and no rider is matched to a car that both prefer less than a free pair

### Requirement: Feedback on parameters
`FeedbackInteraction` SHALL act once per round of its phase, read observables, reporters, and agents through the model
view, and set parameters of any part, with the change recorded in the trajectory.

#### Scenario: Policy rule
- **WHEN** a feedback raises a rate parameter whenever an observable exceeds a target
- **THEN** the parameter changes in the trajectory only at rounds where the observable exceeds the target

### Requirement: Group mechanisms over hyperedges
Mechanisms that act on groups of more than two members at once SHALL run as `HyperedgeInteraction` subclasses over
networks with hyperedges.

#### Scenario: Galam groups
- **WHEN** the Galam discussion model runs with groups of three
- **THEN** its groups are hyperedges of a network, and the majority of every group wins as before the change

### Requirement: One class per equation within a field
A named equation SHALL be a realization of a field, never a core class. Within one field it SHALL be defined once, in
the `common` package of the field when several of its packages use it, and next to its model otherwise.

#### Scenario: Stuart-Landau once in physics
- **WHEN** the physics amplitude-death and oscillator models run
- **THEN** both use one `StuartLandauElement` of `physics/common`, and the core defines none

### Requirement: Neighbour interactions carry member laws, degrees, delays, and cascades
`NeighborInteraction` SHALL take laws of the members at either end of an edge that run once per member, SHALL offer
the degrees of both ends, SHALL let `delayed` read the past state of either end, and SHALL take relaxation, drive, and
local-rule laws, recording cascades as events.

#### Scenario: Recovery beside spreading
- **WHEN** a spreading interaction on a network declares infection per edge and recovery per member
- **THEN** recovery runs once per member, not once per edge, and the fraction of neighbours uses the degree

### Requirement: Reductions read the reading member and reduce to an argmax
A reduction over a group SHALL be able to read the member that reads it apart from the reduced member, and a reduction
SHALL return the state of the neighbour at which an expression is largest.

#### Scenario: Mean field of a general function
- **WHEN** a coupling reads the mean of `sin(other.x - x)` over its group
- **THEN** every member gets the mean of the sine of its own difference to every other member

