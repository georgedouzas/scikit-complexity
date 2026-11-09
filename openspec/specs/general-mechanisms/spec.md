# general-mechanisms Specification

## Purpose
Gives systems of every field the operations they share as general core elements and interactions, configured by the
states they bind, by functions, and by named variants, so a system defines only what is its own.

## Requirements

### Requirement: Core classes name operations
The core SHALL provide one general element or interaction per mathematical operation that systems use. Each SHALL take
the roles it binds, the names of the states it reads and changes, the functions that define it, and its variant as a
named argument. No core class, argument, or variant SHALL name a quantity of a field, such as a force, a position, a
velocity, a mass, a price, an infection, an energy, or a harvest.

#### Scenario: Field terms are rejected
- **WHEN** the field-term test checks the public names of `skcomplex.core.components`
- **THEN** none of them contains a field term

#### Scenario: A force is a rate
- **WHEN** a system adds a constant force `F` to a particle whose position `x` is a state of order 2 with mass `m`
- **THEN** it uses `RateInteraction(state='x', rate=F, divisor='m')`, and the second derivative of `x` is `F / m`

### Requirement: Field cases are subclasses
A field case of a general class SHALL be a subclass of it, even a pure renaming, that lives with its one system, in the
`common` package of its field when several of its packages use it, or in `skcomplex.systems.general` when several
fields use it. No field SHALL define the same case twice.

#### Scenario: Lennard-Jones from the core
- **WHEN** a physics system uses Lennard-Jones particles
- **THEN** `LennardJonesInteraction` of `physics/common/particles` is a `PairInteraction` with its pair potential, and
  its minimum lies at `2 ** (1 / 6)` times its length

### Requirement: Rates, relaxation, gradients, and Lagrangians
`RateInteraction` SHALL add a rate and an optional noise to the highest derivative of a state of any order, divided by
an optional divisor, and a relaxation SHALL be one with the rate `(goal - x) / time`. `GradientInteraction` SHALL move
states down the gradient of a potential, as a gradient flow on states of order 1, as Newtonian motion on states of
order 2, and variationally and conserved on fields.
`EulerLagrangeInteraction` SHALL give the equations of motion of a Lagrangian with dissipation and forcing.

#### Scenario: Ornstein-Uhlenbeck variance
- **WHEN** a `RateInteraction` with the rate `-theta x` and the noise `sigma` runs long
- **THEN** the variance of its state is `sigma**2 / (2 theta)` within 5 percent

#### Scenario: No relaxation class
- **WHEN** a user lists `skcomplex.core.components.interactions`
- **THEN** it holds no relaxation, diffusion, advection, impulse, sign-step, or reinforcement class

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
Advection SHALL be a `ConservationInteraction` with the flux of a drift field times the state. Diffusion SHALL be a
`RateInteraction` with the rate `D laplacian(u)`. The core SHALL have no advection or diffusion class.

#### Scenario: Conserved total
- **WHEN** a state diffuses on a periodic field by a `RateInteraction` with the rate `D laplacian(u)`
- **THEN** its total stays constant and its profile matches the heat kernel

### Requirement: Coupled conservation laws
The conservation laws of several states of one field, from any interactions, SHALL form one system whose numerical flux
reads the wave speeds of all of them, and a flux SHALL read parameters and states that vary across the cells.

#### Scenario: Coupled conservation laws
- **WHEN** the depth and the discharge of shallow water are conserved by two `ConservationInteraction`s
- **THEN** the trajectory equals that of the two flux laws of one element, and the total depth stays constant

#### Scenario: Advection by a drift field
- **WHEN** a `ConservationInteraction` with the flux `a u` carries a state on a periodic field
- **THEN** it gives the trajectory of the conservation law of the flux `a u`, and the total stays constant

### Requirement: Maps, local rules, and cascades
`MapInteraction` SHALL set the next value of a state to a function of the namespace, or add it, updated synchronously,
sequentially, or in random order. Reinforcement of option scores SHALL be a `MapInteraction` of the scores.
`LocalRuleInteraction` SHALL update cells from their ordered neighbourhoods by a table, a totalistic rule, or a
function. `CascadeLaw` SHALL run its drive laws once per step, then its relaxing laws until the lattice is stable,
recording every cascade as an event. Threshold toppling SHALL be a realization of the field that uses it.

#### Scenario: Logistic map
- **WHEN** a `MapInteraction` iterates `r x (1 - x)` at `r = 3.2`
- **THEN** the state settles on the closed-form 2-cycle

#### Scenario: Sandpile avalanches
- **WHEN** grains drop on a 50 by 50 lattice with redistribution to four neighbours, through the physics
  `TopplingInteraction` on `CascadeLaw`
- **THEN** the avalanche sizes follow a power law over two decades

### Requirement: Coupling over structures
`CouplingInteraction` SHALL give every member a reduction over its neighbours, its group, every member of a separate
source group, or the next member of an ordered group, of a function that reads both members, the edge, the parameters,
and the time, by sum, mean, maximum, minimum, normalized mean, or circular mean, with weights given as an edge state, a
matrix, or a function of both members, lag, and delay, and SHALL add it to a rate or combine it, with a function that
reads the member, the parameters, and the time, into the next value. `PairInteraction` SHALL do the same for every pair
within a cutoff, from a function of one or several states or from minus the gradient of a pair potential, and with
neither it SHALL return no laws, so a subclass writes its own. The zones of an agent flock SHALL be a realization of
the field that uses them, not an argument of `CouplingInteraction`.

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

#### Scenario: Pair coupling is the pair interaction
- **WHEN** a user builds `PairInteraction('lj', 'gas', 'gas', cutoff=2.5, potential=potential)`
- **THEN** it adds minus the gradient of the potential to every member of every pair within the cutoff, and the LAMMPS
  backend lowers it to a pair style as it lowered `PairCouplingInteraction`

### Requirement: Transfer flows and pairings
`TransferInteraction` SHALL report every symbolic leg as a flow under its name, under the element when the transfer moves
a quantity within one element and under the interaction otherwise. Its agent form SHALL pair agents into disjoint pairs
of neighbours, and SHALL pair with a fallback group in a step in which no target member lives.

#### Scenario: Named flows keep their reporters
- **WHEN** the Bass diffusion system moves adopters by a `TransferInteraction` within the market with `flow='adoption'`
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
accepts. A step of a state of discrete time by the sign of a signal, within bounds, SHALL be a `MapInteraction` whose
function is `clip(x (1 + step sign(signal)), lower, upper)` or `clip(x + step sign(signal), lower, upper)`, and the
core SHALL have no sign-step class.

#### Scenario: Separation from any employer
- **WHEN** a transition moves the employer of a worker to `-1` from the source `lambda value: value >= 0`
- **THEN** every employed worker loses its job at the probability, whichever firm employs it

#### Scenario: Price search by trial and error
- **WHEN** a shop's price steps by the economics `PriceAdjustmentInteraction(signal=direction, step=adjustment,
  lower=unit_cost)`
- **THEN** the price moves by the relative step in the direction, never below the unit cost

### Requirement: Agent and symbolic lowering
A general class whose operation agent systems use SHALL lower to agent rules in its agent form, with functions of the
agent and the system view, and to symbolic laws in its symbolic form, chosen by a `form` argument.

#### Scenario: Choice reading history
- **WHEN** an agent system's logit choice reads the payoffs the system recorded in earlier rounds
- **THEN** it uses `ChoiceInteraction` of `skcomplex.systems.general.agents` with a utility of the agent and the system
  view, which lowers to a core `RuleLaw`

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
distributed delay of any order with or without conservation, a series of any value through given points, reductions
keyed by a state, and a circular mean. It SHALL offer one function for one operation, so it SHALL NOT offer a function
that only fixes arguments of another.

#### Scenario: Hill function
- **WHEN** a rate reads `hill(x, K, n)` at `x = K`
- **THEN** it evaluates to one half

#### Scenario: Third-order material delay
- **WHEN** a rate reads `distributed_delay(x, T, 3)` of a step input
- **THEN** its output follows the third-order material delay of the step

#### Scenario: Information smoothing
- **WHEN** a rate reads `distributed_delay(x, T, 1, conserved=False)`
- **THEN** its output is the first-order exponential smoothing of `x` with time `T`

#### Scenario: Series of a state
- **WHEN** a rate reads `series(x, [0, 1, 2], [0.0, 2.0, 3.0])` at `x = 0.5` and at `x = 5`
- **THEN** it evaluates to 1.0 and to 3.0

### Requirement: Agent interactions run member by member
An agent rule of an interaction SHALL be able to name the role whose members run it, so every member runs it in its own
activation after the rules of its element and the member rules of the interactions listed before it. Movement,
transfer, life cycle, transition, and map SHALL take the schedule `'member'` for this, transitions SHALL also take the
sequential and random-sequential schedules in the agent form, and transitions SHALL let a member act on the members of
its cell. Transfers SHALL let an actor draw partners of its cell, remove the eaten member, and move amounts into the site
of a field at its cell, and the best step of a movement SHALL look along the directions of a lattice up to a vision.

#### Scenario: Walker infects the cell it reaches
- **WHEN** a walker's movement, a cell transition from sick to healthy walkers, and a death transition after a residence
  run under the schedule `'member'`
- **THEN** a walker infected earlier in a step walks on and infects the cell it reaches in the same step, as one rule of
  the walker did

#### Scenario: Wolf eats a sheep of its cell
- **WHEN** a transfer from sheep to wolves pairs every wolf with one random sheep of its cell and removes it
- **THEN** every wolf with a sheep in its cell gains its energy and the sheep is gone

#### Scenario: Forager in sight of sugar
- **WHEN** a best step with a vision of 3 scores the cells by their sugar
- **THEN** the agent moves to the richest cell up to 3 cells along an axis, the nearest among ties

### Requirement: Coupling of separate elements and hubs
`CouplingInteraction` SHALL couple a set of separate elements to each other, and a single hub element to a group by
any reduction.

#### Scenario: Global inhibition
- **WHEN** a hub element reads the largest state of a lattice group and inhibits every member
- **THEN** it uses `CouplingInteraction`, and the trajectory equals the previous system's

### Requirement: Transfers along edges
`TransferInteraction` SHALL move amounts along the edges of a lattice or network in both directions at parametric
rates, with one draw per edge shared by both ends.

#### Scenario: Parallel exclusion process
- **WHEN** particles hop right on a ring under parallel update with exclusion
- **THEN** the current equals the exact parallel-update current, and no site holds two particles

### Requirement: Visits drawn with replacement
The agent forms SHALL run a phase as N visits by members drawn with replacement, each composing the member
interactions of the visited member.

#### Scenario: Perishables market
- **WHEN** the perishables market runs on the general classes
- **THEN** its magnetization over 20 seeds equals the previous system's at every temperature tested

### Requirement: Agent coupling over radii
The agent form of `CouplingInteraction` SHALL sum terms over several radii with a priority among them and clamp the
resulting turn and speed.

#### Scenario: Couzin zones
- **WHEN** fish repel within one radius, align within a second, and attract within a third
- **THEN** repulsion overrides the other zones, and the polarization over 20 seeds equals the previous system's

### Requirement: Rules read observable history
An agent rule SHALL read the recorded values of an observable at earlier rounds.

#### Scenario: Attendance history
- **WHEN** a choice reads the bar attendance of the previous rounds
- **THEN** no reporter-only interaction records it

### Requirement: One partner-selection spec
Transfers, matching, and groupings SHALL accept one partner-selection spec made of candidates (all, neighbours, a radius,
or a radius read from a state), a choice among them (uniform, by score, nearest), a fallback when there is none (none or
the nearest partner), and a capacity, or a pairing callable instead.

#### Scenario: Within reach, else the nearest
- **WHEN** agents pick a provider within the radius in their `reach` state, falling back to the nearest provider
- **THEN** an agent with providers in reach picks one of them uniformly, and an agent with none picks the nearest

#### Scenario: Continuous space
- **WHEN** the candidates are a radius and the agents live in a continuous space
- **THEN** candidates are the agents within that Euclidean distance, with no grid required

#### Scenario: Pairing callable
- **WHEN** a matching receives a pairing callable instead of a spec
- **THEN** it pairs agents as the callable returns and applies its settlement to each pair

### Requirement: Settlement from one draw credits both ends
A transfer between two partners SHALL be able to credit and debit both ends from one shared draw.

#### Scenario: Meal
- **WHEN** an eater buys a meal whose calories are drawn once
- **THEN** the eater gains those calories and the provider's sales rise by one in the same step

### Requirement: Core names carry no field vocabulary
Core classes, arguments, specs, and law utilities SHALL use the vocabulary of the operation, and a field's vocabulary SHALL
appear only in thin subclasses in `skcomplex.systems`.

#### Scenario: Sampling an objective
- **WHEN** a user samples a function that is not an energy with `SamplingInteraction(objective=..., intensity=...)`
- **THEN** moves are accepted from the change of the objective, with no `temperature` or `energy` argument in the core

#### Scenario: Physics keeps its words
- **WHEN** a statistical mechanics system sweeps the `temperature` of its thin sampling subclass
- **THEN** the subclass gives the core an intensity of one over the temperature

### Requirement: One way to select partners
`PartnerSpec` SHALL be the only spec that selects partners by candidates, a choice, a score, a fallback, and a
capacity, and it SHALL hold no clearing value. Matching algorithms over offers SHALL be realizations named for the
operation, such as batch, continuous, and iterative matching.

#### Scenario: Greedy matching by a partner spec
- **WHEN** a matching uses a partner spec with a score and a capacity
- **THEN** it pairs agents as `GreedyMatchingSpec` did, and `GreedyMatchingSpec` no longer exists

#### Scenario: Partner pairing of a transfer
- **WHEN** a transfer pairs agents by `PartnerSpec('reach', fallback='nearest')`
- **THEN** it pairs them as before, and `PartnerSpec` takes no argument `value`

### Requirement: Bilinear laws are one law
One bilinear law SHALL express gains and losses of a density at the rate of the product of two densities. A conserved
bilinear law SHALL also take the matching loss from its first density, so the total over its classes is unchanged, and
a law without a second density SHALL redistribute the first linearly.

#### Scenario: Encounter outcomes
- **WHEN** an encounter is destructive or proliferative
- **THEN** it lowers to the bilinear law with the sign of its outcome, with the trajectories it had before

#### Scenario: Conservative encounter
- **WHEN** an encounter is conservative
- **THEN** it lowers to a conserved bilinear law, with the trajectories it had before

#### Scenario: Linear redistribution
- **WHEN** a conserved bilinear law with no second density moves a density of two classes from the first class to the
  second at rate 1, starting from all in the first class
- **THEN** the first class decays as `exp(-t)` within 1e-6, and the total stays constant within 1e-12

### Requirement: Operator dynamics are named for their mathematics
The core and the engine SHALL name linear operator dynamics on tensor products by the operation, with no quantum vocabulary
in their class names, and physics SHALL keep its quantum vocabulary in `skcomplex.systems.physics.common.quantum`.

#### Scenario: Quantum system
- **WHEN** a user runs a catalog quantum system
- **THEN** it compiles with the operator compiler and gives the trajectory it gave before

#### Scenario: Core names
- **WHEN** the class names of `skcomplex.core.components`, `skcomplex.core.engine`, and `skcomplex.core.analysis` are searched for `Quantum`
- **THEN** none is found

### Requirement: The law namespace holds only what laws are written with
The public law namespace SHALL re-export only the functions and symbols a law is written with. The helpers that only
the compiler reads, such as the functions that list the reductions, draws, or operators of an expression, and the
constants of the compiler, SHALL NOT be part of it.

#### Scenario: Compiler helper
- **WHEN** a user imports `reductions_in` from the public law namespace
- **THEN** the import raises an `ImportError`

#### Scenario: Operator constructors stay
- **WHEN** a user writes `J * operator_product(a, b) + dagger(b)` from the public law namespace
- **THEN** the expression is a valid operator of a generator law

### Requirement: Draw functions have one import path
The draw functions `uniform`, `normal`, `bernoulli`, `categorical`, and `draw` SHALL be importable from the public law
namespace alone, and the module that defines them SHALL be a module of that namespace.

#### Scenario: One path
- **WHEN** a user imports `uniform` from the public law namespace
- **THEN** the import succeeds, and no other public package of the core re-exports `uniform`

### Requirement: Transfers, groupings, and topology
`TransferInteraction` SHALL move amounts of states between members, cells, sources, and sinks. `GroupingInteraction`
SHALL draw pairs or groups every step. `MatchLaw` SHALL pair the members of agent groups every step for the rules of
the step. `TopologyInteraction` SHALL add, remove, and rewire nodes and edges. Movement, birth and death, matching, and
feedback SHALL be realizations in `skcomplex.systems` built on these primitives.

#### Scenario: Growth from outside a network
- **WHEN** a `TopologyInteraction` adds nodes without a target, with values drawn from the generator
- **THEN** one node joins per step with its edges, as in Barabási-Albert growth, and every new node takes its own draw

#### Scenario: Galam groups
- **WHEN** the Galam discussion system draws groups of three every step
- **THEN** it uses `GroupingInteraction`, and the majority of every group wins as before the change

#### Scenario: Agent recipes leave the core
- **WHEN** a user imports `MovementInteraction`, `BirthDeathInteraction`, `MatchingInteraction`, or
  `FeedbackInteraction` from `skcomplex.core.components.interactions`
- **THEN** the import fails with `ImportError`, since the project ships no alias for a moved class

### Requirement: Bilinear encounters, crossings, impulses, and sampling
`BilinearLaw` SHALL give the gain and loss of encounters of densities over activity classes. `CrossingInteraction`
SHALL reset states or end the run when a guard crosses zero, and periodic impulses SHALL be crossings of
`periodic_guard(t, period, start, count)`. `SamplingInteraction` SHALL run Monte Carlo sweeps by moves accepted from the
change of a symbolic objective. Objectives over blocks of sites SHALL be field subclasses of it.

#### Scenario: Integrate and fire
- **WHEN** a state integrates a constant input and a crossing resets it at a threshold
- **THEN** it fires with the closed-form period

#### Scenario: Ising magnetization
- **WHEN** a `SamplingInteraction` samples a square-lattice Ising energy well below the critical temperature
- **THEN** the absolute magnetization exceeds 0.9

#### Scenario: Periodic impulses by a guard
- **WHEN** a `CrossingInteraction` with the guard `periodic_guard(t, 2.0, 1.0, 3)` adds 1 to a state over a run to 10
- **THEN** it fires at 1, 3, and 5 only, an impulse at the initial time does not fire, and without a count it fires
  at every period to the end

#### Scenario: Impulses at given times
- **WHEN** a `CrossingInteraction` has the guard `t - 2.5`
- **THEN** it fires once at 2.5

#### Scenario: Sampling takes no special scores
- **WHEN** a user passes a row, region, or holonomy score to the core `SamplingInteraction`
- **THEN** checking it raises a `ValidationError` that names the symbolic objective, and the scores live with the
  subclasses of the fields

### Requirement: Structural elements
The core SHALL provide the structural elements `Element`, `GroupElement`, `FieldElement`, and `DeclaredElement`, and
no element that only fixes the states of `Element`. A compartment, a density, a species, or a population SHALL be
`NonnegativeElement` of `skcomplex.systems.general.elements` or a field subclass that keeps the state `x`. The engine
SHALL read the state specs of elements, never these classes.

#### Scenario: Elements of one state leave the core
- **WHEN** a user imports `NonnegativeElement` or `PointElement` from `skcomplex.core.components.elements`
- **THEN** the import fails with `ImportError`, and both import from `skcomplex.systems.general`

#### Scenario: Former elements are gone
- **WHEN** a user imports `CompartmentElement` or `DensityElement` from `skcomplex.core.components.elements`
- **THEN** the import fails with `ImportError`, since the project ships no alias for a removed class

### Requirement: Jumps and transitions
`JumpInteraction` SHALL change states by signed amounts at a rate, with a probability per step, or after drawn
waiting times, and SHALL then reset scalar or vector states. A reaction SHALL be a jump whose changes are its
stoichiometric vector, realized in `skcomplex.systems.general.reactions`. `TransitionInteraction` SHALL move a discrete
state by a rate, a probability, a condition, or a matrix, after an optional residence time.

#### Scenario: Stoichiometric jumps
- **WHEN** a `JumpInteraction` adds `{S: -1, I: +1}` at the rate `beta S I` and another adds `{I: -1, R: +1}` at the
  rate `gamma I`, with 100,000 units
- **THEN** the jump means follow the solution of the rate equations within 2 percent, and `S + I + R` stays constant

#### Scenario: Epidemic threshold
- **WHEN** a contact process runs on a square lattice below and above its critical spreading rate
- **THEN** the active fraction dies out below it and stays positive above it
