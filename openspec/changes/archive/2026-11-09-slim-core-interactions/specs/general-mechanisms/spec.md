## MODIFIED Requirements

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

### Requirement: Encounters, crossings, impulses, and sampling
`EncounterInteraction` SHALL run conservative, destructive, or proliferative encounters over densities on activity
classes. `CrossingInteraction` SHALL reset states or end the run when a guard crosses zero, and periodic impulses
SHALL be crossings of the law utility `periodic_guard(t, period, start, count)`. `SamplingInteraction` SHALL run Monte Carlo sweeps by moves accepted from the change of
an objective at an intensity.

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

#### Scenario: Encounters as a collision operator
- **WHEN** a user reads the documentation of `EncounterInteraction`
- **THEN** it describes the binary kinetic collision operator over the class axes

### Requirement: Choice, imitation, reinforcement, and evolution
`ChoiceInteraction` SHALL draw an option from utilities by the logit, Metropolis, Fermi, or best rule.
`ImitationInteraction` SHALL copy a neighbour's state at random, by majority, from the best, from a random better
neighbour, or by the Fermi rule, as rates, per-step probabilities, or agent rules. Reinforcement of option scores
SHALL be a `MapInteraction` of the scores. `EvolutionInteraction` SHALL replace strategies by selection, replacement,
crossover, and mutation.

#### Scenario: Glauber dynamics
- **WHEN** fully connected spins choose by the logit rule at intensity 2
- **THEN** the magnetization approaches the root of `m = tanh(2 m)`

#### Scenario: Voter consensus
- **WHEN** voters on a ring of 50 imitate a random neighbour
- **THEN** every run reaches consensus, and each opinion wins with its initial fraction

### Requirement: Agents, groups, and exchange
`MovementInteraction` SHALL move agents by a step. `TransferInteraction` SHALL move amounts of states between members,
cells, sources, and sinks. `BirthDeathInteraction` SHALL create and remove members. `GroupingInteraction` SHALL draw
pairs or groups every step. `MatchingInteraction` SHALL pair requesters with providers by its rule spec, reading the
states that `reserve`, `quantity`, and `cap` name. `FeedbackInteraction` SHALL set parameters from observables.
`TopologyInteraction` SHALL add, remove, and rewire nodes and edges.

#### Scenario: Diffusion of walkers
- **WHEN** 10,000 agents step by Gaussian steps of variance `2 D dt` from the origin of a plane
- **THEN** their mean squared displacement after time `t` is `4 D t` within 5 percent

#### Scenario: Matching outside markets
- **WHEN** a ride-sharing system matches riders with idle cars by `StableMatchingSpec`
- **THEN** it uses `MatchingInteraction`, and no rider and car prefer each other to their matches

#### Scenario: Neutral matching names
- **WHEN** a matching binds `reserve=('bid', 'ask')` with `StableMatchingSpec(value='requester')`
- **THEN** every pair settles at the reserve of its requester, and no argument of the matching or its specs is named
  `limit`, `allowance`, or `clearing`

#### Scenario: Life cycle variants
- **WHEN** a birth-death interaction gives births into another group with values of the parent, keeps a share of a
  split state, replaces parents by their offspring, renews dying agents in place, removes kin of another group, or
  brings arrivals without a parent at placed locations
- **THEN** the new agents join the named group with the given values, the parent keeps that share, a parent that gave
  birth is gone, a renewed agent keeps its identity and the states the new values leave out, the first kin member is
  removed, and a Poisson number of arrivals of the given mean appear, with the counts reported as `births`, `deaths`,
  and `arrivals`

#### Scenario: Growth from outside a network
- **WHEN** a `TopologyInteraction` adds nodes without a target, with values drawn from the generator
- **THEN** one node joins per step with its edges, as in Barabási-Albert growth, and every new node takes its own draw

#### Scenario: Galam groups
- **WHEN** the Galam discussion system draws groups of three every step
- **THEN** it uses `GroupingInteraction`, and the majority of every group wins as before the change

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
