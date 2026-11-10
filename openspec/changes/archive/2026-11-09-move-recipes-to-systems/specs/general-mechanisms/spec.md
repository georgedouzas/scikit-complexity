## MODIFIED Requirements

### Requirement: Agent and symbolic lowering
A general class whose operation agent systems use SHALL lower to agent rules in its agent form, with functions of the
agent and the system view, and to symbolic laws in its symbolic form, chosen by a `form` argument.

#### Scenario: Choice reading history
- **WHEN** an agent system's logit choice reads the payoffs the system recorded in earlier rounds
- **THEN** it uses `ChoiceInteraction` of `skcomplex.systems.general.agents` with a utility of the agent and the system
  view, which lowers to a core `RuleLaw`

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

### Requirement: Field cases are subclasses
A field case of a general class SHALL be a subclass of it, even a pure renaming, that lives with its one system, in the
`common` package of its field when several of its packages use it, or in `skcomplex.systems.general` when several
fields use it. No field SHALL define the same case twice.

#### Scenario: Lennard-Jones from the core
- **WHEN** a physics system uses Lennard-Jones particles
- **THEN** `LennardJonesInteraction` of `physics/common/particles` is a `PairInteraction` with its pair potential, and
  its minimum lies at `2 ** (1 / 6)` times its length

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

## ADDED Requirements

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

## REMOVED Requirements

### Requirement: Choice, imitation, reinforcement, and evolution
**Reason**: Choice, imitation, and evolution are recipes on `RuleLaw`, `UpdateLaw`, and `JumpLaw`. Reinforcement moves
to the requirement on maps.
**Migration**: Import `ChoiceInteraction` and `ImitationInteraction` from `skcomplex.systems.general.agents` and
`EvolutionInteraction` from `skcomplex.systems.general.agents`. The behaviour is in `general-realizations`.

### Requirement: Evolution variants
**Reason**: `EvolutionInteraction` is a recipe on `RuleLaw` and leaves the core.
**Migration**: Import it from `skcomplex.systems.general.agents`. The variants are in `general-realizations`.

### Requirement: Encounters take a collision integral
**Reason**: `EncounterInteraction` is a recipe on `BilinearLaw`, `RateLaw`, and `collision_integral`, which stay.
**Migration**: Import it from `skcomplex.systems.general.kinetics`. The behaviour is in `general-realizations`.

### Requirement: Parametric stoichiometry
**Reason**: `ReactionInteraction` is a recipe on `JumpInteraction` and leaves the core.
**Migration**: Import it from `skcomplex.systems.general.reactions`. The behaviour is in `general-realizations`.

### Requirement: Per-member reactions and renamed roles
**Reason**: `ReactionInteraction` is a recipe on `JumpInteraction` and leaves the core.
**Migration**: Import it from `skcomplex.systems.general.reactions`. The behaviour is in `general-realizations`.

### Requirement: Reactions follow a product propensity
**Reason**: The product propensity belongs to the reaction recipe, which leaves the core.
**Migration**: Import `ReactionInteraction` from `skcomplex.systems.general.reactions`. The propensity is stated in
`general-realizations`.

### Requirement: Agents, groups, and exchange
**Reason**: Movement, birth and death, matching, and feedback are recipes and leave the core. The rest is restated as
"Transfers, groupings, and topology".
**Migration**: Import the recipes from `skcomplex.systems.general` or from their field, as `general-realizations` and
`model-library` state.

### Requirement: Encounters, crossings, impulses, and sampling
**Reason**: `EncounterInteraction` is a recipe on `BilinearLaw` and leaves the core. The rest is restated as "Bilinear
encounters, crossings, impulses, and sampling".
**Migration**: Import `EncounterInteraction` from `skcomplex.systems.general.kinetics`.

### Requirement: General elements
**Reason**: `NonnegativeElement` and `PointElement` are recipes and leave the core, and `ReactionNetworkSystem` leaves
with them. The rest is restated as "Structural elements".
**Migration**: Import both elements from `skcomplex.systems.general.elements` and the reaction network from
`skcomplex.systems.general.reactions`.

### Requirement: Jumps, reactions, and transitions
**Reason**: `ReactionInteraction` is a recipe on `JumpInteraction` and leaves the core. The rest is restated as "Jumps
and transitions".
**Migration**: Import `ReactionInteraction` from `skcomplex.systems.general.reactions`. The reaction scenarios are in
`general-realizations`.
