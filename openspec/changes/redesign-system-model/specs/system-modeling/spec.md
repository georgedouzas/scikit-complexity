# Spec Delta

## ADDED Requirements

### Requirement: A system is components, interactions, space, and time
A system SHALL be defined by its label, its components, its interactions, its space, and its time, and by nothing
else. A system SHALL accept no argument about how it runs: no solver, backend, form, noise convention, delay history,
activation schedule outside its time, or observables.

#### Scenario: Four-part system
- **WHEN** a user builds `System('ecology', components=[rabbits, foxes], interactions=[predation],
  space=None, time=ContinuousTime())`
- **THEN** the system compiles and runs, and its parameters list only the parameters of its components and interactions

#### Scenario: Execution arguments are rejected
- **WHEN** a user passes `noise=`, `history=`, `phases=`, or `observables=` to `System`
- **THEN** a `TypeError` is raised, and its message names the argument's new home

### Requirement: Systems are components
A system SHALL be a component, so systems nest to any depth, and an interaction SHALL refer to a component of a nested
system by a slash path relative to the system that holds the interaction. Two direct children of a system with the same
label SHALL be reported by validation.

#### Scenario: Interaction across nested systems
- **WHEN** a parent system holds a predation interaction between `forest/foxes` and `meadow/rabbits`
- **THEN** the compiled system couples the fox count of the forest system to the rabbit count of the meadow system

#### Scenario: Duplicate labels
- **WHEN** two direct children of a system share a label
- **THEN** validation reports one issue whose path is the system and whose message names the duplicated label

### Requirement: Groups have a fixed or variable size and live on parts of the space
A group SHALL hold identical members of one element type as arrays. With `variable=True`, laws SHALL add and remove
members during a run. With `on='sites'`, `on='edges'`, or `on='hyperedges'`, the group SHALL hold one member per site,
edge, or hyperedge of the space, and a variable group on edges SHALL change the edges of the space when it changes size.

#### Scenario: Births and deaths
- **WHEN** a variable group of 10 members has a rule that duplicates a member with probability one each step
- **THEN** the group has 20, 40, and 80 members after one, two, and three steps

#### Scenario: State on edges
- **WHEN** a group of weights lives on the edges of a network with 30 edges
- **THEN** it has 30 members, and an interaction over neighbours reads the weight of the edge joining each pair

#### Scenario: Rewiring
- **WHEN** a law removes an edge member and adds another between two unconnected nodes
- **THEN** the network space has the new edge and not the removed one in the next frame

### Requirement: State domains cover structured values
State domains SHALL include real, integer, Boolean, categorical, angle, and complex scalars, vectors of the space's
dimension, tensors of a declared rank and symmetry, Hilbert spaces of a declared dimension, sequences over an alphabet,
graphs, and tilings, and the domain of a state SHALL decide which laws may target it.

#### Scenario: Hilbert-space element
- **WHEN** an element declares a state on a Hilbert space of dimension 2 and a generator law
- **THEN** the system compiles to unitary dynamics without any quantum-specific element class in the core

#### Scenario: Sequence element
- **WHEN** an element declares a state that is a sequence over the alphabet `{A, B}` with a rewrite law
- **THEN** each step rewrites the sequence in parallel

### Requirement: Interactions bind roles with an arity and a binding
An interaction SHALL declare roles, each binding one component, one member, or a sequence of them. Its binding SHALL
be one of `'members'`, `'pairs'`, `'neighbours'`, or `'hyperedges'`, so one interaction type covers unary, pairwise,
neighbourhood, and higher-order couplings. Interactions SHALL hold no state.

#### Scenario: Neighbour coupling
- **WHEN** an interaction with binding `'neighbours'` couples the members of a group on a lattice
- **THEN** its laws run once per pair of adjacent sites

#### Scenario: Higher-order coupling
- **WHEN** an interaction with binding `'hyperedges'` couples a group on a network with triangles as hyperedges
- **THEN** its laws run once per triangle with the three members bound to its roles

#### Scenario: State on an interaction is rejected
- **WHEN** an interaction type declares a `state`
- **THEN** defining the type raises a named error that suggests a group on the edges of the space

### Requirement: Space is a substrate with boundary conditions
A space SHALL be none (well-mixed), a lattice, a network with optional hyperedges, a continuous space of a dimension, or
a mesh, and boundary conditions SHALL be part of the space. The same space kinds SHALL serve every time class, including
agents.

#### Scenario: Periodic continuous space
- **WHEN** a particle crosses the boundary of a continuous space declared `boundary='periodic'`
- **THEN** its position re-enters on the opposite side, without any box component

#### Scenario: Agents and equations share a space
- **WHEN** a variable group with rules and a field with rates live on the same lattice
- **THEN** both read the same sites and neighbourhoods

### Requirement: Time is continuous, discrete with a schedule, or absent
A system's time SHALL be continuous, discrete with a schedule, or absent. The schedules SHALL be synchronous, random
sequential, or ordered phases, each with its rules and activation policy. Without a declared time, it SHALL be inferred
from the law classes, and an ambiguous inference SHALL be reported by validation. A system without time SHALL hold only
constraint laws and SHALL be solved once.

#### Scenario: Inferred continuous time
- **WHEN** a system holds only rate laws and declares no time
- **THEN** its time is continuous

#### Scenario: Ordered phases
- **WHEN** a discrete-time system declares phases `produce`, then `trade`, then `consume`
- **THEN** every step runs the three phases in that order, each with its own rules

#### Scenario: Static system
- **WHEN** a system with no time holds only algebraic equations
- **THEN** a run returns one frame with the solution

### Requirement: Laws are classified by their time semantics
Every law SHALL belong to one class, continuous, event, discrete, or constraint, and its class SHALL decide the time
under which it is valid, as the design's table of law classes lists. A jump SHALL take an exponential clock by default
and any waiting-time distribution as its clock.

#### Scenario: Renewal as a jump
- **WHEN** a jump law is given a Weibull waiting-time clock
- **THEN** the run reproduces the renewal process the former `Renewal` law produced, within the sampling error

#### Scenario: Noise convention on the law
- **WHEN** a noise law is declared with `convention='stratonovich'`
- **THEN** the compiler adds the drift correction, as the former system argument did

### Requirement: Transfers move quantities between states
A transfer law SHALL move an amount from a source state to a sink state. It SHALL act as a pair of opposite rates in
continuous time, as an atomic update in discrete time, and as a jump increment in event time. A source or sink outside
the system SHALL be declared with `None`.

#### Scenario: Conservation across time classes
- **WHEN** the same two-stock system with a transfer runs once in continuous time and once in discrete time
- **THEN** the sum of the two stocks is constant within 1e-12 in both runs

#### Scenario: Transfer between agents
- **WHEN** a rule transfers 1 unit of wealth from one agent to another
- **THEN** both changes appear in the same frame, and the total wealth is unchanged

### Requirement: Invariants declare conserved quantities
An invariant SHALL declare an expression that stays constant within a tolerance. It SHALL be checked at every frame
with a named error that names the invariant and the frame. When declared enforced, it SHALL be restored by projection
after each step.

#### Scenario: Violated invariant
- **WHEN** a law creates wealth from nothing in a system that declares total wealth invariant
- **THEN** the run raises a named error that names the invariant and the first frame that breaks it

### Requirement: Matching laws pair members of groups
A matching law SHALL pair the members of one group or of two groups each step, with an optional capacity per member and
a pairing function from the two groups to pairs, and the core SHALL provide only uniform random pairing. The pairs SHALL
be readable by the laws of the same step.

#### Scenario: Random pairing
- **WHEN** two groups of 50 and 30 members are matched randomly with capacity one
- **THEN** each step has 30 pairs, and no member appears in two pairs

#### Scenario: Custom pairing
- **WHEN** a library pairing function implements stable matching and is given to a matching law
- **THEN** the core runs it without any matching algorithm of its own

### Requirement: Adaptive parameters change under laws
A parameter declared `adaptive=True` SHALL be changeable by laws of its own component or of interactions bound to it.
It SHALL be recorded in trajectories like a state, and it SHALL remain settable as a parameter before a run.

#### Scenario: Policy rule
- **WHEN** an element raises an adaptive rate parameter of another component whenever an expression exceeds a target
- **THEN** the parameter changes in the trajectory only at steps where the expression exceeds the target

### Requirement: The core is application agnostic
Every public name and module of `skcomplex.core` SHALL be a mathematical, modelling, or engineering term. The core SHALL
hold each primitive in its general form with at most one neutral default. Every specific realization SHALL live in
`skcomplex.models`, and the core SHALL import nothing from `skcomplex.models`. A specific realization is a named
potential, a pairing algorithm, a rule code, a quantum species, a learning heuristic, a tiling, or a distribution beyond
the default.

#### Scenario: Forbidden field terms
- **WHEN** the test suite checks the public names and modules of `skcomplex.core` against the reviewed list of field
  terms
- **THEN** none of them contains a field term

#### Scenario: No import of models
- **WHEN** the import graph of `skcomplex.core` is computed
- **THEN** it contains no module of `skcomplex.models`

## REMOVED Requirements

### Requirement: Systems nest and compose
**Reason**: Replaced by "Systems are components", which makes a system one kind of component.
**Migration**: Pass child systems in `components=` with the elements and groups.

### Requirement: Spaces give vector state its dimension
**Reason**: Replaced by "Space is a substrate with boundary conditions" and the vector domain in "State domains cover
structured values".
**Migration**: Pass the space to `System(space=...)`. A vector state without a space is now reported by validation.

### Requirement: Systems declare observables
**Reason**: Observables are measurements chosen by the user of a system, not part of its definition.
**Migration**: Pass observables to `Simulation(observables=...)` or to the catalog entry of the model.

### Requirement: Pair interactions sum over member pairs
**Reason**: Replaced by the `'pairs'` and `'neighbours'` bindings of "Interactions bind roles with an arity and a
binding".
**Migration**: Replace `PairInteraction` with `Interaction` and the binding `over='pairs'`.
