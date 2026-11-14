## ADDED Requirements

### Requirement: Interactions own the parameters of their relation
A parameter that describes the relation between elements or between an element and its surroundings SHALL belong to
the interaction, and an element SHALL hold only the parameters that describe that element alone.

#### Scenario: Gravity on a particle
- **WHEN** a user models a particle falling under gravity
- **THEN** the gravitational acceleration is a parameter of the gravity interaction, and the particle holds its
  position, its velocity, and its mass

### Requirement: Systems expose the parameters of their parts by nested names
`System.get_params()` SHALL list the parameters of every element, interaction, space, and nested system under the name
`<label>__<parameter>`, nested systems adding their label in front, and `set_params` SHALL accept the same names, as a
scikit-learn `Pipeline` does.

#### Scenario: Tuning a part
- **WHEN** a user calls `system.set_params(gravity__g=1.62)` on a system with an interaction labelled `gravity`
- **THEN** the interaction's `g` is 1.62, and `system.get_params()['gravity__g']` returns 1.62

#### Scenario: An unknown nested name
- **WHEN** a user calls `system.set_params(gravity__mass=2.0)` and the interaction has no `mass`
- **THEN** a `ValidationError` names `gravity`, `mass`, and the parameters the interaction has

### Requirement: Interactions return laws of seven kinds
Interactions SHALL describe every change of state as a list of laws of seven kinds, `Rate`, `Noise`, `Jump`, `Update`,
`Constraint`, `Event`, and `Objective`, each a frozen value that targets states. Every kind SHALL accept a symbolic
expression or a Python callable for the values the engine evaluates numerically. Elements SHALL return no laws.

#### Scenario: Symbolic rate law
- **WHEN** an interaction returns `Rate(s.prey.n, -s.a * s.predator.n * s.prey.n)`
- **THEN** the compiled system includes that term in the time derivative of the prey count

#### Scenario: Callable jump change
- **WHEN** a jump law's change is a callable of the current values
- **THEN** each firing applies the change the callable returns

#### Scenario: Only interactions return laws
- **WHEN** a user writes an element class that defines `_get_laws`
- **THEN** preparing a system with it raises a named error that asks for an interaction with one role

### Requirement: Contributions of every additive kind add up
Rates, noises, jump channels, and constraint balances that several laws give to one state SHALL add up, so adding an
interaction adds a term without changing the others.

#### Scenario: Two forces on one particle
- **WHEN** a particle receives a rate on its velocity from a gravity interaction and another from a spring
- **THEN** the derivative of its velocity is the sum of both rates

### Requirement: One element class makes one member, many members, or one member per site
An element class SHALL create N members when given `size=N`, and one member per site when given `space=`, with
per-member values given as an array whose first dimension is the number of members or as one shared value. Creating
members SHALL reject at once a size below one, an unknown value name, and an array of the wrong length. `size` and
`space` SHALL be reserved names that no state or parameter takes.

#### Scenario: Members with per-member positions
- **WHEN** a user creates `Particle('gas', size=1000, x=positions)` with an array of 1000 two-dimensional positions
- **THEN** the element has 1000 members, and its state `x` has shape `(1000, 2)`

#### Scenario: Wrong number of values
- **WHEN** a user gives 999 positions to an element of size 1000
- **THEN** creating it raises a named error that names the element, the variable, and both sizes

#### Scenario: One member per site
- **WHEN** a user creates `Temperature('T', space=MeshSpace(...))`
- **THEN** the element has one member at every site of the mesh

#### Scenario: A reserved name
- **WHEN** an element class declares a state named `size`
- **THEN** defining the class raises a named error that names `size` and the reserved names

### Requirement: Parts have parameters and systems are prepared and simulated
Elements, interactions, spaces, and neighbourhoods SHALL derive from `BaseComponent`, which gives `label`, parameters,
cloning, and tags, and SHALL have no `prepare` and no `run`. Laws SHALL be frozen values. Systems SHALL have the main
public methods `prepare` and `simulate`.

#### Scenario: A part has parameters only
- **WHEN** a user looks for `prepare` or `run` on an element, an interaction, or a space
- **THEN** neither exists, and `get_params`, `set_params`, and `clone` work

#### Scenario: A system is prepared and simulated
- **WHEN** a user calls `system.prepare().simulate(until=10)`
- **THEN** the call returns a trajectory up to time 10

### Requirement: A system is parts, a space, a solver, and a backend
A system SHALL be defined by its label, its elements, its interactions, its nested systems, its space, its solver, and
its backend, and by nothing else. Its time SHALL be inferred from the kinds of its laws. Observers, observables, initial
states, and random states SHALL be arguments of `simulate`.

#### Scenario: A system with its settings
- **WHEN** a user builds `System(elements=[rabbits, foxes], interactions=[predation], solver=GillespieSolver())`
- **THEN** the system prepares and simulates with the Gillespie solver, and its parameters list `solver` and the nested
  parameters of its parts

#### Scenario: Other execution arguments are rejected
- **WHEN** a user passes `observers=` or `time=` to `System`
- **THEN** a `TypeError` is raised as for any unknown argument

### Requirement: The core holds the building blocks of general systems
Every component of the core SHALL be a building block that general systems of several fields are built from. A law
that one field needs SHALL live in that field with the solver that lowers it, built on the public engine
bases, and the engine core SHALL key only on the seven law kinds.

#### Scenario: A field law with its solver
- **WHEN** the criticality package simulates a sandpile with its cascade law
- **THEN** the law and its solver live in the criticality package, and no module of the core names them

### Requirement: Spatial operators are functions that laws call
Spatial operators, such as `laplacian`, `divergence`, and `gradient`, and the reductions over the neighbours of a member
SHALL be public functions that laws call on states, and SHALL NOT be interactions.

#### Scenario: Heat equation
- **WHEN** an interaction returns `Rate(s.target.T, s.D * laplacian(s.target.T))` for an element placed on a mesh
- **THEN** the system compiles to the discretised heat equation of that mesh

### Requirement: The network changes by updates on its edges
A change of the structure of a network space SHALL be written as `Update` laws on the edges of the space, returned by
an ordinary interaction.

#### Scenario: Rewiring
- **WHEN** an interaction returns updates that remove an edge between disagreeing nodes and add a random one
- **THEN** the network of the trajectory changes at every step, and no dedicated topology class is used

## MODIFIED Requirements

### Requirement: Elements declare typed state and parameters
An element SHALL be data: its states and its own parameters, each declared in its constructor with a `StateSpec` or a
`ParameterSpec`, or with the aliases `State` and `Parameter`, and stored unmodified. A state SHALL declare its domain,
its shape, and the state that is its rate of change, if any.

#### Scenario: Element stores its arguments
- **WHEN** a user creates a population element with label `rabbits` and initial count 40
- **THEN** the element exposes `label == 'rabbits'` and `n == 40`

#### Scenario: Invalid value is rejected at compile time
- **WHEN** a system contains an element whose positive parameter is set to a negative number and it is prepared
- **THEN** a named validation error is raised that names the element, the variable, and the domain

#### Scenario: Velocity declared with its position
- **WHEN** a particle declares `x` with `StateSpec(rate='v')` and a state `v`, and no interaction acts on it
- **THEN** a run moves `x` at the constant velocity `v`, and the trajectory holds both `particle.x` and `particle.v`

### Requirement: Components are defined without subclassing
`DeclaredElement(label, states={...}, parameters={...})` SHALL define an element from data, and
`Interaction(label, roles={...}, laws=callable)` SHALL define an interaction from a function that returns its laws, and
both SHALL behave like subclassed components in members, cloning, and parameters.

#### Scenario: One-off growth system
- **WHEN** a user builds a declared element with a state `n` and an interaction with one role and a rate law, without
  writing a class
- **THEN** the system prepares and simulates exactly like the same system written with classes

#### Scenario: Defined types form groups
- **WHEN** a user creates a declared element with `size=100`
- **THEN** it prepares once for its 100 members, and its parameters appear in the nested parameters of the system

### Requirement: Interactions bind roles with an arity and a binding
The base `Interaction` SHALL take its binding as arguments: named roles bound to elements or members, `among=` with
`within=` a neighbourhood for every pair of close members, or `among=` with `along=` for every edge or hyperedge of a
network. No interaction class SHALL exist only to fix a binding. Interactions SHALL hold no state.

#### Scenario: Neighbour coupling
- **WHEN** an interaction binds `among=` an element placed on a lattice with `within=LatticeNeighborhood()`
- **THEN** its laws run once per pair of adjacent sites

#### Scenario: Pairs within a cutoff
- **WHEN** an interaction is created with `among='gas'` and `within=RadiusNeighborhood(2.5)`
- **THEN** its laws run once per pair of members closer than 2.5

#### Scenario: Hyperedges need a network that has them
- **WHEN** an interaction binds along hyperedges of a lattice
- **THEN** preparing reports that the binding needs a network space with hyperedges

#### Scenario: State on an interaction is rejected
- **WHEN** an interaction type declares a `state`
- **THEN** defining the type raises a named error that suggests an element on the edges of the space

### Requirement: Core interactions are distinct operations
The interactions of the core SHALL be the base `Interaction` and the interactions the engine generates from one
function: `GradientInteraction` from a potential, `HamiltonianInteraction` from a Hamiltonian, and
`EulerLagrangeInteraction` from a Lagrangian. No core interaction SHALL only wrap one law or fix one binding.

#### Scenario: No parametrised copy
- **WHEN** the classes of `skcomplex.core.components.interactions` are listed
- **THEN** none is a subclass that only fixes the rate, the flux, the guard, or the map function of another core class,
  and none of `RateInteraction`, `JumpInteraction`, `MapInteraction`, `PairInteraction`, or `NeighborInteraction` is
  among them

#### Scenario: Specific form without a realization
- **WHEN** a mathematics system needs an Ornstein-Uhlenbeck control or a graph diffusion
- **THEN** it writes an `Interaction` whose law is `Rate` with `(goal - x) / time` or `D laplacian(u)`, with no class
  of its own

## REMOVED Requirements

### Requirement: Laws describe how state changes
**Reason**: Replaced by "Interactions return laws of seven kinds", which names the kinds, makes interactions the only
source of laws, and lets every kind take callables.
**Migration**: Return `Rate`, `Noise`, `Jump`, `Update`, `Constraint`, `Event`, or `Objective` from `get_laws`.

### Requirement: Rate and noise contributions add up
**Reason**: Replaced by "Contributions of every additive kind add up", which also covers jump channels and constraint
balances.
**Migration**: None needed for rates and noises.

### Requirement: Element types create groups
**Reason**: Replaced by "One element class makes one member, many members, or one member per site", with `size=` and
`space=` on the element class.
**Migration**: Replace `Particle.group('gas', 1000, ...)` by `Particle('gas', size=1000, ...)`.

### Requirement: Components share two main public methods
**Reason**: Replaced by "Parts have parameters and systems are prepared and simulated". Parts are descriptions, and
only systems and engine classes are prepared.
**Migration**: Prepare and simulate the system. Read the laws of an interaction through its public `get_laws`.

### Requirement: A system is elements, interactions, space, and time
**Reason**: Replaced by "A system is parts, a space, a solver, and a backend". Time follows from the law kinds, and the
solver and backend are settings of the system.
**Migration**: Drop `time=`, and pass `solver=` and `backend=` to `System`.

### Requirement: The core holds building blocks only
**Reason**: Replaced by "The core holds the building blocks of general systems". Field laws such as matchings,
cascades, and binary encounters move to their fields with their solvers.
**Migration**: Import such laws from their field package.

### Requirement: States declare an order with initial derivatives
**Reason**: A state declares the state that is its rate of change, so the velocity is a declared state and not a
hidden companion named by a convention.
**Migration**: Replace `theta: StateSpec(order=2)` and `theta_d1=...` by `theta: StateSpec(rate='omega')` and a state
`omega`.

### Requirement: Laws of any order superpose on the highest derivative
**Reason**: With explicit rates there is no highest derivative. Forces add as rates on the velocity state.
**Migration**: Write each force as `Rate(s.target.v, force / s.target.m)`, and higher orders as a chain of states.
