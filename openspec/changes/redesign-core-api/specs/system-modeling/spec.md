## ADDED Requirements

### Requirement: States and parameters follow from the laws
A value of an element that some law changes SHALL be a state, and a value that no law changes SHALL be a parameter,
decided by `prepare` from the laws of the system.

#### Scenario: A particle at rest
- **WHEN** a system holds a particle with the values `x` and `m` and no interaction
- **THEN** preparing it finds no state, and `parameters_` lists `ball.x` and `ball.m`

#### Scenario: A particle under gravity
- **WHEN** the system adds a gravity interaction whose law is `RateLaw(derivative(x, 2), -g)`
- **THEN** `states_` lists `ball.x` and `ball.x'`, and `parameters_` lists `ball.m` and `gravity.g`

### Requirement: Derivatives and lags are mathematics in the laws
A law SHALL read and give derivatives of states with `derivative(x, k)` and past values with `lag(x, k)`, the order of
a state SHALL be the highest derivative or lag its laws use, and the compiler SHALL reduce every state to first order.
Derivatives SHALL be named with a prime in initial states and trajectories, such as `"ball.x'"`, and past values with
an index, such as `"economy.y[-1]"`, and they SHALL default to zero.

#### Scenario: A pushed particle
- **WHEN** a user simulates a particle under gravity with `initial_state={"ball.x'": 1.5}`
- **THEN** the run starts with the velocity 1.5, and `trajectory.get_values("ball.x'")` returns the velocity

#### Scenario: A difference equation with two lags
- **WHEN** an interaction returns `UpdateLaw(s.y, a * lag(s.y, 1) + b * lag(s.y, 2))`
- **THEN** the compiled map keeps two past values of `y`, and `states_` lists `economy.y`, `economy.y[-1]`, and
  `economy.y[-2]`

### Requirement: Prepared systems expose what they learned
After `prepare`, a system SHALL expose `states_`, the names of the values its laws change with their derivatives and
lags, `parameters_`, the names of the values no law changes, `equations_`, one readable equation per state, `form_`,
the kind of problem the laws make, and `solver_` and `backend_`, the resolved runners. The compiled problem SHALL be
private.

#### Scenario: A damped spring
- **WHEN** a user prepares a particle under gravity and a damped spring
- **THEN** `form_` is the ordinary differential equations, `equations_` maps `ball.x` to `ball.x'` and `ball.x'` to
  the sum of the forces over the mass, and `solver_` is the default ODE solver

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
Interactions SHALL describe every change of state as a list of laws of seven kinds, `RateLaw`, `NoiseLaw`, `JumpLaw`, `UpdateLaw`,
`ConstraintLaw`, `EventLaw`, and `ObjectiveLaw`, each a frozen value that targets states. Every kind SHALL accept a symbolic
expression or a Python callable for the values the engine evaluates numerically. Elements SHALL return no laws.

#### Scenario: Symbolic rate law
- **WHEN** an interaction returns `RateLaw(s.prey.n, -s.a * s.predator.n * s.prey.n)`
- **THEN** the compiled system includes that term in the time derivative of the prey count

#### Scenario: Callable jump change
- **WHEN** a jump law's change is a callable of the current values
- **THEN** each firing applies the change the callable returns

#### Scenario: Only interactions return laws
- **WHEN** a user writes an element class whose `_prepare` sets `laws_`
- **THEN** preparing a system with it raises a named error that asks for an interaction

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

### Requirement: Components are prepared against their context
Elements, interactions, spaces, neighbourhoods, and systems SHALL derive from `BaseComponent`, which gives `label`,
parameters, cloning, tags, and `prepare(context)`, and no `run`. An interaction SHALL implement `_prepare` against the
namespace of the elements its roles bind and set `laws_`, and the engine SHALL prepare a clone of it for every binding.
Laws SHALL be frozen values, and systems SHALL also have `simulate`.

#### Scenario: An interaction learns its laws
- **WHEN** a user prepares a gravity interaction against a namespace with the symbol `x` of its target
- **THEN** `laws_` holds `RateLaw(derivative(x, 2), -g)`, and the interaction has no `run`

#### Scenario: The user's interaction stays unchanged
- **WHEN** a user prepares a system that binds one interaction object to two groups
- **THEN** the engine prepares a clone per binding, and the user's interaction has no `laws_`

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
- **WHEN** an interaction returns `RateLaw(s.target.T, s.D * laplacian(s.target.T))` for an element placed on a mesh
- **THEN** the system compiles to the discretised heat equation of that mesh

### Requirement: The network changes by updates on its edges
A change of the structure of a network space SHALL be written as `UpdateLaw` laws on the edges of the space, returned by
an ordinary interaction.

#### Scenario: Rewiring
- **WHEN** an interaction returns updates that remove an edge between disagreeing nodes and add a random one
- **THEN** the network of the trajectory changes at every step, and no dedicated topology class is used

## MODIFIED Requirements

### Requirement: Elements declare typed state and parameters
An element SHALL be a plain class whose constructor stores its values unmodified, with no declaration of states or
parameters. The kind of a value SHALL come from its Python type hint or its value, and a constraint SHALL be written as
the allowed values in `Annotated`, only where one is needed.

#### Scenario: Element stores its arguments
- **WHEN** a user creates a population element with label `rabbits` and initial count 40
- **THEN** the element exposes `label == 'rabbits'` and `n == 40`

#### Scenario: Invalid value is rejected at compile time
- **WHEN** a system contains an element whose positive parameter is set to a negative number and it is prepared
- **THEN** a named validation error is raised that names the element, the variable, and the domain

#### Scenario: Velocity as a derivative
- **WHEN** a particle with the values `x` and `m` falls under a gravity interaction whose law gives `derivative(x, 2)`
- **THEN** a run moves `x` with the acceleration `-g`, and the trajectory holds `ball.x` and `ball.x'`

### Requirement: Components are defined without subclassing
`DeclaredElement(label, values={...})` SHALL define an element from data, and
`Interaction(label, roles={...}, laws=callable)` SHALL define an interaction from a function that returns its laws, and
both SHALL behave like subclassed components in members, cloning, and parameters.

#### Scenario: One-off growth system
- **WHEN** a user builds a declared element with a value `n` and an interaction with one role and a rate law, without
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
- **THEN** it writes an `Interaction` whose law is `RateLaw` with `(goal - x) / time` or `D laplacian(u)`, with no class
  of its own

### Requirement: States declare an order with initial derivatives
The order of a state SHALL be the highest derivative its laws give or read, and the compiler SHALL add the derivatives
of orders 1 to `k - 1` as states named with primes. Their initial values SHALL come from the initial state and default
to zero, and they SHALL appear in trajectories by name and be readable in laws with `derivative`.

#### Scenario: Initial angular velocity
- **WHEN** a pendulum interaction gives `derivative(theta, 2)` and a run starts with `initial_state={"bob.theta'": 0.5}`
  and `theta=0.1`
- **THEN** the run starts with angle 0.1 and angular velocity 0.5, and the trajectory returns both `bob.theta` and
  `bob.theta'`

#### Scenario: First order unchanged
- **WHEN** the laws of a state give only its first derivative
- **THEN** the system compiles with no derived states

### Requirement: Laws of any order superpose on the highest derivative
Rate and noise laws on a derivative of order `k` of a state SHALL contribute to it, and the contributions of different
laws SHALL add up. Laws that give different orders of one state SHALL raise a named error.

#### Scenario: Newton's second law from separate forces
- **WHEN** a mass on a spring receives a restoring law on `derivative(x, 2)` from one interaction and a damping law
  that reads `derivative(x)` from another
- **THEN** the run matches the damped oscillator `m x'' = -k x - c x'` within the solver tolerance

#### Scenario: Third order
- **WHEN** Sprott's jerk equation `x''' = -A x'' - x' + |x| - 1` is written as one rate law on `derivative(x, 3)`
- **THEN** it compiles to three first-order equations, and with A = 0.6 its trajectory is bounded and chaotic

#### Scenario: Order mismatch
- **WHEN** one law gives `derivative(x, 1)` and another gives `derivative(x, 2)`
- **THEN** compiling raises a named error that names both laws, the state, and both orders

## REMOVED Requirements

### Requirement: Laws describe how state changes
**Reason**: Replaced by "Interactions return laws of seven kinds", which names the kinds, makes interactions the only
source of laws, and lets every kind take callables.
**Migration**: Set `laws_` in `_prepare` to `RateLaw`, `NoiseLaw`, `JumpLaw`, `UpdateLaw`, `ConstraintLaw`, `EventLaw`, or `ObjectiveLaw`.

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
**Migration**: Prepare and simulate the system. Read the laws of a prepared interaction in `laws_`.

### Requirement: A system is elements, interactions, space, and time
**Reason**: Replaced by "A system is parts, a space, a solver, and a backend". Time follows from the law kinds, and the
solver and backend are settings of the system.
**Migration**: Drop `time=`, and pass `solver=` and `backend=` to `System`.

### Requirement: The core holds building blocks only
**Reason**: Replaced by "The core holds the building blocks of general systems". Field laws such as matchings,
cascades, and binary encounters move to their fields with their solvers.
**Migration**: Import such laws from their field package.
