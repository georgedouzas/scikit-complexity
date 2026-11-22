# system-modeling Specification

## Purpose

Lets a user describe a system from any science as elements that carry state, interactions that change that state
through laws, a space the state lives in, and systems that nest and compose.

## Requirements

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
- **WHEN** a particle with the values `x` and `m` falls under a gravity interaction whose law gives `x.diff(t, 2)`
- **THEN** a run moves `x` with the acceleration `-g`, and the trajectory holds `ball.x` and `ball.x'`

### Requirement: Interactions bind elements through roles
An interaction type SHALL declare named roles and parameters, and an interaction instance SHALL bind each role to an
element label of its system, or a variadic role to a sequence of element paths and group member references.

#### Scenario: Role refers to an unknown element
- **WHEN** a system contains an interaction whose role names a label that no element in reach has
- **THEN** compiling the system raises a named error that names the interaction, the role, and the label

#### Scenario: Variadic entry refers to a missing member
- **WHEN** a variadic role holds the member reference `nodes[12]` and the group `nodes` has 10 members
- **THEN** compiling the system raises a named role error that names the interaction, the role, the entry, and the
  group size

### Requirement: Roles bind to groups
An interaction role SHALL bind to a group, to a single element, or to a set of single elements of one type, and the
interaction's laws SHALL then apply to every member of the group or every element of the set. Roles of one interaction
SHALL bind several groups, and the groups that its laws read outside reductions SHALL have one size and be read member
by member.

#### Scenario: Wall law on a whole gas
- **WHEN** a reflecting wall interaction binds its role to a group of 1000 molecules
- **THEN** every molecule is reflected at the walls

#### Scenario: Role bound to an element set
- **WHEN** an interaction binds its role to the three single elements `unit_0`, `unit_1`, and `unit_2` of one type and
  reads `total(x)` over the role
- **THEN** its law applies to each of the three elements, the total equals the sum of their three states, and the
  elements keep their own state names

#### Scenario: Element set of mixed types
- **WHEN** a role binds to a set of elements of two different types
- **THEN** compiling raises a named error that names the role and both types

#### Scenario: Two groups member by member
- **WHEN** an interaction adds `s.source.n` to the rate of `s.sink.n`, with both roles bound to groups of 5 members
- **THEN** the rate of member `i` of the sink equals the count of member `i` of the source for every `i`

#### Scenario: Groups of different sizes
- **WHEN** an interaction with symbolic laws binds its roles to groups of 5 and 7 members and reads both outside
  reductions
- **THEN** compiling the system raises a named role error that names both groups and both sizes

### Requirement: Boundary laws constrain states after each step
An interaction SHALL be able to return boundary laws, which map states to new values after every internal solver step,
however long the frame, combine with any law kind without changing the inferred form, and describe walls and periodic
boxes. A frame SHALL never report a state that a boundary law would change.

#### Scenario: Reflecting wall in an ODE system
- **WHEN** a particle group with rate laws also has reflecting wall boundaries
- **THEN** the system compiles to the ODE form and no member crosses a wall

#### Scenario: Periodic box
- **WHEN** a member leaves a periodic box through one side
- **THEN** it reappears at the opposite side with the same velocity

#### Scenario: Long frames keep particles in the box
- **WHEN** a particle at 0.5 with velocity 1 in a reflecting box of side 1 runs one frame of duration 5 with `RK4`
  and with `ScipyODESolver`
- **THEN** at time 5 its position is 0.5 and its velocity is -1 within 1e-6, after four reflections

### Requirement: Laws express conditions and remainders
The modelling layer SHALL provide symbolic functions for a value chosen by a condition, for the conjunction and
disjunction of conditions, and for the remainder of a division, which laws use like any other expression and which
every backend lowers. A condition with equal branches SHALL simplify to that branch, and evaluating a condition SHALL
raise no warning from the branch it does not choose.

#### Scenario: Reflection by condition
- **WHEN** a boundary sets a velocity to its negative where a position lies outside a box and keeps it elsewhere
- **THEN** a particle that crosses a wall moves back into the box with its velocity reversed

#### Scenario: Periodic remainder
- **WHEN** a boundary sets a position to its remainder by the box size
- **THEN** a particle that leaves through one side enters through the opposite side

#### Scenario: Conditions in LAMMPS
- **WHEN** a pair energy uses a condition on the distance and the LAMMPS backend runs it
- **THEN** the energies match those of the NumPy backend within a relative tolerance of 1e-10

#### Scenario: Derivative of a condition
- **WHEN** a user differentiates `where(x < 0, 1, 2) * y` twice with respect to `y`
- **THEN** the result is exactly zero

#### Scenario: Branch that is not chosen
- **WHEN** the NumPy backend evaluates `where(x > 0, 1 / x, 0)` at `x = 0` and `x = 2` with warnings turned into errors
- **THEN** it returns 0 and 0.5 and raises nothing

### Requirement: Components are defined without subclassing
`Element(label, values={...})` SHALL define an element from data, and
`Interaction(label, roles={...}, laws=callable)` SHALL define an interaction from a function that returns its laws, and
both SHALL behave like subclassed components in members, cloning, and parameters.

#### Scenario: One-off growth system
- **WHEN** a user builds an element from data with a value `n` and an interaction with one role and a rate law, without
  writing a class
- **THEN** the system prepares and simulates exactly like the same system written with classes

#### Scenario: Defined types form groups
- **WHEN** a user creates an element from data with `size=100`
- **THEN** it prepares once for its 100 members, and its parameters appear in the nested parameters of the system

### Requirement: States take discrete and periodic domains
States SHALL accept boolean, finite-alphabet, and angle domains besides the real and integer domains, and angle states
SHALL stay within one turn after every step.

#### Scenario: Categorical state
- **WHEN** an element declares a state over the alphabet `susceptible`, `infected`, `recovered`
- **THEN** trajectories return the state's values as the alphabet's labels, and a law that produces a value outside the
  alphabet raises a named error in checked mode

#### Scenario: Angle wraps
- **WHEN** an angle state grows past 2π during a step
- **THEN** its value after the step lies in `[0, 2π)`

### Requirement: States declare an order with initial derivatives
The order of a state SHALL be the highest derivative its laws give or read, and the compiler SHALL add the derivatives
of orders 1 to `k - 1` as states named with primes. Their initial values SHALL come from the initial state and default
to zero, and they SHALL appear in trajectories by name and be readable in laws with `derivative`.

#### Scenario: Initial angular velocity
- **WHEN** a pendulum interaction gives `theta.diff(t, 2)` and a run starts with `initial_state={"bob.theta'": 0.5}`
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
- **WHEN** a mass on a spring receives a restoring law on `x.diff(t, 2)` from one interaction and a damping law
  that reads `x.diff(t)` from another
- **THEN** the run matches the damped oscillator `m x'' = -k x - c x'` within the solver tolerance

#### Scenario: Third order
- **WHEN** Sprott's jerk equation `x''' = -A x'' - x' + |x| - 1` is written as one rate law on `x.diff(t, 3)`
- **THEN** it compiles to three first-order equations, and with A = 0.6 its trajectory is bounded and chaotic

#### Scenario: Order mismatch
- **WHEN** one law gives `x.diff(t, 1)` and another gives `x.diff(t, 2)`
- **THEN** compiling raises a named error that names both laws, the state, and both orders

### Requirement: Equations state implicit relations in the highest derivatives
An equation law SHALL relate the highest derivatives of states to other expressions, which MAY contain conditions and
drivers. The compiler SHALL solve all such equations of a system jointly for the highest derivatives, and SHALL raise
a named error when they cannot be solved.

#### Scenario: Pendulum as written
- **WHEN** a pendulum states `m l derivative(θ, 2) == -m g sin θ`
- **THEN** the run equals the one from the rate law `derivative(θ, 2) = -(g / l) sin θ`

#### Scenario: Coupled equations
- **WHEN** two equations each contain the second derivatives of both angles of a double pendulum
- **THEN** the compiler solves them jointly, and the energy stays constant within the solver tolerance

#### Scenario: Unsolvable highest derivative
- **WHEN** an equation contains the square of a highest derivative
- **THEN** compiling raises a named error that names the equation and the derivative

#### Scenario: Driver on the right-hand side
- **WHEN** a mass on a spring states `m derivative(x, 2) == -k x + step(t, 1)`, and another states
  `m derivative(x, 2) == where(x < 0, -k x, -2 k x)`
- **THEN** both compile, and each run equals the run of the matching rate law within the solver tolerance

### Requirement: Difference equations take any number of lags
A state of a discrete map SHALL declare how many past values it keeps. Update laws SHALL read them as lags, and the
initial values of the lags SHALL be given in the constructor.

#### Scenario: Multiplier-accelerator
- **WHEN** Samuelson's system `Y_t = c (1 + v) Y_{t-1} - c v Y_{t-2} + G` runs with c = 0.8 and v = 1.5
- **THEN** income oscillates with growing amplitude, as the complex characteristic roots of modulus `sqrt(c v) > 1`
  predict

#### Scenario: Fibonacci recurrence
- **WHEN** `x_t = x_{t-1} + x_{t-2}` runs from 1 and 1
- **THEN** the trajectory is the Fibonacci sequence

### Requirement: Variadic roles bind sequences of elements
A role declared variadic SHALL bind any number of element paths or group member references in order, laws SHALL read
it as a sequence of namespaces of that length, and a parameter declared per entry of the role SHALL hold one value per
entry, each addressed by path and index. A variadic role SHALL give one law the whole sequence, where an element set
bound to an ordinary role applies the laws once per element.

#### Scenario: Sum over a variable number of inputs
- **WHEN** an interaction whose variadic role binds the elements `a`, `b`, and `c` updates `target.x` to the sum of
  `x` over its inputs
- **THEN** one step sets `target.x` to `a.x + b.x + c.x`

#### Scenario: Member reference
- **WHEN** a variadic role binds the entries `hub` and `nodes[3]`
- **THEN** the second namespace reads the states of member 3 of the group `nodes`

#### Scenario: Per-entry parameter with the wrong length
- **WHEN** a variadic role binds 3 entries and its per-entry weights hold 2 values
- **THEN** compiling the system raises a named validation error that names the interaction, the parameter, and both
  lengths

#### Scenario: Live per-entry parameter
- **WHEN** a user updates the parameter `regulation.weights[1]` of a running simulation
- **THEN** the next step uses the new weight for the second entry only

### Requirement: Laws read group members at an offset
The modelling layer SHALL provide a symbolic function that reads an expression of a group's member variables at the
member a given number of places along the group order, with periodic, reflecting, or fixed ends, and SHALL compile it
into the group's kernel on every backend.

#### Scenario: Ring of members
- **WHEN** a law of a group of 4 members reads `shift_member(s.cars.x, 1)` with periodic ends
- **THEN** member 3 reads the value of member 0, and every other member `i` reads member `i + 1`

#### Scenario: Fixed end
- **WHEN** a law of a group of 4 members reads `shift_member(s.cars.x, 1, boundary='fixed', fill=-1.0)`
- **THEN** member 3 reads -1

#### Scenario: Same result on both backends
- **WHEN** a ring of 1000 members with offset reads runs on the NumPy and the Numba backends
- **THEN** the trajectories agree within a relative tolerance of 1e-10

#### Scenario: Offset of a single element
- **WHEN** a law applies `shift_member` to a state of a single element
- **THEN** compiling the system raises a named validation error that names the component and the state

### Requirement: Reserved variable names are rejected
Defining an element or agent type SHALL reject a state or parameter whose name is reserved by the component or by the
views rules read it through, with a named label error raised when the type is created.

#### Scenario: State named visual
- **WHEN** a user defines an element type with a state named `visual`
- **THEN** a named label error is raised that names the state and lists the reserved names

#### Scenario: Agent state named location
- **WHEN** a user defines an agent type with a state named `location` or `alive`
- **THEN** a named label error is raised that names the state and lists the reserved names

### Requirement: Jump increments are numbers
A jump law SHALL take a number as the increment of every state it changes, and SHALL raise a named validation error
that names the state and the increment when an increment holds a symbol.

#### Scenario: Parameter as increment
- **WHEN** a jump of an element states the increment of its count `n` as its parameter `k`
- **THEN** compiling raises a named validation error that names `n` and `k` and asks for a number, not a `TypeError`

### Requirement: Observables pickle
An observable SHALL raise a named validation error when it is created with an expression or a reduction that cannot
be pickled, naming the observable and asking for a module-level function or a symbolic expression. Every system built
from valid observables SHALL pickle.

#### Scenario: Lambda observable
- **WHEN** a user creates an observable `total` whose expression is a lambda
- **THEN** a named validation error is raised at once that names `total`

#### Scenario: Module-level function
- **WHEN** a system declares an observable whose expression is a module-level function
- **THEN** the system pickles and the restored system records the same observable values

### Requirement: One-member groups take per-member values
A group of one member SHALL read a value or a location given as a one-entry sequence as the value or location of its
member, as a larger group reads one entry per member.

#### Scenario: One agent at a given cell
- **WHEN** a user creates a group of one agent on a 5 by 5 grid with `location=[(1, 1)]` and `n=[3.0]`
- **THEN** the run starts with the agent at cell `(1, 1)` and with `n` equal to the number 3.0

### Requirement: Reductions appear anywhere in interaction laws
An interaction law SHALL accept a reduction in any position of its value, including a denominator or the argument of
a function undefined at zero, and SHALL compile to the same values as the equivalent element law.

#### Scenario: Share of a group total
- **WHEN** an interaction bound to a group of 4 members with `x = 1, 2, 3, 4` declares
  `RateLaw(s.cells.x, s.cells.x / count(over=s.cells))`
- **THEN** the system compiles, and the rate of member `i` is `x_i / 4`

#### Scenario: Weighted reduction in a denominator
- **WHEN** an interaction bound to a group declares `RateLaw(s.cells.x, s.cells.x / total(s.cells.x, weights=W))` with a
  matrix `W` of positive entries
- **THEN** the system compiles on the NumPy and the Numba backends, and both drifts equal `x_i / sum_j W_ij x_j`

### Requirement: Observable member expressions read reductions
The member expression of a group observable SHALL accept reductions over its group, with or without weights, with the
meaning they have in laws. A reduction over another group SHALL raise the named error that laws raise.

#### Scenario: Local field energy
- **WHEN** a group of 4 members declares the observable `energy` as the sum over members of `x_i * total(x, weights=W)`
- **THEN** the trajectory returns `x · W x` at every frame

#### Scenario: Plain reduction
- **WHEN** a group observable's member expression is `m.x - mean(m.x)` with reduction `'var'`
- **THEN** the trajectory returns the variance of `x` at every frame, with no error about a missing NumPy function

### Requirement: Plain roles bind group members
A plain role SHALL bind a member reference `group[i]`, and the interaction's laws SHALL read and change the states of
that member as those of a single element. A reference past the group's size SHALL raise a named role error that names
the interaction, the role, the reference, and the group size.

#### Scenario: Hub reads one site
- **WHEN** an interaction binds its plain role `site` to `sites[3]` of a group with `x = 0, 1, 2, 3, 4` and declares
  `RateLaw(s.hub.y, s.site.x)`
- **THEN** the rate of `hub.y` is 3, the same as with a variadic role bound to `['sites[3]']`

#### Scenario: Member past the end
- **WHEN** a plain role binds `sites[7]` of a group of 5 members
- **THEN** compiling raises a named role error that names the interaction, the role, `sites[7]`, and the size 5

### Requirement: Jumps change members at an offset
A jump of a group law SHALL target a member at an offset, written with the offset read and its periodic, reflecting,
or fixed ends, and SHALL compile to one channel per member. Past a fixed end the jump SHALL change only its other
targets, so the quantity leaves the group.

#### Scenario: Conserved hop on a ring
- **WHEN** a group of 50 sites on a ring declares `Jump(s.n, {s.n: -1, shift_member(s.n, 1): 1})` with 150 particles
- **THEN** every run keeps 150 particles, and each event moves one particle from a site to the next one

#### Scenario: Open end
- **WHEN** a group of 5 sites with fixed ends declares the same hop and holds one particle at the last site
- **THEN** the hop of the last site removes the particle and changes no other site

#### Scenario: Same law as per-link interactions
- **WHEN** a zero-range process of 20 sites is written once with an offset jump and once with one interaction per
  link
- **THEN** the mean occupation profiles of 200 runs agree within their statistical error, on NumPy and on Numba

#### Scenario: Many sites
- **WHEN** a zero-range process of 1000 sites with an offset jump is compiled and run until 5
- **THEN** it compiles one jump kernel, and compiling and running take less than 10 seconds on the Numba backend

### Requirement: Compile time grows linearly with the components
Compiling a system SHALL take time that grows linearly with its number of elements, interactions, and variables.

#### Scenario: Many one-link interactions
- **WHEN** zero-range processes written with one interaction per link are compiled with 400 and with 800 links on the
  NumPy backend
- **THEN** the larger one takes at most 2.5 times as long as the smaller one

### Requirement: Angles wrap into one turn centred on zero
A state of the `ANGLE` domain SHALL hold values in `(-pi, pi]` after every step of every form, and encoding a value in
the `ANGLE` domain SHALL wrap it into the same range.

#### Scenario: Small swing around zero
- **WHEN** a rotor with the angle state `phi' = -1` starts at 0.1 and runs until 1
- **THEN** every recorded angle lies in `(-pi, pi]`, and the last one equals -0.9 within 1e-6

#### Scenario: Encoding an angle
- **WHEN** `Domain.ANGLE.encode` receives `[3 pi / 2, -pi, pi]`
- **THEN** it returns `[-pi / 2, pi, pi]`

### Requirement: Systems replicate into vectorised ensembles
`replicate(system, size, values)` SHALL return a system whose elements are groups of `size` members of the same
element types and labels, with per-member values by dotted path, and SHALL raise a named error for a system with
interactions, subsystems, groups, fields, or agents.

#### Scenario: Many initial conditions in one run
- **WHEN** the logistic map at `r = 3.7` is replicated over 100 initial values and run for 50 iterations
- **THEN** member `i` of `map.x` equals the run of the single map from the `i`-th initial value within 1e-12

#### Scenario: A system with interactions
- **WHEN** a system with an interaction is replicated
- **THEN** a named error says that only systems of plain elements replicate

### Requirement: Groups have a fixed or variable size and live on parts of the space
A group SHALL hold identical members of one element type as arrays. With `variable=True`, laws SHALL add and remove
members during a run. With `on='sites'`, `on='edges'`, or `on='hyperedges'`, the group SHALL hold one member per site,
edge, or hyperedge of the space, and validation SHALL report a group on edges or hyperedges outside a network space.
Interactions that read edge states and laws that change the topology are specified by `add-adaptive-networks`.

#### Scenario: Births and deaths
- **WHEN** a variable group of 10 members has a rule that duplicates a member with probability one each step
- **THEN** the group has 20, 40, and 80 members after one, two, and three steps

#### Scenario: State on edges
- **WHEN** a group of weights lives on the edges of a network with 30 edges
- **THEN** it has 30 members, one per edge in the order of the graph's edges

#### Scenario: Edges outside a network
- **WHEN** a group on edges lives in a lattice space
- **THEN** validation reports that groups on edges need a network space

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
constraint laws; solving it is specified by `add-static-models`.

#### Scenario: Inferred continuous time
- **WHEN** a system holds only rate laws and declares no time
- **THEN** its time is continuous

#### Scenario: Ordered phases
- **WHEN** a discrete-time system declares phases `produce`, then `trade`, then `consume`
- **THEN** every step runs the three phases in that order, each with its own rules

#### Scenario: Static system
- **WHEN** a system declared with static time holds a rate law
- **THEN** validation reports the law, since a static system holds only constraint laws

### Requirement: Laws are classified by their time semantics
Every law SHALL derive from one law base and SHALL declare as class-level facts its kind, its class among continuous,
event, discrete, and constraint, and the traits the compiler and validation read. The class SHALL decide the time under
which the law is valid, as the design's table of law classes lists. A jump SHALL take an exponential clock by default
and any waiting-time distribution as its clock, and a jump SHALL reject at construction any option that does not belong
to the clock it uses.

#### Scenario: Renewal as a jump
- **WHEN** a jump law is given a Weibull waiting-time clock
- **THEN** the run reproduces the renewal process the former `Renewal` law produced, within the sampling error

#### Scenario: Noise convention on the law
- **WHEN** a noise law is declared with `convention='stratonovich'`
- **THEN** the compiler adds the drift correction, as the former system argument did

#### Scenario: Every law shares the base
- **WHEN** the test suite lists every registered law class
- **THEN** each one derives from the law base and declares a kind and a class

#### Scenario: Option of the other clock
- **WHEN** a user creates a jump with a waiting-time clock and a dictionary of changes
- **THEN** construction raises a named error that says changes belong to the exponential clock and resets to the
  waiting-time clock

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

### Requirement: Core packages mirror the system definition
The modelling packages of `skcomplex.core.components` SHALL be `systems`, `elements`, `interactions`, `spaces`, which
holds space and time, and `laws`, each exporting the general components of one kind of building block of a system. A
recipe that serves some fields only SHALL live under `skcomplex.systems`, never in a modelling package.

#### Scenario: Components by kind
- **WHEN** a user imports `CompositeSystem`, `Element`, `Interaction`, `LatticeSpace`, `DiscreteTime`, and
  `ArrayUpdateLaw`
- **THEN** they come from `core.components.systems`, `core.components.elements`, `core.components.interactions`,
  `core.components.spaces`, `core.components.spaces`, and `core.components.laws`

### Requirement: Backends live by their reach
Every backend SHALL live in `skcomplex.core.engine.backends`, an external simulator SHALL be a solver of
`skcomplex.core.engine.solvers`, and `compile` and `solvers` SHALL hold no backend.

#### Scenario: Backends in one package
- **WHEN** the test suite lists the backends and the external solvers
- **THEN** `NumpyBackend`, `NumbaBackend`, and `JaxBackend` are defined in `skcomplex.core.engine.backends`, and
  `MesaSolver` and `LammpsSolver` in `skcomplex.core.engine.solvers`

### Requirement: Noise laws name the Wiener process they share
A noise law SHALL accept an optional process. A law with no process SHALL drive every component and member with its
own independent Wiener process. Laws of one component, or of one member, that name the same local process SHALL share
one increment, and so SHALL the components of a vector target that names one process.

#### Scenario: One process drives two states
- **WHEN** an element drives its states `x` and `y` with unit noise on the same local process and runs from equal values
- **THEN** `x` and `y` have sample correlation 1 over the run, and `x - y` stays at 0 at every frame

#### Scenario: No process named
- **WHEN** a system whose noise laws name no process is compiled
- **THEN** every component and member gets its own independent Wiener process, as before this change

### Requirement: Wiener elements declare correlated processes
A Wiener element SHALL declare a number of Wiener processes and an optional constant covariance matrix, the identity
by default. Noise laws of any component SHALL name its processes through a role, and group laws that name one SHALL
share it across all members. Wiener elements SHALL be components of `skcomplex.core.components`, and their
validation is specified by `system-validation`.

#### Scenario: Covariance reproduced
- **WHEN** two states with zero drift take unit noise from the two processes of a Wiener element with covariance
  `[[1, 0.6], [0.6, 2]]`, over a run of 10 000 unit frames
- **THEN** the sample covariance of their frame increments matches the matrix within 0.1 in every entry

#### Scenario: Perfect correlation
- **WHEN** a Wiener element has covariance `[[1, 1], [1, 1]]` and its two processes drive two states with unit noise
- **THEN** the system compiles, and the two states stay equal at every frame

### Requirement: Components store declared values without generated attributes
A component type SHALL take its label and the values of its declared variables as constructor arguments stored
unmodified, and the estimator contract SHALL list and clone them without generated class attributes. A type with a
state of order `k` or with lags SHALL accept the initial values of the derivatives and of the lags under their derived
names.

#### Scenario: No generated attributes
- **WHEN** a pendulum type declares its angle `theta` with order 2
- **THEN** the class has no attribute `theta_d1`, and `Pendulum('bob', theta=0.1, theta_d1=0.5)` stores both values
  and lists both in `get_params`

#### Scenario: Clone keeps declared values
- **WHEN** a user clones a group of 100 members with per-member values
- **THEN** the clone has the same label, size, and values, and compiles to the same problem

### Requirement: Noise processes are not elements
`WienerProcessSpec` SHALL derive from no component base, and an interaction SHALL bind it through a role that takes
noise processes.

#### Scenario: Correlated noise
- **WHEN** an interaction binds a `WienerProcessSpec` with covariance `C` and two noise laws draw its processes
- **THEN** the increments over a step `h` have the covariance `h C`, and `WienerProcessSpec` is not a subclass of
  `BaseComponent`

### Requirement: Every element derives from one base
Elements and groups SHALL derive from `BaseElement`, which holds a label, declares states, parameters, and roles, and
returns laws. A modeller SHALL define a type of element by subclassing `Element`.

#### Scenario: Hierarchy
- **WHEN** a user checks `Element`, `GroupElement`, and `FieldElement` against `BaseElement`
- **THEN** each is a subclass of `BaseElement`, and `CompositeSystem` and `BaseInteraction` are not

### Requirement: A system is defined by elements, interactions, and laws
A system SHALL be defined by its elements and its interactions, which are components, and by the laws its
interactions hold, which are values, and SHALL declare no other kind of component. A value that only names an expression SHALL be written in the laws, and what a run records or
checks SHALL be an argument of the run.

#### Scenario: Auxiliary as an expression
- **WHEN** a law computes `leverage = s.debt / s.equity` and uses it in a rate, and the run records an observable of
  the same expression
- **THEN** the trajectory holds `leverage`, and the system has no quantity among its components

### Requirement: Runs check invariants
`system.simulate(invariants={name: tolerance})` on a prepared system SHALL check, at every frame, that each named observable stays within its
tolerance of its first value, and SHALL raise `ConservationError` naming the observable and the first frame that
breaks it.

#### Scenario: Violated invariant
- **WHEN** a law creates wealth from nothing in a system whose run declares the observable `total_wealth` invariant
- **THEN** the run raises `ConservationError` naming `total_wealth` and the first frame that breaks it

### Requirement: Hamiltonian interactions generate rate laws
`HamiltonianInteraction` SHALL derive Hamilton's equations symbolically from a function of canonical coordinates and
momenta, states of order 1 that are real or angles, as one `RateLaw` per coordinate and per momentum. The Hamiltonians
of separate interactions on one state SHALL sum. A misuse SHALL raise `ValidationError` naming the state.

#### Scenario: Oscillator from a Hamiltonian
- **WHEN** an element with states `q` and `p` takes `HamiltonianInteraction` with `H = p^2 / (2 m) + k q^2 / 2`
- **THEN** its laws are `RateLaw(q, p / m)` and `RateLaw(p, -k q)`

#### Scenario: Coupling Hamiltonians sum
- **WHEN** two oscillators each take their own `HamiltonianInteraction`, and a third interaction binds both with the
  Hamiltonian `c (q1 - q2)^2 / 2`
- **THEN** the rate of `p1` is `-k q1 - c (q1 - q2)`, the sum of the terms of the two interactions

#### Scenario: Coordinate without a momentum
- **WHEN** a `HamiltonianInteraction` lists two coordinates and one momentum
- **THEN** `prepare` raises `ValidationError` whose message names the unpaired coordinate

#### Scenario: Coordinate of order 2
- **WHEN** a `HamiltonianInteraction` names a coordinate declared with `StateSpec(order=2)`
- **THEN** the compiler's report holds one error at the path of the interaction that names the state and gives the fix
  `StateSpec(order=1)` or `EulerLagrangeInteraction`

### Requirement: Core interactions are distinct operations
The interactions of the core SHALL be `BaseInteraction` and `Interaction` alone. Interactions that generate laws from
one function, such as gradient flows and Hamiltonian or Lagrangian mechanics, SHALL live in the systems packages that
use them, since a law written in Sage gives them in one line. No core interaction SHALL only wrap one law or fix one
binding.

#### Scenario: No parametrised copy
- **WHEN** the classes of `skcomplex.core.components.interactions` are listed
- **THEN** none is a subclass that only fixes the rate, the flux, the guard, or the map function of another core class,
  and none of `RateInteraction`, `JumpInteraction`, `MapInteraction`, `PairInteraction`, or `NeighborInteraction` is
  among them

#### Scenario: Specific form without a realization
- **WHEN** a mathematics system needs an Ornstein-Uhlenbeck control or a graph diffusion
- **THEN** it writes an `Interaction` whose law is `RateLaw` with `(goal - x) / time` or `D laplacian(u)`, with no class
  of its own

### Requirement: The core names mathematics and holds no realization
Every public name and module of `skcomplex.core.components` SHALL be a mathematical, modelling, or engineering term,
and no core name or argument SHALL name a quantity of a field. The core SHALL hold each primitive in its general form,
one class per mathematical operation. Every specific realization and every recipe SHALL live in `skcomplex.systems`,
and the core SHALL import nothing from `skcomplex.systems`.

#### Scenario: Forbidden field terms
- **WHEN** the test suite checks the public names and modules of `skcomplex.core.components` against the reviewed list
  of field terms
- **THEN** none of them contains a field term

#### Scenario: No import of systems
- **WHEN** the import graph of `skcomplex.core.components` is computed
- **THEN** it contains no module of `skcomplex.systems`

#### Scenario: Recipes outside the core
- **WHEN** a user builds a `ChoiceInteraction` with the `'fermi'` rule and a `MatchingInteraction` with
  `StableMatchingSpec`
- **THEN** both come from `skcomplex.systems.general`, built on the core `RuleLaw`, `UpdateLaw`, and `MatchLaw`, and no
  field defines either again

### Requirement: Systems nest as components
A system SHALL hold nested systems among its components, and a nested system SHALL itself be a component, so systems
nest to any depth. An interaction SHALL refer to a component of a nested system by a slash path relative to the system
that holds the interaction. Two direct children of a system with the same label SHALL be reported by validation.

#### Scenario: Interaction across nested systems
- **WHEN** a parent system holds a predation interaction between `forest/foxes` and `meadow/rabbits`
- **THEN** the compiled system couples the fox count of the forest system to the rabbit count of the meadow system

#### Scenario: Duplicate labels
- **WHEN** two direct children of a system share a label
- **THEN** validation reports one issue whose path is the system and whose message names the duplicated label

#### Scenario: A system is a component
- **WHEN** a user checks whether a `CompositeSystem` instance is an instance of `BaseComponent`
- **THEN** the check is true, the instance is an instance of `BaseSystem`, and it is not an instance of `BaseElement`

### Requirement: States and parameters follow from the laws
A value of an element that some law changes SHALL be a state, and a value that no law changes SHALL be a parameter,
decided by `prepare` from the laws of the system.

#### Scenario: A particle at rest
- **WHEN** a system holds a particle with the values `x` and `m` and no interaction
- **THEN** preparing it finds no state, and `parameters_` lists `ball.x` and `ball.m`

#### Scenario: A particle under gravity
- **WHEN** the system adds a gravity interaction whose law is `RateLaw(x.diff(t, 2), -g)`
- **THEN** `states_` lists `ball.x` and `ball.x'`, and `parameters_` lists `ball.m` and `gravity.g`

### Requirement: Derivatives, lags, and delays are written in Sage
A law SHALL be written in Sage, with every state a function of the time `t`: derivatives as `x.diff(t, k)`, past values
as `x(t=t - k)`, and delays as `x(t=t - tau)`. The library SHALL define no functions of its own for them, and the order
of a state SHALL be the highest derivative or lag its laws use.

#### Scenario: A pushed particle
- **WHEN** a user simulates a particle under gravity with `initial_state={"ball.x'": 1.5}`
- **THEN** the run starts with the velocity 1.5, and `trajectory.get_values("ball.x'")` returns the velocity

#### Scenario: A difference equation with two lags
- **WHEN** an interaction returns `UpdateLaw(y, a * y(t=t - 1) + b * y(t=t - 2))`
- **THEN** the compiled map keeps two past values of `y`, and `states_` lists `economy.y`, `economy.y[-1]`, and
  `economy.y[-2]`

### Requirement: Derivatives and past values are named with primes and indices
Derivatives SHALL be named with a prime in initial states and trajectories, such as `"ball.x'"`, and past values with
an index, such as `"economy.y[-1]"`, and they SHALL default to zero.

#### Scenario: A named derivative
- **WHEN** a user reads `trajectory.get_values("ball.x'")` of a falling particle
- **THEN** it returns the velocity, which starts at zero unless the initial state gives it

### Requirement: Prepared systems expose what they learned
After `prepare`, a system SHALL expose `states_`, the names of the values its laws change with their derivatives and
lags, `parameters_`, the names of the values no law changes, `equations_`, one readable equation per state, `form_`,
the kind of problem the laws make, and `solver_` and `backend_`, the resolved runners. The compiled problem SHALL be
private.

#### Scenario: A damped spring
- **WHEN** a user prepares a particle under gravity and a damped spring
- **THEN** `form_` is the ordinary differential equations, `equations_` maps `ball.x` to `ball.x'` and `ball.x'` to
  the sum of the forces over the mass, and `solver_` is the default ODE solver

#### Scenario: Readable reductions
- **WHEN** a user prepares a consensus whose rate sums the differences to the neighbours on a network
- **THEN** `equations_` writes the sum as `neighbors(...)` of the states it reads, and names no internal symbol

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
- **WHEN** a user creates `Temperature('T', space=CellComplexSpace(...))`
- **THEN** the element has one member at every site of the mesh

#### Scenario: A reserved name
- **WHEN** an element class stores a quantity of its own under the name `size`, such as the half side of a table
- **THEN** preparing it raises a named error that names `size` and the reserved names `label`, `size`, `space`, and
  `values`

### Requirement: Components are prepared against their context
Elements, interactions, spaces, neighbourhoods, and systems SHALL derive from `BaseComponent`, which gives `label`,
parameters, cloning, tags, and `prepare(context)`, and no `run`. An interaction SHALL implement `_prepare` against the
namespace of the elements its roles bind and set `laws_`, and the engine SHALL prepare a clone of it for every binding.
Laws SHALL be frozen values, and systems SHALL also have `simulate`.

#### Scenario: An interaction learns its laws
- **WHEN** a user prepares a gravity interaction against a namespace with the symbol `x` of its target
- **THEN** `laws_` holds `RateLaw(x.diff(t, 2), -g)`, and the interaction has no `run`

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

### Requirement: Laws are values that interactions state
A law SHALL be a frozen value with no label, no parameters of its own, and no `prepare` or `run`, and the components
SHALL be elements, interactions, spaces, and systems.

#### Scenario: A law is no component
- **WHEN** a test lists the public classes of `skcomplex.core.components` that derive from `BaseComponent`
- **THEN** none is a law

### Requirement: The core holds general spaces and neighbourhoods
The spaces of the core SHALL be the lattice, network, continuous, phase, and cell complex spaces, a mesh being a cell
complex, and the neighbourhoods SHALL be the lattice, adjacency, radius, nearest, Delaunay, and coface neighbourhoods, a
radius measured in the metric of its space. Spaces and neighbourhoods of one field SHALL live in that field.

#### Scenario: A Cayley graph
- **WHEN** a reviewer looks for `CayleySpace`
- **THEN** it lives in `skcomplex.systems.computer_science.common.automata`, and no module of the core names it

### Requirement: Elements are drawn as their class and values say
An element class SHALL give its default drawing in its class attribute `visual`, whose numeric options MAY be
expressions of the element's parameters. A group SHALL be drawn as its prepared member, or as its space when the space
cannot show that drawing. A visualisation observer SHALL draw chosen elements otherwise by path.

#### Scenario: A radius from a parameter
- **WHEN** a group of cells whose class is drawn with `{'radius': 'R_0 / 2'}` is prepared with `R_0 = 2.0`
- **THEN** the cells are drawn as points of radius 1.0

#### Scenario: A drawing the space cannot show
- **WHEN** a group whose class is drawn as an image lives on the nodes of a graph
- **THEN** it is drawn as the sites of the graph

#### Scenario: A drawing chosen at the run
- **WHEN** a user simulates with `RerunObserver(visuals={'oscillators': VisualSpec('sites', {'states': ['theta']})})`
- **THEN** the oscillators are drawn as sites, whatever their class says

### Requirement: Interactions type-check with either context
The context of an interaction's `_prepare` SHALL be typed as its space or the namespace of its bindings, so an
interaction written in the new style type-checks without suppressions.

#### Scenario: A new-style interaction
- **WHEN** `mypy` checks a test whose interaction reads `context.target.x` in `_prepare`
- **THEN** it reports no error for that `_prepare`
