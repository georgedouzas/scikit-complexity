# system-modeling Specification

## Purpose

Lets a user describe a model from any science as elements that carry state, interactions that change that state
through laws, a space the state lives in, and systems that nest and compose.

## Requirements

### Requirement: Elements declare typed state and parameters
An element type SHALL declare its state variables and parameters with a domain, and an element instance SHALL take
its label, initial state values, and parameter values as constructor arguments stored unmodified.

#### Scenario: Element stores its arguments
- **WHEN** a user creates a population element with label `rabbits` and initial count 40
- **THEN** the element exposes `label == 'rabbits'` and `n == 40`

#### Scenario: Invalid value is rejected at compile time
- **WHEN** a system contains an element whose positive parameter is set to a negative number and it is compiled
- **THEN** a named validation error is raised that names the element, the variable, and the domain

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

### Requirement: Laws describe how state changes
An interaction or element SHALL describe its effect as a list of laws of the kinds rate, noise, jump, and update, each
law targeting state variables, and each law SHALL accept a symbolic expression or a Python callable.

#### Scenario: Symbolic rate law
- **WHEN** an interaction returns a rate law on `prey.n` with the expression `-a * predator.n * prey.n`
- **THEN** the compiled system includes that term in the time derivative of the prey count

#### Scenario: Callable rate law
- **WHEN** an interaction returns a rate law whose value is a Python callable of time and current values
- **THEN** the simulation evaluates the callable at each solver evaluation and adds its result to the derivative

### Requirement: Rate and noise contributions add up
Rate and noise laws on the same state variable SHALL combine by summation, so adding an interaction adds a term
without changing other interactions.

#### Scenario: Two forces on one particle
- **WHEN** a particle is subject to a constant force and a harmonic potential
- **THEN** its velocity derivative equals the sum of both contributions divided by its mass

### Requirement: Element types create groups
An element type SHALL create a group of N identical members with a label and a size given by position, and per-member
values given by keyword as an array whose first dimension is N or as one value shared by every member, so a state of
any name that is not reserved takes values.

#### Scenario: Group with per-member positions
- **WHEN** a user creates a group of 1000 molecules with an array of 1000 positions in a two-dimensional space
- **THEN** the group stores its label, size, and values unmodified, and its position state has shape `(1000, 2)`

#### Scenario: Wrong number of values
- **WHEN** a user gives 999 positions to a group of size 1000
- **THEN** compiling the system raises a named validation error that names the group, the variable, and both sizes

#### Scenario: State named size
- **WHEN** an element type declares a state `size` and a user creates `Cell.group('cells', 3, size=[1.0, 2.0, 3.0])`
- **THEN** the group has 3 members and its `size` state holds 1, 2, and 3

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
  and with `ScipyODE`
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
The modelling layer SHALL build element and interaction types from declared states, parameters, roles, and a function
that returns their laws, and such types SHALL behave like subclassed components in groups, cloning, and parameter
introspection.

#### Scenario: One-off growth model
- **WHEN** a user defines an element type with a state `n` and an interaction type with a role and a rate law in a few
  lines, without writing a class
- **THEN** a system of them compiles and simulates exactly like the same model written with classes

#### Scenario: Defined types form groups
- **WHEN** a user creates a group of 100 members from a defined element type
- **THEN** the group compiles once and its parameters appear in `get_params` like those of a subclassed type

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
A state SHALL declare an order `k`, and the framework SHALL add companion states for its derivatives of orders 1 to
`k - 1`. These SHALL take initial values in the constructor, appear in trajectories by path, and be readable in laws.

#### Scenario: Initial angular velocity
- **WHEN** a pendulum declares its angle with order 2 and is created with `theta=0.1, theta_d1=0.5`
- **THEN** the run starts with angle 0.1 and angular velocity 0.5, and the trajectory returns both `bob.theta` and
  `bob.theta_d1`

#### Scenario: First order unchanged
- **WHEN** a state declares no order
- **THEN** the model compiles exactly as before, with no companion states

### Requirement: Laws of any order superpose on the highest derivative
Rate and noise laws on a state of order `k` SHALL contribute to its `k`-th derivative, and the contributions of
different laws SHALL add up. A law that asserts a different order SHALL raise a named error.

#### Scenario: Newton's second law from separate forces
- **WHEN** a mass on a spring receives a restoring law from one interaction and a damping law that reads `x_d1` from
  another, on a position of order 2
- **THEN** the run matches the damped oscillator `m x'' = -k x - c x'` within the solver tolerance

#### Scenario: Third order
- **WHEN** Sprott's jerk equation `x''' = -A x'' - x' + |x| - 1` is written as one rate law on a state of order 3
- **THEN** it compiles to three first-order equations, and with A = 0.6 its trajectory is bounded and chaotic

#### Scenario: Order mismatch
- **WHEN** a law asserts order 1 on a state declared with order 2
- **THEN** compiling raises a named error that names the law, the state, and both orders

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
- **WHEN** Samuelson's model `Y_t = c (1 + v) Y_{t-1} - c v Y_{t-2} + G` runs with c = 0.8 and v = 1.5
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
  `Rate(s.cells.x, s.cells.x / count(over=s.cells))`
- **THEN** the system compiles, and the rate of member `i` is `x_i / 4`

#### Scenario: Weighted reduction in a denominator
- **WHEN** an interaction bound to a group declares `Rate(s.cells.x, s.cells.x / total(s.cells.x, weights=W))` with a
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
  `Rate(s.hub.y, s.site.x)`
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
An interaction SHALL declare roles, each binding one component, one member, or a sequence of them. Its binding SHALL
be one of `'members'`, `'pairs'`, `'neighbours'`, or `'hyperedges'`, so one interaction type covers unary, pairwise,
neighbourhood, and higher-order couplings, and validation SHALL report a binding the space does not offer.
Interactions SHALL hold no state.

#### Scenario: Neighbour coupling
- **WHEN** an interaction with binding `'neighbours'` couples the members of a quantum group on a lattice
- **THEN** its laws run once per pair of adjacent sites

#### Scenario: Pairs within a cutoff
- **WHEN** an interaction with binding `'pairs'` couples a group of particles
- **THEN** its laws run once per pair of members closer than its cutoff, as the former pair interactions did

#### Scenario: Hyperedges need a network that has them
- **WHEN** an interaction with binding `'hyperedges'` couples a group on a lattice
- **THEN** validation reports that the binding needs a network space with hyperedges

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
An invariant SHALL declare an expression that stays constant within a tolerance. It SHALL be recorded and checked at
every frame of every run, with a named error that names the invariant and the first frame that breaks it.

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

### Requirement: Core packages mirror the system definition
The modelling packages of `skcomplex.core` SHALL be `system`, `components`, `interactions`, `space`, which holds space
and time, and `laws`, each exporting the primitives of its part of a system. Every backend SHALL live in
`skcomplex.core.backends`, and `compile` and `solvers` SHALL hold no backend. A module that serves one backend SHALL be
private to that backend. `skcomplex.core.model` SHALL not exist.

#### Scenario: Primitives by part
- **WHEN** a user imports `System`, `Element`, `Interaction`, `LatticeSpace`, `DiscreteTime`, and `Rate`
- **THEN** they come from `core.system`, `core.components`, `core.interactions`, `core.space`, `core.space`, and
  `core.laws`

#### Scenario: Backends in one package
- **WHEN** the test suite lists the backends registered for solvers
- **THEN** each one is defined in a module of `skcomplex.core.backends`, and `skcomplex.core.backends` has no public
  module that is not a backend or the backend base
