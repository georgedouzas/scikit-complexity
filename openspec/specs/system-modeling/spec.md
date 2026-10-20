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

### Requirement: Systems nest and compose
A system SHALL contain elements, interactions, and child systems, and an interaction SHALL refer to an element of a
child system by a slash-separated path relative to the system that holds the interaction.

#### Scenario: Interaction across subsystems
- **WHEN** a parent system holds a predation interaction between `forest/foxes` and `meadow/rabbits`
- **THEN** the compiled system couples the fox count of the forest subsystem to the rabbit count of the meadow

#### Scenario: Duplicate labels are rejected
- **WHEN** two direct children of a system share a label
- **THEN** compiling the system raises a named error that names the duplicated label

### Requirement: Spaces give vector state its dimension
A system SHALL hold an optional space, and a vector state variable SHALL take its dimension from the space of the
system that holds the element.

#### Scenario: Particle in a plane
- **WHEN** a particle is placed in a system whose Euclidean space has two dimensions
- **THEN** its position and velocity each have two components

#### Scenario: Vector state without a space
- **WHEN** an element with a vector state variable is compiled in a system with no space
- **THEN** a named error is raised that tells the user to give the system a space

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

### Requirement: Pair interactions sum over member pairs
A pair interaction SHALL be written once for one pair of members and SHALL contribute the sum of its law over every
pair of members within a cutoff distance, inside one group or between two groups.

#### Scenario: Pair force inside a group
- **WHEN** a Lennard-Jones pair interaction with cutoff 2.5 applies to a group
- **THEN** the force on each member equals the sum of the pair forces from the members closer than 2.5

#### Scenario: Newton's third law
- **WHEN** a pair interaction acts inside an isolated group
- **THEN** the total momentum of the group stays constant within solver tolerance

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

### Requirement: Systems declare observables
A system SHALL declare observables, named symbolic expressions of its states, parameters, and time or reductions over
its groups, that are evaluated at every frame, readable at the current state by laws of their own partition when they
are symbolic, and readable by laws of other partitions at synchronisation points. The namespaces that observable
expressions receive SHALL hold the time as `t`.

#### Scenario: Order parameter
- **WHEN** a system of phase oscillators declares the observable `order` as the modulus of the mean of `exp(i θ)`
- **THEN** the trajectory returns `order` at every frame with values between 0 and 1

#### Scenario: Reduction of a group
- **WHEN** a system declares the mean of the state `wealth` over a group of agents as an observable
- **THEN** the trajectory returns the mean at every frame, and an ODE in another partition can read it

#### Scenario: Observable read in its own partition
- **WHEN** a law of an ODE system reads the observable `mean_wealth`, the mean of `W` over the system's group of savers
- **THEN** the system compiles to one ODE partition, and the value the law reads equals the recorded observable at
  every frame within 1e-12

#### Scenario: Observable of time
- **WHEN** a system declares the observable `s.a.n * s.t` and the reduction over a group of `member.n + member.t`
- **THEN** both are recorded at every frame with the time of the frame, and no `AttributeError` is raised

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
