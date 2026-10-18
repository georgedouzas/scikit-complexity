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
element label of its system.

#### Scenario: Role refers to an unknown element
- **WHEN** a system contains an interaction whose role names a label that no element in reach has
- **THEN** compiling the system raises a named error that names the interaction, the role, and the label

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
An element type SHALL create a group of N identical members with a label, a size, and per-member values given as an
array whose first dimension is N or as one value shared by every member.

#### Scenario: Group with per-member positions
- **WHEN** a user creates a group of 1000 molecules with an array of 1000 positions in a two-dimensional space
- **THEN** the group stores its label, size, and values unmodified, and its position state has shape `(1000, 2)`

#### Scenario: Wrong number of values
- **WHEN** a user gives 999 positions to a group of size 1000
- **THEN** compiling the system raises a named validation error that names the group, the variable, and both sizes

### Requirement: Roles bind to groups
An interaction role SHALL bind to a group as well as to a single element, and the interaction's laws SHALL then apply
to every member of the group.

#### Scenario: Wall law on a whole gas
- **WHEN** a reflecting wall interaction binds its role to a group of 1000 molecules
- **THEN** every molecule is reflected at the walls

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
An interaction SHALL be able to return boundary laws, which map states to new values after every solver step, combine
with any law kind without changing the inferred form, and describe walls and periodic boxes.

#### Scenario: Reflecting wall in an ODE system
- **WHEN** a particle group with rate laws also has reflecting wall boundaries
- **THEN** the system compiles to the ODE form and no member crosses a wall

#### Scenario: Periodic box
- **WHEN** a member leaves a periodic box through one side
- **THEN** it reappears at the opposite side with the same velocity

### Requirement: Laws express conditions and remainders
The modelling layer SHALL provide symbolic functions for a value chosen by a condition, for the conjunction and
disjunction of conditions, and for the remainder of a division, which laws use like any other expression and which
every backend lowers.

#### Scenario: Reflection by condition
- **WHEN** a boundary sets a velocity to its negative where a position lies outside a box and keeps it elsewhere
- **THEN** a particle that crosses a wall moves back into the box with its velocity reversed

#### Scenario: Periodic remainder
- **WHEN** a boundary sets a position to its remainder by the box size
- **THEN** a particle that leaves through one side enters through the opposite side

#### Scenario: Conditions in LAMMPS
- **WHEN** a pair energy uses a condition on the distance and the LAMMPS backend runs it
- **THEN** the energies match those of the NumPy backend within a relative tolerance of 1e-10

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
A system SHALL declare observables, named symbolic expressions of its states and parameters or reductions over its
groups, that are evaluated at every frame and readable by laws of other partitions.

#### Scenario: Order parameter
- **WHEN** a system of phase oscillators declares the observable `order` as the modulus of the mean of `exp(i θ)`
- **THEN** the trajectory returns `order` at every frame with values between 0 and 1

#### Scenario: Reduction of a group
- **WHEN** a system declares the mean of the state `wealth` over a group of agents as an observable
- **THEN** the trajectory returns the mean at every frame, and an ODE in another partition can read it

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
An equation law SHALL relate the highest derivatives of states to other expressions. The compiler SHALL solve all such
equations of a system jointly for the highest derivatives, and SHALL raise a named error when they cannot be solved.

#### Scenario: Pendulum as written
- **WHEN** a pendulum states `m l derivative(θ, 2) == -m g sin θ`
- **THEN** the run equals the one from the rate law `derivative(θ, 2) = -(g / l) sin θ`

#### Scenario: Coupled equations
- **WHEN** two equations each contain the second derivatives of both angles of a double pendulum
- **THEN** the compiler solves them jointly, and the energy stays constant within the solver tolerance

#### Scenario: Unsolvable highest derivative
- **WHEN** an equation contains the square of a highest derivative
- **THEN** compiling raises a named error that names the equation and the derivative

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
