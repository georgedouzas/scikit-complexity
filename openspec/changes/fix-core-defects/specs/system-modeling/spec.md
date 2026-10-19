## MODIFIED Requirements

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

## ADDED Requirements

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
