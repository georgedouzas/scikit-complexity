# Spec Delta

## MODIFIED Requirements

### Requirement: Core packages mirror the system definition
The modelling packages of `skcomplex.core` SHALL be `system`, `components`, `interactions`, `space`, which holds space
and time, and `laws`, each exporting the general primitives of its part of a system. A primitive that belongs to one
formalism SHALL live in that formalism's package under `skcomplex.core.formalisms`, never in a modelling package. Every
general backend SHALL live in `skcomplex.core.backends`, a backend that serves one formalism SHALL live in that
formalism, and `compile` and `solvers` SHALL hold no backend. `skcomplex.core.model` SHALL not exist.

#### Scenario: Primitives by part
- **WHEN** a user imports `System`, `Element`, `Interaction`, `LatticeSpace`, `DiscreteTime`, and `Rate`
- **THEN** they come from `core.system`, `core.components`, `core.interactions`, `core.space`, `core.space`, and
  `core.laws`

#### Scenario: Formalism primitives by formalism
- **WHEN** a user imports `Particle`, `PairPotential`, `Rule`, and `Field`
- **THEN** they come from `core.formalisms.particles`, `core.formalisms.particles`, `core.formalisms.agents`, and
  `core.formalisms.fields`

#### Scenario: Backends in one package
- **WHEN** the test suite lists the registered backends
- **THEN** NumPy and Numba are defined in `skcomplex.core.backends`, Mesa in `skcomplex.core.formalisms.agents`, and
  LAMMPS in `skcomplex.core.formalisms.particles`

### Requirement: Systems are components
A system SHALL be a component, derived from the same base as elements, groups, and interactions, so systems nest to any
depth, and an interaction SHALL refer to a component of a nested system by a slash path relative to the system that
holds the interaction. Two direct children of a system with the same label SHALL be reported by validation.

#### Scenario: Interaction across nested systems
- **WHEN** a parent system holds a predation interaction between `forest/foxes` and `meadow/rabbits`
- **THEN** the compiled system couples the fox count of the forest system to the rabbit count of the meadow system

#### Scenario: Duplicate labels
- **WHEN** two direct children of a system share a label
- **THEN** validation reports one issue whose path is the system and whose message names the duplicated label

#### Scenario: System is a component
- **WHEN** a user checks whether a `System` instance is an instance of `Component`
- **THEN** the check is true

### Requirement: Element types create groups
An element type SHALL create a group of N identical members with a label and a size given by position, and per-member
values given by keyword as an array whose first dimension is N or as one value shared by every member, so a state of
any name that is not reserved takes values. Creating a group SHALL reject at once a size below one, a value name the
element type does not declare, and an array whose first dimension is not N.

#### Scenario: Group with per-member positions
- **WHEN** a user creates a group of 1000 molecules with an array of 1000 positions in a two-dimensional space
- **THEN** the group stores its label, size, and values unmodified, and its position state has shape `(1000, 2)`

#### Scenario: Wrong number of values
- **WHEN** a user gives 999 positions to a group of size 1000
- **THEN** creating the group raises a named error that names the group, the variable, and both sizes

#### Scenario: Unknown value name
- **WHEN** a user creates `Population.group('herds', 3, foo=1)`
- **THEN** creating the group raises a named error that names `foo` and lists the declared states and parameters

#### Scenario: State named size
- **WHEN** an element type declares a state `size` and a user creates `Cell.group('cells', 3, size=[1.0, 2.0, 3.0])`
- **THEN** the group has 3 members and its `size` state holds 1, 2, and 3

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

## ADDED Requirements

### Requirement: Every component derives from one base
Elements, groups, interactions, and systems SHALL derive from one component base that holds a label, declares variables
and roles, and returns laws.

#### Scenario: Hierarchy
- **WHEN** a user checks `Element`, `Group`, `Interaction`, and `System` against `Component`
- **THEN** each is a subclass of `Component`

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
