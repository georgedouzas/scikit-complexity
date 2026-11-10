## ADDED Requirements

### Requirement: A value outside its domain is reported as outside it

A value of a state that lies outside the domain of the state SHALL be reported with a `ValidationError` that names the
state, the value, and the domain. The error SHALL say that the value has a nonzero imaginary part only when the value
is complex and not real.

#### Scenario: A negative value of a nonnegative state

- **WHEN** a state with the domain `[0, +oo)` is given the value `-1.0` and the system is simulated
- **THEN** a `ValidationError` says that `-1.0` is not in the domain `[0, +oo)`, and it does not mention an imaginary part

#### Scenario: A fractional value of an integer state

- **WHEN** a state whose domain is the nonnegative integers is given `1.5`
- **THEN** the `ValidationError` says that `1.5` is not in that domain

#### Scenario: A complex value of a real state

- **WHEN** a real state is given `1 + 2j`
- **THEN** the `ValidationError` still says that the value has a nonzero imaginary part

### Requirement: An unknown variant lists only the accepted values

When an argument of a part takes one of several named values, or a spec a subclass handles, an unknown value SHALL
raise a `ValidationError` that names the part, the argument, the value given, and the accepted names. The message
SHALL describe an accepted spec in words and SHALL NOT print the name of a base class.

#### Scenario: An unknown mode of a map interaction

- **WHEN** `MapInteraction(..., mode='add')` is prepared
- **THEN** the `ValidationError` names `'add'` and the accepted names `'replace'` and `'increment'`, and its text holds no `BaseSpec`

### Requirement: The order of a law is checked against its state in every system

A rate law or a noise law that gives a derivative of an order other than the order of its state SHALL raise a
`ValidationError` when the system is compiled, whatever the orders of the other states of the system.

#### Scenario: A second-order law on a first-order state, alone

- **WHEN** a system whose only state has order 1 holds `RateLaw(x, -x, order=2)` and is simulated
- **THEN** a `ValidationError` says that the law gives the derivative of order 2 but the state has order 1

### Requirement: The ends of a transfer print by their fields

The source and the sink of a transfer law SHALL print as their class name with their fields, and two ends with equal
fields SHALL be equal, so errors that name them never show a memory address.

#### Scenario: An error that names a sink

- **WHEN** a transfer law is given a sink where a source belongs and is prepared
- **THEN** the error shows `Sink(label='environment')`, not an address in memory

### Requirement: Bad spaces raise library errors

A space SHALL check its arguments when it is prepared and raise a library error that names the space and the
argument at fault. No raw `IndexError`, `KeyError`, `ValueError`, or NumPy error SHALL escape from preparing a space or
from reading its neighbours.

#### Scenario: Mesh points of the wrong shape

- **WHEN** a `MeshSpace` is prepared with points that are not two or three rows of coordinates
- **THEN** a `ValidationError` names `points` and the shape it expects

#### Scenario: Mesh cells that name a missing point

- **WHEN** a `MeshSpace` is prepared with a cell index at or beyond the number of points, or with an empty or one-dimensional array of cells
- **THEN** a `ValidationError` names `cells` and says what is wrong

#### Scenario: A neighbourhood names a missing layer

- **WHEN** an adjacency neighbourhood names the layer `'work'` of a network space that has no such layer and the system is simulated
- **THEN** a `SpaceError` says that the network has no layer `'work'` and lists its layers

### Requirement: A law with no form names its class

A law whose class has no form the engine knows SHALL raise an `UnregisteredLawError` that names the class, before
any check of mixed law classes.

#### Scenario: A law class with no form

- **WHEN** a system holds a law of a class the engine has no form for and is compiled
- **THEN** an `UnregisteredLawError` names that class

### Requirement: Every error class is raised

Every public error class SHALL be raised somewhere in the library. A class that no part can raise SHALL be removed.

#### Scenario: The argument error is gone

- **WHEN** a user imports `ArgumentError` from `skcomplex.errors`
- **THEN** the import fails, and the ridesharing system reports a value of the wrong kind with a `ValidationError`
