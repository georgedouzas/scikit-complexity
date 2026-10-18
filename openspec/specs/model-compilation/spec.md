# model-compilation Specification

## Purpose

Turns a described system into a problem of one mathematical form, with a flat state vector, a parameter vector, and
fast numeric functions that solvers evaluate.

## Requirements

### Requirement: The form is inferred from the law kinds
Compiling a system SHALL infer its form from the law kinds it contains: rate laws alone give an ODE, rate and noise
laws give an SDE, jump laws alone give a jump process, and update laws alone give a discrete map.

#### Scenario: Deterministic ecology
- **WHEN** a system contains only rate laws
- **THEN** the compiled problem has the ODE form

#### Scenario: Stochastic epidemic
- **WHEN** a system contains only jump laws
- **THEN** the compiled problem has the jump process form

#### Scenario: Unsupported mix
- **WHEN** a system contains both update laws and rate laws
- **THEN** compiling raises a named error that lists the law kinds found and the supported combinations

### Requirement: State and parameters are addressed by path
The compiled problem SHALL give every state variable and parameter a unique dotted name built from its element or
interaction path, and SHALL map each name to its position in the flat state or parameter vector.

#### Scenario: Name of a nested state
- **WHEN** the element `rabbits` inside the subsystem `meadow` has the state `n`
- **THEN** the problem names that state `meadow/rabbits.n`

### Requirement: Parameters stay symbolic until run time
The compiled numeric functions SHALL take the parameter vector as an argument, so changing a parameter value does not
require compiling again.

#### Scenario: Same problem with new parameters
- **WHEN** the drift function of a compiled problem is evaluated twice with two parameter vectors
- **THEN** the two results reflect the two parameter values

### Requirement: The compiled problem keeps its equations
The compiled problem SHALL keep the symbolic equations of every symbolic law, so they can be displayed and analysed.

#### Scenario: Equations of an oscillator
- **WHEN** a harmonic oscillator is compiled
- **THEN** the problem lists one equation per state variable, with the velocity equation equal to `-k * x / m`

### Requirement: Groups compile independently of their size
Compiling a system SHALL build one set of numeric functions per group and law, so the compile time of a group does not
grow with its size.

#### Scenario: Large group compiles fast
- **WHEN** a group of 100000 molecules with reflecting walls is compiled
- **THEN** compiling takes no longer than compiling the same system with a group of 10 molecules, within a factor of 2

### Requirement: Group variables are addressed by group path
The compiled problem SHALL name each group variable by the group path and variable name, with a leading dimension of
the group size.

#### Scenario: Name of a group state
- **WHEN** the group `gas` of 500 molecules in a two-dimensional space has the state `x`
- **THEN** the problem has a state named `gas.x` with shape `(500, 2)`

### Requirement: Laws compile to a chosen backend
Compiling a system SHALL accept a backend, `numba`, `numpy` as the reference, or `auto` by default, which picks `numba`
for systems whose groups hold at least 1000 members in total and `numpy` otherwise, and both backends SHALL produce the
same numeric functions for the same system.

#### Scenario: Backends agree
- **WHEN** an ideal gas with pair interactions is compiled with both backends and evaluated on the same state
- **THEN** the drift, boundary, and update results agree to within 1e-12 relative error

#### Scenario: Automatic choice
- **WHEN** a system with a group of 1000 members and a system with two single elements are compiled with `auto`
- **THEN** the first uses the Numba backend and the second uses the NumPy backend

#### Scenario: Unknown backend
- **WHEN** a user asks for a backend that is not registered
- **THEN** a named error is raised that lists the registered backends

### Requirement: The Numba backend is faster on large groups
The Numba backend SHALL evaluate the functions of a large group faster than the NumPy backend.

#### Scenario: Million-member gas
- **WHEN** one step of an ideal gas of 1000000 members is evaluated with both backends after compilation
- **THEN** the Numba backend takes at most a third of the time of the NumPy backend

### Requirement: Compiled callables run inside kernels
A law value that is a Numba-compiled function SHALL be called from inside the compiled kernels, and a plain Python
callable SHALL still work through a Python path.

#### Scenario: Compiled callable law
- **WHEN** a rate law value is a Numba-compiled function of time and member values
- **THEN** the system compiles with the Numba backend and its results equal those of the NumPy backend

#### Scenario: Python callable law
- **WHEN** a rate law value is a plain Python callable and the backend is Numba
- **THEN** the system compiles and runs, with that law evaluated in Python

### Requirement: Compiled kernels are cached
The Numba backend SHALL cache compiled kernels on disk, keyed by the lowered expressions, so compiling the same system
in a new process reuses them.

#### Scenario: Second compilation
- **WHEN** the same system is compiled in two separate processes
- **THEN** the second compilation is at least five times faster than the first
