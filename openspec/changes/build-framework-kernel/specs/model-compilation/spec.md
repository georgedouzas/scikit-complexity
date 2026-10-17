# Spec Delta

## Purpose

Turns a described system into a problem of one mathematical form, with a flat state vector, a parameter vector, and
fast numeric functions that solvers evaluate.

## ADDED Requirements

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
