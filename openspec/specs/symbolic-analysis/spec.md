# symbolic-analysis Specification

## Purpose

Lets a user write laws as Sage symbolic expressions, and gives the equations behind a system and their exact solution
where one exists, using Sage and Maxima.

## Requirements

### Requirement: Systems expose their equations
The analysis layer SHALL return the symbolic equations of a system with symbolic laws, one per state variable, as Sage
relations.

#### Scenario: Equations of a population
- **WHEN** a user asks for the equations of a logistic growth system
- **THEN** the result holds the Sage relation `diff(n(t), t) == r * n(t) * (1 - n(t) / K)`

### Requirement: ODE systems are solved exactly where possible
The analysis layer SHALL solve the equations of an ODE system as functions of time, with integration constants or with
the initial state of the system as initial conditions.

#### Scenario: Free particle
- **WHEN** a user solves a free particle in one dimension without initial conditions
- **THEN** the position is a linear function of time with two integration constants

#### Scenario: Harmonic oscillator with initial conditions
- **WHEN** a user solves a harmonic oscillator using the initial state of the system
- **THEN** the position is a combination of a sine and a cosine of `sqrt(k / m) * t` that matches the initial state

#### Scenario: Not an ODE
- **WHEN** a user solves a system that is not an ODE
- **THEN** a named error is raised that names the form

### Requirement: Sage expressions are accepted
Laws, energies, pair potentials, and boundaries SHALL be written as Sage symbolic expressions, without any optional
installation, and the symbols the framework gives a component SHALL be Sage symbols that carry the domains of their
variables.

#### Scenario: Sage force
- **WHEN** a constant force is given as a Sage expression
- **THEN** the system compiles and simulates with the trajectory of the same force given as a number

#### Scenario: Symbols are Sage symbols
- **WHEN** a component's `laws` method receives its symbols
- **THEN** every scalar symbol is a Sage symbolic expression, and every vector state is a Sage vector with one
  component per dimension of the space

### Requirement: SymPy expressions are rejected
A law that receives a SymPy expression SHALL raise a named validation error that names the component and the law and
tells the user to write the expression in Sage.

#### Scenario: SymPy rate
- **WHEN** an interaction returns a rate law whose value is a SymPy expression
- **THEN** compiling raises a named validation error that names the interaction and suggests a Sage expression
