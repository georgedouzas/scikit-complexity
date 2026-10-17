# Spec Delta

## Purpose

Gives a user the equations behind a system and their exact solution where one exists, using SymPy, and accepts Sage
expressions when the Sage extra is installed.

## ADDED Requirements

### Requirement: Systems expose their equations
The analysis layer SHALL return the symbolic equations of a system with symbolic laws, one per state variable.

#### Scenario: Equations of a population
- **WHEN** a user asks for the equations of a logistic growth system
- **THEN** the result holds the equation `d(n)/dt = r * n * (1 - n / K)`

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
Where the Sage extra is installed, a law SHALL accept a Sage symbolic expression and convert it to SymPy.

#### Scenario: Sage force
- **WHEN** a constant force is given as a Sage expression
- **THEN** the system compiles and simulates as with the equivalent SymPy expression
