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

### Requirement: Equations keep their written order
The analysis layer SHALL return the equations of higher-order states as relations of their highest derivatives, and
those of lagged states as relations of their past values.

#### Scenario: Pendulum equation
- **WHEN** a user asks for the equations of a pendulum with an angle of order 2
- **THEN** the result holds `diff(theta(t), t, 2) == -g sin(theta(t)) / l`, and no separate velocity equation

### Requirement: Equations of any order are solved exactly where possible
The analysis layer SHALL solve single equations of any order with initial values of every derivative, and systems of
them after reduction to first order.

#### Scenario: Small-angle pendulum
- **WHEN** a user solves `θ'' = -(g / l) θ` with `θ(0) = 0.1` and `θ'(0) = 0`
- **THEN** the solution is `0.1 cos(sqrt(g / l) t)`

#### Scenario: Third-order linear equation
- **WHEN** a user solves `x''' = -x'` with `x(0) = 0`, `x'(0) = 1`, and `x''(0) = 0`
- **THEN** the solution is `sin(t)`

### Requirement: No closed form is never reported as a solution
The analysis layer SHALL return a solution only when it contains no unevaluated transforms, integrals, or unknown
functions of time, and SHALL otherwise raise the named error that suggests numeric simulation.

#### Scenario: Nonlinear pendulum
- **WHEN** a user solves the pendulum with `sin θ`
- **THEN** a named error says that no closed form was found, instead of returning unevaluated transforms

### Requirement: Parameter signs reach the solver
Parameters SHALL give the symbolic solver their sign from their domain or, for real domains, from their current value.
A sign question that remains SHALL raise a named error that names the parameters and suggests declaring their domain.

#### Scenario: Plain positive numbers
- **WHEN** a user solves the small-angle pendulum with `g` and `l` given as plain positive numbers
- **THEN** the solution is returned without a sign question

### Requirement: Fixed points and their stability are derived
The analysis layer SHALL find the fixed points of ODE systems and maps, derive their Jacobians, and classify each fixed
point's stability from its eigenvalues.

#### Scenario: Lotka-Volterra centre
- **WHEN** a user analyses prey with exponential growth rate `r`, predators with mortality `d`, and predation with
  rates `a` for the prey and `b` for the predators
- **THEN** the interior fixed point has prey `d / b` and predators `r / a`, and its eigenvalues are purely imaginary,
  classified as a centre

#### Scenario: Logistic map period one
- **WHEN** a user analyses the logistic map at r = 2.5
- **THEN** the fixed point `1 - 1 / r` is classified as stable

### Requirement: Vector-field divergence is derived
The analysis layer SHALL derive the divergence of a system's vector field.

#### Scenario: Volume preservation
- **WHEN** a user derives the divergence of a frictionless harmonic oscillator
- **THEN** the divergence is 0
