# Spec Delta

## ADDED Requirements

### Requirement: SymPy expressions are rejected
A law that receives a SymPy expression SHALL raise a named validation error that names the component and the law and
tells the user to write the expression in Sage.

#### Scenario: SymPy rate
- **WHEN** an interaction returns a rate law whose value is a SymPy expression
- **THEN** compiling raises a named validation error that names the interaction and suggests a Sage expression

## MODIFIED Requirements

### Requirement: Sage expressions are accepted
Laws, energies, pair potentials, and boundaries SHALL be written as Sage symbolic expressions, without any optional
installation, and the symbols the framework gives a component SHALL be Sage symbols that carry the domains of their
variables.

#### Scenario: Sage force
- **WHEN** a constant force is given as a Sage expression
- **THEN** the system compiles and simulates with the trajectory of the same model before this change, within a
  relative tolerance of 1e-10

#### Scenario: Symbols are Sage symbols
- **WHEN** a component's `laws` method receives its symbols
- **THEN** every scalar symbol is a Sage symbolic expression, and every vector state is a Sage vector with one
  component per dimension of the space

### Requirement: Systems expose their equations
The analysis layer SHALL return the symbolic equations of a system with symbolic laws, one per state variable, as Sage
relations.

#### Scenario: Equations of a population
- **WHEN** a user asks for the equations of a logistic growth system
- **THEN** the result holds the Sage relation `diff(n(t), t) == r * n(t) * (1 - n(t) / K)`
