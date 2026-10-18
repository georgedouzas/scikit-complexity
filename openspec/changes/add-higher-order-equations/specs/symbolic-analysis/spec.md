# Spec Delta

## ADDED Requirements

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
