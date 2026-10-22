# Spec Delta

## Purpose

Lets a user write a model whose states are closed forms of its parameters or solutions of algebraic equations, and
solve it again at every frame instead of integrating it in time.

## ADDED Requirements

### Requirement: Definitions give states as closed forms
Components and interactions SHALL define a state as a symbolic expression of parameters and other defined states with
the definition law, a law of the constraint class. A system with static time whose laws are only definitions and
algebraic equations SHALL compile to the static form, and SHALL evaluate the definitions in dependency order at every
frame from the current parameters.

#### Scenario: Survival at a dose
- **WHEN** a state is defined as `exp(-alpha*D - beta*D**2)` with `alpha = 0.3`, `beta = 0.03`, and `D = 2`, in a
  system with `StaticTime()`
- **THEN** the frame holds the value `exp(-0.72)` within 1e-12

#### Scenario: Cyclic definitions
- **WHEN** two definitions each read the state the other defines
- **THEN** validation reports one issue that names both states and whose fix suggests an algebraic equation instead

### Requirement: Algebraic equations are solved for their unknowns
Algebraic equation laws SHALL declare the states they solve for. The compiler SHALL solve them exactly when the
symbolic layer finds a unique closed-form solution, and SHALL otherwise solve them numerically at every frame,
starting from the solution of the previous frame. Validation SHALL report an unknown that is not a state of the system.

#### Scenario: Exact linear solution
- **WHEN** a system declares the two Cournot best-response conditions of Forrest (2018 Eq. 21.5) for the sales `q_i`
  and `q_j`
- **THEN** the compiled solution is Eq. 21.6, `q_i = (alpha - 2 c_i/(1 + lambda_i) + c_j/(1 + lambda_j))/3`,
  symbolically

#### Scenario: Numerical solution
- **WHEN** a system declares the equation `x = cos(x)` for `x`
- **THEN** every frame holds `x = 0.7390851` within 1e-9

#### Scenario: No solution
- **WHEN** the numerical solve does not converge at a frame
- **THEN** the run raises a named error that names the equations, the frame, and the residual

#### Scenario: Unknown that is not a state
- **WHEN** an algebraic equation names an unknown that no component of the system declares
- **THEN** validation reports the equation's path and the missing state

### Requirement: The static solver declares its support
The evaluation solver SHALL declare that it supports static time, definitions, algebraic equations, and objectives,
and validation SHALL report a static system whose laws it does not support, before compilation.

#### Scenario: Unsupported law under static time
- **WHEN** a system with `StaticTime()` holds a rate law and is validated with the evaluation solver
- **THEN** the report names the rate law and says that static systems hold only constraint laws
