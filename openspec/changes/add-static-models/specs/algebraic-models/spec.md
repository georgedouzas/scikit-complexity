# Spec Delta

## Purpose

Lets a user write a model whose states are closed forms of its parameters or solutions of algebraic equations, and
solve it again at every frame instead of integrating it in time.

## ADDED Requirements

### Requirement: Definitions give states as closed forms
Interactions SHALL define a state as a symbolic expression of parameters and other defined states. A system whose
laws are only definitions and algebraic equations SHALL compile to the static form, and SHALL evaluate the
definitions in dependency order at every frame from the current parameters.

#### Scenario: Survival at a dose
- **WHEN** a state is defined as `exp(-alpha*D - beta*D**2)` with `alpha = 0.3`, `beta = 0.03`, and `D = 2`
- **THEN** the frame holds the value `exp(-0.72)` within 1e-12

#### Scenario: Cyclic definitions
- **WHEN** two definitions each read the state the other defines
- **THEN** compiling raises a named error that names both states and suggests an algebraic equation instead

### Requirement: Algebraic equations are solved for their unknowns
Interactions SHALL declare algebraic equations and the states they solve for. The compiler SHALL solve them exactly
when the symbolic layer finds a unique closed-form solution, and SHALL otherwise solve them numerically at every frame,
starting from the solution of the previous frame.

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
