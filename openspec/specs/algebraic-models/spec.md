# algebraic-systems Specification

## Purpose
Lets a user write a system whose states are closed forms of its parameters or solutions of algebraic equations, under
static time, and solve it once per run.

## Requirements

### Requirement: Static systems compile to the static form
A system under `StaticTime` whose laws are `EquationLaw`s with a target and no time derivative SHALL compile to the
form `FormKind.STATIC` and run on `RootSolver`, which solves the equations once and gives one frame. A target on a group
SHALL give one unknown per member, and the equations of one target from several elements and interactions SHALL add.

#### Scenario: Survival at a dose
- **WHEN** an `EquationLaw` with target `S` reads `S == exp(-alpha*D - beta*D**2)` with `alpha = 0.3`, `beta = 0.03`,
  and `D = 2`, in a system with `StaticTime()`
- **THEN** the one frame holds `S = exp(-0.72)` within 1e-12

#### Scenario: Static system without equations
- **WHEN** a system with `StaticTime()` holds a group of quenched states and no laws, and runs with an observable of
  their total
- **THEN** the trajectory has one frame holding the initial states and the observable

### Requirement: Algebraic equations are solved exactly or numerically
The compiler SHALL solve the equations of a static system exactly when Sage finds a unique explicit solution, and
`RootSolver` SHALL otherwise solve them numerically from the initial state. A numerical solve that does not converge
SHALL raise a `SolverError` that names the equations and the largest residual.

#### Scenario: Exact linear solution
- **WHEN** a system declares the two Cournot best-response conditions of Forrest (2018 Eq. 21.5) with the targets `q_i`
  and `q_j`
- **THEN** the compiled solution is Eq. 21.6, `q_i = (alpha - 2 c_i/(1 + lambda_i) + c_j/(1 + lambda_j))/3`,
  symbolically

#### Scenario: Numerical solution
- **WHEN** a system declares the equation `x == cos(x)` with target `x`
- **THEN** the frame holds `x = 0.7390851` within 1e-9

#### Scenario: Cyclic equations
- **WHEN** two equations each give one target as a function of the other, `x == y / 2 + 1` and `y == x / 2 + 1`
- **THEN** the frame holds `x = y = 2` within 1e-12

#### Scenario: No solution
- **WHEN** the numerical solve of `x**2 == -1` with target `x` does not converge
- **THEN** the run raises a `SolverError` that names the equation and the residual

### Requirement: Static equations name their targets
`PassCompiler` SHALL report, with the code `equation-target`, an equation under static time that names no target or
names a target that is not a state of the system, with the path of the law. A law that does not run under static time
SHALL keep its `law-time` issue.

#### Scenario: Target that is not a state
- **WHEN** an equation under static time names the target `z`, which no element declares
- **THEN** the report holds an issue with the code `equation-target` that names the law and `z`

#### Scenario: Rate law under static time
- **WHEN** a system with `StaticTime()` holds a rate law
- **THEN** the report holds an issue with the code `law-time` that names the rate law
