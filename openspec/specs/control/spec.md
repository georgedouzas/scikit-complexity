# control Specification

## Purpose
Lets a user add bounded control inputs to any system, compute optimal controls, and analyse linearised
controllability.

## Requirements

### Requirement: Controls are bounded adaptive parameters
A parameter declared adaptive SHALL accept bounds, so it is a control input that laws use like a parameter, supplied by
a schedule, an optimal solution, or a controller, and recorded in trajectories. Validation SHALL report an initial value
outside the bounds.

#### Scenario: Scheduled control
- **WHEN** a bounded adaptive parameter follows a schedule of values over time
- **THEN** the laws read the scheduled value at every step, and the trajectory records it

#### Scenario: Control outside its bounds
- **WHEN** an adaptive parameter with bounds `(0, 1)` is given the initial value 2
- **THEN** `validate()` reports one error whose path is the parameter and whose fix names its bounds

### Requirement: Optimal control solvers minimise costs
The solvers `Pontryagin` and `Collocation` SHALL compute the values of a system's bounded adaptive parameters that
minimise its objective over the horizon of a run, by Pontryagin's principle and by direct collocation. Each SHALL
declare its support for continuous time and for rate, objective, and boundary laws, and validation SHALL report a
system that the chosen solver cannot solve.

#### Scenario: Linear-quadratic agreement
- **WHEN** a user solves a scalar linear-quadratic problem with both solvers
- **THEN** both controls match the analytic Riccati solution within 1e-3

#### Scenario: Bang-bang harvesting
- **WHEN** a user maximises the discounted harvest of a logistic resource with bounded effort
- **THEN** the optimal effort switches between its bounds and holds the resource near the level that maximises
  discounted yield in between

#### Scenario: Unsupported system
- **WHEN** a system with a noise law is validated with the solver `Pontryagin`
- **THEN** the report says that the solver does not support noise laws

### Requirement: Linearisations are analysed for control
The analysis layer SHALL linearise systems into state-space matrices with the controls named by their adaptive
parameters, report controllability and observability ranks, and compute linear-quadratic regulator gains.

#### Scenario: Uncontrollable mode
- **WHEN** a user linearises two oscillators of which only one receives the control
- **THEN** the controllability rank is 2 of 4

### Requirement: Optimal control solvers take their objective
`PontryaginSolver` and `CollocationSolver` SHALL take `objective`, a running cost over the dotted names of states and
parameters, with `terminal` and `sense`, and SHALL raise a named error for a missing objective or a system under
discrete time.

#### Scenario: Objective under discrete time
- **WHEN** a collocation solver with an objective is run on a system declared with discrete time
- **THEN** the run raises a named error that says objectives need static or continuous time
