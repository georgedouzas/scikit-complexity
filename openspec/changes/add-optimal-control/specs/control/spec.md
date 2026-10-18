# Spec Delta

## Purpose

Lets a user add control inputs to any system, compute optimal controls, close feedback loops, and analyse linearised
controllability.

## ADDED Requirements

### Requirement: Controls are bounded inputs of laws
A system SHALL declare control inputs with bounds that laws use like parameters, supplied by a schedule, an optimal
solution, or a controller.

#### Scenario: Scheduled control
- **WHEN** a control follows a schedule of values over time
- **THEN** the laws read the scheduled value at every step, and the trajectory records it

### Requirement: Optimal controls minimise costs
The control layer SHALL compute controls that minimise a running and terminal cost over a horizon, with Pontryagin's
principle and with direct collocation.

#### Scenario: Linear-quadratic agreement
- **WHEN** a user solves a scalar linear-quadratic problem with both methods
- **THEN** both controls match the analytic Riccati solution within 1e-3

#### Scenario: Bang-bang harvesting
- **WHEN** a user maximises the discounted harvest of a logistic resource with bounded effort
- **THEN** the optimal effort switches between its bounds and holds the resource near the level that maximises
  discounted yield in between

### Requirement: Feedback controllers act during runs
Systems SHALL include PID and state-feedback controllers whose internal states are part of the compiled system.

#### Scenario: Inverted pendulum
- **WHEN** a cart with an inverted pendulum runs from a tilt of 0.1 radians with a tuned PID controller
- **THEN** the angle returns within 0.01 radians of upright within 5 seconds

### Requirement: Linearisations are analysed for control
The control layer SHALL linearise systems into state-space matrices, report controllability and observability ranks,
and compute linear-quadratic regulator gains.

#### Scenario: Uncontrollable mode
- **WHEN** a user linearises two oscillators of which only one receives the control
- **THEN** the controllability rank is 2 of 4
