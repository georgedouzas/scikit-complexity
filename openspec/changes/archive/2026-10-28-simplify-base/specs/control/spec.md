## ADDED Requirements

### Requirement: Optimal control solvers take their objective
`PontryaginSolver` and `CollocationSolver` SHALL take `objective`, a running cost over the dotted names of states and
parameters, with `terminal` and `sense`, and SHALL raise a named error for a missing objective or a system under
discrete time.

#### Scenario: Objective under discrete time
- **WHEN** a collocation solver with an objective is run on a system declared with discrete time
- **THEN** the run raises a named error that says objectives need static or continuous time

## REMOVED Requirements

### Requirement: Objectives define costs over a horizon
**Reason**: An objective defines a control problem, not a part of a system.
**Migration**: Pass the running cost, the terminal cost, and the sense to the optimal control solver.
