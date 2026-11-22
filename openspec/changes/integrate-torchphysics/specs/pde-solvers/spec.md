## ADDED Requirements

### Requirement: Static fields given by an energy solve by the Deep Ritz method

The framework SHALL find the static state of a field system whose laws give its energy as an objective, by training a
neural network that minimises the energy integral over the domain of the space, with the fixed values at the boundary
as a penalty. The result SHALL be the field on the points of the space, read like the result of any other solver.

#### Scenario: Loaded membrane

- **WHEN** a square membrane on a space fixed at zero on its boundary carries the energy `∫ (|∇u|² / 2 - f u)` with the
  load `f = 2π² sin(πx) sin(πy)`, and is prepared with `DeepRitzSolver` as its solver and a fixed random seed
- **THEN** the field agrees with the exact solution `sin(πx) sin(πy)` within 1e-2 in the relative L2 norm on a 33 by
  33 grid

#### Scenario: Same answer as finite elements

- **WHEN** the same membrane is solved by the Deep Ritz method and by finite elements on a mesh of the same square
- **THEN** both fields agree within 2e-2 in the relative L2 norm on the nodes of the mesh

### Requirement: The Deep Ritz method rejects laws that are not an energy

A field system whose laws are not one energy over a field, or that changes in time, SHALL be rejected when it is
prepared for the Deep Ritz method, with an error that names the laws the method cannot read.

#### Scenario: Laws that are not an energy

- **WHEN** a heat equation given by rate laws is prepared with `DeepRitzSolver` as its solver
- **THEN** preparing the system raises an error that names its rate laws and points to a solver for fields in time
