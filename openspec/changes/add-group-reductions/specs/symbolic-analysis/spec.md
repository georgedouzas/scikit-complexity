# Spec Delta

## MODIFIED Requirements

### Requirement: Fixed points and their stability are derived
The analysis layer SHALL find the fixed points of ODE systems and maps, derive their Jacobians, and classify each fixed
point's stability from its eigenvalues, also for systems whose groups of at most 64 members read reductions, which it
expands into one equation per member.

#### Scenario: Lotka-Volterra centre
- **WHEN** a user analyses prey with exponential growth rate `r`, predators with mortality `d`, and predation with
  rates `a` for the prey and `b` for the predators
- **THEN** the interior fixed point has prey `d / b` and predators `r / a`, and its eigenvalues are purely imaginary,
  classified as a centre

#### Scenario: Logistic map period one
- **WHEN** a user analyses the logistic map at r = 2.5
- **THEN** the fixed point `1 - 1 / r` is classified as stable

#### Scenario: Hawks and doves as a group
- **WHEN** a user analyses replicator dynamics of hawks and doves, two members of a group, with `V < C`
- **THEN** a fixed point has the hawk share `V / C` and is classified as stable

#### Scenario: Group above the limit
- **WHEN** a user derives the Jacobian of a group of 1000 members that reads reductions
- **THEN** a named error is raised that states the limit of 64 members and suggests analysing a smaller group
