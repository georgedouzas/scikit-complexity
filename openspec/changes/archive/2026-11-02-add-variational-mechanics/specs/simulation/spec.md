# Spec Delta

## ADDED Requirements

### Requirement: A symplectic solver of fourth order
`Yoshida4Solver` SHALL integrate continuous-time systems by three velocity Verlet steps with the Yoshida coefficients,
after splitting the states into two blocks whose rates each read only the other block, a state of order 2 into its
position and velocity and states of order 1 by the sparsity of the Jacobian of the problem. A system with no such split
SHALL raise `SolverError` naming the state that prevents it and a solver that needs no split.

#### Scenario: Fourth order
- **WHEN** a harmonic oscillator written with `HamiltonianInteraction` runs to time 10 with steps of 0.1 and of 0.05
- **THEN** the error at the end shrinks by a factor between 14 and 18

#### Scenario: Non-separable Hamiltonian
- **WHEN** the restricted three-body problem, whose Hamiltonian couples coordinates and momenta in one term, is prepared
  with `Yoshida4Solver`
- **THEN** `SolverError` is raised that names a momentum whose rate reads a momentum and suggests `RK4Solver`
