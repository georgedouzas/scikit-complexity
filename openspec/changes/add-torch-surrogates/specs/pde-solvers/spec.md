## ADDED Requirements

### Requirement: Physics-informed networks solve fields
The physics-informed solver SHALL solve a system of fields with rate laws on a rectangular lattice in one or two
dimensions over the time of the run, by training a network on the residuals of the declared laws, the initial fields,
and the boundary conditions, and SHALL return the network solution on the lattice at the frames of the run.

#### Scenario: Heat equation
- **WHEN** a one-dimensional heat diffusion system runs to time 0.5 with the physics-informed solver and with the
  spectral solver
- **THEN** the relative L2 difference of the fields at the final time is below 0.02

#### Scenario: Allen-Cahn benchmark
- **WHEN** the one-dimensional Allen-Cahn system of the PINN benchmark runs to time 1 with Fourier features
- **THEN** the relative L2 error against the spectral solution at the final time is below 0.05

### Requirement: Physics-informed residuals come from the declared laws
The physics-informed solver SHALL build its residuals from the symbolic rate laws of the system, with spatial
derivatives and Laplacians taken by automatic differentiation of the network, so no law is written twice, and SHALL
reject before training any law it cannot express.

#### Scenario: Unsupported law
- **WHEN** a field system with a noise law is given to the physics-informed solver
- **THEN** it raises `SolverError` before training, whose message names the solver and the noise law

#### Scenario: Periodic boundary
- **WHEN** a field on a periodic lattice runs with the physics-informed solver
- **THEN** the solution at both ends of every periodic axis agrees within 1e-3 at every frame
