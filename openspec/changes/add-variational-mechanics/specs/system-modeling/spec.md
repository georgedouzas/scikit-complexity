# Spec Delta

## ADDED Requirements

### Requirement: Hamiltonians and Lagrangians generate laws
An interaction SHALL declare a Hamiltonian over coordinates and momenta, or a Lagrangian over coordinates and
velocities, and the framework SHALL derive its rate laws symbolically.

#### Scenario: Pendulum from a Lagrangian
- **WHEN** a pendulum declares `L = m l² ω² / 2 + m g l cos θ` with an angle coordinate `θ`
- **THEN** the derived equation is `dω/dt = -(g / l) sin θ`

#### Scenario: Double pendulum chaos
- **WHEN** a double pendulum declared by its Lagrangian runs from two initial angles that differ by 1e-9
- **THEN** the trajectories separate exponentially, and the Hamiltonian stays constant within 1e-6 relative drift with
  the Yoshida stepper

### Requirement: Symmetries are declared and checked
A system SHALL declare symmetries as infinitesimal transformations of its coordinates, which the framework checks
against its Lagrangian or Hamiltonian.

#### Scenario: Rotation invariance
- **WHEN** the Kepler problem declares rotation symmetry in the plane
- **THEN** the check confirms invariance, and the Noether charge equals the angular momentum
