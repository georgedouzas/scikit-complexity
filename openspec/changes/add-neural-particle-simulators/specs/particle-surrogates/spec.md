# Spec Delta

## Purpose

Lets a user learn a runnable particle system from particle trajectories, as a graph network of one step, a symplectic
Hamiltonian network, or a force-matched potential of coarse-grained beads.

## ADDED Requirements

### Requirement: Graph networks learn one-step particle updates
`graph_network(source, radius=...)` SHALL train a graph network on the particle trajectories of `source`, over the
graph of the pairs closer than `radius` at every frame, across periodic sides, to predict the acceleration of every
particle from its recent velocities, its kind, and the displacements to its neighbours, and SHALL return a frozen result
with the network, the training and validation errors, and a rollout `forecast`.

#### Scenario: Dam break rollout
- **WHEN** the graph network is trained on runs of `SphDamBreakSystem` and rolled out for 20 frames from a held-out
  frame
- **THEN** the mean squared position error of the rollout is below that of particles kept at constant velocity

#### Scenario: Not a particle trajectory
- **WHEN** the source holds no particle or agent group with positions
- **THEN** `AnalysisError` is raised and says that positions are needed

### Requirement: Learned graph networks are systems
The result of `graph_network` SHALL hold `system`, a system of one particle group under discrete time whose update moves
every particle by the predicted acceleration with semi-implicit Euler steps, and that system SHALL run through
`simulate` and `sweep` and be read by the agent analyses like any particle system.

#### Scenario: Forecast of the system
- **WHEN** the `system` of the result runs from the frame the `forecast` starts at
- **THEN** its positions equal the forecast within 1e-10

### Requirement: Hamiltonian networks learn symplectic many-body laws
`hamiltonian_network(source, cutoff=...)` SHALL learn the potential of a separable Hamiltonian, the kinetic energy plus
a sum of a network over the distances of the pairs within `cutoff`, by matching velocity Verlet steps to the
trajectories, and SHALL return a frozen result with the network, the potential on a grid of distances, and the errors.
The potential SHALL be invariant under permutations of the particles and rotations of the frame.

#### Scenario: Lennard-Jones potential recovered
- **WHEN** the network is trained on a trajectory of `LennardJonesFluidSystem`
- **THEN** the learned pair potential agrees with the Lennard-Jones potential within 5% of the well depth between
  `0.95 σ` and the cutoff

#### Scenario: Invariance
- **WHEN** the learned energy is evaluated on a configuration, on the same configuration rotated, and with its particles
  permuted
- **THEN** the three values agree within 1e-10

### Requirement: Learned Hamiltonians run symplectically
The result of `hamiltonian_network` SHALL hold `system`, a particle group under continuous time whose accelerations are
minus the gradient of the learned potential over the masses, which the velocity Verlet and fourth-order symplectic
solvers accept.

#### Scenario: Bounded energy error
- **WHEN** the `system` of a network trained on a chain of masses and springs runs for 10,000 steps with
  `VelocityVerletSolver`
- **THEN** the learned total energy stays within 1e-3 of its initial value relative to it, without drift

### Requirement: Force matching fits coarse-grained potentials
`force_matching(source, forces=..., mapping=...)` SHALL map the particle positions and forces of every frame to beads,
and SHALL fit a pair potential of the beads to the mapped forces, as a linear combination of radial basis functions by
least squares or as a network, and SHALL return a frozen result with the coefficients or the network, the potential on
a grid of distances, the relative force error, and a `system` of the beads with the fitted potential.

#### Scenario: Identity mapping
- **WHEN** force matching runs with the identity mapping on frames and exact forces of a Lennard-Jones fluid
- **THEN** the fitted potential agrees with the Lennard-Jones potential within 2% of the well depth between `0.95 σ`
  and the cutoff

#### Scenario: Dimers to beads
- **WHEN** a fluid of dimers bound by stiff springs is mapped to the centres of mass of its dimers and force matched
- **THEN** the `system` of the result reproduces the first peak of the radial distribution of the mapped centres within
  5% of its position

#### Scenario: Mismatched forces
- **WHEN** the forces differ in shape from the positions of the source
- **THEN** `AnalysisError` is raised and names both shapes
