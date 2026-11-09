## MODIFIED Requirements

### Requirement: Mobility tensors map forces and torques to velocities

The physics library SHALL provide `StokesianMobilityInteraction` in `skcomplex.systems.physics.common.particles`, an
interaction over a group of members with a vector position, and optionally a unit vector
orientation, that sets the velocity and angular velocity of every member to the product of a grand mobility tensor with
the forces and torques on all members. The forces and torques SHALL be functions of the namespace of a member. The
tensor SHALL be the Oseen tensor, the Rotne-Prager-Yamakawa tensor, or the free-draining tensor that couples no pair, of spheres of one radius in a fluid of one
viscosity, with the self mobilities `1 / (6 pi eta a)` and `1 / (8 pi eta a^3)` and the translation-rotation couplings
of the rotlet. The Rotne-Prager-Yamakawa tensor SHALL use its regularized form for overlapping spheres. Positions in the
plane SHALL be read as positions in the plane `z = 0` of a three-dimensional fluid, with a scalar torque and angular
velocity about the normal.

#### Scenario: Stokes drag of one sphere

- **WHEN** one member of radius `a` in a fluid of viscosity `eta` feels a constant force `F`
- **THEN** it moves at the velocity `F / (6 pi eta a)`

#### Scenario: Oseen correlation of two spheres

- **WHEN** a force acts on one of two members at a distance `r` along their line of centres
- **THEN** the other member moves along the force at the velocity `F / (4 pi eta r)` of the Oseen tensor, or
  `F (1 / (4 pi eta r) - a^2 / (6 pi eta r^3))` of the Rotne-Prager-Yamakawa tensor

#### Scenario: Rotlet coupling

- **WHEN** a torque acts on one member
- **THEN** the other members move at the velocity `T x r / (8 pi eta r^3)` and rotate as the rotlet flow turns them

#### Scenario: Not in the core

- **WHEN** a user lists `skcomplex.core.components.interactions`
- **THEN** it holds no mobility interaction, and the squirmers and the coupled traps import
  `StokesianMobilityInteraction` from `skcomplex.systems.physics.common.particles`
