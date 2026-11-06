# Spec Delta

## Purpose

Couple the members of a group through a Stokes fluid, turning forces, torques, and active slip into velocities and
angular velocities by a mobility tensor, with Brownian displacements consistent with that tensor.

## ADDED Requirements

### Requirement: Mobility tensors map forces and torques to velocities

The core SHALL provide an interaction over a group of members with a vector position, and optionally a unit vector
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

### Requirement: Active slip couples through the mobility

The interaction SHALL add to every member an intrinsic velocity given as a function of its namespace, and SHALL carry
to every other member the flow and half the vorticity of a stresslet and of a potential dipole of every member, each a
function of its namespace. Pair forces of a potential of the distance within a cutoff SHALL enter the mobility like
external forces. A member SHALL not feel its own stresslet or dipole.

#### Scenario: Stresslet flow of a puller

- **WHEN** a member carries the stresslet `sigma (e e - I / 3)` with `sigma > 0`
- **THEN** a member at a distance `r` along `e` moves towards it at the speed `sigma / (4 pi eta r^2)`

#### Scenario: Orientations rotate with the angular velocity

- **WHEN** members carry an orientation state
- **THEN** every orientation turns at the rate `Omega x p` of its angular velocity and stays a unit vector

### Requirement: Brownian displacements consistent with the mobility

In discrete time the interaction SHALL give the Ermak-McCammon step `x + M F dt + sqrt(2 kT dt) B xi` of a step `dt`
at the temperature `kT`, where `B B^T` is the grand mobility, drawn from the random stream of the run, and the
orientations SHALL turn by the angular part of the step and be normalized. The square root SHALL be a Cholesky factor
or a Lanczos approximation of the matrix square root applied to the normal vector. In continuous time a positive
temperature SHALL be rejected with a message that names the discrete form.

#### Scenario: Equilibrium covariance

- **WHEN** two members in harmonic traps of stiffness `k` run at the temperature `kT` with either square root
- **THEN** the variance of every coordinate approaches `kT / k`, and the displacements of the free members over a short
  time correlate as the mobility tensor predicts

#### Scenario: Oseen tensor that is not positive

- **WHEN** members of the Oseen tensor overlap so that the mobility is not positive definite and noise is drawn
- **THEN** the run raises a validation error that suggests the Rotne-Prager-Yamakawa tensor

#### Scenario: Unknown variants

- **WHEN** the tensor, the form, or the square root is not one of the names of the module
- **THEN** preparing the interaction raises a validation error that lists the valid names
