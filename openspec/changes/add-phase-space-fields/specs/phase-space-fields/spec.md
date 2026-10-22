# Spec Delta

## Purpose

Lets a user write fields over position and velocity or orientation, transport them through phase space, relax and
collide them in velocity, and read their velocity moments as fields over position.

## ADDED Requirements

### Requirement: Phase spaces are products of position and velocity axes
The modelling layer SHALL define phase spaces as the product of a position lattice and one or more velocity axes, each
a uniform grid over a bounded interval, or orientation axes, each a periodic grid of angles. A phase space SHALL be a
lattice space, so a group on its sites, a phase-space field, SHALL hold one value per point of the product, and its
laws SHALL be written for one point. Validation SHALL report a phase space larger than the declared size limit, with
the size it would take.

#### Scenario: Size of a phase space
- **WHEN** a user creates a field over a periodic position lattice of 64 sites and a velocity axis of 128 points on
  `[-6, 6]`
- **THEN** the field holds 8192 values, and its initial values can be given as a function of position and velocity

#### Scenario: Orientation axis is periodic
- **WHEN** a field over an orientation axis of 32 angles is transported along that axis by a constant angular speed
  for one full turn
- **THEN** the profile returns to its initial values within the discretization error of the scheme

### Requirement: Laws read phase coordinates
Laws over a phase space SHALL read the velocity components and the orientation of the point they update as
expressions, so rates and initial values can depend on them.

#### Scenario: Maxwellian start
- **WHEN** a user sets the initial value to `exp(-v**2 / 2) / sqrt(2 * pi)` with `v` the velocity coordinate
- **THEN** the density moment equals 1 at every site within the quadrature error of the velocity grid

### Requirement: Transport operators stream and accelerate
Laws SHALL use free streaming along the position axes at the velocity of each point, and acceleration along the
velocity axes by a force that depends on position, velocity, time, or other fields. Both SHALL stay symbolic in the
equations and lower to the field's discretization. Transport SHALL conserve the total of the field on periodic
position axes and on velocity axes whose edges carry no inflow.

#### Scenario: Free streaming is exact
- **WHEN** a collisionless field streams on a periodic position lattice for time `t`
- **THEN** every velocity slice equals its initial profile shifted by `v t`, within the error of the scheme, and the
  total stays constant within 1e-12 relative error

#### Scenario: Zero force leaves velocity unchanged
- **WHEN** a spatially uniform field is accelerated by a force of zero
- **THEN** the field does not change

### Requirement: Velocity moments are fields over position
Laws, and the observables of simulations and catalog entries, SHALL read the moments of a phase-space field over its
velocity or orientation axes as reductions: the density,
the mean velocity or polarization, and the temperature or second angular moment. A moment SHALL be a field over the
position lattice, computed by a quadrature over the velocity axes that is exact for the discrete conservation laws.

#### Scenario: Moments of a shifted Maxwellian
- **WHEN** a field is initialised to a Maxwellian of density 2, mean velocity 0.5, and temperature 1.5 on a velocity
  grid wide enough to hold its tails
- **THEN** the density, mean velocity, and temperature moments return 2, 0.5, and 1.5 within 1e-8 at every site

### Requirement: BGK collisions relax toward the local equilibrium
A BGK collision law SHALL relax a phase-space field toward the Maxwellian built from its own local moments at a
declared rate, which may depend on the moments. The discrete equilibrium SHALL carry exactly the density, momentum,
and energy of the field at every site, so collisions conserve them to round-off.

#### Scenario: Conservation under BGK
- **WHEN** a spatially uniform field starting from two separated Maxwellians relaxes by BGK collisions for 50 collision
  times
- **THEN** the density, momentum, and energy stay constant within 1e-12 relative error, the field converges to the
  Maxwellian with those moments, and the discrete entropy `-Σ f log f` never decreases

### Requirement: Boltzmann collisions use a collision kernel
A Boltzmann collision law SHALL evaluate the binary collision integral of a declared collision kernel over the velocity
axes. It SHALL conserve density exactly and momentum and energy within the spectral accuracy of the method, and SHALL
accept a factor of the local density that scales the collision frequency, for the local Enskog equation.

#### Scenario: BKW solution
- **WHEN** a spatially homogeneous field of Maxwell molecules in two velocity dimensions starts from the
  Bobylev-Krook-Wu profile and relaxes by Boltzmann collisions
- **THEN** the field matches the exact BKW solution within 1e-4 in the maximum norm at time 5, and its density stays
  constant within 1e-12 relative error

#### Scenario: Enskog factor speeds relaxation
- **WHEN** the same field relaxes once with an Enskog density factor of 2 and once without
- **THEN** the run with the factor reaches every intermediate state in half the time, within 1%

### Requirement: Angular collisions act on orientation axes
Collision laws on orientation axes SHALL model self-diffusion, where a particle's orientation jumps at a declared rate
by an angle drawn from a declared noise distribution, and binary alignment, where two particles at the same position
collide at a rate set by their relative orientation and leave with their mean orientation plus noise. Both SHALL
conserve the density at every site.

#### Scenario: Self-diffusion erases polarization
- **WHEN** a spatially uniform field with polarization 0.8 evolves by self-diffusion alone
- **THEN** the density stays constant within 1e-12 relative error, and the polarization decays exponentially at the
  rate given by the first Fourier coefficient of the noise distribution

### Requirement: Phase-space fields couple with position fields
A law over a phase space SHALL read a field over its position lattice as a value broadcast along the velocity axes, and
a law over the position lattice SHALL read a moment of a phase-space field, so a force can be solved from a moment
every step.

#### Scenario: Landau damping
- **WHEN** a Vlasov-Poisson field starts from a Maxwellian with density `1 + 0.01 cos(0.5 x)` on a periodic domain
  of length `4 π`, with the electric field solved from the density every step
- **THEN** the electric energy oscillates and decays at the linear Landau damping rate 0.1533 within 5%, up to time 30

### Requirement: Transport and collisions split in time
The solvers SHALL advance a phase-space field by a semi-Lagrangian transport step that shifts each slice along its
characteristics, and SHALL alternate transport and collision steps by Strang splitting, with second-order accuracy in
the time step. The method-of-lines steppers SHALL also accept phase-space fields.

#### Scenario: Large transport steps stay stable
- **WHEN** a collisionless field streams with the semi-Lagrangian step at a Courant number of 5
- **THEN** the run stays bounded, and the total stays constant within 1e-12 relative error

#### Scenario: Second order splitting
- **WHEN** a BGK field runs with Strang splitting at steps `h` and `h / 2`
- **THEN** the error against a reference run at step `h / 64` shrinks by a factor between 3.5 and 4.5

### Requirement: Phase-space fields stream to Rerun
Phase-space fields SHALL stream to Rerun as images of declared two-dimensional slices, such as the position-velocity
plane, and their moments SHALL stream as images or series over the position lattice.

#### Scenario: Two-stream instability in the viewer
- **WHEN** a Vlasov-Poisson two-stream model streams to Rerun
- **THEN** the viewer shows the position-velocity image of the field and the density over position, both updating
  every frame

### Requirement: Phase-space laws declare their requirements
The collision laws SHALL be of the law kind `rate` and SHALL declare that they require continuous time, a lattice
space, and real states. Validation SHALL report a collision law or transport operator whose group lies on a lattice
without the axes it needs, with the law's path and a fix.

#### Scenario: Missing velocity axis
- **WHEN** a BGK law targets a site group on a position lattice with no velocity axis
- **THEN** validation reports an issue whose path is the law and whose fix declares a `PhaseSpace` with a
  `VelocityAxis`

### Requirement: Phase-space steppers declare their support
The semi-Lagrangian and Strang splitting steppers SHALL declare that they support continuous time, lattice spaces, site
groups, real states, and the law kinds they advance. Validation SHALL report either stepper chosen for a system it does
not support.

#### Scenario: Unsupported stepper
- **WHEN** a system whose site groups live on a plain lattice is validated with the semi-Lagrangian stepper
- **THEN** validation reports that the stepper needs a phase space and lists the steppers that support the system
