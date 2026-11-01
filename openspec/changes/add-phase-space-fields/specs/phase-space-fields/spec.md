# Spec Delta

## Purpose

Lets a user write densities over position and class axes such as velocity or orientation, transport them through
phase space, let them encounter and collide over their class axes, and read their moments as fields over position.

## ADDED Requirements

### Requirement: Phase spaces are products of position and class axes
`PhaseSpace` SHALL be the product of a position lattice and one or more class axes, each a uniform grid over an
interval with zero-flux edges or a periodic grid. It SHALL be a lattice space, so a `FieldElement` of `DensityElement`
on it holds one value per point of the product, and its laws are written for one point. The compile report SHALL hold
a phase space larger than its size limit, with the size it would take.

#### Scenario: Size of a phase space
- **WHEN** a user creates a field over a periodic position lattice of 64 sites and a class axis of 128 points on
  `[-6, 6]`
- **THEN** the field holds 8192 values, and its initial values can be given as an array over position and class

#### Scenario: Periodic class axis
- **WHEN** a field over a periodic class axis of 32 angles is advected along that axis by a constant drift for one full
  turn
- **THEN** the profile returns to its initial values within the discretization error of the scheme

### Requirement: Laws read class coordinates
Laws over a phase space SHALL read the coordinate of the point they update along a named class axis, so drifts, rates,
kernels, and initial values can depend on it. The compile report SHALL hold a coordinate read in a law whose group is
not on a phase space, or of an axis the space does not hold.

#### Scenario: Maxwellian start
- **WHEN** a user sets the initial value to `exp(-v**2 / 2) / sqrt(2 * pi)` with `v` the coordinate of the class axis
- **THEN** the density moment equals 1 at every position within the quadrature error of the grid

### Requirement: Advection streams and accelerates phase-space fields
`AdvectionInteraction` SHALL carry a phase-space field along the position axes at a drift that reads the class
coordinates, and along the class axes at a drift that depends on position, class, time, or other fields, in the
field's discretization. Transport SHALL conserve the total on periodic axes and on axes whose edges carry no inflow.

#### Scenario: Free streaming is exact
- **WHEN** a field without collisions streams on a periodic position lattice for time `t`
- **THEN** every class slice equals its initial profile shifted by `v t`, within the error of the scheme, and the
  total stays constant within 1e-12 relative error

#### Scenario: Zero drift leaves classes unchanged
- **WHEN** a spatially uniform field is advected along its class axis by a drift of zero
- **THEN** the field does not change

### Requirement: Moments over class axes are fields over position
`moment` with `axes` SHALL reduce a phase-space field over the named class axes into a field over position, by a
quadrature that is exact for the discrete conservation laws, which laws read and observables reduce.

#### Scenario: Moments of a shifted Maxwellian
- **WHEN** a field is initialised to a Maxwellian of density 2, mean velocity 0.5, and temperature 1.5 on a grid wide
  enough to hold its tails
- **THEN** the normalized and central moments return 2, 0.5, and 1.5 within 1e-8 at every position

### Requirement: Encounters run at every position
An `EncounterInteraction` on phase-space fields SHALL pair classes at the same position over the class axes. Without a
field it SHALL redistribute the candidate linearly at a rate per class with a kernel over destination classes. Both
SHALL conserve the density at every position in the conservative outcome.

#### Scenario: Linear redistribution erases polarization
- **WHEN** a spatially uniform field over a periodic angle axis with polarization 0.8 evolves by a linear encounter
  whose kernel is a wrapped normal noise
- **THEN** the density stays constant within 1e-12 relative error, and the polarization decays exponentially at the
  rate given by the first Fourier coefficient of the noise

### Requirement: Relaxation towards the maximum-entropy equilibrium conserves moments
`maximum_entropy` SHALL return the density on the class axes with the largest entropy whose discrete moments up to its
order equal those of the expression at every position, so a `RelaxationInteraction` towards it conserves those
moments to round-off.

#### Scenario: Conservation under BGK
- **WHEN** a spatially uniform field starting from two separated Maxwellians relaxes towards its maximum-entropy
  equilibrium for 50 relaxation times
- **THEN** the density, momentum, and energy stay constant within 1e-12 relative error, the field converges to the
  Maxwellian with those moments, and the discrete entropy `-Σ f log f` never decreases

### Requirement: Collisions evaluate the binary collision integral
`CollisionInteraction` SHALL evaluate the binary collision integral of a declared cross-section over the interval class
axes of a phase space. It SHALL conserve density exactly and momentum and energy within the spectral accuracy of the
method, and SHALL accept a factor of the local density that scales the collision frequency. The compile report SHALL
hold a collision on a phase space without interval class axes.

#### Scenario: BKW solution
- **WHEN** a spatially homogeneous field of Maxwell molecules in two velocity dimensions starts from the
  Bobylev-Krook-Wu profile and collides
- **THEN** the field matches the exact BKW solution within 1e-4 in the maximum norm at time 5, and its density stays
  constant within 1e-12 relative error

#### Scenario: Density factor speeds relaxation
- **WHEN** the same field collides once with a density factor of 2 and once without
- **THEN** the run with the factor reaches every intermediate state in half the time, within 1%

#### Scenario: Collision without velocity axes
- **WHEN** a `CollisionInteraction` targets a field on a position lattice with no class axis
- **THEN** the compile report holds a `law-space` issue whose fix declares an interval `AxisSpec`

### Requirement: Phase-space fields couple with position fields
A drift over a phase space SHALL read a moment of its own field and a field over the position lattice through its
`sources`, broadcast along the class axes, and a rate over the position lattice SHALL read a moment of a phase-space
field, so a force can be solved from a moment every step.

#### Scenario: Landau damping
- **WHEN** a Vlasov-Poisson field starts from a Maxwellian with density `1 + 0.01 cos(0.5 x)` on a periodic domain
  of length `4 π`, with the electric field solved from the density moment in its drift
- **THEN** the electric energy oscillates and decays at the linear Landau damping rate 0.1533 within 5%, up to time 30

### Requirement: Transport and collisions split in time
`SemiLagrangianSolver` SHALL advance fields whose fluxes are linear drifts by shifting each slice along its
characteristics, and `StrangSplittingSolver` SHALL alternate transport and collision steps with second-order accuracy
in the time step. Both SHALL declare their support, and the method-of-lines solvers SHALL also accept phase-space
fields.

#### Scenario: Large transport steps stay stable
- **WHEN** a field without collisions streams with `SemiLagrangianSolver` at a Courant number of 5
- **THEN** the run stays bounded, and the total stays constant within 1e-12 relative error

#### Scenario: Second order splitting
- **WHEN** a BGK field runs with `StrangSplittingSolver` at steps `h` and `h / 2`
- **THEN** the error against a reference run at step `h / 64` shrinks by a factor between 3.5 and 4.5

#### Scenario: Nonlinear flux
- **WHEN** `SemiLagrangianSolver` prepares on Burgers' equation
- **THEN** it raises a named error saying that the flux of the field is not a linear drift

### Requirement: Phase-space fields stream to Rerun
`ImageVisual` SHALL log declared two-dimensional slices of a phase-space field, such as the position-velocity plane,
and its moments over the class axes as series or images over the position lattice.

#### Scenario: Two-stream instability in the viewer
- **WHEN** a Vlasov-Poisson two-stream model streams to Rerun
- **THEN** the recording holds the position-velocity image of the field and the density over position, both updated
  every frame
