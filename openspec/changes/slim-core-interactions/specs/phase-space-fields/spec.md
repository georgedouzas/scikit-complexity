## MODIFIED Requirements

### Requirement: Advection streams and accelerates phase-space fields
`ConservationInteraction` with the flux of a drift times the field SHALL carry a phase-space field along the position
axes at a drift that reads the class coordinates, and along the class axes at a drift that depends on position, class,
time, or other fields, in the field's discretization. Transport SHALL conserve the total on periodic axes and on axes
whose edges carry no inflow. Physics SHALL name this flux in its `AdvectionInteraction` of
`skcomplex.systems.physics.common.fields`.

#### Scenario: Free streaming is exact
- **WHEN** a field without collisions streams on a periodic position lattice for time `t`
- **THEN** every class slice equals its initial profile shifted by `v t`, within the error of the scheme, and the
  total stays constant within 1e-12 relative error

#### Scenario: Zero drift leaves classes unchanged
- **WHEN** a spatially uniform field is advected along its class axis by a drift of zero
- **THEN** the field does not change

#### Scenario: Same trajectory as the physics realization
- **WHEN** a field streams once by `ConservationInteraction` with the flux `a f` and once by the physics
  `AdvectionInteraction` with the drift `a`
- **THEN** both trajectories are equal

### Requirement: Relaxation towards the maximum-entropy equilibrium conserves moments
`maximum_entropy` SHALL return the density on the class axes with the largest entropy whose discrete moments up to its
order equal those of the expression at every position, so a `RateInteraction` with the rate `(equilibrium - f) / time`
towards it conserves those moments to round-off.

#### Scenario: Conservation under BGK
- **WHEN** a spatially uniform field starting from two separated Maxwellians relaxes towards its maximum-entropy
  equilibrium for 50 relaxation times
- **THEN** the density, momentum, and energy stay constant within 1e-12 relative error, the field converges to the
  Maxwellian with those moments, and the discrete entropy `-Σ f log f` never decreases
