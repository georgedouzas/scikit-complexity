# Spec Delta

## ADDED Requirements

### Requirement: Audited physics systems
The library SHALL provide, as registered factories built from the general classes of the core, the physics systems of
the reference audit that the core supports: `nuclear_quadrupole`, `forced_toda`, and `impulsive_oscillator` in
`physics/nonlinear_oscillators`, `beam_beam_map` and `friction_oscillator` in `physics/mechanics`,
`keplerian_dumbbell` and `relativistic_orbit` in `physics/celestial_mechanics`, `forest_dieback`,
`social_forest_dieback`, `leading_following_cascade`, and `population_pollution` in `physics/climate`,
`rossler_driven_lorenz` and `adaptive_kuramoto` in `physics/synchronization`, `fixed_energy_sandpile`,
`zhang_sandpile`, and `self_organized_branching` in `physics/criticality`, `hpp_lattice_gas` in `physics/fluids`,
`inertial_active_brownian_particles`, `active_polymer`, and `deformable_active_particles` in `physics/active_matter`,
and `anisotropic_brownian_particles` and `rigid_dimers` in `physics/soft_matter`. Every one SHALL have tests of a known
result of its source, a gallery example, and an inventory entry with its sources and sections.

#### Scenario: Conserved quantities
- **WHEN** `nuclear_quadrupole`, `keplerian_dumbbell`, or `fixed_energy_sandpile` runs with its defaults
- **THEN** the quadrupole energy, the dumbbell energy and angular momentum, or the number of grains stay constant
  within the tolerance of the solver

#### Scenario: Published fixed points and orbits
- **WHEN** `impulsive_oscillator`, `relativistic_orbit`, `deformable_active_particles`, or `forest_dieback` runs from
  the states of its analytic solution
- **THEN** it follows the exact periodic orbit, the precession period, the straight or circular motion, or the root of
  its balance equation that the source derives

#### Scenario: Self-organization and synchronization
- **WHEN** `self_organized_branching`, `rossler_driven_lorenz`, or `adaptive_kuramoto` runs long enough
- **THEN** the branching probability settles at 1/2, the two driven Lorenz copies converge for a coupling above the
  critical value and not below it, and identical oscillators split into frequency clusters

#### Scenario: Statistical laws of particles
- **WHEN** `inertial_active_brownian_particles`, `active_polymer`, `anisotropic_brownian_particles`, or `rigid_dimers`
  runs
- **THEN** the projection of the velocity on the orientation, the motion of the centre of mass, the decorrelation of
  the axes and the body-frame diffusion, and the diffusion of the centre and the rod match their analytic forms

### Requirement: Variants as options of existing systems
`billiards` SHALL take the shape `'three_disk'`, the open billiard of three discs, `parametric_pendulum` SHALL take the
direction `'horizontal'` of the pivot, `aggregate_averaging_synchronization` SHALL take the Bernoulli map and a delay
of the coupling, `topological_kuramoto` SHALL take the model `'NLT'`, and `vicsek` SHALL take a number of nearest
neighbours, a vision cone, and a delay of the alignment, negative for a linear prediction. The defaults SHALL keep the
earlier behaviour of every system.

#### Scenario: Chaotic scattering
- **WHEN** a particle comes towards the centre of the three-disk table along the line through the centre of a disc
- **THEN** it bounces once and goes straight back, and other directions give several bounces

#### Scenario: Delayed synchronization
- **WHEN** two Bernoulli maps of slope 1.5 are coupled with a delay
- **THEN** they synchronize exactly for the couplings whose linear transverse map is stable, and not outside them

#### Scenario: Nodes-Links-Triangles
- **WHEN** `topological_kuramoto` runs at a weak coupling with the models `'NL'` and `'NLT'`
- **THEN** the node phases synchronize in the first and stay incoherent in the second

#### Scenario: Vicsek variants
- **WHEN** `vicsek` aligns with the 8 nearest neighbours rather than the nearest one, or with a delay of 25 steps or
  a prediction 2 steps ahead
- **THEN** the flock orders more with more neighbours, and the long and negative delays leave fewer neighbours around
  every particle than no delay

#### Scenario: Variants that do not combine
- **WHEN** `vicsek` gets both a number of nearest neighbours and a vision cone
- **THEN** a parameter error names the two options
