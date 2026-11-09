## MODIFIED Requirements

### Requirement: Vector fields project onto divergence-free fields
The law utilities SHALL give the Leray projection of a vector expression, its divergence-free part, built from the
existing `gradient`, `divergence`, and `inverse_helmholtz` with no screening, in two and three dimensions on spectral
fields of periodic lattices, including the walls that the `symmetry` setting of `FieldElement` gives.

#### Scenario: Divergence-free projection
- **WHEN** `leray_projection` of a random smooth vector field is evaluated on a spectral field of a periodic 3D lattice
- **THEN** its discrete divergence is below 1e-10 times its largest component

#### Scenario: Stokes flow between walls
- **WHEN** the overdamped Stokes velocity `leray_projection(inverse_helmholtz(-f / eta))` of a uniform force along a
  channel is evaluated with the velocity odd about both walls of a symmetry-extended periodic lattice
- **THEN** the profile across the channel matches the Poiseuille parabola within 1% of its peak
