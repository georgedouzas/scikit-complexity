## MODIFIED Requirements

### Requirement: Collisions evaluate the binary collision integral
`EncounterInteraction` with a `cross_section` SHALL evaluate the binary collision integral of a declared cross-section over the interval class
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
- **WHEN** an `EncounterInteraction` with a `cross_section` targets a field on a position lattice with no class axis
- **THEN** the compile report holds a `law-space` issue whose fix declares an interval `AxisSpec`
