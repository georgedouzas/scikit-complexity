## ADDED Requirements

### Requirement: SPH dam break
The library SHALL provide `SphDamBreakSystem` in `physics/fluids`, a two-dimensional weakly compressible SPH fluid of
fluid particles and fixed wall particles, with the continuity equation for the density, the Tait equation of state,
artificial viscosity, and gravity, whose water column of height `H` and width `2 H` collapses in a tank, as in the dam
break of 2309.16342.

#### Scenario: Surge front
- **WHEN** the system runs with its defaults
- **THEN** the front of the surge stays behind the dry-bed front `2 H + 2 sqrt(g H) t` of shallow water at every frame
  before it reaches the right wall, and it reaches that wall before `t sqrt(g / H) = 3.5`

#### Scenario: Mass and density
- **WHEN** the system runs until the surge reaches the right wall
- **THEN** no fluid particle leaves the tank, and the density of every fluid particle stays within 3% of the rest
  density

### Requirement: The SPH dam break is recorded in the inventory
The inventory SHALL record `sph_dam_break` as implemented with 2309.16342 as its source and the primitives `particles`,
`interactions`, and `particle-surrogates`, and the gallery SHALL hold its example, which compares the system with a
graph network trained on it.

#### Scenario: Inventory and gallery
- **WHEN** the inventory check and the strict gallery coverage run
- **THEN** they report no problem for `sph_dam_break`
