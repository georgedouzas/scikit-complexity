# Spec Delta

## ADDED Requirements

### Requirement: Dry active nematics
The library SHALL provide `dry_active_nematics` under the field path `physics/active_matter` as a registered factory
built only from public names, a `FieldElement` on a lattice under continuous time: a conserved density coupled to a
symmetric traceless alignment tensor, with an active current along the divergence of the tensor (1207.2929 §II.2). The
inventory SHALL list it as implemented.

#### Scenario: Dry active nematic conserves density
- **WHEN** `dry_active_nematics` runs on a periodic lattice for 100 time units
- **THEN** the total density stays constant within 1e-10 relative error

### Requirement: Active nematic and polar gels
The library SHALL provide `active_gels` under the field path `physics/active_matter` as a registered factory: a
symmetric traceless tensor, with `order='nematic'`, or a polarisation vector, with `order='polar'`, transported by its
co-rotational derivative in an overdamped Stokes flow driven by the elastic and active stresses (1207.2929 §III), with
the Landau-de Gennes energy in its own package. The inventory SHALL list it as implemented.

#### Scenario: Passive nematic order
- **WHEN** `active_gels` runs with `order='nematic'` at zero activity from a uniform tensor with Landau coefficients
  `a < 0` and `c`
- **THEN** the velocity stays below 1e-8, and the scalar order relaxes to `sqrt(-2 a / c)` within 1e-6

#### Scenario: Active turbulence conserves topological charge
- **WHEN** the extensile nematic gel runs on a periodic 2D lattice from a nearly aligned state with an activity well
  above the bend instability threshold
- **THEN** `DefectAnalysis` finds pairs of `+1/2` and `-1/2` defects, the total charge stays 0 at every frame, and the
  mean kinetic energy settles at a nonzero value

#### Scenario: Spontaneous flow of a polar film
- **WHEN** the polar gel runs in a film between walls of fixed polarisation, given by the `symmetry` setting, at
  activities 20% below and 20% above the threshold of the linear stability analysis
- **THEN** the flow decays below the threshold and grows to a steady nonzero flow above it
