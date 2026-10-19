# Spec Delta

## ADDED Requirements

### Requirement: Dry active nematic
The library SHALL provide the dry active nematic under the field path `physics/active_matter` as a registered factory
built only from the public names of the core: a conserved density coupled to a symmetric traceless alignment tensor,
with an active current along the divergence of the tensor (1207.2929 §II.2). The inventory SHALL list it as
implemented.

#### Scenario: Dry active nematic conserves density
- **WHEN** the dry active nematic runs on a periodic lattice for 100 time units
- **THEN** the total density stays constant within 1e-10 relative error

### Requirement: Active nematic gel
The library SHALL provide the active nematic gel under the field path `physics/active_matter` as a registered factory:
a symmetric traceless alignment tensor transported by its co-rotational derivative in an incompressible Stokes flow
driven by the elastic and active stresses (1207.2929 §III). The inventory SHALL list it as implemented.

#### Scenario: Passive nematic order
- **WHEN** the active nematic gel runs at zero activity from a uniform tensor with Landau coefficients `a < 0` and `c`
- **THEN** the velocity stays below 1e-8, and the scalar order relaxes to `sqrt(-2 a / c)` within 1e-6

#### Scenario: Active turbulence conserves topological charge
- **WHEN** the extensile active nematic gel runs on a periodic 2D lattice from a nearly aligned state with an activity
  well above the bend instability threshold
- **THEN** pairs of `+1/2` and `-1/2` defects appear, the total charge of all defects stays 0 at every frame, and the
  mean kinetic energy settles at a nonzero value

### Requirement: Active polar gel
The library SHALL provide the active polar gel under the field path `physics/active_matter` as a registered factory:
a polarisation vector transported by its co-rotational derivative in an incompressible Stokes flow driven by the active
stress `ζ p p` (1207.2929 §III). The inventory SHALL list it as implemented.

#### Scenario: Spontaneous flow of a polar film
- **WHEN** the active polar gel runs in a film between walls with fixed polarisation, at activities 20% below and 20%
  above the threshold of the linear stability analysis
- **THEN** the flow decays below the threshold and grows to a steady nonzero flow above it
