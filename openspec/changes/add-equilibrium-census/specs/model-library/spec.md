## ADDED Requirements

### Requirement: Relaxation under a random force field
The library SHALL provide `RandomForceFieldRelaxationSystem` in `physics/statistical_mechanics` (Lacroix-A-Chez-Toine,
Le Doussal, and Ros 2026, arXiv 2611.08324): a group of units with `dx_i/dt = -mu x_i + F_i(x)`, where `F` is a
quenched Gaussian random force field drawn from its `random_state`, the sum of a gradient part and a divergence-free
part weighted by the reciprocity `tau`, scaled by the rescaled disorder `gamma`.

#### Scenario: Gradient field
- **WHEN** the system is drawn with `tau = 1`
- **THEN** the Jacobian of `F` at random points is symmetric within 1e-8

#### Scenario: Divergence-free field
- **WHEN** the system is drawn with `tau = 0`
- **THEN** the divergence of `F` at random points is 0 within 1e-8

#### Scenario: Relaxation without disorder
- **WHEN** the system runs with `gamma = 0`
- **THEN** every unit decays to 0 at the rate `mu` within 1e-6

#### Scenario: Recorded in the inventory
- **WHEN** the inventory check runs
- **THEN** the system registers as `random_force_field_relaxation` under `physics.statistical_mechanics` with an
  implemented entry and a gallery example
