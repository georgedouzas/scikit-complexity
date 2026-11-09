## ADDED Requirements

### Requirement: Many-body alignment swarm
The library SHALL provide `ManyBodyAlignmentSwarmSystem` in `biology/collective_behaviour` (Gutleb 2026, arXiv
2611.08314, Eq. 2): self-propelled particles in two or three dimensions whose acceleration is the self-propulsion
`(alpha - beta |v|^2) v`, a Morse force, Cucker-Smale alignment, a 3-body flanking term weighted by `kappa_v`, and a
4-body encirclement term weighted by `kappa_4`, all under the smooth cutoff `(1 - r^2 / R^2)^2`.

#### Scenario: Pairwise collective
- **WHEN** the system runs with `kappa_v = 0` and `kappa_4 = 0`
- **THEN** its drift equals the sum of its 1-body and 2-body terms within 1e-12

#### Scenario: Body orders against the paper
- **WHEN** the drift of a fixed configuration of 10 particles is evaluated with `kappa_v = 2` and `kappa_4 = 4`
- **THEN** each body-order term equals a direct evaluation of the matching term of Eq. 2 within 1e-12

#### Scenario: Bounded motion
- **WHEN** the system runs with its defaults, 40 particles in two dimensions, for 200 time units
- **THEN** every position and velocity stays finite and every speed stays within 20% of `sqrt(alpha / beta)`

#### Scenario: Recorded in the inventory
- **WHEN** the inventory check runs
- **THEN** the system registers as `many_body_alignment_swarm` under `biology.collective_behaviour` with an
  implemented entry and a gallery example
