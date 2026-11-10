## ADDED Requirements

### Requirement: Particle life
The library SHALL provide `ParticleLifeSystem` in `computer_science/automata`: particles of `S` species in a periodic
square move under friction and a force between pairs that repels within a core radius and, beyond it, attracts or
repels with the strength of an entry of an asymmetric `S` by `S` matrix and a triangular profile up to a cutoff. It is
one of the substrates of Kumar et al. (2024, arXiv 2412.17799).

#### Scenario: No interaction beyond the core
- **WHEN** the system runs with a matrix of zeros from a uniform start
- **THEN** the nearest-neighbour distances stay at least the core radius times 0.5 and no cluster forms

#### Scenario: Chasing species
- **WHEN** two species run with the matrix `[[0, 1], [-1, 0]]`
- **THEN** the sum of the pair forces over all particles is not zero, while it is zero within 1e-12 for a symmetric
  matrix

### Requirement: Particle life is recorded in the inventory
The inventory SHALL record `particle_life` as implemented with the source 2412.17799 and the primitives it uses, and
the gallery SHALL hold its example, which searches the matrix with `diversity_search`.

#### Scenario: Inventory and gallery
- **WHEN** the inventory check and the strict gallery coverage run
- **THEN** they report no problem for `particle_life`
