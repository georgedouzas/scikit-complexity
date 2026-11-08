## ADDED Requirements

### Requirement: Hyperedges form from current neighbours
The reduction `hyperedges` with `over` set to a neighbourhood of a continuous space SHALL reduce over the hyperedges
around every member formed by the member and each set of `size - 1` of its current neighbours, with the other members
in increasing member order, and SHALL equal a direct loop over those sets within 1e-12 on the NumPy and Numba backends.

#### Scenario: Three-body term against a loop
- **WHEN** a rate law of 30 members in a periodic square reads `hyperedges` of `(1 - dot(r_ij, r_ik)) / 2` with
  `size=3` over `RadiusNeighborhood(1.5)`
- **THEN** the compiled drift equals a loop over every member and every pair of its neighbours within 1e-12 on both
  backends

#### Scenario: Four-body sets
- **WHEN** a member has four neighbours within the radius and a law counts its hyperedges of `size=4`
- **THEN** the count is 4, the number of triples of its neighbours

#### Scenario: Fewer neighbours than the size
- **WHEN** a member has one neighbour within the radius and a law sums over its hyperedges of `size=3`
- **THEN** the sum is 0

### Requirement: Neighbour hyperedges follow the motion
The sets of neighbours a hyperedge reduction or coupling reads SHALL be recomputed from the positions at every
evaluation of the laws, so a member that leaves the neighbourhood leaves every set at once.

#### Scenario: Neighbour leaves the radius
- **WHEN** one of the two neighbours of a member moves out of the radius during a run
- **THEN** the 3-body term of that member is 0 from the first step after it left

### Requirement: Neighbour hyperedges need a size and a continuous space
A hyperedge reduction or coupling over a neighbourhood SHALL take a size of 2 or more and SHALL run in a continuous
space, and the compiler SHALL report an issue otherwise. A backend without the neighbour search SHALL report an issue
that names the NumPy and Numba backends.

#### Scenario: Missing size
- **WHEN** a law reads `hyperedges` over `RadiusNeighborhood(1.0)` without a size
- **THEN** a `ValidationError` asks for a size of 2 or more

#### Scenario: Neighbourhood on a network
- **WHEN** a law reads `hyperedges` over `RadiusNeighborhood(1.0)` in a network space
- **THEN** the compiler reports an issue with the code `law-space`

### Requirement: Hyperedge couplings run over current neighbours
`CouplingInteraction(mode='hyperedges')` with a `neighborhood` SHALL couple every member through the hyperedges that
the member forms with its current neighbours, with the same function, weights, and reductions as on a network.

#### Scenario: Flanking alignment
- **WHEN** a hyperedge coupling of size 3 over a radius neighbourhood adds `(1 - dot(r_ij, r_ik)) / 2 (u_ij + u_ik) / 2`
  to the acceleration of a member whose two neighbours sit on opposite sides of it
- **THEN** the member gains the mean of their relative velocities, and it gains nothing when both sit on the same side
