## MODIFIED Requirements

### Requirement: Lattices define sites and ordered neighbourhoods
The modelling layer SHALL define lattices of any dimension with periodic, fixed, or reflecting edges, and neighbourhoods
as ordered offsets. One ball neighbourhood SHALL give the offsets within a radius under a named metric, Chebyshev,
Manhattan, or hexagonal, with the site itself left out, put first, or kept in its lexicographic place. The Chebyshev and
Manhattan offsets SHALL come in lexicographic order, and the hexagonal offsets ring by ring, counterclockwise. A lattice
space SHALL name the metric of the neighbourhood agents query, and SHALL take no second name for it.

#### Scenario: Moore neighbourhood on a torus
- **WHEN** a user asks for the Chebyshev ball of radius 1 around the corner cell of a periodic 10 × 10 lattice
- **THEN** the result holds the eight cells around it, wrapping across both edges

#### Scenario: Von Neumann neighbourhood
- **WHEN** a user asks for the Manhattan ball of radius 1 in two dimensions
- **THEN** the offsets are `(-1, 0)`, `(0, -1)`, `(0, 1)`, and `(1, 0)`, in that order

#### Scenario: Wolfram order
- **WHEN** a user asks for the Chebyshev ball of radius 1 in one dimension with the site kept in its place
- **THEN** the offsets are `(-1,)`, `(0,)`, and `(1,)`, the order of Wolfram's rule numbers

#### Scenario: Hexagonal ball
- **WHEN** a user asks for the hexagonal ball of radius 1
- **THEN** the offsets are the six triangular neighbours counterclockwise from the direction of the second axis, so
  the offset `k + 3` modulo 6 is opposite to the offset `k`

#### Scenario: Former neighbourhoods are gone
- **WHEN** a user imports `MooreNeighborhood`, `VonNeumannNeighborhood`, `BoxNeighborhood`, `TriangularNeighborhood`,
  or `GeneratorNeighborhood` from `skcomplex.core.components.spaces`
- **THEN** the import fails with `ImportError`, since the project ships no alias for a removed class

## ADDED Requirements

### Requirement: Groups on sites read values in the shape of their space
A group on the sites of a lattice or of a network SHALL be a `GroupElement` on `'sites'`, with no second element class
for it. It SHALL take a value in the shape of the lattice, or a mapping from node to value on a network, as one row per
site, and SHALL take the default visual of its space when its members declare none.

#### Scenario: Lattice-shaped values
- **WHEN** a user builds a group on the sites of a 3 × 4 lattice with a state given as a 3 × 4 array
- **THEN** the prepared group has 12 members and the state has one row per site, in row-major order

#### Scenario: Node mapping on a network
- **WHEN** a user builds a group on the sites of a network with a state given as a mapping from node to value
- **THEN** the state has one row per node, in the order of the nodes of the graph
