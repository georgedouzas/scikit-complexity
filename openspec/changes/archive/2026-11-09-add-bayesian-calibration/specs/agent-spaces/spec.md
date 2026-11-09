## ADDED Requirements

### Requirement: Delaunay neighbourhoods of continuous spaces
`DelaunayNeighborhood` SHALL be a neighbourhood of continuous spaces, next to the radius, nearest, and cone
neighbourhoods, whose neighbours of a member are the members joined to it by an edge of the Delaunay triangulation of
the positions, computed with `scipy.spatial.Delaunay`, and reductions over it SHALL run as over the others.

#### Scenario: Neighbours of a known point set
- **WHEN** the neighbours of the members of a square with points at its four corners and its centre are counted
- **THEN** the centre has four neighbours and every corner has three

### Requirement: Delaunay neighbours cross periodic boundaries
In a periodic space the triangulation SHALL cover the periodic images of the positions, so neighbours connect across
the boundary.

#### Scenario: Neighbours across a periodic boundary
- **WHEN** two members lie near opposite edges of a periodic square with no member between them
- **THEN** each is a Delaunay neighbour of the other

### Requirement: Delaunay neighbours follow moving members
The Delaunay neighbours SHALL be computed again at every step, as members move and as variable groups gain or lose
members.

#### Scenario: Neighbours follow moving members
- **WHEN** a member moves from one side of a point set to the other between two steps
- **THEN** its neighbours at the second step are those of its new position
