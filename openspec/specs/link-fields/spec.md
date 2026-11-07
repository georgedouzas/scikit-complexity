# link-fields Specification

## Purpose

Lets a user place groups on the links and plaquettes of a lattice or on the edges and faces of a network, write their
laws with discrete exterior calculus and reductions over the faces of a link, sample angle-valued link states by Monte
Carlo, and analyse and draw the result.

## Requirements

### Requirement: Lattices have cell complexes
`CellComplexSpace` SHALL derive from a `LatticeSpace` a network space whose nodes are its sites, whose edges are its
links oriented along the positive direction of each axis, and whose hyperedges are its plaquettes oriented by pairs of
axes in increasing order, in cyclic node order, with the oriented incidence matrix between every two consecutive
degrees.

#### Scenario: Periodic square lattice
- **WHEN** a user prepares the `CellComplexSpace` of a periodic 8 × 8 lattice
- **THEN** it has 64 sites, 128 links, and 64 plaquettes, and the product of the link-to-plaquette incidence matrix and
  the site-to-link incidence matrix is zero

#### Scenario: Fixed edges
- **WHEN** a user prepares the `CellComplexSpace` of a 4 × 4 lattice with fixed edges
- **THEN** it has 16 sites, 24 links, and 9 plaquettes, with no cell that leaves the box

### Requirement: Networks have cell complexes
`CellComplexSpace` SHALL derive from a `NetworkSpace` a network space whose faces are given node cycles, the hyperedges
of the network, the cycles of a minimum cycle basis, or the cliques of the network up to a chosen dimension, each
oriented by its node order.

#### Scenario: Faces of a network
- **WHEN** a user prepares the `CellComplexSpace` of a 4-node cycle graph with one chord and no faces given
- **THEN** it has 5 edges and 2 faces from the minimum cycle basis, each oriented by its node cycle

#### Scenario: Clique complex of the complete graph
- **WHEN** a user prepares the `CellComplexSpace` of the complete graph on four nodes with `faces='cliques'` and
  `dimension=3`
- **THEN** it has 4 nodes, 6 edges, 4 triangles, and 1 tetrahedron, and the product of consecutive incidence matrices is
  zero

#### Scenario: Simplicial hyperedges
- **WHEN** a user prepares the `CellComplexSpace` of a network whose hyperedges are triangles of its edges with
  `faces='hyperedges'`, one of them given in reversed node order
- **THEN** the edge-to-face incidence matrix equals XGI's `boundary_matrix` of order 2 with the orientation of the
  reversed triangle flipped, and the product of consecutive incidence matrices is zero

#### Scenario: Incidence parity
- **WHEN** the incidence matrices of the clique complex of a random graph on 12 nodes are compared with those of XGI and
  of TopoNetX built from the same simplices
- **THEN** they agree entry by entry after matching the cells and the orientation of each cell

#### Scenario: Invalid face
- **WHEN** a user gives a face whose node cycle uses a pair of nodes that the graph does not join
- **THEN** preparing the space raises a `SpaceError` that names the face and the missing edge

### Requirement: Link and plaquette states are edge and hyperedge groups
A state on links SHALL be held by a `GroupElement` on the edges of a `CellComplexSpace`, and a state on plaquettes by a
`GroupElement` on its hyperedges, with one member per cell.

#### Scenario: Potential and field strength
- **WHEN** a user places a group with a potential on the edges and a group with a field strength on the hyperedges of
  the `CellComplexSpace` of a periodic 16 × 16 lattice
- **THEN** the groups hold 512 potential values and 256 field strength values

### Requirement: Exterior calculus maps between degrees
The law utilities SHALL give the exterior derivative, which maps a state of a group on sites, edges, or hyperedges of a
cell complex to the next degree, the Hodge star of the lattice spacings or of the group weights, and the codifferential,
which maps to the previous degree. The exterior derivative applied twice SHALL be exactly zero, and the NumPy and Numba
backends SHALL give the same values.

#### Scenario: Curl of a gradient
- **WHEN** a law computes `exterior_derivative` of `exterior_derivative` of a random site state on the complex of a
  periodic 3D lattice
- **THEN** every plaquette value is exactly zero on the NumPy and the Numba backends

#### Scenario: Divergence of a curl
- **WHEN** a law computes `codifferential` of `codifferential` of a random plaquette state
- **THEN** every site value is zero within 1e-12

### Requirement: Reductions run over the faces of a link
`neighbors` SHALL reduce over the faces that contain an edge, or the edges that contain a site, with
`CofaceNeighborhood`, and inside such a reduction `exterior_derivative` of an edge state SHALL be the oriented sum of
the edge values around the one face.

#### Scenario: Abelian plaquette
- **WHEN** the plaquette sums of a random angle state on the links of a periodic lattice complex are read through a
  coface reduction
- **THEN** they equal `exterior_derivative` of the link angles at the same plaquettes

### Requirement: Monte Carlo samples link states
`SamplingInteraction` SHALL sample a state of a group on the edges of a cell complex, with a local objective that reduces
over `CofaceNeighborhood`, under discrete time, one sweep of all links per unit of time. Its `'colored'` schedule SHALL
update together only links that share no face, and every draw SHALL come from the simulation's random stream.

#### Scenario: Two-dimensional U(1)
- **WHEN** the energy `-beta * neighbors(cos(exterior_derivative(s.theta)), over=CofaceNeighborhood())` with
  `beta = 2` is sampled with the `'rotate'` move on the complex of a periodic 32 × 32 lattice for 4000 sweeps after
  1000 sweeps of transient
- **THEN** the mean cosine of the plaquette sums is within 0.01 of the exact value `I_1(2) / I_0(2)`

#### Scenario: Reproducible sampling
- **WHEN** a Z_3 link system sampled with the `'random-value'` move and the `'heat-bath'` rule runs twice with the same
  seed
- **THEN** both runs give identical link values at every frame

#### Scenario: Colored schedule on an odd lattice
- **WHEN** the colored schedule colors the links of the complex of a periodic lattice with 15 sites along one axis
- **THEN** no two links of one class share a plaquette

### Requirement: Exterior operations check the degree
`PassCompiler` SHALL report, with the code `law-degree`, a codifferential of a site state, an exterior derivative of a
state of the highest degree of a complex, and a sampling of a hyperedge group, each with the path of the law and a fix
that names the operation of the right degree. It SHALL report a coface reduction outside a cell complex with the code
`law-space`. A reduction over a whole group, such as a mean, SHALL add no degree to the value it appears in, while the
exterior operations inside its argument SHALL still be checked.

#### Scenario: Several link issues at once
- **WHEN** a system applies `codifferential` to a site state and samples a hyperedge group
- **THEN** the report holds two issues with the code `law-degree`, each with its own path and fix

#### Scenario: Order parameter of another degree
- **WHEN** a rate on links is scaled by the mean over the faces of a function of the exterior derivative of the links
- **THEN** the report holds no issue with the code `law-degree`

#### Scenario: Misapplied operation inside a mean
- **WHEN** a rate on sites is scaled by the mean of the codifferential of a site state
- **THEN** the report holds an issue with the code `law-degree` for that law

### Requirement: Link fields evolve in real time
Edge groups SHALL run with rate laws under continuous time and the existing solvers, so the evolution of a potential
and an electric field on links follows `A_t = -E` and `E_t = δ dA - J`. Without a current, the codifferential of the
electric field SHALL stay equal to the site charge that the initial state sets.

#### Scenario: Gauss law
- **WHEN** a lattice Maxwell field on the complex of a periodic 32 × 32 lattice with a static charge pair runs 1000
  steps with `VelocityVerletSolver`
- **THEN** the largest absolute value of the codifferential of the electric field minus the charge stays below 1e-10 at
  every frame

#### Scenario: Lattice dispersion
- **WHEN** a plane wave of wave number `k` along one axis runs on a lattice complex of spacing `a`
- **THEN** its measured angular frequency is within 1% of `(2 / a) sin(k a / 2)`

### Requirement: Cell complexes are analysed
`CellComplexAnalysis` SHALL give, with the kind `'hodge'`, the decomposition of an edge state into the exterior
derivative of a site potential, the codifferential of a face potential, and a harmonic remainder, which are orthogonal
and sum to the state. With the kind `'betti'` it SHALL give the Betti numbers of a complex, and with the kind
`'curvature'` the share and the energy of the faces whose oriented sum of an angle state is not a multiple of its
period.

#### Scenario: Consistent exchange rates
- **WHEN** the logarithms of exchange rates `r_ij = p_j / p_i` among 6 currencies are decomposed on the clique complex
  of the complete graph
- **THEN** the curl and harmonic parts are zero within 1e-12, and the gradient part recovers `log p` up to a constant

#### Scenario: Flow around a torus
- **WHEN** a uniform flow along the first axis of the complex of a periodic 8 × 8 lattice is decomposed
- **THEN** it is purely harmonic, and the Betti numbers of the complex are 1, 2, and 1

#### Scenario: Flat and curved connections
- **WHEN** the curvature of a flat angle state and of an angle state with a sum of pi around one plaquette is computed
- **THEN** the first has no curved face and zero energy, and the second has exactly one curved face

### Requirement: Cell states stream to Rerun
`CellVisual` SHALL stream an edge group as segments colored by value, with a cyclic color map for angle states, and a
hyperedge group as an image on a 2D lattice complex or as colored polygons on a network complex. A lattice complex of
more than two dimensions SHALL be drawn as a chosen 2D slice.

#### Scenario: Plaquette image
- **WHEN** a 2D U(1) system on a 32 × 32 lattice complex runs with a `RerunObserver` and `CellVisual`
- **THEN** the recording holds the plaquette sums as a 32 × 32 image with a cyclic color map and the links as 2048
  colored segments at every recorded frame

### Requirement: Link states take values in a matrix group
A state SHALL be declarable with a compact matrix group, `SU(n)` or `U(n)`, which stores the `2 n^2` real components of
the matrix of every member, the real and imaginary part of every entry in row order. The group SHALL convert matrices
to and from their components, draw matrices from the Haar measure of the group and near its identity, and project
matrices onto the group, the unitary factor of their polar decomposition, divided by the principal root of its
determinant for `SU(n)`. A matrix group state SHALL be neither a vector nor a tensor, of order 1 without lags.

#### Scenario: SU(3) links store their components
- **WHEN** a group of links of the complex of a periodic 4 × 4 lattice declares an SU(3) state with the identity as
  default
- **THEN** the state holds 18 components for each of the 32 links, which are the identity matrix of every link

#### Scenario: Draws stay in the group
- **WHEN** matrices of SU(2), SU(3), U(1), or U(2) are drawn from the Haar measure or near the identity
- **THEN** they are unitary, and of unit determinant for the special unitary groups, within 1e-12

#### Scenario: Projection undoes rounding
- **WHEN** matrices of a group are perturbed by 1e-6 and projected
- **THEN** the projected matrices lie in the group and within 1e-5 of the original matrices

### Requirement: Holonomies are path-ordered products of link matrices
The matrix group SHALL give the holonomy of paths of oriented edges, the product of the matrix of every edge along its
orientation and of its inverse against it, from the first edge of a path to the last, with padding that contributes the
identity. A `CellComplexSpace` SHALL list the edges around every face in the cyclic order of its nodes with the sign of
their traversal, whose signed sum is the row of the face in the edge-to-face incidence.

#### Scenario: Ordered products do not commute
- **WHEN** the holonomy of a path of three SU(2) links is computed, once forward and once along the reversed path
- **THEN** the first is the ordered product of the links with the inverse of the link traversed backward, and the second
  is its inverse

#### Scenario: Gauge-invariant trace
- **WHEN** a random gauge transformation `g_x U g_y^-1` is applied to the SU(3) links of a closed loop
- **THEN** the trace of the holonomy of the loop is unchanged within 1e-12

#### Scenario: Face boundaries agree with the incidence
- **WHEN** the boundaries of the faces of a periodic 3 × 4 × 2 lattice, a fixed 4 × 4 lattice, a clique complex, or a
  cycle basis of a network are read
- **THEN** every boundary is a closed walk, and its signed edges sum to the row of the face in the incidence

### Requirement: Monte Carlo samples matrix link states
`SamplingInteraction` SHALL sample a matrix group state on the edges of a cell complex with a holonomy score, a class
function of the holonomy of every face, whose local objective of an edge sums the score over the holonomies of the faces
around it started at the edge. A move SHALL multiply a link by a random matrix of the group near the identity, draw it
from the Haar measure, or be a function of the link matrices, accepted by the Metropolis or the Glauber rule, with a
colored or random-sequential schedule. Every sweep SHALL project the links onto the group. A drive SHALL change a state
of several components per member as a whole, with the components on a last axis.

#### Scenario: Two-dimensional SU(2)
- **WHEN** the Wilson action `-(beta / 2) Re tr U_p` with `beta = 2` is sampled on a periodic 16 × 16 lattice for 300
  sweeps after 100 sweeps of transient
- **THEN** the mean of `Re tr U_p / 2` is within 0.01 of the exact value `I_2(2) / I_1(2)`

#### Scenario: Two-dimensional SU(3)
- **WHEN** the Wilson action `-(beta / 3) Re tr U_p` with `beta = 4` is sampled on a periodic 12 × 12 lattice
- **THEN** the mean of `Re tr U_p / 3` is within 0.01 of the one-plaquette integral over the Haar measure of SU(3)

#### Scenario: Links stay in the group
- **WHEN** SU(3) links are sampled for 20 sweeps with the colored or the random-sequential schedule
- **THEN** they are unitary of unit determinant within 1e-12

#### Scenario: Moves of scalars are refused
- **WHEN** a holonomy score is given the flip move, the heat-bath rule, or the checkerboard schedule
- **THEN** checking the interaction raises a `ValidationError` that names the moves, rules, or schedules it takes
