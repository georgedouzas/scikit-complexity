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
`law-space`.

#### Scenario: Several link issues at once
- **WHEN** a system applies `codifferential` to a site state and samples a hyperedge group
- **THEN** the report holds two issues with the code `law-degree`, each with its own path and fix

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
