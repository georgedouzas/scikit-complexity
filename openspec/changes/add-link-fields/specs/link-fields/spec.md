# Spec Delta

## Purpose

Lets a user place groups on the links and plaquettes of a lattice or on the edges and faces of a network, write their
laws with discrete exterior calculus and products around loops, sample compact group-valued link variables by Monte
Carlo, and analyse and draw the result.

## ADDED Requirements

### Requirement: Lattices have cell complexes
The space layer SHALL build the cell complex of a lattice as a network space: its sites as nodes, its links oriented
along the positive direction of each axis as edges, its plaquettes oriented by pairs of axes in increasing order as
hyperedges in cyclic node order, and its cubes, with the incidence matrix between every two consecutive degrees.

#### Scenario: Periodic square lattice
- **WHEN** a user builds the cell complex of a periodic 8 × 8 lattice
- **THEN** it has 64 sites, 128 links, and 64 plaquettes, and the product of the link-to-plaquette incidence matrix and
  the site-to-link incidence matrix is zero

#### Scenario: Fixed edges
- **WHEN** a user builds the cell complex of a 4 × 4 lattice with fixed edges
- **THEN** it has 16 sites, 24 links, and 9 plaquettes, with no cell that leaves the box

### Requirement: Networks have cell complexes
The space layer SHALL build the cell complex of a network: its nodes, its edges oriented from the first to the second
node in node order, and its faces, given as closed node cycles, taken from the network's hyperedges, or taken from a
minimum cycle basis of the graph.

#### Scenario: Faces of a network
- **WHEN** a user builds the cell complex of a 4-node cycle graph with one chord and no faces given
- **THEN** it has 5 edges and 2 faces from the minimum cycle basis, each face oriented by its node cycle

#### Scenario: Invalid face
- **WHEN** a user gives a face whose node cycle uses a pair of nodes that the graph does not join
- **THEN** building the complex raises a named error that names the face and the missing edge

### Requirement: Link and plaquette states are edge and hyperedge groups
A state on links or edges SHALL be held by a group on the edges of a cell complex, and a state on plaquettes or faces by
a group on its hyperedges, with one member per cell. No component kind other than the redesign's groups SHALL be needed.

#### Scenario: Potential and field strength
- **WHEN** a user places a group with a potential on the edges and a group with a field strength on the hyperedges of
  the cell complex of a periodic 16 × 16 lattice
- **THEN** the groups hold 512 potential values and 256 field strength values, and the trajectory of the potential has
  the shape `(frames, 2, 16, 16)`

#### Scenario: Values of the wrong shape
- **WHEN** a user gives values of the shape `(16, 16)` to a group on the edges of the complex of a 16 × 16 lattice
- **THEN** creating the group raises a named error that names the state, the part of the space, and the expected shape
  `(2, 16, 16)`

### Requirement: Exterior calculus maps between degrees
Laws SHALL apply the exterior derivative, which maps a state of a group on sites, edges, or hyperedges to the next
part, the Hodge star of the lattice spacings or of declared network weights, and the codifferential, which maps to the
previous part. The exterior derivative applied twice SHALL be exactly zero, and both backends SHALL give the same
results.

#### Scenario: Curl of a gradient
- **WHEN** a law computes the curl of the gradient of a random site state on the complex of a periodic 3D lattice
- **THEN** every plaquette value is exactly zero on the NumPy and the Numba backends

#### Scenario: Divergence of a curl
- **WHEN** a law computes the codifferential of the codifferential of a random plaquette state
- **THEN** every site value is zero within 1e-12

### Requirement: Field operators read edge and hyperedge states
On states of edge and hyperedge groups, `gradient` SHALL lower to the exterior derivative of a site state, `curl` to
the exterior derivative of a link state, `divergence` to the codifferential of a link state, and `laplacian` to the
Hodge Laplacian of the state's degree. On site groups, the operators SHALL keep their current discretizations.

#### Scenario: Site groups unchanged
- **WHEN** the Gray-Scott model runs after this change with the same seed and settings
- **THEN** its trajectory equals the trajectory before the change

#### Scenario: Degree mismatch
- **WHEN** a law applies `curl` to a state of a site group
- **THEN** validation reports an issue with the code `law-degree` whose path is the law and whose message names the
  state and its degree

### Requirement: States take values in compact groups
A state SHALL take values in a declared compact group domain: U(1) as an angle in `(-π, π]`, Z_N as an integer in
`[0, N)`, or SU(2) as a unit quaternion. The group product, the inverse, and the normalised real trace SHALL be
available to laws, and every update SHALL keep each value in its group.

#### Scenario: Angles wrap
- **WHEN** a U(1) link at angle 3.0 is rotated by 0.5
- **THEN** its stored value is `3.5 - 2π`

#### Scenario: SU(2) stays unitary
- **WHEN** a 4D SU(2) model runs 1000 Monte Carlo sweeps
- **THEN** every link quaternion has a norm within 1e-12 of one at every recorded frame

#### Scenario: Rate on a compact group
- **WHEN** a rate law targets a state of an SU(2) domain
- **THEN** validation reports an issue with the code `law-domain` that suggests a real or angle state for real-time
  dynamics

### Requirement: Holonomies multiply links around closed paths
Laws and analysis SHALL compute the holonomy of a closed path, the ordered product of its link variables with the
inverse on links traversed against their orientation, for every plaquette and for rectangular and straight loops given
by their extents. Laws SHALL compute the staple sum of a link, the sum over its plaquettes of the product of the other
three links. For an abelian group, the plaquette holonomy SHALL equal the exterior derivative of the link angles.

#### Scenario: Abelian plaquette
- **WHEN** the plaquette holonomies of a random U(1) configuration are computed
- **THEN** their angles equal the curl of the link angles wrapped to `(-π, π]`

#### Scenario: Non-abelian order
- **WHEN** the plaquette holonomy of an SU(2) configuration is computed along the reversed path
- **THEN** it equals the inverse of the holonomy along the original path

### Requirement: Monte Carlo samples link variables
A `Sample` law SHALL accept the local energy of a link from the plaquettes or staples that contain it, under discrete
time. It SHALL sample the energy with Metropolis acceptance, exact heat-bath draws for U(1), Z_N, and SU(2), and
overrelaxation, at a coupling that changes live, with one sweep of all links per unit of time.

#### Scenario: Two-dimensional U(1)
- **WHEN** a U(1) plaquette energy with coupling β = 2 on the complex of a periodic 32 × 32 lattice is sampled by
  heat-bath for 2000 sweeps after 200 sweeps of transient
- **THEN** the mean of the cosine of the plaquette angles is within 0.01 of the exact value `I_1(β) / I_0(β)`

#### Scenario: Two-dimensional SU(2)
- **WHEN** an SU(2) plaquette energy with coupling β = 4 on the complex of a periodic 16 × 16 lattice is sampled by
  heat-bath
- **THEN** the mean of `½ tr U_p` is within 0.01 of the exact value `I_2(β) / I_1(β)`

### Requirement: Link updates are scheduled and reproducible
The colored schedule SHALL update together only links that share no plaquette, and every draw of a link update SHALL
come from the simulation's random stream.

#### Scenario: Reproducible sampling
- **WHEN** a Z_3 link model runs twice with the same seed
- **THEN** both runs give identical link values at every frame

#### Scenario: Schedule on an odd lattice
- **WHEN** a user asks for the colored schedule on the complex of a periodic lattice with 15 sites along one axis
- **THEN** validation reports an issue with the code `sample-schedule` that names the axis and suggests an even size or
  the random-sequential schedule

### Requirement: Link primitives declare their requirements and support
The cell complex SHALL declare the network space kind with sites, edges, and hyperedges. The compact group domain SHALL
declare the law kinds that may target it. The exterior calculus and holonomies SHALL declare the degrees they accept.
The Monte Carlo solver and the NumPy and Numba backends SHALL declare their support for compact group domains and edge
groups, so `validate()` reports every unsupported combination before compilation.

#### Scenario: Several link issues at once
- **WHEN** a system applies `curl` to a site state, samples a hyperedge group, and uses the colored schedule on an odd
  lattice complex
- **THEN** the report holds three errors with the codes `law-degree`, `solver-unsupported`, and `sample-schedule`, each
  with its own path and fix

### Requirement: Link fields evolve in real time
Edge groups SHALL run with rate laws under continuous time and the existing ODE steppers, so the Hamiltonian evolution
of a potential and an electric field on links follows `A_t = -E` and `E_t = δ dA - J`, written with the exterior
calculus. Without a current, the codifferential of the electric field SHALL stay equal to the site charge that the
initial state sets.

#### Scenario: Gauss law
- **WHEN** a lattice Maxwell field on the complex of a periodic 32 × 32 lattice with a static charge pair runs 1000 steps
  with velocity Verlet
- **THEN** the largest absolute value of the codifferential of the electric field minus the charge stays below 1e-10 at
  every frame

#### Scenario: Lattice dispersion
- **WHEN** a plane wave of wave number `k` along one axis runs on a lattice of spacing `a`
- **THEN** its measured angular frequency is within 1% of `(2 / a) sin(k a / 2)`

### Requirement: Edge flows decompose into gradient, curl, and harmonic parts
The analysis layer SHALL decompose a state of an edge group into the exterior derivative of a site potential, the
codifferential of a face potential, and a harmonic remainder, which are orthogonal and sum to the state.

#### Scenario: Consistent exchange rates
- **WHEN** the logarithms of exchange rates `r_ij = p_j / p_i` among 6 currencies are decomposed on the complete graph
  with its triangles as faces
- **THEN** the curl and harmonic parts are zero within 1e-12, and the gradient part recovers `log p` up to a constant

#### Scenario: Flow around a torus
- **WHEN** a uniform flow along the first axis of a periodic 8 × 8 lattice is decomposed
- **THEN** it is purely harmonic, and the harmonic space has dimension 2

### Requirement: Link fields stream to Rerun
Edge and hyperedge groups SHALL stream to Rerun with a link visual that draws links as segments or arrows colored by
value, with a cyclic color map for angles, and plaquettes as an image on a 2D lattice complex or as colored faces on a
network complex. A lattice of more than two dimensions SHALL be drawn as a declared 2D slice.

#### Scenario: Plaquette image
- **WHEN** a 2D U(1) model on a 32 × 32 lattice complex runs with a `RerunObserver`
- **THEN** the viewer shows the plaquette angles as a 32 × 32 image with a cyclic color map and the links as 2048
  colored segments at every recorded frame

### Requirement: Networks have clique complexes
A network SHALL yield its clique complex up to a chosen dimension, with its triangles as faces and its larger cliques as
oriented simplices, and the incidence matrices between them, so edge and hyperedge groups and exterior calculus work on
agent interaction networks as on lattices.

#### Scenario: Complete graph on four nodes
- **WHEN** the clique complex of the complete graph on four nodes is built up to dimension 3
- **THEN** it has 4 vertices, 6 edges, 4 triangles, and 1 tetrahedron, and the product of consecutive boundary matrices
  is zero

### Requirement: Frustration and topology of link fields
The analysis SHALL report the frustration of a link field of any compact group, the share and the energy of plaquettes
whose holonomy is not the identity, and the Betti numbers of a complex from the ranks of its boundary matrices.

#### Scenario: Flat and frustrated connections
- **WHEN** the frustration of a flat U(1) field and of a field with flux pi through one plaquette is computed
- **THEN** the first has no frustrated plaquette and zero energy, and the second has exactly one frustrated plaquette

#### Scenario: Betti numbers of a torus
- **WHEN** the Betti numbers of the cell complex of a periodic 8 × 8 lattice are computed
- **THEN** they are 1, 2, and 1
