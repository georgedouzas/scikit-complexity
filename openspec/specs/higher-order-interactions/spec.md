# higher-order-interactions Specification

## Purpose
Lets a user run laws over hyperedges of more than two members, on hypergraphs and simplicial complexes, run symbolic
laws once per group of a grouping, and let nodes regulate links.

## Requirements

### Requirement: Network spaces take hyperedges from XGI
`NetworkSpace` SHALL take as its hyperedges an XGI hypergraph, an XGI simplicial complex, or sequences of nodes, and
SHALL derive its incidence matrix and the hyperedges around every node.

#### Scenario: Triangle hypergraph
- **WHEN** a network space takes the triangles of a complete graph on five nodes as its hyperedges
- **THEN** it has 10 hyperedges of size 3 and every node lies in 6 of them

### Requirement: Laws run once per hyperedge
The laws of a `HyperedgeInteraction` SHALL run once per hyperedge of its declared size, SHALL read the members of the
hyperedge, the focal member, and the member of a group on hyperedges or edges that a second role binds, and SHALL
compile to gathers and scatter-adds on the NumPy and Numba backends.

#### Scenario: Hyperedge law against a loop
- **WHEN** a `HyperedgeInteraction` adds `total(x, over=members) - size * x` to the rate of every focal member on a
  hypergraph with hyperedges of sizes 3 and 4
- **THEN** the compiled drift equals a direct loop over the hyperedges within 1e-12 on both backends

#### Scenario: Missing hyperedge size
- **WHEN** a `HyperedgeInteraction` of size 4 runs in a network space whose hyperedges all have size 3
- **THEN** the compiler reports an issue with the code `binding` that names the size and the sizes the space has

### Requirement: Laws reduce over the hyperedges of a member
The reduction `hyperedges` SHALL reduce an expression of one hyperedge over the hyperedges around every member, and
`CouplingInteraction` SHALL couple through it in the mode `'hyperedges'`.

#### Scenario: Three-body Kuramoto
- **WHEN** phase oscillators couple only through the triangles of a hypergraph by
  `CouplingInteraction(mode='hyperedges', size=3)` with the term `sin(theta_j + theta_k - 2 theta_i)`
- **THEN** the compiled drift equals a direct loop over the triangles within 1e-12 on both backends

#### Scenario: Hyperedge reduction outside a hypergraph
- **WHEN** a law reads `hyperedges` in a lattice space
- **THEN** the compiler reports an issue with the code `law-space`

### Requirement: Groupings take a symbolic form
`GroupingInteraction(form='symbolic')` SHALL compile a function of one group as an update of a state of its members,
reduced over the groups that hold a member, with pairs and partitions drawn every step from the random stream of the
simulation. The agent form SHALL stay the default.

#### Scenario: Majority in partitions
- **WHEN** a symbolic grouping of partitions of size 3 sets every member to the majority opinion of its group
- **THEN** after one step every group of the drawn partition holds one opinion, and two runs with one seed draw the
  same partitions

#### Scenario: Agent setting in the symbolic form
- **WHEN** a symbolic grouping is given `act`
- **THEN** `prepare` raises `ValidationError` that names `act` as a setting of the agent form

### Requirement: Symbolic updates run in ordered phases
An `UpdateLaw` SHALL take a `phase`, and the phases of a `PhasesScheduleSpec` of the discrete time SHALL run one after
the other within a step on the NumPy and Numba backends, each reading the states the phases before it set, with laws
without a phase in the default phase, first unless the schedule places it. Without phases, every update law SHALL read
the states at the start of the step, as before.

#### Scenario: Payoffs then imitation in one step
- **WHEN** a symbolic grouping of neighbourhoods writes the payoffs in a first phase and `ImitationInteraction`
  copies the best neighbour by those payoffs in a second phase
- **THEN** the payoffs and strategies of every step equal a loop that computes the payoffs and then the copies, on both
  backends

#### Scenario: Update law in an undeclared phase
- **WHEN** an update law names a phase that the schedule of the discrete time does not declare
- **THEN** compilation raises `ValidationError` naming the phase

### Requirement: Nodes regulate links
A `HyperedgeInteraction` SHALL bind a group on the edges of the same network through its second role, whose member is
the edge joining the first two members of the hyperedge, so a regulating node switches the activity of a link with an
update law every step.

#### Scenario: Triadic percolation
- **WHEN** links are regulated by the activity of their regulating nodes as in Millán et al. (2023)
- **THEN** the giant component shows period two above the bifurcation point of the paper within the finite-size error

### Requirement: Networks grow by hyperedges
`TopologyInteraction(action='add-node', hyperedge=k)` SHALL add every new node together with a hyperedge of size `k`
holding it and `k - 1` existing nodes.

#### Scenario: Triangle attachment
- **WHEN** a growing network adds 50 nodes with `hyperedge=3` from one triangle
- **THEN** it holds 51 triangles, and every new node lies in at least one of them

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
