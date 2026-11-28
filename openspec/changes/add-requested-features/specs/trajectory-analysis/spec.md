## ADDED Requirements

### Requirement: Interaction graph of a fixed space
`Trajectory.get_network(frame)` SHALL return, for a system with no changing network, the nodes and edges of the
neighbourhood its interactions read: the neighbour pairs of a lattice, or the pairs of agents within the interaction
radius of a continuous space at that frame. A system with neither SHALL raise a `KeyError` that names both cases.

#### Scenario: Ising lattice graph
- **WHEN** the network of a run of `IsingSystem(5)` is read at frame 0
- **THEN** it has 25 nodes and 50 edges, each site joined to its four von Neumann neighbours

#### Scenario: Vicsek graph changes
- **WHEN** the network of a Vicsek run is read at two frames
- **THEN** every edge joins two agents within the interaction radius across periodic edges, and the edge sets differ

### Requirement: Pairs and degrees within a radius
`neighbors(source, kind='radius', r_max=)` SHALL return, for every frame, the pairs of agents within `r_max` of each
other, across the periodic edges of the box, and the degree of every agent.

#### Scenario: Pairs across a periodic edge
- **WHEN** two agents sit at 0.1 and 9.9 of a periodic box of side 10 and `r_max=0.5`
- **THEN** they form one pair and each has degree 1

### Requirement: Attractors of finite deterministic systems
`automaton(system, kind='attractors')` SHALL map every configuration of a finite automaton or Boolean network in one
vectorised pass, and SHALL return the transient length and the attractor of every configuration, and the cycles with
their lengths and basin sizes. It SHALL raise an `AnalysisError` that names the count above `2^24` configurations.

#### Scenario: Rule 90 on a ring
- **WHEN** the attractors of `ElementaryAutomatonSystem(90, size=8)` are found
- **THEN** the basin sizes sum to 256 and every cycle length divides the known periods of rule 90 on a ring of 8

#### Scenario: Yeast cell cycle
- **WHEN** the attractors of `YeastCellCycleSystem` are found
- **THEN** the largest basin holds the stationary G1 state, as Li and coauthors report

### Requirement: Damage spreading in Boolean networks
`automaton(kind='difference')` SHALL accept a Boolean network and return the Hamming distance between two copies that
start from configurations differing at given nodes.

#### Scenario: Ordered network heals
- **WHEN** a network with `k=1` starts two copies one node apart
- **THEN** the distance stays small over time
