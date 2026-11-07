# Spec Delta

## ADDED Requirements

### Requirement: Laws add and remove hyperedges
Topology requests SHALL add hyperedges with `'add-hyperedge'`, run for a node that becomes the last member of a
hyperedge drawn by the weights of the request, and remove them with `'remove-hyperedge'`, run for every hyperedge of
the variable group on hyperedges. A variable group on hyperedges SHALL take a `rank` and hold hyperedges of every size
up to it, padded with -1, and laws per hyperedge on a ranked group SHALL raise `RoleError`.

#### Scenario: Hyperedges of several sizes
- **WHEN** new nodes form hyperedges of two, three, and four members in a group of rank 4 for 10 steps
- **THEN** the group holds 10 hyperedges of each size, each with the new node as its last member

#### Scenario: Forming and dissolving hyperedges
- **WHEN** three nodes form triangles with probability 0.2 per step and triangles dissolve with probability 0.25
- **THEN** the mean number of triangles over the last 100 of 200 steps is within 30 percent of 2.4

#### Scenario: Hyperedge beyond the rank
- **WHEN** a request forms a hyperedge larger than the rank of the group
- **THEN** preparing the simulation raises `ValidationError`

### Requirement: New ends join independently and in a direction
Topology requests SHALL take `independent`, under which every candidate becomes a new end with its weight as a
probability, and `direction`, which sends new edges out of the node, into it, or both ways on a directed layer.

#### Scenario: Independent ends both ways
- **WHEN** new nodes of a directed network join independent ends both ways with probability 0.5 for 60 steps
- **THEN** the edges join a share of the ordered pairs within 0.05 of 0.5

### Requirement: Laws read shortest paths of the current network
`shortest_paths(active, over=None, reduce=...)` SHALL reduce over the shortest paths of the active links with the
reductions `'betweenness'` and `'closeness'` of NetworkX, on the network present at every evaluation, computed again
only when the active links change, and the Numba backend SHALL run its laws in Python with a `BackendWarning`.

#### Scenario: Agreement with NetworkX
- **WHEN** the drift of every node is the betweenness or the closeness of the open links of a random graph
- **THEN** it equals NetworkX's measure on the graph of the open links

#### Scenario: Removed nodes
- **WHEN** nodes are removed at random while a map reads their betweenness, on the NumPy and Numba backends
- **THEN** the load read at every step is the betweenness of the network present at its start

### Requirement: Neighbours read the member sets of nodes
`AgentView.neighbors` SHALL take a `group`, so an agent reads the agents of one group on the nodes around it, the
collection of every neighbouring node, apart from the agents of other groups.

#### Scenario: Tokens around a holder
- **WHEN** holders on a path of four nodes count the tokens of their neighbouring nodes
- **THEN** they count the tokens alone, and `neighbors()` counts the holders too
