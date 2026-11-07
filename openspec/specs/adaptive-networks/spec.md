# adaptive-networks Specification

## Purpose
Lets a user build networks whose structure and edge states evolve together with the states of their nodes, across
several relation layers, with neighbourhood reductions in any space.

## Requirements

### Requirement: Generators create reproducible networks
The space layer SHALL generate Erdős-Rényi, Watts-Strogatz, Barabási-Albert, configuration-system, stochastic block,
lattice, and complete network spaces from the simulation's random stream.

#### Scenario: Same seed, same network
- **WHEN** a Watts-Strogatz network is generated twice with the same seed
- **THEN** both networks have identical edges

### Requirement: Interactions read and change edge states
An interaction with the binding `'neighbours'` SHALL bind an edge group of the same network, run its laws once per
edge, and read and target the states of the edge member and of the members at its source and target.

#### Scenario: Hebbian weights
- **WHEN** an interaction grows the weights of an edge group with the product of the activities at both ends
- **THEN** edges between co-active nodes end with higher weights than edges between inactive ones

### Requirement: Layers separate relation types
A network space SHALL hold several named layers over the same nodes, and each interaction and edge group SHALL bind to
one layer.

#### Scenario: Two relations
- **WHEN** a system has a contact layer for infection and a work layer for wages
- **THEN** infection spreads only along contact edges, and wages flow only along work edges

#### Scenario: Missing layer
- **WHEN** an interaction names a layer the network space does not have
- **THEN** validation reports an issue with the code `binding` that lists the layers of the space

### Requirement: Laws change the topology
Update, jump, and rule laws SHALL request edge and node additions, removals, and rewirings, which add and remove members
of the variable groups on the edges and sites of the network, so groups on network nodes resize with the nodes.

#### Scenario: Preferential attachment
- **WHEN** a Barabási-Albert growth system adds 10,000 nodes with two edges each
- **THEN** the degree distribution's tail exponent lies within 0.3 of 3

#### Scenario: Adaptive epidemic
- **WHEN** susceptible nodes rewire away from infected neighbours at a high rate
- **THEN** the epidemic threshold rises above its value without rewiring

#### Scenario: Request on a fixed group
- **WHEN** a law requests a new edge on a network whose edge group is not variable
- **THEN** validation reports an issue with the code `group-fixed` whose fix is `variable=True`

### Requirement: Variable groups run on compiled backends
The NumPy and Numba backends SHALL run variable groups with capacity layouts and active masks, and SHALL declare their
support for variable and edge groups, so validation accepts them with solvers that support variable groups.

#### Scenario: Growth outside Mesa
- **WHEN** a network that grows by rule laws from 10 to 1,000 nodes runs under discrete time on the Numba backend
- **THEN** the run completes, every frame holds the active nodes only, and it agrees with the NumPy backend within 1e-12

### Requirement: Laws reduce over neighbourhoods
Laws SHALL use reductions of expressions over neighbours: network neighbours of a layer weighted by an edge state,
lattice neighbours, neighbours within a radius, the k nearest neighbours, or neighbours within a vision cone. Validation
SHALL report a neighbourhood that the system's space does not offer.

#### Scenario: Kuramoto synchronisation
- **WHEN** 1000 Kuramoto oscillators with Lorentzian frequencies of half-width γ run on a complete graph with coupling
  above and below `2 * γ`
- **THEN** the order parameter stays near 0 below the critical coupling and grows above it

#### Scenario: Flocking
- **WHEN** the Vicsek system runs at low noise and high density
- **THEN** the polar order parameter exceeds 0.9

#### Scenario: Metric neighbourhood on a network
- **WHEN** a law reduces over `Radius(1.0)` in a system on a network space
- **THEN** validation reports an issue with the code `law-space` that suggests a continuous space or a layer

### Requirement: Networks are recorded and drawn
Trajectories SHALL hold network snapshots whenever the topology changes, and Rerun SHALL draw them as graphs with
node and edge colors from states.

#### Scenario: Snapshot by frame
- **WHEN** a user asks a trajectory of a growing network for frame 10
- **THEN** it returns the nodes and edges present at frame 10

### Requirement: New nodes attach by weights that read them
In the action `'add-node'`, the attachment weight, the condition, and the probability of `TopologyInteraction` SHALL
read the drawn values of the node being added as `namespace.source`, beside the symbols of the candidate node.

#### Scenario: Attachment by distance
- **WHEN** a network grows by nodes with a drawn angle that attach with weight 1 to candidates within an angular
  distance of 0.1 and with weight 0 to all others
- **THEN** every edge of a new node joins nodes whose angles differ by at most 0.1

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

### Requirement: New edges are chosen among candidates
An `'add-edge'` request SHALL take `candidates`, a number of candidate edges, and `score`, a function of the namespace
of a candidate end that reads the node the edge starts from as `node.source`, and SHALL add the candidate of the
smallest score. Without a target, `'add-edge'` SHALL run once per step, and every candidate SHALL join two distinct
nodes not yet joined, drawn uniformly.

#### Scenario: Product rule
- **WHEN** edges are added one per step to 2000 isolated nodes, each the one of two candidates with the smaller
  product of the sizes of the components it joins
- **THEN** the largest component holds less than 10 percent of the nodes after 0.8 edges per node and more than half
  of them after 0.95 edges per node

#### Scenario: One candidate
- **WHEN** the same process runs with one candidate
- **THEN** the largest component after 0.8 edges per node is within 0.06 of the Erdős-Rényi value `S = 1 - exp(-1.6 S)`

### Requirement: Triangles close around a node
A `'close-triangle'` request SHALL run for a node and join two of its neighbours that are not yet joined, drawn
uniformly among such pairs. A node without such a pair, with fewer than two neighbours or all of them joined, SHALL
instead join `edges` other nodes drawn by `toward`, as `'add-edge'` does.

#### Scenario: Closure keeps the neighbours
- **WHEN** the centre of a star of five leaves closes one triangle
- **THEN** the network gains one edge between two leaves and the centre keeps its edges

#### Scenario: Closed neighbourhood
- **WHEN** a node of a complete graph on four nodes with one more isolated node closes a triangle
- **THEN** it joins the isolated node

### Requirement: Edges contract
A `'contract-edge'` request SHALL run for a node, draw a neighbour by `toward`, and merge the neighbour into the node,
which keeps the union of their links on every layer without self-loops or repeated links, while the neighbour leaves
the network.

#### Scenario: Union of links
- **WHEN** node 0 of the path 3-0-1-2 contracts its edge with node 1
- **THEN** node 1 leaves and node 0 joins nodes 2 and 3

### Requirement: Nodes duplicate
A `'duplicate-node'` request SHALL run for a node and add a copy with its values, which inherits each of its links with
probability `inherit`, joins it with probability `parent`, and joins further ends among the nodes joined to neither as a
new node does.

#### Scenario: Exact copy
- **WHEN** a node of a cycle duplicates with `inherit=1`, `parent=0`, and no further ends
- **THEN** the copy has the neighbours of the node

### Requirement: Replacements remove and add together
A `'replace-node'` request SHALL remove the node it runs for and add a new node that joins new ends, in one request, so
a replacement fired with a probability removes and adds together.

#### Scenario: Turnover keeps the size
- **WHEN** every node of a network of 50 nodes is replaced with probability 0.3 per step for 20 steps
- **THEN** the network holds 50 nodes at every step

### Requirement: Requests act on one drawn member
A request with `one=True` SHALL run at every step for one member drawn uniformly among the members present, node, edge,
or hyperedge, and fire with the probability of its law for that member. A jump law SHALL NOT request it.

#### Scenario: One removal per step
- **WHEN** a network of 30 nodes removes one drawn node per step with probability one
- **THEN** it holds `30 - t` nodes after `t` steps

#### Scenario: Jump with one member
- **WHEN** a jump law requests a change with `one=True`
- **THEN** preparing the simulation raises `ValidationError`
