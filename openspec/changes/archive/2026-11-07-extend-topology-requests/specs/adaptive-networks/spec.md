# Spec Delta

## ADDED Requirements

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
