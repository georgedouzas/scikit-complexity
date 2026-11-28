## ADDED Requirements

### Requirement: Directed Erdős-Rényi graphs
`RandomGraphEnsemblesSystem(ensemble='erdos_renyi', directed=True)` SHALL grow a directed graph in which every node
adds out-edges to nodes drawn uniformly, so that the mean out-degree and the mean in-degree reach `mean_degree` after
`steps` steps. Any other ensemble with `directed=True` SHALL raise a `ValidationError` that names the ensemble.

#### Scenario: Mean in and out degrees
- **WHEN** `RandomGraphEnsemblesSystem(500, directed=True, mean_degree=4.0, steps=20)` runs 20 steps
- **THEN** the network of the last frame is directed, and its mean out-degree and mean in-degree both lie within 10
  percent of 4

#### Scenario: Directed ring refused
- **WHEN** `RandomGraphEnsemblesSystem(ensemble='watts_strogatz', directed=True)` is prepared
- **THEN** a `ValidationError` names `'watts_strogatz'` and `directed`

### Requirement: Directed configuration model
`ConfigurationModelSystem(directed=True)` SHALL take `degrees` as a pair of an in-degree and an out-degree sequence of
equal sums, or draw both from the power law, and SHALL pair out-stubs with in-stubs uniformly, rewiring self-loops and
repeated links away when `simple` is true so that every in-degree and out-degree is kept.

#### Scenario: Degrees are kept
- **WHEN** `ConfigurationModelSystem(([1, 2, 1, 0], [2, 0, 1, 1]), directed=True)` is prepared
- **THEN** the graph is directed, simple, and its in-degrees and out-degrees equal the given sequences

#### Scenario: Unequal sums
- **WHEN** the in-degrees sum to 4 and the out-degrees to 5
- **THEN** a `ValidationError` names both sums

### Requirement: Newman-Watts shortcuts
`RandomGraphEnsemblesSystem(ensemble='newman_watts')` SHALL start from the ring of `mean_degree` nearest neighbours and
SHALL let every lattice edge add at most one shortcut from its lattice end to a node drawn uniformly, with total
probability `rewiring` over `steps` steps, so that no lattice edge is removed.

#### Scenario: Ring is kept and shortcuts are added
- **WHEN** `RandomGraphEnsemblesSystem(1000, ensemble='newman_watts', mean_degree=4, rewiring=0.2, steps=10)` runs 10
  steps
- **THEN** every edge of the ring is still present, and the number of added edges lies within 10 percent of
  `0.2 * 2000`

### Requirement: Preferential attachment with an offset
`BarabasiAlbertSystem(offset=C)` SHALL draw the ends of a new node with probability proportional to `k + C`, and SHALL
raise a `ValidationError` when `k + C` is not positive for the smallest degree of the network.

#### Scenario: Exponent of the shifted law
- **WHEN** `BarabasiAlbertSystem(edges=2, offset=2.0)` grows to 20,000 nodes
- **THEN** the fitted degree exponent lies within 0.3 of `3 + C / m = 4`

#### Scenario: Offset too negative
- **WHEN** `BarabasiAlbertSystem(edges=2, offset=-2.0)` is prepared
- **THEN** a `ValidationError` names the offset and the smallest degree

### Requirement: Growth with uniform attachment
`BarabasiAlbertSystem(attachment='uniform')` SHALL draw the ends of a new node uniformly among the nodes present, while
`attachment='preferential'` SHALL stay the default, and an unknown attachment SHALL raise a `ValidationError`.

#### Scenario: Exponential degree tail
- **WHEN** `BarabasiAlbertSystem(edges=2, attachment='uniform')` grows to 10,000 nodes
- **THEN** the largest degree stays below 40, and the share of nodes of degree `k` falls with `k` as
  `exp(-k / m)` within 20 percent for `k` from 3 to 10

### Requirement: Individual epidemics on a given graph
`ContactNetworkEpidemicSystem(graph, compartments=...)` SHALL run the SIR or SIS epidemic of one host per node of the
graph in continuous time, a susceptible host infected at the rate `beta` per infected neighbour and an infected host
recovering at the rate `gamma`, with the state `hosts.state` holding 0, 1 or 2 for susceptible, infected or removed.

#### Scenario: Start from chosen nodes
- **WHEN** `ContactNetworkEpidemicSystem(nx.path_graph(10), compartments='sir', gamma=0.0)` runs with
  `initial_state={'hosts.state': x0}` where only node 0 is infected
- **THEN** the infected hosts at every frame form one path that starts at node 0

#### Scenario: Final size of SIR
- **WHEN** 200 runs on an Erdős-Rényi graph of 1000 nodes and mean degree 5 with `beta=0.5` and `gamma=1.0` reach the
  end of the epidemic
- **THEN** the mean final share of removed hosts among large outbreaks lies within 0.05 of the root `S` of
  `S = 1 - exp(-5 T S)` with the transmissibility `T = beta / (beta + gamma) = 1 / 3`

#### Scenario: Endemic SIS
- **WHEN** `compartments='sis'` runs above the threshold on a random regular graph of degree 6
- **THEN** no host is ever removed and the share of infected hosts settles above zero

### Requirement: Network walkers start at chosen nodes
`NetworkRandomWalkSystem(start=...)` SHALL place every walker at the node `start`, at the node of its own entry when
`start` is a sequence of nodes, and at a node drawn uniformly when `start` is `None`, and the node of every walker
SHALL be the state `walkers.node`.

#### Scenario: Walkers start together
- **WHEN** `NetworkRandomWalkSystem(nx.cycle_graph(10), n_walkers=5, start=3)` runs one step
- **THEN** every walker is at node 3 in the first frame and at node 2 or 4 in the second

### Requirement: Master equation of a network walk
`NetworkRandomWalkSystem(mode='master')` SHALL evolve the probability `nodes.rank` of a walker at every node by
`p(t + 1) = p(t) P` with `P = D^-1 A`, starting from `start` or from the uniform law, while `mode='walkers'` SHALL stay
the default. A directed graph or one with an isolated node SHALL raise a `ValidationError` in this mode.

#### Scenario: Stationary law of a connected graph
- **WHEN** the master equation runs 500 steps on Zachary's karate club from node 0
- **THEN** `nodes.rank` sums to 1 at every frame and ends within `1e-6` of `k_i / 2L`

### Requirement: Voters start from given opinions and update one at a time
`VoterSystem` SHALL take its opinions from `initial_state={'voters.opinion': x0}`, and `VoterSystem(schedule='single')`
SHALL update one voter drawn uniformly per step, while `schedule='random-sequential'` SHALL stay the default.

#### Scenario: Degree-weighted martingale on a star
- **WHEN** 4000 runs of `VoterSystem(nx.star_graph(10), schedule='single')` start with the hub and five leaves at
  opinion 1 and the other leaves at 0, and run to consensus
- **THEN** the share of runs that end at opinion 1 lies within 0.03 of 0.75, the initial degree-weighted
  magnetisation `(10 + 5) / 20`

### Requirement: Synchronous Watts cascades
`WattsCascadesSystem(update='synchronous')` SHALL activate every node whose share of active neighbours at the start of
the step reaches its threshold, while `update='random-sequential'` SHALL stay the default, and the docstring SHALL
state the order of both updates.

#### Scenario: Cycle grows by two per step
- **WHEN** `WattsCascadesSystem(nx.cycle_graph(30), threshold=0.5, update='synchronous')` starts from one active node
- **THEN** the active counts of the first frames are 1, 3, 5 and 7

### Requirement: DebtRank exposes its exposures
`DebtrankSystem` SHALL hold after `prepare` the liabilities matrix it uses as `liabilities_` and the capital of every
bank as `capital_`, whether given or drawn.

#### Scenario: Drawn exposures reproduce the cascade
- **WHEN** `DebtrankSystem(random_state=1)` is prepared and simulated
- **THEN** `liabilities_` and `capital_` are arrays of shapes `(10, 10)` and `(10,)`, and a `DebtrankSystem` given them
  explicitly gives the same distress at every frame
