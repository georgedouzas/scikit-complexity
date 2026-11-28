# Design

## Context

See proposal.md for the motivation. The network models are ready-made systems in `systems/mathematics/networks`. The
ensembles in `_ensembles.py` grow their graph step by step through `TopologyRequestSpec` requests (`'add-edge'` for
Erdős-Rényi, `'rewire'` for Watts-Strogatz), while `ConfigurationModelSystem` builds a static graph at `prepare`. The
growth models in `_growth.py` attach new nodes through `TopologyRequestSpec('add-node', toward=...)`, whose `toward`
gives the weight of a candidate end. Directed layers already exist: `TopologyRequestSpec(direction='out')` adds
out-edges, and `NetworkSpace` reads successors on a `nx.DiGraph`.

`NetworkRandomWalkSystem`, `VoterSystem` and `WattsCascadesSystem` are agent systems that run on Mesa. A walker is
placed through the position value `node`, which the trajectory records as `walkers.node` when it is given, but an
agent problem has no named states, so `PassCompiler._copy_with_initial_state` refuses every name. This is why both
`initial_state={'voters.opinion': ...}` in #73 and a chosen start in #72 fail. The initial values of agents live in
`_CompiledAgents.values`, which `MesaSolver._create_agent` reads.

`SpreadingInteraction` of `biology/common/networks` already runs SIR and SIS jumps on the edges of a network, with an
inactive, an active and an optional removed value, and `AdaptiveSISSystem` uses it on a drawn graph. `passage` of
`core/analysis/_passage.py` already reads transition matrices through `pydtmc` for its matrix kinds.

The change `fix-network-model-bugs` is in progress and owns the Holme-Kim edges, the Watts-Strogatz rewiring, the
Erdős-Rényi step check and the docstrings of run lengths and seeds. This change builds on its versions of
`_ensembles.py` and `_growth.py` and plans none of it again.

## Goals / Non-Goals

**Goals:**

- Every request of the issues through options of the existing systems, one new epidemic system, two new kinds of
  `passage`, and one engine fix for agent initial states.
- Initial values that users set are states, so `initial_state` reaches them.

**Non-Goals:**

- The directed degree measures and reciprocity of #66 and the Katz centrality of #72.
- A general random choice of a neighbour in the symbolic laws. The walkers stay agents.

## Decisions

### Placement

- The agent initial states are an engine fix in `core/engine`, by step 2 of the placement rule: the compiler handles
  them by the type of the problem, for agents of every field.
- The two new kinds of `passage` go to `core/analysis`, since mean first-passage and return times of a Markov chain
  serve walks, molecular kinetics and queues alike.
- `ContactNetworkEpidemicSystem` goes to `systems/biology/epidemiology/_adaptive.py`, the module of the epidemics on
  networks, by step 3. Its element `ContactHostElement` is used by this one system, so it stays in that module.
- Every other change is an option of the system that already exists, in its own module.

### Variants are string options

- The new variants follow the scikit-learn convention of the project: `ensemble='newman_watts'`,
  `attachment='uniform'`, `compartments='sis'`, `mode='master'`, `schedule='single'` and `update='synchronous'`, each
  checked in the private `_check` or `_prepare` of its owner with a `ValidationError` that lists the choices.
- `directed` is a boolean, like `simple` of `ConfigurationModelSystem` and `fitness` of `BarabasiAlbertSystem`, since
  it switches one property of the graph rather than naming a model.

### Directed graphs (#66)

- Directed Erdős-Rényi starts from `nx.empty_graph(n_nodes, create_using=nx.DiGraph)` and every node fires
  `TopologyRequestSpec('add-edge', direction='out')` with probability `mean_degree / steps` per step. An out-edge adds
  one to the mean out-degree and one to the mean in-degree, so the rate is twice the undirected one, and the step check
  of `fix-network-model-bugs` becomes `steps >= ceil(mean_degree)`.
- `mean_degree` means the mean out-degree, equal to the mean in-degree, and the docstring says so.
- The directed configuration model draws or takes the pair `(in_degrees, out_degrees)`. Drawn sequences come from the
  same power law, and one stub is added to random nodes of the smaller sum until the sums agree. The pairing permutes
  the in-stubs against the out-stubs. With `simple=True` a bad link `a -> b` and a link `c -> d` drawn at random swap
  their targets into `a -> d` and `c -> b`, which keeps every in-degree and out-degree, the directed form of the
  rewiring that the undirected model already uses.
- Libraries surveyed: NetworkX gives the pairing as `nx.directed_configuration_model`, but its edge swaps
  (`nx.directed_edge_swap`) act on simple graphs only and cannot remove the loops and multilinks of a pairing. The
  pairing may use NetworkX with the seed drawn from `random_state`, and the rewiring stays the small in-house loop the
  undirected model already has.
- The nodes keep the states `degree`, the total degree, and `clustering`. The in-degrees and out-degrees are read from
  the network of the trajectory, and the measures of #66 are a later change.

### Newman-Watts and attachment (#70)

- `ensemble='newman_watts'` builds the same ring and `LatticeLinkElement` as Watts-Strogatz. Every lattice link gains a
  mark `added`, and the interaction on its edges fires `TopologyRequestSpec('add-edge')` from its lattice end with the
  per-step rate `1 - (1 - rewiring) ** (1 / steps)` while the mark is 0, then sets the mark. A shortcut has no lattice
  places, so it never fires. `rewiring` is reused for the shortcut probability, as `p` of
  `nx.newman_watts_strogatz_graph`, rather than adding a parameter that one ensemble reads, and the docstring says what
  it means for each ensemble.
- `offset` changes the weight `toward` of `BarabasiAlbertSystem` from `neighbors(1)` to `neighbors(1) + offset`, also
  under `fitness`, where the weight is `fitness * (k + offset)`. The default 0 keeps today's growth. `_check` raises
  when `offset <= -min(edges, seed_nodes - 1)`, the smallest degree a node holds when it can be drawn.
- `attachment='uniform'` gives `toward=None`, the equal weights of `TopologyRequestSpec`, so a new node joins nodes
  drawn uniformly, the Model A of Barabási (2016). `offset` is unused with uniform attachment, and the
  docstring says so.

### Summary components and superposition (#70)

- `NetworkMeasuresResult.components` becomes `list[IntArray]`, one sorted array per frame, built like `degrees`. A
  ragged list of arrays keeps one entry per frame, since the number of components changes between frames.
- The docstring of `NetworkSuperpositionSystem` says that the network is the disjoint union of the subgraphs, with
  `n_nodes * n_subgraphs` nodes in at least `n_subgraphs` components, and the summary line drops "union".

### Individual epidemics on a given graph (#71)

- A new system rather than a `graph=` option of `NetworkSIRSystem`, because `NetworkSIRSystem` is a metapopulation of
  travellers between cities. Renaming it to free the name was rejected, since its users did not ask for that.
- `ContactNetworkEpidemicSystem(graph=None, *, compartments='sir', beta=0.5, gamma=1.0, infected=0.05,
  random_state=0)` holds one `ContactHostElement('hosts', state=..., space=NetworkSpace(graph))` and one
  `SpreadingInteraction(state='state', beta=beta, gamma=gamma, removed=2.0 if sir else None)`. The default graph is the
  Erdős-Rényi graph of `AdaptiveSISSystem`.
- It runs in continuous time with the Gillespie jumps of `SpreadingInteraction`, the exact individual-level process,
  like `AdaptiveSISSystem`. The discrete steps of Datseris Fig. 10.9 are a choice of that book, and the continuous rates
  keep one law for SIR and SIS.
- `infected` is the share of hosts drawn infected at `prepare` from `random_state`, as in `AdaptiveSISSystem`, and
  chosen nodes come through `initial_state={'hosts.state': x0}`. The group lives on the nodes of a network space and
  is no agent group, so its state already takes an initial value.

### Agent initial states (#72, #73)

- `PassCompiler._copy_with_initial_state` accepts, for an agent problem, the dotted names `<group path>.<state>` of
  `_CompiledAgents.states` and of its position states. It checks that a value is a scalar or has one entry per
  agent of the group, and returns the problem with that spec's `values` replaced. The Mesa solver then creates the
  agents from the new values, and places them by their position values as it already does.
- The error names follow the ones of other problems, so the message of an unknown name lists the dotted names of the
  agent states rather than an empty list.

### Walks on networks (#72)

- `NetworkRandomWalkSystem(start=None)` places the walkers by the position value `node`: `start` itself, the entries of
  a sequence of nodes, or nodes drawn uniformly from `random_state` when it is `None`. The system gains `random_state`
  for that draw. Giving the value makes the trajectory record `walkers.node`, which `passage(kind='first-passage',
  target=...)` then reads.
- `mode='master'` reuses the parts of `PagerankSystem` in the same module: `NodeRankElement('nodes', rank=p0,
  degree=k)` on the network and `PagerankInteraction(damping=1.0)`, whose update `r_i' = Σ_j A_ij r_j / k_j` is
  `p(t + 1) = p(t) P` when nothing jumps. A new element and law for the same update were rejected, since the
  constitution asks to reuse a capability the code already has. The probability is therefore the state `nodes.rank`,
  and the docstring says that it is the probability of a walker at every node. `p0` is the indicator of `start`, the
  share of `start` over a sequence of nodes, or the uniform law. `n_walkers` is unused in this mode, and the docstring
  says so.
- A directed graph or a graph with isolated nodes raises a `ValidationError` in the master mode, since `P` is then not
  stochastic.
- `passage(kind='mean-first-passage-matrix')` returns the matrix of `MarkovChain.first_passage_times_to` for every
  target, with the diagonal from `MarkovChain.recurrence_times`, both of `pydtmc`, the library `passage` already uses.
  `passage(kind='mean-return-times')` returns `recurrence_times`. A NetworkX graph as the source is turned into
  `P = D^-1 A` in node order. The result holds them as `mean_first_passage_matrix` and `mean_return_times`.

### Voters (#73)

- With the engine fix, `voters.opinion` is a state that `initial_state` sets, and the docstring of `VoterSystem` shows
  `initial_state={'voters.opinion': x0}`.
- `schedule` is a parameter of `VoterSystem` and of `VoterImitationInteraction`. `'random-sequential'` keeps the sweeps
  of today. `'single'` draws one voter uniformly per step and lets it copy a random neighbour, so the degree-weighted
  magnetisation is a martingale, and one unit of time is one update. The user named this option `schedule='single'`.
  The local rules of lattices use `'random-site'` for as many random cells as there are cells, which is not the same
  thing, so `'single'` is a new value of the voter alone.

### Cascades and outputs (#74)

- `WattsCascadesSystem(update='random-sequential')` keeps today's order, and `update='synchronous'` sets every node from
  the activations at the start of the step. The name and values follow `_update_agents` of `BaseInteraction`, which
  already takes `'synchronous'` and `'random-sequential'`, and `update=` of `RandomBooleanNetworkSystem`.
  `ThresholdActivationInteraction` gains `update` and calls `_update_agents`.
- `DebtrankSystem._prepare` sets `liabilities_` and `capital_` to the arrays it builds the impact matrix from, given
  or drawn. They are derived state with the trailing underscore, set by `prepare`, like `system_`.
- The docstring of `NetworkDiffusionSystem` says that frames come every `dt = 0.1` units of time by default, so a frame
  index is not a time, and that `trajectory.times` gives the time of every frame.
- The docstring of `MotterLaiOverloadSystem` says that `nodes.giant` holds, for every node present, the size of the
  largest component at the start of the step, the same for all of them, and that a node that has failed reads `nan`,
  so the size at a frame is the largest value of its row.
- The docstring of `WattsCascadesSystem` states the order of both updates, with the counts 1, 5, 11 of the sequential
  update on a cycle against 1, 3, 5 of the synchronous one.

## Risks / Trade-offs

- [The `'add-edge'` request may not run from a law on the edges of a network] → The Newman-Watts shortcuts then fire
  from the nodes, every node counting its unused lattice links in a state, which gives the same law.
- [Agent initial states touch the compiler that every agent system runs through] → The change only replaces values a
  name selects, and the agent tests of `tests/core/engine/compile/test_pass_compiler.py` and
  `tests/core/engine/solvers/test_mesa.py` cover a system with and without them.
- [The voter martingale scenario is statistical] → It uses 4000 runs of an 11-node star, which take seconds, and a
  tolerance of three standard errors.
- [`fix-network-model-bugs` changes the same modules] → This change is applied after it, and its tasks start by
  checking that the bug fix is merged.
