# Proposal

## Why

Thurner et al. (2018 §1.5) define complex systems as co-evolving multilayer networks: states change through
interactions, and the interactions change through the states. The references build on this:
- **Growing and rewiring networks:** Barabási-Albert growth, Watts-Strogatz rewiring, and adaptive epidemics with
  rewiring (Thurner et al. 2018 ch. 4; Gros 2015 ch. 1).
- **Relations as edges:** credit, employment, and imitation links that agents create and break (Caiani et al. 2016
  ch. 2-3; Chesney 2021 ch. 5).
- **Synchronisation on networks:** Kuramoto oscillators coupled through adjacency (Jensen 2023 ch. 7; Datseris and
  Parlitz 2022 ch. 9-10).
- **Neighbourhood reductions:** flocking aligns with the normalised mean velocity of neighbours, a reduction over a
  neighbourhood rather than a sum of pair forces (Argun et al. 2021 ch. 8; Charbonneau 2017 ch. 10). Some variants
  count only the k nearest neighbours or a vision cone.

A `NetworkSpace` is built by hand from a NetworkX graph, interactions read no edge state, the topology never changes
outside the Mesa backend, and neighbourhoods in continuous space are radius searches inside pair interactions.

**Rebased on `redesign-system-model`.** The redesign already declares state on edges as `Group(on='edges')` on a
`NetworkSpace`, groups whose size laws change as `Group(variable=True)`, the bindings `'neighbours'` and `'hyperedges'`,
and their validation, and it runs variable groups on the Mesa backend. It leaves to this change the reading of edge
states by interactions, the laws that change the topology, the compiled execution of variable groups outside Mesa, and
reductions over neighbourhoods. This change therefore no longer adds an `Edges` component: edge states are edge groups,
the topology changes by adding and removing members of variable site and edge groups, and node groups resize through
the same mechanism. It still adds the network generators, multilayer networks, interactions over edges that read and
write edge states, topology requests, compiled variable groups, neighbourhood reductions, network snapshots, and the
library models.

## What Changes

- Add network generators in `core.space` as space factories: Erdős-Rényi, Watts-Strogatz, Barabási-Albert,
  configuration model, stochastic block model, lattices, complete graphs, and the spatially embedded Waxman and random
  geometric graphs, whose node positions laws can read, seeded from the simulation. The spatial generators come from
  `survey-thin-references`, for the spatial variant of `triadic_percolation`.
- Let interactions with the binding `'neighbours'` bind an edge group of the same network, so their laws run once per
  edge, read the states of the source and target members and of the edge member, and target any of them. Directed
  networks give each edge its source and target.
- Add multilayer networks: `Layer` in `core.space` names a relation over the same nodes, a network space holds several
  layers, and each interaction and edge group binds to one layer.
- Add topology requests for `Update`, `Jump`, and `Rule` laws: add or remove edges and nodes, and rewire edges. A
  request adds or removes members of a variable group on the edges or sites of the network, so groups on network nodes
  resize with the nodes and edge groups with the edges.
- Compile variable groups for the NumPy and Numba backends with capacity layouts and active masks, so topology changes
  run outside Mesa.
- Add neighbourhood reductions inside laws in `core.laws`: sum, mean, count, max, and normalised mean of neighbours'
  values, over network neighbours weighted by an edge state, over a radius, over the k nearest neighbours, or within a
  vision cone.
- Declare every new primitive for validation, so `validate()` reports a topology request on a fixed group, a reduction
  or interaction over a layer the space lacks, and a metric neighbourhood outside a continuous space, and solvers list
  their support for variable and edge groups.
- Record network snapshots in trajectories and log them to Rerun as graphs with node and edge colors.
- Add library models: Kuramoto on complete graphs and networks, the power-grid swing equation, SIS and SIR on networks,
  adaptive SIS with rewiring, Barabási-Albert growth, voter and majority models on networks, random Boolean networks,
  coupled logistic maps, Vicsek flocking, and DebtRank interbank contagion.

## Capabilities

### New Capabilities

- `adaptive-networks`: network generators, interactions over edges, layers, topology requests, compiled variable
  groups, neighbourhood reductions, network snapshots, and their validation declarations.

### Modified Capabilities

- `model-library`: network models.

## Impact

- `skcomplex.core.space` gains `generators`, `Layer`, and the neighbourhoods `Radius`, `Nearest`, and `Cone`.
- `skcomplex.core.interactions` lets the `'neighbours'` binding bind an edge group of the same network and layer.
- `skcomplex.core.laws` gains the topology requests `add_edge`, `remove_edge`, `rewire`, `add_node`, and `remove_node`,
  and the general neighbourhood reduction `neighbors`, of which the existing lattice `neighbors` and `neighbor_sum` are
  the lattice case.
- `skcomplex.core.compile` and the NumPy and Numba backends in `skcomplex.core.backends` support variable groups with
  capacity-based layouts and active masks, and declare that support.
- Network measures come from the existing network analysis.
