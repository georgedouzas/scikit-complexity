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

`NetworkSpace` is a static substrate today, edges carry no state, and neighbourhoods are radius searches.

## What Changes

- Add network generators as space factories: Erdős-Rényi, Watts-Strogatz, Barabási-Albert, configuration model,
  stochastic block model, lattices, and complete graphs, seeded from the simulation.
- Add edge states: weights and other variables on edges, with their own laws, and directed edges.
- Add multilayer networks: named relation layers over the same nodes, each bound to its own interactions.
- Add topology laws: add or remove edges and nodes, and rewire edges, from Update, Jump, and Rule laws. Element groups
  on network nodes resize with them.
- Add neighbourhood reductions inside laws: sum, mean, count, max, and normalised mean of neighbours' values, over
  network neighbours weighted by edges, over a radius, over the k nearest neighbours, or within a vision cone.
- Record network snapshots in trajectories and log them to Rerun as graphs with node and edge colors.
- Add library models: Kuramoto on complete graphs and networks, the power-grid swing equation, SIS and SIR on networks,
  adaptive SIS with rewiring, Barabási-Albert growth, voter and majority models on networks, random Boolean networks,
  coupled logistic maps, Vicsek flocking, and DebtRank interbank contagion.

## Capabilities

### New Capabilities

- `adaptive-networks`: network generators, edge states, layers, topology laws, neighbourhood reductions, and network
  snapshots.

### Modified Capabilities

- `model-library`: network models.

## Impact

- `skcomplex.core.model` gains `generators`, `Edges`, `Layer`, `neighbors`, and the topology laws `AddEdge`, `RemoveEdge`,
  `Rewire`, `AddNode`, and `RemoveNode`.
- Group sizes become variable outside Mesa. The compiler supports this with capacity-based layouts and active masks.
- Depends on `add-general-model-interface` for observables and on `add-analysis-toolkit` for network measures.
