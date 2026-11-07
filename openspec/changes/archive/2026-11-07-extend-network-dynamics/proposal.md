# Proposal

## Why

Three systems of the reference audit need network dynamics the core lacks. The interacting urns of Iacopini et al.
(2020) give every node a collection of balls that grows and that neighbours read. The TAP model with catalysis of
Hordijk and Kauffman (2022) adds production functions, hyperedges of several sizes, during a run, while topology
requests add and remove only nodes and edges, and a variable group on hyperedges holds one size. The overload
cascades of Motter and Lai read the betweenness of the network as it is after every failure, while betweenness is
available only as an analysis before the run.

## What Changes

- Member sets per node: the agents of a variable group placed on the nodes of a network are the collection of each
  node. `AgentView.neighbors` takes a `group`, as `agents_here` does, so an agent reads the collections of the nodes
  around it. This extends agents on network nodes, the owner index of the audit, rather than adding a concept.
- Hyperedge requests: `TopologyRequestSpec` and `TopologyInteraction` gain the kinds `'add-hyperedge'`, which runs for
  a node, and `'remove-hyperedge'`, which runs for every hyperedge of the variable group on hyperedges. A variable group
  on hyperedges takes a `rank`, the largest size it holds, and pads smaller hyperedges with -1. A new node with a
  hyperedge and no edges of its own draws every member by its weights. Requests take `independent` ends, every
  candidate joining with its weight as a probability, and a `direction` on directed layers.
- XGI 1.0: the project already depends on `xgi>=1.0.0,<2` and locks 1.0.0, so no dependency change is left. The tests
  of hypergraphs and higher-order interactions pass on it.
- Shortest paths: `shortest_paths(active, over=None, reduce='betweenness' | 'closeness')` of
  `skcomplex.core.components.laws.utils` reduces over the shortest paths of the active links, computed by NetworkX on
  the network present and computed again only when the links change. The Numba backend runs the laws that read it in
  Python with a warning.
- `AutocatalyticSetAnalysis` finds the largest RAF of a catalytic reaction network, hyperedges as reactions and edges
  as catalysis, by the algorithm of Hordijk and Steel.
- Catalog systems `interacting_urns_network` and `tap_catalysis` in `economics/innovation`, and `motter_lai_overload` in
  `mathematics/networks`, with tests, inventory entries, and gallery examples.

## Capabilities

### New Capabilities

### Modified Capabilities
- `adaptive-networks`: hyperedge requests, independent and directed new ends, shortest-path reductions, and member sets
  on the nodes.
- `network-analysis`: autocatalytic sets of catalytic reaction networks.
- `model-library`: the interacting urns, TAP catalysis, and Motter-Lai systems.

## Impact

- Core: `TopologyRequestSpec`, `TopologyInteraction`, `GroupElement(rank=...)`, the topology kernel, the layout,
  hyperedge, and topology function passes, `NeighborhoodKernel`, the Numba fallback, `AgentView.neighbors`, a new
  module `laws/utils/_paths.py`, and `core/analysis/_autocatalytic.py`. No new package or dependency.
- Systems: `skcomplex.systems.economics.innovation` and `skcomplex.systems.mathematics.networks`.
- Docs: the networks and agent-based guides, the inventory, and three gallery examples.
