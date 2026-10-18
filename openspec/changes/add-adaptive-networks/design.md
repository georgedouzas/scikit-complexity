# Design

## Context

Groups have fixed sizes and compile once, pair interactions find neighbours by radius with cKDTree or cell lists, and
`NetworkSpace` places Mesa agents on NetworkX nodes. Kernels evaluate members or pairs over arrays.

## Goals / Non-Goals

**Goals:**

- Thurner et al. (2018 §1.5): states and interactions co-evolve, with interactions of several types as layers.
- Neighbourhood reductions usable in any space and any law kind.

**Non-Goals:**

- Hypergraphs and simplicial interactions (Thurner et al. 2018 §4.9). Layers leave room for them.
- Distributed graphs beyond one machine.

## Decisions

### Networks as spaces with arrays

A network space holds CSR adjacency arrays per layer, plus edge-state arrays aligned with the edges. NetworkX graphs
convert in and out. Generators take the simulation's random stream so networks are reproducible.

### Edge states

`Edges(label, layer, **states)` declares states per edge of a layer, laid out as a group whose members are edges, with
symbols for the source and target nodes' states in its laws. Edge laws of any kind update weights, as in adaptive
weights and plasticity (Gros 2015 ch. 7).

### Neighbourhood reductions

`neighbors(expression, over=layer | Radius(r) | Nearest(k) | Cone(r, angle), reduce='sum' | 'mean' | 'count' | 'max'
| 'normalized-mean', weight=None)` is a Sage function whose argument is written for the neighbour, such as
`sin(s.other.theta - s.theta)`. The lowering evaluates it with segment reductions over CSR rows for networks, and over
neighbour lists from cKDTree or cell lists for metric spaces. The Kuramoto coupling becomes
`Rate(s.theta, s.omega + K * neighbors(sin(s.other.theta - s.theta), reduce='mean'))`.

### Topology laws and variable groups

Topology laws return requests that the stepper applies between steps: edge insertions and deletions, rewiring, and
node creation and removal. Groups on network nodes get a capacity, an active mask, and a free list, so compiled kernels
keep fixed array shapes and skip inactive members. When the capacity is exceeded, the layout grows and the kernels are
compiled again. Trajectories store node ids per frame, as agent snapshots do.

### Snapshots and visualization

Networks are recorded every frame whose topology changed, and Rerun logs them as `GraphNodes` and `GraphEdges`, with
node colors from states and edge colors from edge states.

## Risks / Trade-offs

- [Variable sizes in compiled kernels] → Capacities with active masks keep shapes fixed, and growth is amortised by
  doubling.
- [Reductions over large neighbourhoods] → Segment reductions are vectorised on NumPy and fused on Numba.
