# Design

## Context

After `redesign-system-model`, a system is its components, interactions, space, and time. A `NetworkSpace(graph,
hyperedges=)` holds a NetworkX graph; `Group(on='sites')`, `Group(on='edges')`, and `Group(on='hyperedges')` hold one
member per node, edge, or hyperedge; `Group(variable=True)` lets laws add and remove members, which only the Mesa
backend runs today; and an `Interaction` binds its laws `over` `'members'`, `'pairs'` within a cutoff found with cKDTree
or cell lists, `'neighbours'` of the space, or `'hyperedges'`. Kernels evaluate members or pairs over arrays. Every
primitive, solver, and backend declares `Requires` or `Supports`, and `validate()` reports what does not fit.

## Goals / Non-Goals

**Goals:**

- Thurner et al. (2018 §1.5): states and interactions co-evolve, with interactions of several types as layers.
- Neighbourhood reductions usable in any space and any law kind.
- Build every feature on the redesign's groups and bindings, adding no component kind and no law kind.

**Non-Goals:**

- Laws run once per hyperedge, reductions over hyperedges, and regulatory hyperedges, which `add-higher-order-
  interactions` specifies.
- Distributed graphs beyond one machine.

## Decisions

### Networks as spaces with arrays

A network space holds CSR adjacency arrays per layer. Edge groups hold arrays aligned with the edges of their layer.
NetworkX graphs convert in and out. Generators in `core.space` take the simulation's random stream so networks are
reproducible.

### Layers

`Layer(label, graph)` names a relation over the nodes of a network space, and `NetworkSpace(graph, layers=[...])` holds
several layers over the same nodes, the plain graph being the default layer. An edge group and an interaction with the
binding `'neighbours'` name their layer, and its neighbours and edges are the only ones they see.

Declaration: a layer is part of the network space kind. Validation reports an edge group or an interaction that names a
layer the space does not have, with the existing code `binding`, and lists the layers of the space.

### Interactions over edges

An interaction with `over = 'neighbours'` may bind, besides its node roles, an edge group of the same network and layer.
Its laws then run once per edge and read `s.source`, `s.target`, and `s.edge`, the members at both ends and the edge
member, and may target any of them, so edge laws of any kind update weights, as in adaptive weights and plasticity (Gros
2015 ch. 7), and node laws read edge weights. On a directed graph `source` and `target` follow the edge's direction; on
an undirected graph they follow node order. Interactions stay stateless, as the redesign requires.

Declaration: the binding declares that it reads an edge group. Validation reports an interaction that binds a group
on edges of another space or layer with the code `binding`.

### Neighbourhood reductions

`neighbors(expression, over=layer | Radius(r) | Nearest(k) | Cone(r, angle), reduce='sum' | 'mean' | 'count' | 'max'
| 'normalized-mean', weight=None)` in `core.laws` is a Sage function whose argument is written for the neighbour, such
as `sin(s.other.theta - s.theta)`, and `weight` may be a state of an edge group. The existing lattice `neighbors` and
`neighbor_sum` become its lattice case, with the lattice's neighbourhood as `over`. The lowering evaluates it with
segment reductions over CSR rows for networks, and over neighbour lists from cKDTree or cell lists for continuous
spaces. The Kuramoto coupling becomes `Rate(s.theta, s.omega + K * neighbors(sin(s.other.theta - s.theta),
reduce='mean'))`.

Declarations: `Radius`, `Nearest`, and `Cone` declare `Requires(space={SpaceKind.CONTINUOUS})`, a layer requires
`SpaceKind.NETWORK`, and a lattice neighbourhood requires `SpaceKind.LATTICE`. Validation reports a reduction whose
neighbourhood the space does not offer with the existing code `law-space`, and a cone without a vector velocity or
heading state with `law-domain`.

### Topology requests and variable groups

Topology requests are values that `Update`, `Jump`, and `Rule` laws return, `add_edge`, `remove_edge`, `rewire`,
`add_node`, and `remove_node`, and the stepper applies them between steps. Each request adds or removes members of a
variable group: an edge request changes the members of the variable edge group of its layer and the graph with it, and
a node request changes the members of every variable group on the network's sites, with the new members' states from
the request or their element's defaults. No law kind is added: the requests change the targets of existing laws.

Variable groups compile for the NumPy and Numba backends with a capacity, an active mask, and a free list, so compiled
kernels keep fixed array shapes and skip inactive members. When the capacity is exceeded, the layout grows and the
kernels are compiled again. Trajectories store node and edge ids per frame, as agent snapshots do.

Declarations: the NumPy and Numba backends and the ODE, SDE, jump, and discrete-map solvers that support them add
`ComponentKind.VARIABLE_GROUP` and `ComponentKind.EDGE_GROUP` to their supported components. Validation reports:
- a topology request on a fixed group, with the code `group-fixed` and the fix `variable=True`;
- a node request on a network whose site groups are not all variable, naming the fixed ones;
- a solver that does not support variable groups, with the existing `solver-unsupported`, listing the solvers that do.

### Snapshots and visualization

Networks are recorded every frame whose topology changed, and Rerun logs them as `GraphNodes` and `GraphEdges`, with
node colors from states and edge colors from edge states.

## Risks / Trade-offs

- [Variable sizes in compiled kernels] → Capacities with active masks keep shapes fixed, and growth is amortised by
  doubling.
- [Reductions over large neighbourhoods] → Segment reductions are vectorised on NumPy and fused on Numba.
- [Generalising the lattice `neighbors` changes its signature] → The lattice call keeps working, with the lattice
  neighbourhood passed as `over`, and the existing lattice tests run unchanged.
