# Design

## Context

- `NetworkSpace(graph, hyperedges=())` stores hyperedges as sequences of at least two nodes and checks them against the
  graph. `GroupElement(on='hyperedges')` has one member per hyperedge.
- `HyperedgeInteraction(label)` is the base of interactions whose laws run per hyperedge. The engine compiles its laws
  once per member and ignores the hyperedges, as `general-elements-and-interactions` recorded. The pass compiler
  reports a hyperedge interaction on a space without hyperedges with the issue code `binding`.
- `NeighborInteraction` runs laws once per edge with `namespace.source`, `namespace.target`, and `namespace.edge`, and
  the reduction `neighbors(expression, over, reduce, weight)` reads `namespace.other`. `CouplingInteraction` builds on
  both, with the modes `'edges'`, `'group'`, and `'next'`.
- `GroupingInteraction(target, groups, act, size, by, capacity, phase)` draws `'pairs'`, `'partition'`,
  `'neighborhood'`, or `'state'` groups every step and runs `act(members, model)` as an agent `RuleLaw`. Galam
  (`DiscussionsInteraction`), Kolkata (`RestaurantsInteraction`), the tournaments, and the random-pair opinion models
  use it.
- The reductions `total`, `mean`, `count`, and their keyed form `by=` exist in `skcomplex.core.laws.utils`.

## Goals / Non-Goals

**Goals:**
- Run laws over hyperedges of any size on the same symbolic and compiled path as the laws over edges.
- Give the group mechanisms that agent groupings run a symbolic form where the mechanism is a law of the group.
- Let a node gate a link, and let a tensor couple k members.

**Non-Goals:**
- Persistent homology of hypergraphs, which belongs to analysis libraries.
- A symbolic form of Kolkata and of the minority game. Kolkata serves one random diner per restaurant, a choice inside a
  group that is not a law of the group. The minority game reads one group of all players, which `total` and `mean`
  already reduce. Both stay as they are.
- Cell complexes and link fields, which `add-link-fields` adds.

## Decisions

### Hyperedges from XGI

XGI (CompleXGroupInteractions) is the maintained library for hypergraphs and simplicial complexes, with generators,
statistics, and incidence matrices. `NetworkSpace(graph, hyperedges=...)` takes an XGI `Hypergraph` or
`SimplicialComplex` as well as sequences of nodes, stored unmodified, and its `prepare` derives `incidence_`, the
sparse incidence matrix, and `memberships_`, the hyperedges around every node. A simplicial complex gives its faces
of size at least two. An XGI hyperedge is a set, so its members follow the order of the nodes of the graph, and
`prepare` also derives `hyperedges_`, the node indices of every hyperedge in order, which the incidence matrix does not
keep and the gather tables read. Sequences of nodes keep their own order.

Survey (November 2026):

| | XGI 0.10.2 | HyperNetX 2.4.3 |
|---|---|---|
| Licence | BSD-3-Clause | BSD-3-Clause |
| Wheels | pure Python `py3-none-any`, every platform | pure Python `py3-none-any`, every platform |
| Python | 3.10 or later | 3.10 to 3.x |
| Maturity | 47 releases since 2021, last release May 2026, CompleX Group Interactions team | 50 releases, last release July 2026, PNNL |
| Simplicial complexes | `SimplicialComplex`, a subclass of `Hypergraph` | none, homology behind an optional extra |
| Incidence and generators | `incidence_matrix`, random and simplicial generators, dynamics such as the simplicial Kuramoto | incidence matrix, few generators |
| Dependencies | `numpy`, `scipy`, `pandas`, `networkx`, `matplotlib`, `requests`, `xopen` | `igraph`, `pandas<3`, `concepts`, `bitsets`, `scikit-learn`, `matplotlib`, and more |

XGI is chosen. Both are mature and installable everywhere, but HyperNetX pins `pandas<3`, which conflicts with the
pandas 3 the project already locks, adds `igraph` and two lattice libraries the core does not use, and has no
simplicial complex. XGI is a runtime dependency, since `skcomplex.core.spaces` imports it, locked with its new
dependencies `xopen` and `backports-zstd` and no other change to the lock.

### Laws per hyperedge

`HyperedgeInteraction(label, size=None)` gains the size of the hyperedges it runs on, `None` for every size. Its laws
are written for one hyperedge and read:
- `namespace.members`, the members of the hyperedge, which `total`, `mean`, `count`, and `maximum` reduce;
- `namespace.member`, the focal member whose state a rate, noise, or jump law changes;
- `namespace.others`, the other members in hyperedge order, a tuple when `size` is given;
- `namespace.hyperedge`, the member of a group on hyperedges that a second role binds;
- `namespace.edge`, the member of a group on the edges of the same network joining the first two members of the
  hyperedge, when the second role binds an edge group.

A law of the focal member runs once per member of every hyperedge and scatter-adds into that member, the pattern pair
interactions already use. A law of the hyperedge or edge member runs once per hyperedge, with the first member of the
hyperedge as the focal member. `engine/compile/passes/_hyperedges.py` holds `HyperedgeTables`, which builds the
tables of every size and expands the reductions over `namespace.members`, and `SpecsPass` builds two kernel specs per
size from them, the laws of the focal member over `size * E` rows and the laws of the hyperedge over `E` rows, marked
by the field `hyperedge` of `KernelSpec`. Every symbol of a slot is a gather reference, `ReadKind.NEIGHBOR` with a
`NeighborReadSpec` whose table holds the member of every row and no fill value, so the existing readers of the NumPy
backend gather it and `_add_into` and the update writers scatter into it. The Numba printer prints a gather table as
integer data of the layout, `x[start + L[table + row] * columns + column]`, and runs such kernels in one thread, since
their scatter-adds share members. A reduction over `namespace.members` of a hyperedge of size `k` is written out as a
sum, a mean, a count, or an extremum over the focal member and the `k - 1` other members before lowering.

Issues: an interaction whose `size` no hyperedge of the space has gets the existing code `binding`, naming the size and
the sizes the space has. An edge role whose first two members are not joined by an edge of the bound group gets
`binding` too.

### Reductions over the hyperedges of a member

`hyperedges(expression, reduce='sum', size=None, weight=None)` in `skcomplex.core.laws.utils` reduces an expression
written for one hyperedge around every member, with `namespace.others` and `namespace.members` inside and the reading
member outside, as `neighbors` does for one neighbour. The reductions are those of `neighbors`. A law of any group on
the nodes of a network with hyperedges may use it. Outside such a network the compiler reports `law-space`.

`CouplingInteraction(mode='hyperedges', size=...)` uses it: member `i` gets the reduction over its hyperedges of
`function(ns)`, so the three-body Kuramoto coupling is
`CouplingInteraction(mode='hyperedges', size=3, read='theta', write='theta',
function=lambda s: sin(s.others[0].theta + s.others[1].theta - 2 * s.target.theta))`.

### Symbolic groupings

`GroupingInteraction` gains `form='agent'`, the default that keeps `act`, and `form='symbolic'` with the arguments
`write`, the state it sets, `function`, a function of the namespace of one group that returns the contribution of the
group to its member, and `reduce`, the reduction over the groups that hold a member, `'sum'` by default. The symbolic
form compiles on the hyperedge path: the groups of a step are the hyperedges of that step. Pairs and partitions are
drawn every step from the simulation's random stream and fill a per-step index table, neighbourhoods are fixed by the
space and built once, and groups by state are the keyed reductions `by=`. The form lowers to an `UpdateLaw` of `write`
in discrete time, and `act` and `capacity` belong to the agent form alone, which `_check` enforces.

- Galam: `GroupingInteraction(form='symbolic', groups='partition', size=3, write='opinion',
  function=lambda s: heaviside(mean(s.members.opinion, over=s.members) - 1 / 2))`, with ties going to 0 through the
  strict threshold.
- Spatial public goods: `groups='neighborhood'` on the von Neumann lattice, whose groups of five each pay
  `synergy * total(strategy) / 5 - strategy` to every member, summed over the five groups a member belongs to, followed
  by the Fermi imitation of `ImitationInteraction`.

### Triadic regulation

Regulation is a composition. The links carry a Boolean activity in a group on the edges of the network. Each regulated
link and its regulating node form a hyperedge of size three, whose first two members are the ends of the link and whose
third is the regulator. A `HyperedgeInteraction` subclass in `mathematics/networks` binds the edge group as its second
role, and its `UpdateLaw` sets `namespace.edge.active` from the regulator, so percolation alternates with regulation
every step, as in Millán et al.

### Coupling tensors

A k-body tensor is a parameter of an interaction declared `ParameterSpec(shape=('target',) * k)`, the shaped
parameters `add-learning-dynamics` adds, whose value a factory draws with its `random_state`. No core class draws it.
`contraction(tensor, *expressions)` in `skcomplex.core.laws.utils` returns `sum_jk T_ijk a_j b_k` for every member `i`,
with one expression per index after the first, and lowers to `numpy.einsum` and a Numba loop. A shaped parameter whose
role binds a single element gets the existing issue code `parameter-shape`, with the fix of binding a group.

### Growth by hyperedges

`TopologyInteraction(action='add-node', hyperedge=None)` gains `hyperedge`, the size of the hyperedge every new node
forms with `hyperedge - 1` existing nodes, drawn along an edge for size three, as the triangle attachment of a growing
simplicial complex does. The hyperedge tables of a network whose groups are variable grow with its nodes.

### Models

| Model | Field path | Built on |
|---|---|---|
| `hypergraph_kuramoto` | `physics/synchronization` | `CouplingInteraction(mode='hyperedges', size=3)` |
| `higher_order_random_lotka_volterra` | `biology/ecology` | `RateInteraction` with `contraction` |
| `hypergraph_consensus` | `social/opinion_dynamics` | `HyperedgeInteraction`, a hyperedge group of activities |
| `simplicial_social_dilemmas` | `social/cooperation` | `CouplingInteraction` on edges and hyperedges |
| `hypergraph_public_goods` | `social/cooperation` | `HyperedgeInteraction`, `ImitationInteraction` |
| `simplicial_contagion` | `social/spreading` | `TransitionInteraction` with `hyperedges` |
| `hypergraph_exposure_sis` | `social/spreading` | `HyperedgeInteraction` on a hyperedge group of doses |
| `triadic_percolation` | `mathematics/networks` | `HyperedgeInteraction` with an edge role |
| `hypergraph_random_walk` | `mathematics/networks` | `JumpInteraction` with `hyperedges` weights |
| `growing_simplicial_complex` | `mathematics/networks` | `TopologyInteraction(hyperedge=3)` |

`galam_majority` and `public_goods_game` are rebuilt on the symbolic grouping, `DiscussionsInteraction` becoming a thin
subclass of the symbolic form, and are compared over 20 seeds with their previous trajectories.

## Risks / Trade-offs

- [Large hyperedges make gathers wide] → Hyperedges are grouped by size, one index table per size.
- [A coupling tensor of order k over N members holds N^k entries] → The documentation states the memory, and the probe
  model keeps N at 100 for order 3.
- [Rebuilt groupings draw in a different order] → Galam and the public goods game are accepted only when they agree
  statistically over 20 seeds.
