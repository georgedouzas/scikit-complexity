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

- **Connected components are a core reduction.** Percolation and the giant component are a fundamental graph operation
  used across domains (epidemic thresholds, infrastructure failure, habitat connectivity, social cohesion), so
  `skcomplex.core.laws.utils` gains a reduction over the connected components of a network's active links, giving
  component labels and the size of the largest component. `triadic_percolation` builds on it. Decided on 2026-11-01.
- **The components reduction is a neighbourhood reduction.** `components(active, over=None, reduce='label')` in its
  own module of `skcomplex.core.laws.utils` writes the activity of one link as `neighbors` writes the value of one
  neighbour, reading `namespace.other`, `namespace`, and the states of the edge group of the layer, and a link is
  active where the expression is not 0, so a condition or an edge weight selects the links. The reductions are
  `'label'`, the smallest member of the component, `'size'`, the size of the member's component, `'largest'`, the size
  of the largest component, and `'giant'`, 1 for the members of the largest component, the one of the smallest label
  among ties, so a node tests its membership of the giant component without comparing sizes. The Sage function and
  its kinds sit beside `neighbors` in `_reductions.py`, and `neighborhoods_in` returns its calls with the kinds
  `'component-label'`, `'component-size'`, `'component-largest'`, and `'component-giant'`, so every pass that finds,
  validates, and lowers neighbourhood reductions takes it unchanged. `NeighborhoodKernel` evaluates the activity of
  every pair and runs SciPy's `connected_components` on the active pairs, read without direction, and the Numba
  printer collects the active pairs in one serial loop and calls a union-find in `_numba_components.py`. Any
  neighbourhood works, so lattices and radius neighbourhoods give site and continuum percolation. networkx was
  rejected for the NumPy side, since it builds a Python graph every step, and SciPy is already a runtime dependency.
  Decided on 2026-11-01.
- **Triadic percolation alternates by step parity.** Update laws of one step read the states at its start, so the
  regulation of the links and the giant component of the nodes cannot run in one step. `triadic_percolation` runs one
  time step of Millán et al. as two steps: the links follow their regulators in even steps,
  `where(mod(t, 2) < 1, ...)`, and the nodes take the giant component of the active links in odd steps. Every link has
  one regulator, as the hyperedge of size three of the design holds, positive or negative, so the mean-field map is
  `R' = S(c p ((1 - nu) R + nu (1 - R)))` for the share `nu` of negative regulators. With negative regulators alone the
  map decreases, its bifurcation point is the percolation point `p = 1 / c`, and above it the giant component has
  period two, which the tests verify. Several regulators per link, which period doubling to chaos needs, would need
  update laws of one edge from several hyperedges to combine, which the design does not cover. Decided on 2026-11-01.

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
function=lambda s: sin(s.others[0].theta + s.others[1].theta - 2 * s.member.theta))`.

Implementation (task 2.3):
- The namespace of a member of a group on a network with hyperedges, `s` of its own laws, `s.<role>` of an
  interaction, and both ends `s.source` and `s.target` of an interaction over neighbours, holds `others`, as many
  slots as the largest hyperedge has other members, and `members`, read as `ReadKind.OTHER`. A group that declares a
  variable named `others` or `members` keeps it.
- The reduction lowers to the existing `NeighborhoodKernel`, one per size, whose `StaticSearch` rows are the
  hyperedges of that size around every member, grouped by member (`HyperedgeTables._incidences`). The new field
  `table` of `PairReadSpec` gives the other member of every row for one position, so the NumPy and Numba kernels gather
  `others[p]` through it; the member reads itself. Reductions over `members` are written out with the expansion of the
  hyperedge pass over the member and the other members.
- Without a size and with several sizes in the network, the sum and the count add the kernels of every size and the
  mean divides their sum by the number of hyperedges around the member; the other reductions need a size. Reading
  more other members than a size has, `members` outside a reduction over them, or the neighbour `other` raises
  `ValidationError`.
- Issues: a reduction in a component outside a network with hyperedges gets `law-space`, and a size the network does
  not have gets `binding`, both from `PassCompiler._around_issues`.
- The coupling function reads `s.member`, as the other modes do, `s.others`, cut to `size - 1` when the size is given,
  and `s.members`. The default term is the sum over the other members of `x_j - x_i`, with the lag on `x_i`, a
  callable `weights` gives the weight of one hyperedge, and the mode takes no `include_self`, `delay`, or agent form.
- `hypergraph_kuramoto` keeps `SimplexCouplingInteraction(HyperedgeInteraction)`. Two couplings of sizes 2 and 3 give
  its drift within 9e-16, but summing in another order moves its trajectory by up to 4e-4 over its default run, so the
  trajectories are not identical, and `K1` and `K2` would stop being parameters of the interaction.

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

Implementation (November 2026):

- `SpecsPass` routes a symbolic grouping of pairs, partitions, or neighbourhoods to `HyperedgeTables._groups_of`,
  which returns the groups by size, and builds one spec of the focal member per size with
  `SpecsPass._hyperedge_spec`. A partition of `N` members into groups of `k` has a last group of `N mod k` members, and
  an odd member forms no pair, as in the agent form.
- Pairs and partitions hold positions, not members. `NeighborReadSpec` gains `order`, an array of the members at every
  position, and the property `indices`, the member every row reads. `DrawLowering` collects the orders of the specs
  into `DrawKernel(orders=...)`, which draws every order again in place with `rng.permutation` after the other draws,
  so a problem with orders is stochastic. The Numba printer reads a position through the order as
  `L[order + L[table + row]]`, and the Numba backend copies every order into the layout before every call of a bound
  function. Since the orders are drawn in Python, the compiled loop `iterate_random` is not used for such problems.
- A member takes the sum or the mean of its groups through shares, so the kernels of the sizes stay independent:
  `KernelSpec` and `OutputSpec` gain `reduce` and `counts`, the number of groups that hold the member of every row,
  and every row adds `v - x / m` for a sum, or `(v - x) / m` for a mean, to the copy of the state. A member in no group
  keeps its value. `KernelsPass` exempts these update laws from the check of one update law per state.
- Groups by state stay in the core: `GroupingInteraction._get_laws` runs the function on the namespace of the target,
  with `members` the other member of its reductions and no `others`, and adds the condition that the other member
  shares the value of `by` to every reduction over the members.
- The NumPy writer of a combined update law of a gathered target now adds with `np.add.at`, as the Numba writer adds
  row by row, so the changes of several rows of one member add on both backends.
- `galam_majority` runs on the symbolic grouping, `DiscussionsInteraction` setting `opinion` to
  `where(mean(opinion) > 1/2, 1, 0)` over partitions. Its test and gallery example read the states of the group, since
  a symbolic group has no agent snapshots. Over 20 seeds from shares 0.4 and 0.6, the mean share at every step agrees
  with the agent form within 1.1 standard errors.
- `public_goods_game` keeps its `LocalRuleInteraction`. A discrete map updates every state from the state at the start
  of the step, so a grouping that writes the payoffs and an imitation that reads them imitate the payoffs of the step
  before. Over 20 seeds the rebuilt model ends with a share of cooperators of 0.95 against 0.49 at synergy 4.5. Also,
  `ImitationInteraction(rule='fermi', form='probability')` is biased in `development`: with equal payoffs, a member
  of state 1 never switches and a member of state 0 copies a neighbour of state 1 with probability `k / degree`
  instead of `k / (2 degree)`, although its weights and its categorical draw are right when evaluated alone.

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

- Storage: the hyperedges that grow are the rows of `GroupElement(on='hyperedges', variable=True, capacity=n)`, a
  capacity group like the variable groups on nodes and edges. The layout pass gives it the hidden parameter
  `_members`, one column per member, beside `_active`, so the initial hyperedges of the network must have one size,
  which every new hyperedge also has. `CapacityRowsSpec.members` holds the entry, `NetworkRowsSpec.hyperedges` the
  group, and `NetworkSnapshotSpec.hyperedges` the members of the hyperedges present at a frame.
- Drawing: a new node draws an edge of its layer uniformly, whose ends are the first two members, then the other
  `hyperedge - 3` members by `toward`, and joins each member by an edge; it is the last member, so an edge joins the
  first two members of every hyperedge. The request ignores `edges`. Without edges, all members are drawn by `toward`;
  with too few nodes the new node stays alone. A removed node clears the hyperedges that hold it. The topology pass
  reports a request with a hyperedge on a network without a variable group on its hyperedges, or of another size.
- Gather tables at run time: on such a network the tables of the member slots of a hyperedge interaction hold positions
  in `_members`, and `NeighborReadSpec.entry` names that entry, so the NumPy readers and writers, which now take the
  parameter vector, and the Numba printer, `int(p[start + L[table + row]])`, read the members of every row at every
  evaluation. The spec also reads the activity of the hyperedge of every row, by which `HyperedgeTables._masked`
  multiplies rates, noises, and jumps. An update of the hyperedge member writes only its own row and stays as it is;
  an update of the focal member is reported, since free rows would write node 0, as is an edge role and a jump that
  targets a gathered member, whose channels are fixed.

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
