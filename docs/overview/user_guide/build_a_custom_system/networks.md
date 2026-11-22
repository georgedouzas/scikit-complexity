# Networks

This page builds systems on a graph, for models where who meets whom matters more than where each member stands. A
[`NetworkSpace`](../core_components/spaces.md#networkspace) holds a NetworkX graph. Its nodes carry the members of
groups on its sites, and its edges carry the members of groups on its edges. Here the space holds the karate club of
Zachary, a social network of 34 members, and returns the first five edges as their sources and targets.

```python
import networkx as nx

from skcomplex.core.components.spaces import NetworkSpace

space = NetworkSpace(nx.karate_club_graph()).prepare()
space.edges_[None][0][:5].tolist(), space.edges_[None][1][:5].tolist()
```

The first five edges join member 0 to the members 1 to 5, so the sources read `[0, 0, 0, 0, 0]` and the targets
`[1, 2, 3, 4, 5]`.

The states and the structure of a network can change together. Interactions read and change the states of edges, laws
reduce over the neighbours of a node and over its shortest paths, and topology requests add and remove nodes, edges,
and hyperedges. The sections below take these in turn.

## Generators and layers

This section builds a network with a NetworkX generator and adds a second relation over the same nodes as a layer. A
generator takes a seed, so a network drawn for an experiment is reproducible. Here two small-world graphs of
Watts and Strogatz drawn with the same seed are compared.

```python
import networkx as nx

from skcomplex.core.components.spaces import NetworkSpace
from skcomplex.core.components.spaces import LayerSpec

first = NetworkSpace(nx.watts_strogatz_graph(100, 6, 0.1, seed=7))
second = NetworkSpace(nx.watts_strogatz_graph(100, 6, 0.1, seed=7))
sorted(first.graph.edges) == sorted(second.graph.edges)
```

The two graphs have the same edges, so the block returns `True`.

A `LayerSpec` names a second relation over the same nodes, such as the
colleagues at work beside the contacts at home. Here four people form a path at home and all meet at work.

```python continuation
space = NetworkSpace(nx.path_graph(4), layers=[LayerSpec('work', nx.complete_graph(4))]).prepare()
list(space.graphs_), [item.tolist() for item in space.edges_['work']]
```

The prepared space holds the graphs of the layers `None` and `'work'`. The work layer has all six edges of four nodes,
from `[0, 0, 0, 1, 1, 2]` to `[1, 2, 3, 2, 3, 3]`.

- A `NetworkSpace` takes any NetworkX graph, so the generators of NetworkX build it, such as `nx.erdos_renyi_graph`,
  `nx.watts_strogatz_graph`, or `nx.barabasi_albert_graph`.
- Node positions in the node attribute `pos`, as `nx.random_geometric_graph` sets them, become the fitted `positions_`.

Preparing a network turns every layer into arrays, keyed by its label, with `None` for the graph itself. `graphs_`
holds the graph of each layer, `edges_` the source and target of every edge, and `adjacency_` the neighbours of every
node in compressed sparse row form.

## Interactions over edges

This section gives every edge a value of its own, such as the weight of a synapse, and changes it from the values at
both ends. An [`Element`](../core_components/elements.md#element) with `on='edges'` holds one member per edge of the
network. An [`Interaction`](../core_components/interactions.md#interaction) with `among`, the group on the nodes, and
`along=True`, the edges of its network, runs its laws once per edge. In the Hebbian rule, the weight of a synapse grows
with the product of the activities at its two ends. Here the first three neurons of a ring of six are active.

```python
import networkx as nx
import numpy as np

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import RateLaw
from skcomplex.core.components.spaces import NetworkSpace
from skcomplex.core.systems import System
from skcomplex.core.types import Role


class Hebbian(Interaction):
    def __init__(self, label, *, among: Role = 'neurons', along=True, eta=1.0):
        super().__init__(label, among=among, along=along)
        self.eta = eta

    def _prepare(self, s):
        super()._prepare(s)
        self.laws_ = [RateLaw(s.edge.w, s.eta * s.source.a * s.target.a)]


space = NetworkSpace(nx.cycle_graph(6))
neurons = Element('neurons', values={'a': [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]}, space=space)
synapses = Element('synapses', values={'w': 0.1}, space=space, on='edges')
system = System(elements=[neurons, synapses], interactions=[Hebbian('hebb')]).prepare()
trajectory = system.simulate(until=1.0, dt=0.5)
np.round(trajectory.get_values('synapses.w')[-1], 3).tolist()
```

The two synapses that join active neurons, the first and the third edge, grow from 0.1 to 1.1 in one unit of time. The
other four stay at 0.1.

- The laws read `s.source` and `s.target`, the members at both ends, and `s.edge`, the member of the edge.
- Rate, noise, and jump laws may target any of the three, and update laws the edge member.
- `along` takes the name of a layer in place of `True`, such as `along='work'`, for the edges of that layer.

## Hypergraphs and laws per hyperedge

This section couples more than two nodes at once, such as three oscillators that pull on each other only as a triad.
A hyperedge is a set of nodes that interact together, and a hypergraph is a set of hyperedges.
`NetworkSpace(graph, hyperedges=...)` takes one of these forms.

- sequences of at least two nodes,
- an XGI `Hypergraph`,
- an XGI `SimplicialComplex`, whose faces of at least two nodes are its hyperedges.

An [`Interaction`](../core_components/interactions.md#interaction) with `among`, the group on the nodes, and
`along=('hyperedges', 3)` runs its laws once per hyperedge of three members, and `along='hyperedges'` once per hyperedge
of every size. Here five phase oscillators on a complete graph pull on each other through the three-body term
`K sin(theta_j + theta_k - 2 theta_i)` over all triangles. The `problem` of a trajectory is the compiled system. Its
`drift` gives the rates of every state at a time, a state, and parameters, and `problem.states.read` picks the rates of
one state.

```python
import itertools

import networkx as nx
import numpy as np
import xgi
from sage.all__sagemath_symbolics import sin

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import RateLaw
from skcomplex.core.components.spaces import NetworkSpace
from skcomplex.core.systems import System
from skcomplex.core.types import Role


class Triads(Interaction):
    def __init__(self, label, *, among: Role = 'phases', along=('hyperedges', 3), K=1.0):
        super().__init__(label, among=among, along=along)
        self.K = K

    def _prepare(self, s):
        super()._prepare(s)
        first, second = (other.theta for other in s.others)
        self.laws_ = [RateLaw(s.member.theta, s.K * sin(first + second - 2 * s.member.theta))]


triangles = xgi.Hypergraph([list(nodes) for nodes in itertools.combinations(range(5), 3)])
space = NetworkSpace(nx.complete_graph(5), hyperedges=triangles)
phases = Element('phases', values={'theta': np.linspace(0, 1, 5)}, space=space)
system = System(elements=[phases], interactions=[Triads('triads')]).prepare()
problem = system.simulate(until=0.0).problem
drift = problem.drift(0.0, problem.initial_state, problem.initial_parameters)
len(space.prepare().hyperedges_), np.round(problem.states.read(drift, 'phases.theta'), 3).tolist()
```

The space holds 10 triangles, all sets of three of five nodes. The rates of the phases are `[5.403, 3.199, 0.0, -3.199,
-5.403]`, so the oscillators at the ends move toward the middle one and the phases draw together.

The members of an XGI hyperedge follow the order of the nodes of the graph. The prepared space holds `hyperedges_`, the
node indices of every hyperedge, `incidence_`, the sparse incidence matrix of nodes and hyperedges, and `memberships_`,
the hyperedges around every node. The laws of an interaction along hyperedges read three names.

- `s.member`, the focal member,
- `s.others`, the other members in hyperedge order,
- `s.members`, which `total`, `mean`, `count`, `maximum`, and `minimum` reduce with `over=s.members`.

A law of the focal member runs once per member of every hyperedge and adds into that member. The compiler builds one
gather table per size of hyperedge, so the laws run as gathers and scatter-adds on the NumPy and Numba backends. The
blocks below read rates through `drift` as this one does.

An element on the hyperedges is read as `s.hyperedge`, or else an element on the edges as `s.edge`, the edge that joins
the first two members of the hyperedge. A law that targets either runs once per hyperedge. Hyperedges that share an
edge would set it several times, so a link that several nodes act on lists them all in one hyperedge. The catalog
system `SpatialTriadicPercolationSystem` lists the negative and then the positive regulators of every link, whose
numbers the link holds, and pads the hyperedges to one size with members its law leaves out.

Validation rejects two mistakes with the issue code `binding`, an interaction whose size no hyperedge of the network
has, and an element on the edges whose edge does not join the first two members of a hyperedge.

### Random groups and neighbourhoods

This section lets members meet in small groups that form anew at every step. An
[`Interaction`](../core_components/interactions.md#interaction) with `among`, a group, and `along=RandomPartition(k)`
splits the group into random groups of `k` members at every step, and runs its laws once for every member of every
group. In the majority rule of Galam, voters talk in random groups of three, and every group adopts the opinion of its
majority. Here each of 30 voters starts with opinion 1 with a probability of 0.7.

```python
import numpy as np

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction, RandomPartition
from skcomplex.core.components.laws import UpdateLaw
from skcomplex.core.components.laws.utils import mean, where
from skcomplex.core.components.spaces import LatticeSpace
from skcomplex.core.systems import System


def majority(s):
    return [UpdateLaw(s.member.s, where(mean(s.members.s, over=s.members) > 1 / 2, 1, 0))]


opinions = (np.random.default_rng(0).random(30) < 0.7).astype(float)
voters = Element('voters', values={'s': opinions}, size=30)
galam = Interaction('galam', among='voters', along=RandomPartition(3), laws=majority)
run = System(elements=[voters], interactions=[galam]).prepare().simulate(until=5, random_state=0)
run.get_values('voters.s').mean(axis=1).round(2).tolist()
```

The share of opinion 1 rises from 0.67 to 0.7, 0.8, 0.9, and then 1.0, where it stays. The majority takes over.

The groups can also be fixed by the space. In the public goods game, each player in a group pays 1 to cooperate, and
the group shares the total multiplied by a factor. With `along='neighborhoods'`, every player of a lattice forms a
group with its four neighbours, contributes its strategy `s`, and sums its payoffs over the groups it belongs to.

```python continuation
def goods(s):
    return [UpdateLaw(s.member.payoff, 4.5 * mean(s.members.s, over=s.members) - s.member.s)]


strategies = (np.random.default_rng(1).random(100) < 0.5).astype(float)
lattice = LatticeSpace((10, 10), metric='manhattan')
players = Element('players', values={'s': strategies, 'payoff': 0.0}, space=lattice)
game = Interaction('goods', among='players', along='neighborhoods', laws=goods)
run = System(elements=[players], interactions=[game]).prepare().simulate(until=1)
run.get_values('players.payoff')[-1][:5].round(2).tolist()
```

The first five players earn `[6.3, 9.9, 8.5, 14.4, 11.2]` after one round, the sums of the shares of the five groups
that hold each of them.

The laws of a group run on the same path as the laws per hyperedge, and the groups of a step are its hyperedges.

- The laws read the namespace of one group: `s.member`, the member they change, `s.others`, and `s.members`.
- An update law gives the value of the member in the next step, so the system runs in discrete time.
- A member takes the sum of the values of the groups that hold it. A member in no group keeps its value.
- `RandomPartition(k)` draws the groups again at every step from the random stream of the simulation, and
  `RandomPartition(2)` draws pairs. Two runs with one seed draw the same groups, on the NumPy and the Numba backends
  alike.
- `'neighborhoods'` takes every member and the members around it in the space, fixed for the run.

### Coupling over the hyperedges of a member

This section couples every member to the hyperedges around it with one reduction. The reduction
`hyperedges(expression, reduce='sum', size=None, weight=None, over=None)` of `skcomplex.core.components.laws.utils`
reduces an expression written for one hyperedge over the hyperedges around every member of a group on the nodes of a
network with hyperedges, as `neighbors` does for one neighbour. Here it repeats the three-body coupling of the
oscillators above, as one rate of a member.

```python
import itertools

import networkx as nx
import numpy as np
from sage.all__sagemath_symbolics import sin

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import RateLaw, UpdateLaw
from skcomplex.core.components.laws.utils import count, hyperedges, total
from skcomplex.core.components.spaces import NetworkSpace
from skcomplex.core.systems import System


def triads(s):
    theta, others = s.target.theta, s.target.others
    return [RateLaw(theta, hyperedges(sin(others[0].theta + others[1].theta - 2 * theta), size=3))]


triangles = [list(nodes) for nodes in itertools.combinations(range(5), 3)]
space = NetworkSpace(nx.complete_graph(5), hyperedges=triangles)
oscillators = Element('oscillators', values={'theta': np.linspace(0, 1, 5)}, space=space)
coupling = Interaction('triads', roles={'target': 'oscillators'}, laws=triads)
problem = System(elements=[oscillators], interactions=[coupling]).prepare().simulate(until=0.0).problem
drift = problem.drift(0.0, problem.initial_state, problem.initial_parameters)
np.round(problem.states.read(drift, 'oscillators.theta'), 3).tolist()
```

The rates `[5.403, 3.199, 0.0, -3.199, -5.403]` equal those of the interaction along hyperedges above.

An update law with the same reduction sets the next value of a state. Here five players meet in three groups of
different sizes, and each earns three times the mean contribution of each group, less its own contribution, summed over
its groups.

```python continuation
def goods(s):
    members = s.target.members
    share = 3 * total(members.s, over=members) / count(over=members) - s.target.s
    return [UpdateLaw(s.target.payoff, hyperedges(share))]


groups = NetworkSpace(nx.complete_graph(5), hyperedges=[[0, 1, 2], [1, 2, 3, 4], [0, 4]])
players = Element('players', values={'s': [1.0, 0.0, 1.0, 1.0, 0.0], 'payoff': 0.0}, space=groups)
game = Interaction('goods', roles={'target': 'players'}, laws=goods)
run = System(elements=[players], interactions=[game]).prepare().simulate(until=1)
run.get_values('players.payoff')[-1].round(2).tolist()
```

The payoffs are `[1.5, 3.5, 1.5, 0.5, 3.0]`. The second player, a defector in two groups with two cooperators each,
earns the most.

Inside the expression these names hold.

- `s.target.others` holds the other members of the hyperedge in its order,
- `s.target.members` holds all its members, which `total`, `mean`, `count`, `maximum`, and `minimum` reduce with
  `over=s.target.members`,
- `s.target` is the reading member.

The reductions are those of `neighbors`. Without a size the expression reads `s.target.members`, and only the sum, the
mean, and the count run over hyperedges of several sizes. A law that reads it outside a network with hyperedges gets
the issue code `law-space`, and a size the network lacks gets the code `binding`.

### Hyperedges of current neighbours

This section forms the hyperedges of a member from its neighbours in a continuous space, so they change as the members
move. With `over`, a neighbourhood of the space, `hyperedges(expression, size=k, over=RadiusNeighborhood(r))` reduces
over the hyperedges that every member forms with each set of `k - 1` of its current neighbours, the other members of a
set in increasing member order. The sets are found again from the positions at every evaluation, so a member that
enters or leaves the neighbourhood joins or leaves every set at once.
[`NearestNeighborhood(k)`](../core_components/spaces.md#nearestneighborhood) and
[`RadiusNeighborhood(r, angle=angle)`](../core_components/spaces.md#radiusneighborhood) work too. Such sets carry the
many-body terms of flocks and schools, three-body potentials such as Axilrod-Teller and Stillinger-Weber, and group
interactions in crowds and social models.

Here four gliders move in a [`ContinuousSpace`](../core_components/spaces.md#continuousspace), and a three-body
alignment acts only on a member whose two neighbours flank it. The value `x` of a glider is its position, since it is
named after an axis of the space. The weight `(1 - cos) / 2` is 1 when the neighbours sit on opposite sides and 0 when
they sit on the same side.

```python
import numpy as np
from sage.all__sagemath_symbolics import sqrt

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import RateLaw
from skcomplex.core.components.laws.utils import hyperedges
from skcomplex.core.components.spaces import ContinuousSpace, RadiusNeighborhood
from skcomplex.core.systems import System


def flank(s):
    member, others = s.target, s.target.others
    first, second = others[0].x - member.x, others[1].x - member.x
    cosine = first * second / sqrt((first * first) * (second * second))
    return (1 - cosine) / 2 * ((others[0].u - member.u) + (others[1].u - member.u)) / 2


def glide(s):
    alignment = hyperedges(flank(s), size=3, over=RadiusNeighborhood(1.0))
    return [RateLaw(s.target.x, s.target.u), RateLaw(s.target.u, alignment)]


plane = ContinuousSpace(extent=(10.0, 10.0))
x = np.array([[5.0, 5.0], [4.5, 5.0], [5.5, 5.0], [8.0, 8.0]])
u = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 0.0]])
gliders = Element('gliders', values={'x': x, 'u': u}, size=4, space=plane)
system = System(elements=[gliders], interactions=[Interaction('glide', roles={'target': 'gliders'}, laws=glide)])
problem = system.prepare().simulate(until=0.0).problem
drift = problem.drift(0.0, problem.initial_state, problem.initial_parameters)
problem.states.read(drift, 'gliders.u').round(3).tolist()
```

The first glider sits between the second and the third, so it gains `[1.0, 0.5]`, the mean of their relative
velocities. The second and the third have their neighbours on one side, and the last has none, so they gain nothing.

The number of sets is `C(d, k - 1)` for `d` neighbours, so the cost grows as `N d^(k - 1)` for `N` members. Hyperedges
of four members at ten neighbours are 84 sets per member, so a neighbourhood of a few neighbours keeps the cost low.
The size is required, between 2 and 5. The NumPy and Numba backends run these reductions, and the Numba backend reduces
every set as it visits it, without storing the sets. A neighbourhood outside a continuous space gets the issue code
`law-space`, and another backend the code `backend-unsupported`.

### Ordered phases of update laws

This section lets a later rule of a step read what an earlier rule of the same step wrote. By default every update law
of a step reads the states at the start of the step, so a payoff and an imitation that reads it would need two steps.
Phases order the laws within a step. Here the public goods game of a lattice plays in the phase `play`. In the phase
`imitate`, a player whose payoff is below the mean payoff of its neighbours adopts the strategy of their majority.

```python
import numpy as np

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import UpdateLaw
from skcomplex.core.components.laws.utils import mean, neighbors, where
from skcomplex.core.components.spaces import LatticeSpace
from skcomplex.core.systems import System


def play(s):
    return [UpdateLaw(s.member.payoff, 5.5 * mean(s.members.s, over=s.members) - s.member.s, phase='play')]


def imitate(s):
    player = s.target
    behind = player.payoff < neighbors(player.other.payoff, reduce='mean')
    majority = where(neighbors(player.other.s, reduce='mean') > 0.5, 1, 0)
    return [UpdateLaw(player.s, where(behind, majority, player.s), phase='imitate')]


strategies = (np.random.default_rng(0).random(400) < 0.5).astype(float)
lattice = LatticeSpace((20, 20), metric='manhattan')
players = Element('players', values={'s': strategies, 'payoff': 0.0}, space=lattice)
interactions = [
    Interaction('play', among='players', along='neighborhoods', laws=play),
    Interaction('imitation', roles={'target': 'players'}, laws=imitate),
]
run = System(elements=[players], interactions=interactions).prepare().simulate(until=30, random_state=0)
round(float(run.get_values('players.s')[0].mean()), 2), round(float(run.get_values('players.s')[-1].mean()), 2)
```

The share of cooperators falls from 0.46 to 0.04 over 30 steps.

- [`UpdateLaw(state, value, phase=...)`](../core_components/laws.md#updatelaw) names a phase.
- The phases run one after the other within a step, on the NumPy and Numba backends. Each reads the states the phases
  before it set.
- Laws without a phase run in the default phase.
- The draws of a step are made once, before the first phase.
- The phases run in the order their laws first appear, so the order of the interactions of a system orders them.

## Reductions over neighbourhoods

This section sums, averages, or counts a quantity over the neighbours of every member, weighted by the state of the
edge between them. `neighbors(expression, over=..., reduce=..., weight=...)` reduces an expression written for one
neighbour, through `s.other`, over the neighbours of every member. Here heat diffuses along a path of three nodes whose
second edge conducts ten times better than the first, and the interaction is a class of its own derived from
[`Interaction`](../core_components/interactions.md#interaction).

```python
import networkx as nx

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import RateLaw
from skcomplex.core.components.laws.utils import neighbors
from skcomplex.core.components.spaces import NetworkSpace
from skcomplex.core.systems import System
from skcomplex.core.types import Role


class BondExchange(Interaction):
    def __init__(self, label, *, nodes: Role = 'nodes', links: Role = 'links'):
        super().__init__(label)
        self.nodes = nodes
        self.links = links

    def _prepare(self, s):
        super()._prepare(s)
        self.laws_ = [RateLaw(s.nodes.u, neighbors(s.nodes.other.u - s.nodes.u, weight=s.links.w))]


space = NetworkSpace(nx.path_graph(3))
nodes = Element('nodes', values={'u': [0.0, 1.0, 4.0]}, space=space)
links = Element('links', values={'w': [1.0, 10.0]}, space=space, on='edges')
system = System(elements=[nodes, links], interactions=[BondExchange('diffusion')])
problem = system.prepare().simulate(until=0.0).problem
drift = problem.drift(0.0, problem.initial_state, problem.initial_parameters)
problem.states.read(drift, 'nodes.u').tolist()
```

The rates are `[1.0, 29.0, -30.0]`. The middle node gains 1 times its difference of minus 1 to the left and 10 times its
difference of 3 to the right, so 29.

The neighbours come from the space.

- the neighbours of a network, by default,
- a layer, with [`AdjacencyNeighborhood(layer=label)`](../core_components/spaces.md#adjacencyneighborhood),
- the neighbourhood of a lattice,
- in a continuous space, `RadiusNeighborhood(r)` with an optional `angle` for a cone, `NearestNeighborhood(k)`, and
  [`DelaunayNeighborhood()`](../core_components/spaces.md#delaunayneighborhood), the members joined by an edge of the
  Delaunay triangulation of the positions, such as cells joined by springs, computed again every step and across
  periodic boundaries, on the NumPy backend.

The reductions are `'sum'`, `'mean'`, `'count'`, `'max'`, `'min'`, and `'normalized-mean'`. A weight may read the state
of the edge group of the layer through a role of an interaction, as the weight of the edge that joins the member to its
neighbour. The Kuramoto coupling on a network is
`RateLaw(s.theta, s.omega + s.K * neighbors(sin(s.other.theta - s.theta), reduce='mean'))`.

`simulate` rejects a reduction whose neighbourhood the space does not offer, such as `RadiusNeighborhood(1.0)` on a
network, with the code `law-space`, and a layer the network lacks with the code `binding`.

### Connected components of a network

This section finds the clusters of connected members, for percolation or for the giant component of an epidemic. A
connected component is a set of members that links join, directly or through others. Here the six sites of a path are
occupied except the fourth, and every site reads the size of its cluster of occupied sites.

```python
import networkx as nx

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import RateLaw
from skcomplex.core.components.laws.utils import components, where
from skcomplex.core.components.spaces import NetworkSpace
from skcomplex.core.systems import System


def clusters(s):
    site = s.target
    both = where(site.occupied > 0, 1, 0) * where(site.other.occupied > 0, 1, 0)
    return [RateLaw(site.occupied, components(both, reduce='size'))]


occupied = [1.0, 1.0, 1.0, 0.0, 1.0, 1.0]
sites = Element('sites', values={'occupied': occupied}, space=NetworkSpace(nx.path_graph(6)))
system = System(elements=[sites], interactions=[Interaction('clusters', roles={'target': 'sites'}, laws=clusters)])
problem = system.prepare().simulate(until=0.0).problem
drift = problem.drift(0.0, problem.initial_state, problem.initial_parameters)
problem.states.read(drift, 'sites.occupied').tolist()
```

The sizes are `[3.0, 3.0, 3.0, 1.0, 2.0, 2.0]`. The first three sites form one cluster and the last two another, and
the empty site is a component of its own.

`components(active, over=None, reduce='label')` reduces over the connected components of the active links between the
members of a group. The links are the pairs of neighbours of the neighbourhood, the graph of the network by default.
The expression `active` is written for one link, as in `neighbors`. It reads `s.other`, `s`, and the edge group of a
role, so a condition on both ends gives site percolation and an edge weight gives bond percolation. The reductions are
these.

- `'label'`, the smallest member of the component of every member,
- `'size'`, the size of that component,
- `'largest'`, the size of the largest component,
- `'giant'`, 1 for the members of the largest component.

The NumPy backend runs the `connected_components` of SciPy on the active links, and the Numba backend a union-find.

### Shortest paths of a network

This section reads a quantity of the shortest paths on the network as it is now, such as the load of a node, the number
of shortest paths through it. In the cascade of Motter and Lai (2002), every node has a capacity a little above its
initial load, and a node fails once its load exceeds it. An
[`UpdateLaw`](../core_components/laws.md#updatelaw) whose target is the request `TopologyRequestSpec('remove-node')`
removes such a node with the probability its value gives, here 1 for an overloaded node and 0 for the others. Here
the node of the largest load on a scale-free graph of 100 nodes fails first, with a capacity of -1.

```python
import networkx as nx

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import TopologyRequestSpec, UpdateLaw
from skcomplex.core.components.laws.utils import shortest_paths, where
from skcomplex.core.components.spaces import NetworkSpace
from skcomplex.core.systems import System


def overload(s):
    failure = where(shortest_paths(1) > s.target.capacity, 1, 0)
    return [UpdateLaw(TopologyRequestSpec('remove-node'), failure)]


graph = nx.barabasi_albert_graph(100, 2, seed=0)
loads = nx.betweenness_centrality(graph, normalized=False)
capacity = [1.2 * loads[node] for node in graph.nodes]
capacity[max(loads, key=loads.get)] = -1.0
space = NetworkSpace(graph)
nodes = Element('nodes', values={'capacity': capacity}, space=space)
links = Element('links', space=space, on='edges')
cascade = Interaction('overload', roles={'target': 'nodes'}, laws=overload)
trajectory = System(elements=[nodes, links], interactions=[cascade], space=space).prepare().simulate(until=10)
[len(trajectory.get_network(frame).nodes) for frame in range(4)]
```

The network keeps 100 nodes at the start, 99 after the first failure, and 36 after the cascade that follows, where it
stays.

`shortest_paths(active, over=None, reduce='betweenness')` reduces over the shortest paths of the active links between
the members of a group, read without direction, with the links and the activity of `components`. The reductions are
these.

- `'betweenness'`, the number of shortest paths between other members through every member, as
  `networkx.betweenness_centrality(graph, normalized=False)` gives it,
- `'closeness'`, the closeness of every member, as `networkx.closeness_centrality(graph)` gives it.

NetworkX computes the reduction, again only when the active links changed since the last evaluation, so it follows a
network whose topology changes during a run. The Numba backend runs the laws that read it in Python, with a
`BackendWarning`. The catalog system `MotterLaiOverloadSystem` removes the node of the largest load, of the largest
degree, or a random node, and lets the cascade run.

## Topology requests

This section grows, shrinks, and rewires a network while it runs. A change of structure is a `TopologyRequestSpec`, and
an [`UpdateLaw`](../core_components/laws.md#updatelaw) fires it with a probability at every step. In the growth of
Barabási and Albert, one node joins with two edges at every step, and the new edges prefer nodes of high degree. Here
the growth starts from a triangle and runs 200 steps. The links carry no value, so their element on the edges takes
no values. An interaction without roles, such as `growth` below, runs its laws once per step for the whole network.

```python
import networkx as nx
import numpy as np

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import TopologyRequestSpec, UpdateLaw
from skcomplex.core.components.laws.utils import neighbors
from skcomplex.core.components.spaces import NetworkSpace
from skcomplex.core.systems import System


def degree(node):
    return neighbors(1)


space = NetworkSpace(nx.complete_graph(3))
nodes = Element('nodes', values={'age': 0.0}, space=space)
links = Element('links', space=space, on='edges')
ageing = Interaction('ageing', roles={'target': 'nodes'}, laws=lambda s: [UpdateLaw(s.target.age, s.target.age + 1)])
growth = Interaction('growth', laws=[UpdateLaw(TopologyRequestSpec('add-node', edges=2, toward=degree), 1)])
system = System(elements=[nodes, links], interactions=[ageing, growth], space=space)
trajectory = system.prepare().simulate(until=200, dt=50.0, random_state=1)
snapshot = trajectory.get_network(-1)
degrees = np.bincount(np.concatenate([snapshot.sources, snapshot.targets]))
snapshot.nodes.size, snapshot.sources.size, int(degrees.max()) > 10
```

The network ends with 203 nodes and 403 edges, the triangle plus 200 nodes with two edges each, and its largest degree
exceeds 10, the hubs that preferential attachment grows.

A network that changes needs no setting to grow. When a law requests a change of the network, the elements on its
nodes and edges become variable, and the engine enlarges their arrays as a Python list grows. The NumPy and Numba
backends run them with fixed array shapes between enlargements, and mark the members present as active.

- laws leave the free rows unchanged,
- reductions read the members present,
- neighbourhoods and interactions over neighbours read the edges present.

A request has the kind `'add-node'`, `'remove-node'`, `'replace-node'`, `'add-edge'`, `'remove-edge'`, `'rewire'`,
`'add-hyperedge'`, `'remove-hyperedge'`, `'close-triangle'`, `'contract-edge'`, or `'duplicate-node'`. It is the target
of update and jump laws.

- `UpdateLaw(request, probability)` fires the request with a probability at every step, for every member or edge its
  law runs for.
- [`JumpLaw(rate, {request: 1})`](../core_components/laws.md#jumplaw) fires it at a rate.

The iteration and Gillespie steppers apply the requests between steps. A request that joins nodes draws the new ends
with weights from `toward`. This is a module-level function of the namespace of a candidate node, such as `degree`
above, which returns `neighbors(1)`.

Trajectories hold the values of absent members as NaN, and `trajectory.get_network(frame)` returns the nodes and edges
present at a frame. Rerun draws the nodes present, and the edges at every frame where they change.

New nodes can also attach by their distance to the others. Here every new node draws an angle and links to the nodes
whose angle lies within 0.1 of its own.

```python
import networkx as nx
import numpy as np

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import TopologyRequestSpec, UpdateLaw
from skcomplex.core.components.laws.utils import where
from skcomplex.core.components.spaces import NetworkSpace
from skcomplex.core.systems import System

space = NetworkSpace(nx.empty_graph(80))
nodes = Element('nodes', values={'angle': np.linspace(0.0, 6.0, 80)}, space=space)
links = Element('links', space=space, on='edges')
request = TopologyRequestSpec(
    'add-node',
    values=(('angle', lambda rng: rng.uniform(0.0, 6.0)),),
    toward=lambda node: where(abs(node.angle - node.source.angle) <= 0.1, 1, 0),
)
growth = Interaction('growth', laws=[UpdateLaw(request, 1)])
system = System(elements=[nodes, links], interactions=[growth], space=space)
trajectory = system.prepare().simulate(until=50, random_state=0)
network = trajectory.get_network(-1)
placed = trajectory.get_values('nodes.angle')[-1]
network.sources.size, bool((abs(placed[network.sources] - placed[network.targets]) <= 0.1).all())
```

After 50 steps the network holds 50 edges, and every one of them joins two nodes whose angles differ by at most 0.1.

- The weight `toward` reads the node a new edge starts from as `node.source`, beside the candidate. That node is the new
  node of `'add-node'`, with the values it drew, the adding node of `'add-edge'`, and the kept end of `'rewire'`.
- An interaction without roles fires its requests once per step, and an interaction with a role once for every member
  it binds, with the probability its law gives that member.
- Where fewer candidates have a positive weight than the edges to draw, the new ends are drawn uniformly.

The catalog systems `PopularitySimilarityNetworkSystem` and `UrbanScalingInteractionsSystem` grow by the hyperbolic
distance. The second adds internal links at a fixed rate per resident, and the exponent of its meetings rises with that
rate. It is near 1 without internal links, and near the 1.15 that Ribeiro and Netto (2024) review at the rate 0.002 over
sizes from 100 to 800.

A network can also grow by hyperedges. Here a network grows by triangles from one triangle, and every new node forms a
triangle with the two ends of an existing edge.

```python
import networkx as nx

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import TopologyRequestSpec, UpdateLaw
from skcomplex.core.components.spaces import NetworkSpace
from skcomplex.core.systems import System

space = NetworkSpace(nx.complete_graph(3), hyperedges=[[0, 1, 2]])
elements = [
    Element('nodes', space=space),
    Element('links', space=space, on='edges'),
    Element('triangles', space=space, on='hyperedges'),
]
growth = Interaction('growth', laws=[UpdateLaw(TopologyRequestSpec('add-node', hyperedge=3), 1)])
trajectory = System(elements=elements, interactions=[growth], space=space).prepare().simulate(until=20, random_state=0)
trajectory.get_network(-1).hyperedges.shape, trajectory.get_network(-1).sources.size
```

After 20 steps the network holds 21 triangles of 3 members and 43 edges, the first triangle and its 3 edges plus 20
triangles with 2 new edges each.

`TopologyRequestSpec('add-node', hyperedge=k)` makes every new node form a hyperedge of `k` members with `k - 1`
existing nodes, and join each of them by an edge.

- The first two members are the ends of an edge drawn uniformly, the others are drawn by `toward`, and the new node is
  the last member. So an edge joins the first two members of every hyperedge.
- The hyperedges are the members of the element on the hyperedges, `Element(label, space=space, on='hyperedges')`.
  The initial hyperedges and the new ones have one size.
- Interactions along hyperedges read the hyperedges present at every step, and
  `trajectory.get_network(frame).hyperedges` holds the members of those present at a frame.

Hyperedges also come and go on their own, and hyperedges of several sizes share one group. Here nodes form random
triangles that dissolve, and new nodes join hyperedges of two to four members.

```python
import networkx as nx

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import TopologyRequestSpec, UpdateLaw
from skcomplex.core.components.spaces import NetworkSpace
from skcomplex.core.systems import System

space = NetworkSpace(nx.empty_graph(10))
elements = [
    Element('nodes', space=space),
    Element('links', space=space, on='edges'),
    Element('groups', space=space, on='hyperedges'),
]
form, dissolve = TopologyRequestSpec('add-hyperedge', hyperedge=3), TopologyRequestSpec('remove-hyperedge')
interactions = [
    Interaction('form', roles={'target': 'nodes'}, laws=lambda s: [UpdateLaw(form, 0.1)]),
    Interaction('dissolve', roles={'target': 'groups'}, laws=lambda s: [UpdateLaw(dissolve, 0.2)]),
    *(
        Interaction(f'join_{size}', laws=[UpdateLaw(TopologyRequestSpec('add-node', hyperedge=size, edges=0), 1)])
        for size in (2, 3, 4)
    ),
]
system = System(elements=elements, interactions=interactions, space=space)
trajectory = system.prepare().simulate(until=10, random_state=0)
trajectory.get_network(-1).hyperedges[:3].tolist()
```

The first three hyperedges at the end are `[[16, 28, -1, -1], [18, 19, -1, -1], [13, 37, -1, -1]]`, three pairs,
with -1 in the unused places.

- `'add-hyperedge'` runs for every node, which forms a hyperedge of
  size `hyperedge` with nodes drawn by `toward` and is its last member.
- `'remove-hyperedge'` runs for every hyperedge of the element on the hyperedges that the role binds.
- An element on the hyperedges holds hyperedges of every size. Its rows hold as many members as the largest
  hyperedge, and a smaller hyperedge fills its last ones with -1, as `trajectory.get_network(frame).hyperedges` shows.
- With `edges=0` or `independent=True`, a new node with a hyperedge draws every other member by `toward`, and joins
  them by no edge of its own.

A new node can also join a random number of nodes. With `independent=True`, every candidate becomes a new end on its
own with the probability `toward`, and on a directed layer `direction` sends the new edges `'out'` of the node, `'in'`
to it, or `'both'` ways, each way drawn apart. The catalog system `TapCatalysisSystem` grows goods whose production
functions are hyperedges of every size up to the number of parents, with catalysis edges both ways, and
`autocatalytic_sets` finds the largest autocatalytic set of its production functions.

### Choices, closures, contractions, and copies

This section chooses a new edge among candidates and applies the other requests that run for a node. In explosive
percolation, each step draws two candidate edges and adds the one whose product of the sizes of the components it
joins is smaller, which delays the giant component. Here 400 edges join 500 nodes.

```python
import networkx as nx

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import TopologyRequestSpec, UpdateLaw
from skcomplex.core.components.laws.utils import components
from skcomplex.core.components.spaces import NetworkSpace
from skcomplex.core.systems import System


def product(node):
    return node.cluster_size * node.source.cluster_size


space = NetworkSpace(nx.empty_graph(500))
nodes = Element('nodes', values={'cluster_size': 1.0}, space=space)
links = Element('links', space=space, on='edges')
size = Interaction(
    'size', roles={'target': 'nodes'}, laws=lambda s: [UpdateLaw(s.target.cluster_size, components(1, reduce='size'))]
)
linking = Interaction('linking', laws=[UpdateLaw(TopologyRequestSpec('add-edge', candidates=2, score=product), 1)])
system = System(elements=[nodes, links], interactions=[size, linking], space=space)
trajectory = system.prepare().simulate(until=400, random_state=0)
largest = int(trajectory.get_values('nodes.cluster_size')[-1].max())
largest < 100
```

The largest cluster holds fewer than 100 nodes, 33 in this run. With 400 random edges the mean degree would be 1.6,
above the threshold of 1 where a giant component of random graphs appears.

- `candidates` draws that many candidate ends for every new edge of `'add-edge'`, and `score`, a function of the
  namespace of a candidate end that reads the other end as `node.source`, picks the candidate of the smallest score.
  A negated score picks the largest.
- Without a target, `'add-edge'` runs once per step, and every candidate joins two distinct nodes not yet joined,
  drawn uniformly. With one candidate the edges form an Erdős-Rényi graph.

Further requests run for a node.

- `'close-triangle'` joins two neighbours of the node that are not yet joined, drawn uniformly among such pairs. A node
  without such a pair joins `edges` nodes by `toward` instead, as `'add-edge'` does, and `edges=0` turns this off.
- `'contract-edge'` draws a neighbour by `toward` and merges it into the node, which keeps the union of their links on
  every layer, without loops or repeated links. The neighbour leaves with its hyperedges.
- `'duplicate-node'` adds a copy of the node with its values, overridden by `values`. The copy inherits each link with
  probability `inherit` and joins the node with probability `parent`, and joins further ends like a new node, among
  the nodes joined to neither.
- `'replace-node'` removes the node and adds a new one in one request, so a replacement fired with a probability
  removes and adds together.

With `one=True`, a request runs at every step for one member drawn uniformly among the members present, node, edge, or
hyperedge, and fires with the probability of its law there. The member is drawn when the request applies, after the
requests before it, so a deletion followed by a duplication never duplicates the deleted node. A jump law cannot take
it, since a jump channel already belongs to one member. Here a network of fixed size closes triangles and replaces
nodes.

```python
import networkx as nx

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import TopologyRequestSpec, UpdateLaw
from skcomplex.core.components.spaces import NetworkSpace
from skcomplex.core.systems import System

closure = TopologyRequestSpec('close-triangle', one=True)
turnover = TopologyRequestSpec('replace-node', one=True)
space = NetworkSpace(nx.gnm_random_graph(50, 50, seed=0))
interactions = [
    Interaction('closure', roles={'target': 'nodes'}, laws=lambda s: [UpdateLaw(closure, 1)]),
    Interaction('turnover', roles={'target': 'nodes'}, laws=lambda s: [UpdateLaw(turnover, 0.2)]),
]
elements = [Element('nodes', space=space), Element('links', space=space, on='edges')]
system = System(elements=elements, interactions=interactions, space=space)
trajectory = system.prepare().simulate(until=500, random_state=0)
len(trajectory.get_network(-1).nodes)
```

After 500 steps the network still holds 50 nodes, because each replacement removes one node and adds one.

The catalog systems `ExplosivePercolationSystem`, `DavidsenEbelBornholdtSystem`, `NodeMergingSystem`, and
`DuplicationDeletionSystem` build on these requests.

## Sets of members on the nodes

This section gives every node a collection that grows during the run, such as the balls of an urn, and lets the
neighbours read it. The members of a variable agent group placed on the nodes are the collection of each node.

- `agent.get_agents_here(group)` reads the collection of the node of an agent,
- `agent.get_neighbors(group=group)` reads the collections of the nodes around it, without the agents of other groups,
- `agent.create(group)` adds a member to the collection of its node, and `remove` takes one out.

The catalog system `InteractingUrnsNetworkSystem` places an urn with triggering on every node of a network, with one
ball agent per colour and its multiplicity, and every urn draws from its own balls and from the colours of the urns
around it. [Agent-based systems](agent_based_systems.md) describes the rules and phases that act on such collections.
