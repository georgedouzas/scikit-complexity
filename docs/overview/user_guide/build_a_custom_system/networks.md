# Networks

Put your elements on a graph when who meets whom matters more than where they stand. A `NetworkSpace` holds a NetworkX
graph. Its nodes carry the members of groups on its sites, and its edges carry the members of groups on its edges.

```python
import networkx as nx

from skcomplex.core.components.spaces import NetworkSpace

space = NetworkSpace(nx.karate_club_graph()).prepare()
space.edges_[None][0][:5].tolist(), space.edges_[None][1][:5].tolist()
```

The states and the structure of a network can change together. Interactions read and change the states of edges, laws
reduce over the neighbours of a node and over its shortest paths, and topology requests add and remove nodes, edges,
and hyperedges.

## Generators and layers

Build a network with any NetworkX generator, and add a second relation over the same nodes as a layer.

```python
import networkx as nx

from skcomplex.core.components.spaces import NetworkSpace
from skcomplex.core.specs import LayerSpec

first = NetworkSpace(nx.watts_strogatz_graph(100, 6, 0.1, seed=7))
second = NetworkSpace(nx.watts_strogatz_graph(100, 6, 0.1, seed=7))
sorted(first.graph.edges) == sorted(second.graph.edges)

space = NetworkSpace(nx.path_graph(4), layers=[LayerSpec('work', nx.complete_graph(4))]).prepare()
list(space.graphs_), [item.tolist() for item in space.edges_['work']]
```

- A `NetworkSpace` takes any NetworkX graph, so the generators of NetworkX build it, such as `nx.erdos_renyi_graph`,
  `nx.watts_strogatz_graph`, or `nx.barabasi_albert_graph`.
- A generator takes a seed, so a network drawn for an experiment is reproducible.
- Node positions in the node attribute `pos`, as `nx.random_geometric_graph` sets them, become the fitted `positions_`.
- A `LayerSpec` names a second relation over the same nodes, such as work next to contacts.

Preparing a network lowers every layer to arrays, keyed by its label, with `None` for the graph itself. `graphs_` holds
the graph of each layer, `edges_` the source and target of every edge, and `adjacency_` the neighbours of every node in
compressed sparse row form.

## Interactions over edges

Give every edge a state of its own, such as the weight of a synapse, and change it from the states at both ends.
Hebbian weights grow with the product of the activities at the ends of each edge:

```python
from typing import Annotated

import networkx as nx
import numpy as np

from skcomplex.core.components.elements import Element, GroupElement, LatticeElement
from skcomplex.core.components.interactions import NeighborInteraction
from skcomplex.core.components.laws import RateLaw
from skcomplex.core.components.spaces import NetworkSpace
from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.specs import ParameterSpec, RoleSpec, StateSpec


class NeuronElement(Element):
    def __init__(self, label, *, a: Annotated[float, StateSpec()] = 0.0):
        super().__init__(label)
        self.a = a


class SynapseElement(Element):
    def __init__(self, label, *, w: Annotated[float, StateSpec()] = 0.1):
        super().__init__(label)
        self.w = w


class HebbianInteraction(NeighborInteraction):
    def __init__(
        self,
        label,
        *,
        neurons: Annotated[str, RoleSpec()],
        synapses: Annotated[str, RoleSpec()],
        eta: Annotated[float, ParameterSpec()] = 1.0,
    ):
        super().__init__(label)
        self.neurons = neurons
        self.synapses = synapses
        self.eta = eta

    def _get_laws(self, s):
        return [RateLaw(s.edge.w, s.eta * s.source.a * s.target.a)]


space = NetworkSpace(nx.cycle_graph(6))
system = System(
    elements=[
        LatticeElement('neurons', element=NeuronElement, space=space, values={'a': [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]}),
        GroupElement('synapses', element=SynapseElement, space=space, on='edges'),
    ],
    interactions=[HebbianInteraction('hebb', neurons='neurons', synapses='synapses')],
)
trajectory = Simulation(system, dt=0.5).prepare().run(until=1.0).trajectory_
np.round(trajectory.values('synapses.w')[-1], 3).tolist()
```

- A group on the edges of a network, `GroupElement(..., on='edges')`, holds one member per edge of its layer.
- A `NeighborInteraction` binds a group on the nodes and, optionally, a group on the edges of the same network.
- Its laws run once per edge. They read `s.source` and `s.target`, the members at both ends, and `s.edge`, the member
  of the edge.
- Rate, noise, and jump laws may target any of the three, and update laws the edge member.
- The argument `layer` chooses the layer of an interaction without an edge group.

## Hypergraphs and laws per hyperedge

Couple more than two nodes at once, such as three oscillators that pull on each other only as a triad. Three-body phase
coupling over the triangles of a hypergraph:

```python
import itertools
from typing import Annotated

import networkx as nx
import numpy as np
import xgi
from sage.all__sagemath_symbolics import sin

from skcomplex.core.components.elements import Element, LatticeElement
from skcomplex.core.components.interactions import HyperedgeInteraction
from skcomplex.core.components.laws import RateLaw
from skcomplex.core.components.spaces import NetworkSpace
from skcomplex.core.components.system import System
from skcomplex.core.engine.compile import PassCompiler
from skcomplex.core.specs import ParameterSpec, RoleSpec, StateSpec


class PhaseElement(Element):
    def __init__(self, label, *, theta: Annotated[float, StateSpec()] = 0.0):
        super().__init__(label)
        self.theta = theta


class TriadInteraction(HyperedgeInteraction):
    def __init__(self, label, *, phases: Annotated[str, RoleSpec()], K: Annotated[float, ParameterSpec()] = 1.0):
        super().__init__(label, size=3)
        self.phases = phases
        self.K = K

    def _get_laws(self, s):
        first, second = (other.theta for other in s.others)
        return [RateLaw(s.member.theta, s.K * sin(first + second - 2 * s.member.theta))]


triangles = xgi.Hypergraph([list(nodes) for nodes in itertools.combinations(range(5), 3)])
space = NetworkSpace(nx.complete_graph(5), hyperedges=triangles)
system = System(
    elements=[LatticeElement('phases', element=PhaseElement, space=space, values={'theta': np.linspace(0, 1, 5)})],
    interactions=[TriadInteraction('triads', phases='phases')],
)
problem = PassCompiler().prepare(system).run().problem_
drift = problem.drift(0.0, problem.initial_state, problem.initial_parameters)
len(space.prepare().hyperedges_), np.round(drift, 3).tolist()
```

A network space holds hyperedges beside its edges. `NetworkSpace(graph, hyperedges=...)` takes one of:

- sequences of at least two nodes,
- an XGI `Hypergraph`,
- an XGI `SimplicialComplex`, whose faces of at least two nodes are its hyperedges.

The members of an XGI hyperedge follow the order of the nodes of the graph. The prepared space holds `hyperedges_`, the
node indices of every hyperedge, `incidence_`, the sparse incidence matrix of nodes and hyperedges, and `memberships_`,
the hyperedges around every node.

A `HyperedgeInteraction` binds a group on the nodes and runs its laws once per hyperedge. It runs on the hyperedges of
its `size`, or of every size when the size is `None`. The laws read:

- `s.member`, the focal member,
- `s.others`, the other members in hyperedge order,
- `s.members`, which `total`, `mean`, `count`, `maximum`, and `minimum` reduce with `over=s.members`.

A law of the focal member runs once per member of every hyperedge and adds into that member. The compiler builds one
gather table per size of hyperedge, so the laws run as gathers and scatter-adds on the NumPy and Numba backends.

A second role may bind a group on the hyperedges, read as `s.hyperedge`, or a group on the edges, read as `s.edge`, the
edge that joins the first two members of the hyperedge. A law that targets either runs once per hyperedge.

The compiler reports two mistakes with the issue code `binding`: an interaction whose size no hyperedge of the network
has, and an edge group whose edge does not join the first two members of a hyperedge.

### Symbolic groupings

Let members meet in small groups that form anew at every step, such as voters who talk in random groups of three. The
Galam majority rule on random groups of three, and the public goods of the groups of a lattice summed into a payoff:

```python
from typing import Annotated

import numpy as np

from skcomplex.core.components.elements import Element, GroupElement, LatticeElement
from skcomplex.core.components.interactions import GroupingInteraction
from skcomplex.core.components.laws.utils import mean, where
from skcomplex.core.components.spaces import LatticeSpace
from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.specs import StateSpec


class PlayerElement(Element):
    def __init__(
        self,
        label,
        *,
        s: Annotated[float, StateSpec()] = 0.0,
        payoff: Annotated[float, StateSpec()] = 0.0,
    ):
        super().__init__(label)
        self.s = s
        self.payoff = payoff


opinions = (np.random.default_rng(0).random(30) < 0.7).astype(float)
voters = GroupElement('voters', element=PlayerElement, size=30, values={'s': opinions})
galam = GroupingInteraction(
    'galam',
    target='voters',
    groups='partition',
    size=3,
    form='symbolic',
    write='s',
    function=lambda g: where(mean(g.members.s, over=g.members) > 1 / 2, 1, 0),
)
run = Simulation(System(elements=[voters], interactions=[galam]), random_state=0).prepare().run(until=5)
run.trajectory_.values('voters.s').mean(axis=1).round(2).tolist()

players = LatticeElement(
    'players',
    element=PlayerElement,
    space=LatticeSpace((10, 10), neighborhood='von_neumann'),
    values={'s': (np.random.default_rng(1).random(100) < 0.5).astype(float)},
)
goods = GroupingInteraction(
    'goods',
    target='players',
    groups='neighborhood',
    form='symbolic',
    write='payoff',
    function=lambda g: 4.5 * mean(g.members.s, over=g.members) - g.member.s,
)
run = Simulation(System(elements=[players], interactions=[goods])).prepare().run(until=1)
run.trajectory_.values('players.payoff')[-1][:5].round(2).tolist()
```

`GroupingInteraction(form='symbolic', write=..., function=..., reduce='sum')` runs a function of one group on the same
path as the laws per hyperedge. The groups of a step are its hyperedges.

- The function reads the namespace of one group: `s.member`, the member it gives a value to, `s.others`, and
  `s.members`.
- It returns the value of the state `write` for that member in the next step, so the system runs in discrete time.
- A member takes the sum of the values of the groups that hold it, or the mean with `reduce='mean'`. A member in no
  group keeps its value.

The groups come in four kinds:

- `'pairs'` and `'partition'` are drawn again at every step from the random stream of the simulation. Two runs with one
  seed draw the same groups, on the NumPy and the Numba backends alike.
- `'neighborhood'` takes every member and the members around it in the space, fixed for the run.
- `'state'` groups the members that share a value of the state `by`, and every reduction over `s.members` reads only
  them.

`act` and `capacity` belong to the agent form, and `prepare` rejects them in the symbolic form.

### Coupling over the hyperedges of a member

Couple every member to the hyperedges around it with one ready-made interaction, without writing an interaction
class. The three-body Kuramoto coupling over the triangles of a hypergraph, and the payoff of the public goods game
summed over the hyperedges of every size around a player:

```python
import itertools
from typing import Annotated

import networkx as nx
import numpy as np
from sage.all__sagemath_symbolics import sin

from skcomplex.core.components.elements import Element, LatticeElement
from skcomplex.core.components.interactions import CouplingInteraction
from skcomplex.core.components.laws.utils import count, total
from skcomplex.core.components.spaces import NetworkSpace
from skcomplex.core.components.system import System
from skcomplex.core.engine.compile import PassCompiler
from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.specs import StateSpec


class PlayerElement(Element):
    def __init__(
        self,
        label,
        *,
        theta: Annotated[float, StateSpec()] = 0.0,
        s: Annotated[float, StateSpec()] = 0.0,
        payoff: Annotated[float, StateSpec()] = 0.0,
    ):
        super().__init__(label)
        self.theta = theta
        self.s = s
        self.payoff = payoff


triangles = [list(nodes) for nodes in itertools.combinations(range(5), 3)]
space = NetworkSpace(nx.complete_graph(5), hyperedges=triangles)
triads = CouplingInteraction(
    'triads',
    target='oscillators',
    mode='hyperedges',
    size=3,
    read='theta',
    write='theta',
    function=lambda s: sin(s.others[0].theta + s.others[1].theta - 2 * s.member.theta),
)
oscillators = LatticeElement('oscillators', element=PlayerElement, space=space, values={'theta': np.linspace(0, 1, 5)})
problem = PassCompiler().prepare(System(elements=[oscillators], interactions=[triads])).run().problem_
drift = problem.states.read(problem.drift(0.0, problem.initial_state, problem.initial_parameters), 'oscillators.theta')
np.round(drift, 3).tolist()

groups = NetworkSpace(nx.complete_graph(5), hyperedges=[[0, 1, 2], [1, 2, 3, 4], [0, 4]])
players = LatticeElement('players', element=PlayerElement, space=groups, values={'s': [1.0, 0.0, 1.0, 1.0, 0.0]})
goods = CouplingInteraction(
    'goods',
    target='players',
    mode='hyperedges',
    form='map',
    write='payoff',
    function=lambda s: 3 * total(s.members.s, over=s.members) / count(over=s.members) - s.member.s,
)
run = Simulation(System(elements=[players], interactions=[goods])).prepare().run(until=1)
run.trajectory_.values('players.payoff')[-1].round(2).tolist()
```

The reduction `hyperedges(expression, reduce='sum', size=None, weight=None)` of `skcomplex.core.components.laws.utils`
does the work. It reduces an expression written for one hyperedge over the hyperedges around every member of a group on
the nodes of a network with hyperedges, as `neighbors` does for one neighbour. Inside the expression:

- `s.others` holds the other members of the hyperedge in its order,
- `s.members` holds all its members, which `total`, `mean`, `count`, `maximum`, and `minimum` reduce with
  `over=s.members`,
- `s` is the reading member.

The reductions are those of `neighbors`. Without a size the expression reads `s.members`, and only the sum, the mean,
and the count run over hyperedges of several sizes. A law that reads it outside a network with hyperedges gets the
issue code `law-space`, and a size the network lacks gets the code `binding`.

`CouplingInteraction(mode='hyperedges', size=...)` couples every member through this reduction. The function reads
`s.member`, `s.others`, cut to `size - 1` members when the size is given, and `s.members`. The default term is the sum
over the other members of the difference of `read`. In the map form it sets the next value of `write` in discrete time.

### Ordered phases of update laws

Let a later rule of a step read what an earlier rule of the same step wrote, such as an imitation that reads the
payoff just played. The public goods game of a lattice plays in the phase `play` and copies a random neighbour by the
Fermi rule in the phase `imitate`:

```python
from typing import Annotated

import numpy as np

from skcomplex.core.components.elements import Element, LatticeElement
from skcomplex.core.components.interactions import GroupingInteraction, ImitationInteraction
from skcomplex.core.components.laws.utils import mean
from skcomplex.core.components.spaces import DiscreteTime, LatticeSpace
from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.specs import PhaseSpec, PhasesScheduleSpec, StateSpec


class PlayerElement(Element):
    def __init__(
        self,
        label,
        *,
        s: Annotated[float, StateSpec()] = 0.0,
        payoff: Annotated[float, StateSpec()] = 0.0,
    ):
        super().__init__(label)
        self.s = s
        self.payoff = payoff


players = LatticeElement(
    'players',
    element=PlayerElement,
    space=LatticeSpace((20, 20), neighborhood='von_neumann'),
    values={'s': (np.random.default_rng(0).random(400) < 0.5).astype(float)},
)
play = GroupingInteraction(
    'play',
    target='players',
    groups='neighborhood',
    form='symbolic',
    write='payoff',
    function=lambda g: 5.5 * mean(g.members.s, over=g.members) - g.member.s,
    phase='play',
)
imitation = ImitationInteraction(
    'imitation', target='players', state='s', rule='fermi', form='probability', intensity=2.0, phase='imitate'
)
time = DiscreteTime(schedule=PhasesScheduleSpec((PhaseSpec('play'), PhaseSpec('imitate'))))
system = System(elements=[players], interactions=[play, imitation], time=time)
run = Simulation(system, random_state=0).prepare().run(until=30)
round(float(run.trajectory_.values('players.s')[-1].mean()), 2) > 0.5
```

By default every update law of a step reads the states at the start of the step, so a payoff and an imitation that
reads it would need two steps. Phases fix this.

- `UpdateLaw(state, value, phase=...)` names a phase. The interactions with update laws pass a `phase` to theirs:
  `GroupingInteraction` in its symbolic form, `ImitationInteraction` in its probability form, and `CouplingInteraction`
  in its map form.
- The phases of the `PhasesScheduleSpec` of the `DiscreteTime` run one after the other within a step, on the NumPy and
  Numba backends. Each reads the states the phases before it set.
- Laws without a phase run in the default phase, first unless the schedule places it.
- The draws of a step are made once, before the first phase.
- A law in a phase the schedule does not declare raises `ValidationError`.

## Reductions over neighbourhoods

Sum, average, or count a quantity over the neighbours of every member, and weight it by the state of the edge between
them. Diffusion on a path whose second edge conducts ten times better:

```python
from typing import Annotated

import networkx as nx

from skcomplex.core.components.elements import Element, GroupElement, LatticeElement
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import RateLaw
from skcomplex.core.components.laws.utils import neighbors
from skcomplex.core.components.spaces import NetworkSpace
from skcomplex.core.components.system import System
from skcomplex.core.engine.compile import PassCompiler
from skcomplex.core.specs import RoleSpec, StateSpec


class NodeElement(Element):
    def __init__(self, label, *, u: Annotated[float, StateSpec()] = 0.0):
        super().__init__(label)
        self.u = u


class LinkElement(Element):
    def __init__(self, label, *, w: Annotated[float, StateSpec()] = 1.0):
        super().__init__(label)
        self.w = w


class DiffusionInteraction(Interaction):
    def __init__(self, label, *, nodes: Annotated[str, RoleSpec()], links: Annotated[str, RoleSpec()]):
        super().__init__(label)
        self.nodes = nodes
        self.links = links

    def _get_laws(self, s):
        return [RateLaw(s.nodes.u, neighbors(s.nodes.other.u - s.nodes.u, weight=s.links.w))]


space = NetworkSpace(nx.path_graph(3))
system = System(
    elements=[
        LatticeElement('nodes', element=NodeElement, space=space, values={'u': [0.0, 1.0, 4.0]}),
        GroupElement('links', element=LinkElement, space=space, on='edges', values={'w': [1.0, 10.0]}),
    ],
    interactions=[DiffusionInteraction('diffusion', nodes='nodes', links='links')],
)
problem = PassCompiler().prepare(system).run().problem_
drift = problem.drift(0.0, problem.initial_state, problem.initial_parameters)
problem.states.read(drift, 'nodes.u').tolist()
```

`neighbors(expression, over=..., reduce=..., weight=...)` reduces an expression written for one neighbour, through
`s.other`, over the neighbours of every member. The neighbours are:

- the neighbours of a network, by default,
- a layer, with `AdjacencyNeighborhood(layer=label)`,
- the neighbourhood of a lattice,
- in a continuous space, `RadiusNeighborhood(r)`, `NearestNeighborhood(k)`, and `ConeNeighborhood(r, angle)`.

The reductions are `'sum'`, `'mean'`, `'count'`, `'max'`, `'min'`, and `'normalized-mean'`. A weight may read the state
of the edge group of the layer through a role of an interaction, as the weight of the edge that joins the member to its
neighbour. The Kuramoto coupling on a network is
`RateLaw(s.theta, s.omega + s.K * neighbors(sin(s.other.theta - s.theta), reduce='mean'))`.

Validation reports a reduction whose neighbourhood the space does not offer, such as `RadiusNeighborhood(1.0)` on a
network, with the code `law-space`, and a layer the network lacks with the code `binding`.

### Connected components of a network

Find the clusters of connected members, for percolation or for the giant component of an epidemic. The size of the
cluster of occupied sites around every site of a path:

```python
from typing import Annotated

import networkx as nx

from skcomplex.core.components.elements import Element, LatticeElement
from skcomplex.core.components.laws import RateLaw
from skcomplex.core.components.laws.utils import components, where
from skcomplex.core.components.spaces import NetworkSpace
from skcomplex.core.components.system import System
from skcomplex.core.engine.compile import PassCompiler
from skcomplex.core.specs import StateSpec


class SiteElement(Element):
    def __init__(self, label, *, occupied: Annotated[float, StateSpec()] = 0.0):
        super().__init__(label)
        self.occupied = occupied

    def _get_laws(self, s):
        both = where(s.occupied > 0, 1, 0) * where(s.other.occupied > 0, 1, 0)
        return [RateLaw(s.occupied, components(both, reduce='size'))]


sites = LatticeElement(
    'sites',
    element=SiteElement,
    space=NetworkSpace(nx.path_graph(6)),
    values={'occupied': [1.0, 1.0, 1.0, 0.0, 1.0, 1.0]},
)
problem = PassCompiler().prepare(System(elements=[sites])).run().problem_
drift = problem.drift(0.0, problem.initial_state, problem.initial_parameters)
problem.states.read(drift, 'sites.occupied').tolist()
```

`components(active, over=None, reduce='label')` reduces over the connected components of the active links between the
members of a group. The links are the pairs of neighbours of the neighbourhood, the graph of the network by default.
The expression `active` is written for one link, as in `neighbors`. It reads `s.other`, `s`, and the edge group of a
role, so a condition on both ends gives site percolation and an edge weight gives bond percolation. The reductions are:

- `'label'`, the smallest member of the component of every member,
- `'size'`, the size of that component,
- `'largest'`, the size of the largest component,
- `'giant'`, 1 for the members of the largest component.

The NumPy backend runs SciPy's `connected_components` on the active links, and the Numba backend a union-find.

### Shortest paths of a network

Read a quantity of the shortest paths on the network as it is now, such as the load of a node, the number of shortest
paths through it. Nodes fail once their load exceeds a capacity, the cascade of Motter and Lai:

```python
from typing import Annotated

import networkx as nx

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.interactions import TopologyInteraction
from skcomplex.core.components.laws.utils import shortest_paths
from skcomplex.core.components.spaces import DiscreteTime, NetworkSpace
from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.specs import StateSpec


class NodeElement(Element):
    def __init__(self, label, *, capacity: Annotated[float, StateSpec()] = 0.0):
        super().__init__(label)
        self.capacity = capacity


graph = nx.barabasi_albert_graph(100, 2, seed=0)
loads = nx.betweenness_centrality(graph, normalized=False)
capacity = [1.2 * loads[node] for node in graph.nodes]
capacity[max(loads, key=loads.get)] = -1.0
nodes = GroupElement('nodes', element=NodeElement, on='sites', variable=True, values={'capacity': capacity})
links = GroupElement('links', element=Element, on='edges', variable=True)
overload = TopologyInteraction(
    'overload', target='nodes', action='remove-node', condition=lambda s: shortest_paths(1) > s.source.capacity
)
system = System(elements=[nodes, links], interactions=[overload], space=NetworkSpace(graph), time=DiscreteTime())
trajectory = Simulation(system).prepare().run(until=10).trajectory_
[len(trajectory.network(frame).nodes) for frame in range(4)]
```

`shortest_paths(active, over=None, reduce='betweenness')` reduces over the shortest paths of the active links between
the members of a group, read without direction, with the links and the activity of `components`. The reductions are:

- `'betweenness'`, the number of shortest paths between other members through every member, as
  `networkx.betweenness_centrality(graph, normalized=False)` gives it,
- `'closeness'`, the closeness of every member, as `networkx.closeness_centrality(graph)` gives it.

NetworkX computes the reduction, again only when the active links changed since the last evaluation, so it follows a
network whose topology changes during a run. The Numba backend runs the laws that read it in Python, with a
`BackendWarning`. The catalog system `motter_lai_overload` removes the node of the largest load, of the largest degree,
or a random node, and lets the cascade run.

## Topology requests

Grow, shrink, and rewire a network while it runs. Barabási-Albert growth adds one node with two edges at every step,
and the new edges prefer nodes of high degree:

```python
from typing import Annotated

import networkx as nx
import numpy as np

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.laws import UpdateLaw
from skcomplex.core.components.laws.utils import neighbors
from skcomplex.core.components.spaces import DiscreteTime, NetworkSpace
from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.specs import StateSpec, TopologyRequestSpec


class NodeElement(Element):
    def __init__(self, label, *, age: Annotated[float, StateSpec()] = 0.0):
        super().__init__(label)
        self.age = age

    def _get_laws(self, s):
        return [UpdateLaw(s.age, s.age + 1)]


class LinkElement(Element):
    pass


def degree(node):
    return neighbors(1)


class GrowthElement(Element):
    def _get_laws(self, s):
        return [UpdateLaw(TopologyRequestSpec('add-node', edges=2, toward=degree), 1)]


system = System(
    elements=[
        GroupElement('nodes', element=NodeElement, on='sites', variable=True, capacity=203),
        GroupElement('links', element=LinkElement, on='edges', variable=True, capacity=403),
        GrowthElement('growth'),
    ],
    space=NetworkSpace(nx.complete_graph(3)),
    time=DiscreteTime(),
)
trajectory = Simulation(system, dt=50.0, random_state=1).prepare().run(until=200).trajectory_
snapshot = trajectory.network(-1)
degrees = np.bincount(np.concatenate([snapshot.sources, snapshot.targets]))
snapshot.nodes.size, snapshot.sources.size, int(degrees.max()) > 10
```

A network that changes needs room to grow. A variable group on the sites or edges of a network,
`GroupElement(..., on='sites', variable=True, capacity=n)`, holds rows for up to `capacity` members, and marks the
members present as active. The NumPy and Numba backends run such groups with fixed array shapes:

- laws leave the free rows unchanged,
- reductions read the members present,
- neighbourhoods and interactions over neighbours read the edges present.

A change of structure is a `TopologyRequestSpec` of the kind `'add-node'`, `'remove-node'`, `'add-edge'`,
`'remove-edge'`, `'rewire'`, `'add-hyperedge'`, or `'remove-hyperedge'`. It is the target of update and jump laws:

- `UpdateLaw(request, probability)` fires the request with a probability at every step, for every member or edge its
  law runs for.
- `JumpLaw(rate, {request: 1})` fires it at a rate.

The iteration and Gillespie steppers apply the requests between steps. A request that joins nodes draws the new ends
with weights from `toward`. This is a module-level function of the namespace of a candidate node, such as one that
returns `neighbors(1)` for its degree. A request on a group that is not variable gets the validation issue
`group-fixed`, and a full capacity raises `CapacityError`.

Trajectories hold the values of absent members as NaN, and `trajectory.network(frame)` returns the nodes and edges
present at a frame. Rerun draws the nodes present, and the edges at every frame where they change.

New nodes can also attach by their distance to the others. Here every new node draws an angle and links to the nodes
whose angle lies within 0.1 of its own:

```python
from typing import Annotated

import networkx as nx
import numpy as np

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.interactions import TopologyInteraction
from skcomplex.core.components.laws.utils import where
from skcomplex.core.components.spaces import DiscreteTime, NetworkSpace
from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.specs import StateSpec


class PlacedElement(Element):
    def __init__(self, label, visual=None, *, angle: Annotated[float, StateSpec()] = 0.0):
        super().__init__(label, visual=visual)
        self.angle = angle


angles = np.linspace(0.0, 6.0, 80).tolist()
nodes = GroupElement('nodes', element=PlacedElement, on='sites', variable=True, capacity=130, values={'angle': angles})
links = GroupElement('links', element=Element, on='edges', variable=True, capacity=130)
growth = TopologyInteraction(
    'growth',
    action='add-node',
    values={'angle': lambda rng: rng.uniform(0.0, 6.0)},
    toward=lambda node: where(abs(node.angle - node.source.angle) <= 0.1, 1, 0),
)
space = NetworkSpace(nx.empty_graph(80))
system = System(elements=[nodes, links], interactions=[growth], space=space, time=DiscreteTime())
trajectory = Simulation(system, random_state=0).prepare().run(until=50).trajectory_
network = trajectory.network(-1)
placed = trajectory.values('nodes.angle')[-1]
network.sources.size, bool((abs(placed[network.sources] - placed[network.targets]) <= 0.1).all())
```

- The weight `toward` reads the node a new edge starts from as `node.source`, beside the candidate. That node is the new
  node of `'add-node'`, with the values it drew, the adding node of `'add-edge'`, and the kept end of `'rewire'`.
- Without a target, the condition and a probability of `'add-node'` read the drawn values of the node being added as
  `s.source`, so a node that does not fire is never added. A rate does not read them, since no node exists before the
  jump, and neither does a reduction.
- With a target, the condition keeps reading the node it runs for.
- Where fewer candidates have a positive weight than the edges to draw, the new ends are drawn uniformly.

The catalog systems `popularity_similarity_network` and `urban_scaling_interactions` grow by the hyperbolic distance.
The second adds internal links at a fixed rate per resident, and the exponent of its meetings rises with that rate. It
is near 1 without internal links, and near the 1.15 that Ribeiro and Netto (2024) review at the rate 0.002 over sizes
from 100 to 800.

A network can also grow by hyperedges. A network grows by triangles from one triangle:

```python
import networkx as nx

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.interactions import TopologyInteraction
from skcomplex.core.components.spaces import DiscreteTime, NetworkSpace
from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import Simulation

system = System(
    elements=[
        GroupElement('nodes', element=Element, on='sites', variable=True, capacity=23),
        GroupElement('links', element=Element, on='edges', variable=True, capacity=43),
        GroupElement('triangles', element=Element, on='hyperedges', variable=True, capacity=21),
    ],
    interactions=[TopologyInteraction('growth', action='add-node', hyperedge=3)],
    space=NetworkSpace(nx.complete_graph(3), hyperedges=[[0, 1, 2]]),
    time=DiscreteTime(),
)
trajectory = Simulation(system, random_state=0).prepare().run(until=20).trajectory_
trajectory.network(-1).hyperedges.shape, trajectory.network(-1).sources.size
```

`TopologyInteraction(action='add-node', hyperedge=k)`, or `TopologyRequestSpec('add-node', hyperedge=k)`, makes every
new node form a hyperedge of `k` members with `k - 1` existing nodes, and join each of them by an edge.

- The first two members are the ends of an edge drawn uniformly, the others are drawn by `toward`, and the new node is
  the last member. So an edge joins the first two members of every hyperedge.
- The hyperedges are the rows of a variable group on the hyperedges, `GroupElement(on='hyperedges', variable=True,
  capacity=n)`. The initial hyperedges and the new ones have one size.
- Hyperedge interactions read the hyperedges present at every step, and `trajectory.network(frame).hyperedges` holds
  the members of those present at a frame.

Hyperedges also come and go on their own, and hyperedges of several sizes share one group. Nodes form random triangles
that dissolve, and new nodes join hyperedges of two to four members:

```python
import networkx as nx

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.interactions import TopologyInteraction
from skcomplex.core.components.spaces import DiscreteTime, NetworkSpace
from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import Simulation

system = System(
    elements=[
        GroupElement('nodes', element=Element, on='sites', variable=True, capacity=40),
        GroupElement('links', element=Element, on='edges', variable=True, capacity=40),
        GroupElement('groups', element=Element, on='hyperedges', variable=True, size=0, capacity=200, rank=4),
    ],
    interactions=[
        TopologyInteraction('form', target='nodes', action='add-hyperedge', hyperedge=3, probability=0.1),
        TopologyInteraction('dissolve', target='groups', action='remove-hyperedge', probability=0.2),
        *(TopologyInteraction(f'join_{size}', action='add-node', hyperedge=size, edges=0) for size in (2, 3, 4)),
    ],
    space=NetworkSpace(nx.empty_graph(10)),
    time=DiscreteTime(),
)
trajectory = Simulation(system, random_state=0).prepare().run(until=10).trajectory_
trajectory.network(-1).hyperedges[:3].tolist()
```

- `'add-hyperedge'`, as an action or a `TopologyRequestSpec` kind, runs for every node, which forms a hyperedge of
  size `hyperedge` with nodes drawn by `toward` and is its last member.
- `'remove-hyperedge'` runs for every hyperedge of the variable group on the hyperedges that the target binds, and its
  laws read the hyperedge member as `s.source`.
- A variable group with a rank, `GroupElement(on='hyperedges', variable=True, rank=k)`, holds hyperedges of every size
  up to `k`. Its rows hold `k` members, and a smaller hyperedge fills its last ones with -1, as
  `trajectory.network(frame).hyperedges` shows. Hyperedge interactions run on groups of one size, without a rank.
- With `edges=0` or `independent=True`, a new node with a hyperedge draws every other member by `toward`, and joins
  them by no edge of its own.

A new node can also join a random number of nodes: with `independent=True`, every candidate becomes a new end on its
own with the probability `toward`, and on a directed layer `direction` sends the new edges `'out'` of the node, `'in'`
to it, or `'both'` ways, each way drawn apart. The catalog system `tap_catalysis` grows goods whose production
functions are hyperedges of every size up to the number of parents, and catalysis edges both ways, and
`AutocatalyticSetAnalysis` finds the largest autocatalytic set of its production functions.

## Sets of members on the nodes

Give every node a collection that grows during the run, such as the balls of an urn, and let the neighbours read it.
The members of a variable agent group placed on the nodes are the collection of each node:

- `agent.agents_here(group)` reads the collection of the node of an agent,
- `agent.neighbors(group=group)` reads the collections of the nodes around it, without the agents of other groups,
- `agent.create(group)` adds a member to the collection of its node, and `remove` takes one out.

The catalog system `interacting_urns_network` places an urn with triggering on every node of a network, with one ball
agent per colour and its multiplicity, and every urn draws from its own balls and from the colours of the urns around
it. [Agent-based systems](agent_based_systems.md) describes the rules and phases that act on such collections.
