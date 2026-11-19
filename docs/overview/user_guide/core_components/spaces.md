# Spaces, neighbourhoods, and times

A space places the members of a group and says who neighbours whom. A neighbourhood picks, around one member, the
members that a law reads. A time says whether the states flow, step, or are solved once. Elements hold the states,
interactions return the laws that change them, and a system composes all of these components into one model. [How the
components fit](index.md) shows the families together, and the [spaces page](../../../components/spaces.md)
compares every class of this page and holds its fact sheet. All of them live in `skcomplex.core.components.spaces`. The
page takes the spaces first, then the neighbourhoods, then the times, and each family ends with its base class.

## LatticeSpace

A `LatticeSpace` is a box of a regular grid, such as a row of cells or a square of sites, with a boundary. Here a ring
of five cells conducts heat. An `Element` with `space=` holds one member per site, here the temperature `u` of
each cell. The first cell starts at 10 degrees, and along every edge of the lattice, `along=True`, heat flows from the
warmer end to the cooler one at the rate of their difference.

```python
from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import RateLaw
from skcomplex.core.components.spaces import LatticeSpace
from skcomplex.core.systems import System

ring = LatticeSpace(5, boundary='periodic')
rod = Element('rod', values={'u': [10.0, 0.0, 0.0, 0.0, 0.0]}, space=ring)


def conduct(s):
    flow = s.source.u - s.target.u
    return [RateLaw(s.target.u.diff(s.t), flow), RateLaw(s.source.u.diff(s.t), -flow)]


conduction = Interaction('conduction', among='rod', laws=conduct, along=True)
system = System(elements=[rod], interactions=[conduction]).prepare()
system.simulate(until=4, dt=2.0).get_values('rod.u').round(2).tolist()
```

The heat spreads around the ring. After four time units every cell is within 0.02 of the mean, 2 degrees.

The same lattice holds the cells of a cellular automaton in computer science. In rule 90 of Wolfram a cell becomes
live when exactly one of its two neighbours is live, and a single live cell grows into the Sierpinski triangle. An
`UpdateLaw` with `neighborhood=` is a local rule: it reads the neighbours of every cell, and an update law makes the
system step in discrete time.

```python continuation
from skcomplex.core.components.laws import TotalisticSpec, UpdateLaw
from skcomplex.core.components.spaces import RadiusNeighborhood

cells = Element('cells', values={'c': [0, 0, 0, 1, 0, 0, 0]}, space=LatticeSpace(7))
pair = RadiusNeighborhood(1, metric='manhattan')
rule = Interaction(
    'rule', roles={'target': 'cells'}, laws=lambda s: [UpdateLaw(s.target.c, TotalisticSpec({1: 1}), neighborhood=pair)]
)
automaton = System(elements=[cells], interactions=[rule]).prepare()
automaton.simulate(until=2).get_values('cells.c').astype(int).tolist()
```

The live cell at the centre splits into two cells that move one site apart at every step.

**How the engine runs it.** `prepare` numbers the sites in row-major order, so site `(i, j)` of a lattice of shape
`(n, m)` is member `i * m + j`, and it derives the position of every site. The compiler turns a neighbourhood into a
table of the neighbour of every site in every slot. A periodic boundary wraps the table around, a fixed one reads the
value `fill`, and a reflecting one reads the mirror image of the box.

```python continuation
grid = LatticeSpace((2, 3), boundary='fixed').prepare()
grid.size_, grid.shape_, grid.positions_[:3].tolist()
```

The grid has 6 sites in the shape `(2, 3)`, and the first row sits at `(0, 0)`, `(0, 1)`, and `(0, 2)`.

**How it combines.** An `Element` with `space=` holds one member per site, and its laws may read the operators
of space, such as the Laplacian. The lattice says how a field on it becomes
numbers: `discretization` is `'finite-difference'`, `'spectral'`, or `'finite-volume'`, `scheme` is the
`FiniteVolumeSpec` of a finite-volume field, `symmetry` keeps a state odd or even along an axis, and `boundary` may give
the edge of each state, such as `LatticeSpace(64, discretization='spectral', boundary={'T': ('fixed', 0.0)})`. A
local rule reads a `RadiusNeighborhood` or a `LatticeNeighborhood` around each site, an interaction with `along=True`
reads the two ends of every edge, and the `neighbors` reduction reads the nearest neighbours. `CellComplexSpace` and
`PhaseSpace` build on a lattice.

**Limits and errors.** Every site has the same pattern of neighbours, so an irregular pattern of links needs a
`NetworkSpace`, and an irregular domain needs a `CellComplexSpace` of the points and cells of a mesh. The boundary is
one of three names, and `prepare` raises a `ValidationError` for any other.

```python continuation
from skcomplex.core.errors import ValidationError

try:
    LatticeSpace((4, 4), boundary='open').prepare()
except ValidationError as error:
    message = str(error)
message
```

The message reads "Unknown lattice boundary 'open'. Use one of periodic, fixed, reflecting, or the edge of each
state, such as {'T': ('fixed', 0.0)}." The fix is `boundary='fixed'` for edges that read the value `fill`.

**A field realization.** A field fixes the shape and the boundary of its lattice, such as the grid of a forest fire
whose edges do not burn.

```python continuation
class ForestSpace(LatticeSpace):
    def __init__(self, rows=10, columns=10, *, label='forest'):
        super().__init__((rows, columns), boundary='fixed', label=label)


ForestSpace(4, 5).prepare().size_
```

See its [fact sheet](../../../components/spaces.md#latticespace)
and its [API entry][skcomplex.core.components.spaces.LatticeSpace].

## NetworkSpace

A `NetworkSpace` places the members on the nodes of a graph, so each member has the neighbours its edges give. Here
an epidemic spreads on a contact network shaped as a star. The person at the hub is infected, and each person is
infected at a rate of 0.5 times their own susceptibility and the infection of each contact. The interaction runs
`along=True`, once for every edge of the network, and its laws read the two ends of the edge as `s.source` and
`s.target`.

```python
import networkx as nx

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import RateLaw
from skcomplex.core.components.spaces import NetworkSpace
from skcomplex.core.systems import System


def infect(s):
    return [
        RateLaw(s.target.x.diff(s.t), 0.5 * (1 - s.target.x) * s.source.x),
        RateLaw(s.source.x.diff(s.t), 0.5 * (1 - s.source.x) * s.target.x),
    ]


contacts = NetworkSpace(nx.star_graph(3))
people = Element('people', values={'x': [1.0, 0.0, 0.0, 0.0]}, space=contacts)
infection = Interaction('infection', among='people', laws=infect, along=True)
system = System(elements=[people], interactions=[infection]).prepare()
system.simulate(until=4, dt=2.0).get_values('people.x').round(2).tolist()
```

The three leaves catch the infection from the hub, and their share of infection reaches 0.63 after two days and 0.86
after four, which is `1 - exp(-0.5 t)`.

The same class holds a power grid of four generators wired in a ring. Each generator pulls its phase toward those of
the generators it is wired to by the sine of their difference, as in the model of Kuramoto.

```python continuation
from sage.all__sagemath_symbolics import sin


def pull(s):
    gap = sin(s.source.theta - s.target.theta)
    return [RateLaw(s.target.theta.diff(s.t), gap), RateLaw(s.source.theta.diff(s.t), -gap)]


grid = NetworkSpace(nx.cycle_graph(4))
generators = Element('generators', values={'theta': [0.0, 0.4, 0.8, 1.2]}, space=grid)
sync = Interaction('sync', among='generators', laws=pull, along=True)
system = System(elements=[generators], interactions=[sync]).prepare()
system.simulate(until=6, dt=3.0).get_values('generators.theta').round(2).tolist()
```

The generators lock to the common phase 0.6, the mean of their initial phases.

**How the engine runs it.** `prepare` numbers the nodes in the order of `graph.nodes` and stores the neighbours of
every node in compressed sparse row form, the attribute `adjacency_`. An interaction along the edges writes its
laws once per edge, and each law reads the states of both ends. A local rule reads the neighbours of every node padded to the
largest degree, the largest number of edges at a node.

**How it combines.** An element with `space=` sits on its nodes, with `on='edges'` on its edges, and with
`on='hyperedges'` on its hyperedges, links of more than two nodes. A `LayerSpec` adds a named relation over the same
nodes, such as the colleagues of each person at work, which an `AdjacencyNeighborhood` reads by its label and an
interaction runs along with `along='work'`. An `UpdateLaw` of a `TopologyRequestSpec` adds and removes edges during a
run.

**Limits and errors.** The nodes are fixed for a run, and a node has no position of its own beyond the layout of the
graph. Every layer holds the nodes of the graph, and `prepare` raises a `ValidationError` for a layer with other
nodes.

```python continuation
from skcomplex.core.errors import ValidationError
from skcomplex.core.components.spaces import LayerSpec

try:
    NetworkSpace(nx.path_graph(3), layers=[LayerSpec('work', nx.path_graph(4))]).prepare()
except ValidationError as error:
    message = str(error)
message
```

The message reads "The layer 'work' has other nodes than the graph. Give it the nodes of the graph." The fix is a
layer graph built on the same nodes, such as `nx.empty_graph(3)` with the edges of work added to it.

**A field realization.** A field builds the graph from its own arguments, such as a random contact network of a
given size and density.

```python continuation
class ContactNetworkSpace(NetworkSpace):
    def __init__(self, people=100, density=0.05, *, seed=0, label='contacts'):
        super().__init__(nx.erdos_renyi_graph(people, density, seed=seed), label=label)


ContactNetworkSpace(people=20, density=0.2).prepare().size_
```

See its [fact sheet](../../../components/spaces.md#networkspace)
and its [API entry][skcomplex.core.components.spaces.NetworkSpace].

## ContinuousSpace

A `ContinuousSpace` is a box of real positions, or the whole of `R^d`, through which members move. The positions are
states of the members, so the space has no sites. A value named after an axis of the space, here `x`, is a position,
and a vector `x` of two entries is a position in the plane. Here two atoms of a gas coast in a periodic unit box, one
toward the right wall and one upward.

```python
from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import RateLaw
from skcomplex.core.components.spaces import ContinuousSpace
from skcomplex.core.systems import System

moving = {'x': [[0.9, 0.5], [0.2, 0.5]], "x'": [[1.0, 0.0], [0.0, 1.0]]}
coast = Interaction('coast', roles={'target': 'atoms'}, laws=lambda s: [RateLaw(s.target.x.diff(s.t, 2), 0.0)])
box = ContinuousSpace(extent=(1.0, 1.0), boundary='periodic')
atoms = Element('atoms', values=moving, size=2, space=box)
gas = System(elements=[atoms], interactions=[coast]).prepare()
gas.simulate(until=0.2, dt=0.2).get_values('atoms.x')[-1].round(2).tolist()
```

The law `x.diff(t, 2) = 0` makes the position a state of order 2, so it has a velocity `"x'"`, given with the
starting values. The first atom leaves through the right wall at 1.1 and enters again at 0.1, and the second moves
from 0.5 to 0.7.

In geography the same space holds the districts of a city as polygons, called regions, which agents ask about.

```python continuation
from shapely import Point
from shapely import box as rectangle

districts = [rectangle(0.0, 0.0, 2.0, 2.0), rectangle(2.0, 0.0, 4.0, 2.0)]
city = ContinuousSpace(extent=(4.0, 2.0), regions={'districts': districts}).prepare()
city.regions_['districts'].query(Point(3.0, 1.0), predicate='intersects').tolist()
```

The point `(3, 1)` lies in district 1, the eastern one.

**Starts in a region.** A region may also be one named polygon, such as `regions={'west': polygon}`. Agents whose value
`region` names a region start at a point the run draws inside it from its own random stream, so a census tract can
place its residents. Here ten residents live in the western district and twenty in the eastern one, and each counts the
steps it has lived there.

```python continuation
from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import UpdateLaw
from skcomplex.core.systems import System


def settle(agent, system):
    agent.steps = agent.steps + 1


town = ContinuousSpace(extent=(4.0, 2.0), regions={'west': districts[0], 'east': districts[1]})
residents = Element('residents', values={'steps': 0.0, 'region': ['west'] * 10 + ['east'] * 20}, size=30, space=town)
living = Interaction('living', roles={'target': 'residents'}, laws=lambda s: [UpdateLaw(s.target, settle)])
start = System(elements=[residents], interactions=[living]).prepare().simulate(until=1, random_state=0)
x = start.get_agents('residents', 0).locations[:, 0]
bool((x[:10] <= 2.0).all() and (x[10:] >= 2.0).all())
```

Every resident starts inside the district its value names. The same seed draws the same points again.

**How the engine runs it.** The space returns no laws. After every step the solver applies the boundary to every
position state: a periodic box wraps it, and a reflecting box mirrors it and reverses the velocity across the wall.
Pair distances in a periodic box use the nearest periodic image.

```python continuation
walls = ContinuousSpace(extent=(1.0, 1.0), boundary='reflecting')
atoms = Element('atoms', values=moving, size=2, space=walls)
trajectory = System(elements=[atoms], interactions=[coast]).prepare().simulate(until=0.2, dt=0.2)
trajectory.get_values('atoms.x')[-1].round(2).tolist(), trajectory.get_values("atoms.x'")[-1].round(2).tolist()
```

In the reflecting box the first atom bounces back to 0.9, and its velocity turns from 1 to -1.

**How it combines.** The members are agents or particles whose positions are states. An interaction with
`within=RadiusNeighborhood(r)` reads every pair within a cutoff, and the `neighbors` reduction reads a
`RadiusNeighborhood`, a `NearestNeighborhood`, or a `DelaunayNeighborhood`.

**Limits and errors.** The space has no sites, so it holds no field of one member per site, and regions need two
dimensions. A
space needs the sides of its box or a number of dimensions, and `prepare` raises a `ValidationError` otherwise.

```python continuation
from skcomplex.core.errors import ValidationError

try:
    ContinuousSpace().prepare()
except ValidationError as error:
    message = str(error)
message
```

The message reads "A continuous space needs the sides of its box or a number of dimensions." The fix is
`ContinuousSpace(extent=(10.0, 10.0))` for a box, or `ContinuousSpace(n_dim=2)` for the whole plane.

**A field realization.** A field fixes the box, such as the periodic box of a molecular simulation.

```python continuation
class PeriodicBoxSpace(ContinuousSpace):
    def __init__(self, side=10.0, *, label='box'):
        super().__init__(extent=(side, side, side), boundary='periodic', label=label)


PeriodicBoxSpace(5.0).prepare().extent_
```

See its [fact sheet](../../../components/spaces.md#continuousspace)
and its [API entry][skcomplex.core.components.spaces.ContinuousSpace].

## CellComplexSpace

A `CellComplexSpace` adds the links between the sites of a lattice, a network, or a mesh and the faces that the links
bound, each with an orientation. Cells of degree 0 are sites, of degree 1 links, and of degree 2 faces. In a lattice
gauge theory of physics the states live on the links of a lattice.

```python
from skcomplex.core.components.spaces import CellComplexSpace, LatticeSpace

links = CellComplexSpace(LatticeSpace((3, 3))).prepare()
[matrix.shape for matrix in links.incidences_], int((links.incidences_[1] @ links.incidences_[0]).count_nonzero())
```

The periodic lattice of 9 sites has 18 links and 9 faces, called plaquettes. The first incidence matrix gives the
sites at the ends of each link, the second the links around each face, and their product is zero, since the boundary
of a boundary is empty.

The same class holds the flows of a network, which the Hodge decomposition splits into gradient and cyclic parts. A
ring of four nodes with a chord bounds two triangles.

```python continuation
import networkx as nx

from skcomplex.core.components.spaces import NetworkSpace

ring = nx.cycle_graph(4)
ring.add_edge(0, 2)
CellComplexSpace(NetworkSpace(ring)).prepare().hyperedges_
```

Without given faces, the space takes a minimum cycle basis, here the triangles `(0, 1, 2)` and `(0, 2, 3)`.

A mesh of triangles or tetrahedra is a cell complex too, built from the points and cells of a scikit-fem mesh. Its
nodes hold the members of a field, its edges are the sides of the cells, and its faces their triangles. Here heat
leaves an L-shaped machine part whose walls are held at zero.

```python continuation
import skfem

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import RateLaw
from skcomplex.core.components.laws.utils import laplacian
from skcomplex.core.specs import Role
from skcomplex.core.systems import System


class Conduction(Interaction):
    def __init__(self, label, *, target: Role = 'bracket', D=1.0):
        super().__init__(label)
        self.target = target
        self.D = D

    def _prepare(self, s):
        super()._prepare(s)
        u = s.target.u
        self.laws_ = [RateLaw(u.diff(s.t), s.D * laplacian(u))]


shape = skfem.MeshTri.init_lshaped().refined(1)
mesh = CellComplexSpace(points=shape.p, cells=shape.t, boundary={'u': ('fixed', 0.0)})
bracket = Element('bracket', values={'u': 1.0}, space=mesh)
system = System(elements=[bracket], interactions=[Conduction('conduction')]).prepare()
system.simulate(until=1, dt=0.5).get_values('bracket.u').max(axis=1).round(3).tolist()
```

The hottest node falls from 1 to 0.013 after half a time unit, and after one unit it rounds to 0.

In hydrology the same class holds a block of an aquifer cut into tetrahedra, through which groundwater seeps.

```python continuation
import numpy as np

block = skfem.MeshTet.init_tensor(*[np.linspace(0.0, 1.0, 3)] * 3)
aquifer = CellComplexSpace(points=block.p, cells=block.t).prepare()
aquifer.size_, aquifer.n_dim_
```

The block has 27 nodes in three dimensions.

**How the engine runs it.** The space is a network whose edges are the oriented links and whose hyperedges are the
oriented faces. The law utilities `exterior_derivative`, `codifferential`, and `hodge_star` read its incidence
matrices and its Hodge stars, the ratios of dual to primal cell volumes, so the laws of discrete exterior calculus
compile to sparse products. A field on a mesh takes the finite-element discretization. Its stiffness and mass
matrices come from scikit-fem, so `laplacian(u)` becomes a sparse product, and the fixed values of the argument
`boundary` hold at the nodes of `boundary_nodes_`. A state without a fixed value has zero flux through the edge.

**How it combines.** An `Element` with `on='edges'` holds a value per link, and one with `on='hyperedges'` a
value per face. A `CofaceNeighborhood` reads the faces around each link, as the action of a gauge field on its links
does. An `Element` with `space=` a mesh holds a field on its nodes, with the same laws as on a lattice.

**Limits and errors.** It takes exactly one source, a square lattice, a network, or the points and cells of a mesh,
and it has no layers. Cells of degree 3 take part in incidence only, since a group lives on sites, edges, or
hyperedges. A mesh does not change during a run, and a field on it is solved by finite elements. A
triangular lattice raises a `SpaceError` in `prepare`.

```python continuation
from skcomplex.core.errors import SpaceError

try:
    CellComplexSpace(LatticeSpace((3, 3), geometry='triangular')).prepare()
except SpaceError as error:
    message = str(error)
message
```

The message reads "A cell complex of a lattice takes a square lattice and no faces, which the lattice gives." The fix
is a square lattice, or a `NetworkSpace` of the triangular graph with its faces given as node cycles.

**A field realization.** A field names the complex after its use, such as the links of a gauge theory on a square
lattice of a given side.

```python continuation
class GaugeLatticeSpace(CellComplexSpace):
    def __init__(self, side=8, *, label='links'):
        super().__init__(LatticeSpace((side, side)), label=label)


len(GaugeLatticeSpace(4).prepare().hyperedges_)
```

Another builds the mesh of its domain, such as an L-shaped plate refined a given number of times.

```python continuation
class PlateSpace(CellComplexSpace):
    def __init__(self, refinements=2, *, label='plate'):
        plate = skfem.MeshTri.init_lshaped().refined(refinements)
        super().__init__(points=plate.p, cells=plate.t, label=label)


PlateSpace(1).prepare().size_
```

See its [fact sheet](../../../components/spaces.md#cellcomplexspace)
and its [API entry][skcomplex.core.components.spaces.CellComplexSpace].

## PhaseSpace

A `PhaseSpace` pairs every position of a lattice with a value on one or more extra axes, called class axes, such as
velocity. A class axis is a range cut into equal cells. In the kinetic theory of gases a density varies over the
position and the velocity of the molecules.

```python
from skcomplex.core.components.spaces import LatticeSpace, PhaseSpace
from skcomplex.core.components.spaces import AxisSpec

gas = PhaseSpace(LatticeSpace((16,), spacing=0.5), [AxisSpec('v', 32, -4.0, 4.0)]).prepare()
gas.size_, gas.shape_, gas.coordinates_['v'][:2].round(3).tolist()
```

The space has 512 sites, 16 positions times 32 velocities, and the first velocity cells are centred at -3.875 and
-3.625.

In biology bacteria are counted over position and swimming direction. A direction is an angle, so its axis is
periodic.

```python continuation
import math

direction = AxisSpec('angle', 8, 0.0, 2 * math.pi, periodic=True)
bacteria = PhaseSpace(LatticeSpace((10,)), [direction]).prepare()
bacteria.size_, bacteria.axes_periodic_
```

The 80 sites have a periodic position axis, from the ring of the lattice, and a periodic angle axis.

**How the engine runs it.** The position axes come first, so the sites of one position come one after the other. An
`Element` on the space holds one density per site, and its laws read `coordinate('v')`, the velocity of the site,
for streaming. Moments such as the density at each position, `moment(u, 0, axes=('v',))`, reduce over the class axes.

**How it combines.** The usual partners are an `Element` on the space with the finite-volume discretization, a
rate law of a divergence, `RateLaw(u, -divergence(flux))`, for streaming and forces, and a rate law for collisions. A phase space whose class
axes are frequencies that sum to at most one, as in `TrimorphicMultilevelSelectionSystem` of
`skcomplex.systems.social.cooperation`, cuts the cells along its slanted face.

**Limits and errors.** The number of sites is the product of the sizes of all axes, so it grows fast. The argument
`limit` guards memory, and `prepare` raises a `PhaseSpaceTooLargeError` above it.

```python continuation
from skcomplex.core.errors import PhaseSpaceTooLargeError

try:
    PhaseSpace(LatticeSpace((100, 100)), [AxisSpec('v', 64, -4.0, 4.0)], limit=100_000).prepare()
except PhaseSpaceTooLargeError as error:
    message = str(error)
message
```

The message reads "The phase space would take 640000 sites, above its limit of 100000." The fix is fewer cells per
axis, or a higher `limit` when the memory allows.

**A field realization.** A field names the class axis, such as the velocity of the molecules of a gas in one
dimension.

```python continuation
class VelocitySpace(PhaseSpace):
    def __init__(self, positions=32, velocities=64, *, speed=6.0, label='gas'):
        super().__init__(LatticeSpace((positions,)), [AxisSpec('v', velocities, -speed, speed)], label=label)


VelocitySpace(8, 16).prepare().shape_
```

See its [fact sheet](../../../components/spaces.md#phasespace)
and its [API entry][skcomplex.core.components.spaces.PhaseSpace].

## BaseSpace

`BaseSpace` is the base of every space. A new kind of structure that members sit on, beyond lattices, networks,
continuous boxes, and meshes, derives from it and answers the hooks the engine reads, such as the neighbours of a
site. A space with sites says so with `_has_sites`, and the solvers, local rules, and reductions over neighbours then
read it as they read a lattice. The Cayley graph of a group, `CayleySpace` of
[`skcomplex.systems.computer_science.common.automata`][skcomplex.systems.computer_science.common.automata], is such a
space, written outside the core.

```python
from skcomplex.core.components.spaces import BaseSpace, ContinuousSpace, LatticeSpace

issubclass(LatticeSpace, BaseSpace), issubclass(ContinuousSpace, BaseSpace)
```

See its [fact sheet](../../../components/spaces.md#basespace)
and its [API entry][skcomplex.core.components.spaces.BaseSpace].

## LatticeNeighborhood

A `LatticeNeighborhood` takes any list of lattice offsets, in the order a rule receives their values. Rule 90 of an
elementary cellular automaton reads the left and the right neighbour of each cell and returns their sum modulo 2.

```python
from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import UpdateLaw
from skcomplex.core.components.spaces import LatticeNeighborhood, LatticeSpace
from skcomplex.core.systems import System

cells = Element('cells', values={'c': [0, 0, 0, 1, 0, 0, 0]}, space=LatticeSpace(7))
sides = LatticeNeighborhood([(-1,), (1,)])
parity = lambda left, right: (left + right) % 2
rule = Interaction(
    'rule', roles={'target': 'cells'}, laws=lambda s: [UpdateLaw(s.target.c, parity, neighborhood=sides)]
)
system = System(elements=[cells], interactions=[rule]).prepare()
system.simulate(until=2).get_values('cells.c').astype(int).tolist()
```

The rows are those of rule 90 in the `LatticeSpace` section, written here with the two offsets in a chosen order.

In a model of advection a pulse of pollutant moves one cell downstream at every step. The rule reads only the upwind
cell, the cell at offset -1.

```python continuation
river = Element('river', values={'c': [0, 1, 1, 0, 0, 0]}, space=LatticeSpace(6))
upwind = LatticeNeighborhood([(-1,)])
flow = Interaction(
    'flow',
    roles={'target': 'river'},
    laws=lambda s: [UpdateLaw(s.target.c, lambda upstream: upstream, neighborhood=upwind)],
)
system = System(elements=[river], interactions=[flow]).prepare()
system.simulate(until=2).get_values('river.c').astype(int).tolist()
```

The pulse of two cells moves one cell to the right at each step.

**How the engine runs it.** `prepare` stores the offsets as tuples of integers in `offsets_`. The space builds the
table of the neighbour of every site at every offset, following its boundary, and the rule receives one array per
offset.

**How it combines.** It fills the same places as `RadiusNeighborhood` on a lattice: a local rule
`UpdateLaw(..., neighborhood=...)` and the `neighbors` reduction on a `LatticeSpace`.

**Limits and errors.** Every offset has the number of dimensions of the lattice, and `prepare` raises a
`ValidationError` for offsets of different lengths.

```python continuation
from skcomplex.core.errors import ValidationError

try:
    LatticeNeighborhood([(-1,), (1, 0)]).prepare()
except ValidationError as error:
    message = str(error)
message
```

The message reads "Every offset of a neighbourhood has the same number of dimensions." The fix is `[(-1,), (1,)]` on a
line, or `[(-1, 0), (1, 0)]` on a grid.

**A field realization.** A field names its offsets, such as the upwind cell of a model of transport.

```python continuation
class UpwindNeighborhood(LatticeNeighborhood):
    def __init__(self, *, label='upwind'):
        super().__init__([(-1,)], label=label)


UpwindNeighborhood().prepare().offsets_
```

See its [fact sheet](../../../components/spaces.md#latticeneighborhood)
and its [API entry][skcomplex.core.components.spaces.LatticeNeighborhood].

## AdjacencyNeighborhood

An `AdjacencyNeighborhood` takes the nodes joined to a node by the edges of a network, so each node may have a
different number of neighbours. Here people in a line of contacts catch a disease when any contact is infected.

```python
import networkx as nx
import numpy as np

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import UpdateLaw
from skcomplex.core.components.spaces import AdjacencyNeighborhood, LatticeSpace, LayerSpec, NetworkSpace
from skcomplex.core.systems import System


def catch(infected, *contacts):
    return np.maximum(infected, sum(contacts) > 0)


def spread_over(neighborhood):
    return lambda s: [UpdateLaw(s.target.x, catch, neighborhood=neighborhood)]


people = Element('people', values={'x': [1, 0, 0, 0, 0]}, space=NetworkSpace(nx.path_graph(5)))
spread = Interaction('spread', roles={'target': 'people'}, laws=spread_over(AdjacencyNeighborhood(center=True)))
system = System(elements=[people], interactions=[spread]).prepare()
system.simulate(until=2).get_values('people.x').astype(int).tolist()
```

With `center=True` the rule receives the node itself first. The infection moves one person along the line at every
step.

At work the same people have other contacts. A `LayerSpec` adds the network of colleagues over the same nodes, and
the neighbourhood reads it by its label.

```python continuation
office = nx.empty_graph(5)
office.add_edges_from([(0, 4), (4, 3)])
town = NetworkSpace(nx.path_graph(5), layers=[LayerSpec('work', office)])
colleagues = Element('people', values={'x': [1, 0, 0, 0, 0]}, space=town)
at_work = AdjacencyNeighborhood(center=True, layer='work')
spread = Interaction('spread', roles={'target': 'people'}, laws=spread_over(at_work))
system = System(elements=[colleagues], interactions=[spread]).prepare()
system.simulate(until=2).get_values('people.x').astype(int).tolist()
```

At work the infection goes from person 0 to person 4 and then to person 3, along the edges of the layer.

**How the engine runs it.** The space gives the neighbours of every node from its adjacency in compressed sparse row
form. A local rule receives their values padded to the largest degree with the value `fill` of the network. The
default fill is 0, so a sum is not changed by the padding.

**How it combines.** It is the neighbourhood of a local rule on a `NetworkSpace`, and the default of the
`neighbors` reduction on a network.

**Limits and errors.** It reads a network, not a lattice, and a local rule on a lattice raises a `ValidationError`
when it is given one.

```python continuation
from skcomplex.core.errors import ValidationError

row = Element('people', values={'x': [1, 0, 0, 0, 0]}, space=LatticeSpace(5))
spread = Interaction('spread', roles={'target': 'people'}, laws=spread_over(AdjacencyNeighborhood()))
try:
    System(elements=[row], interactions=[spread]).prepare().simulate(until=1)
except ValidationError as error:
    message = str(error)
message
```

The message says that a lattice reads a lattice neighbourhood of offsets, but was given an `AdjacencyNeighborhood`.
The fix is a `RadiusNeighborhood` on a lattice, or a `NetworkSpace` for the adjacency.

**A field realization.** A field names the layer it reads, such as the household of each person in an epidemic.

```python continuation
class HouseholdNeighborhood(AdjacencyNeighborhood):
    def __init__(self, *, label='household'):
        super().__init__(center=True, layer='household', label=label)


HouseholdNeighborhood().layer
```

See its [fact sheet](../../../components/spaces.md#adjacencyneighborhood)
and its [API entry][skcomplex.core.components.spaces.AdjacencyNeighborhood].

## CofaceNeighborhood

A `CofaceNeighborhood` takes the cells of the next degree that contain a cell: the faces around a link, or the links
at a site. Here a state on the links of a network sums the circulation around the faces that contain each link. The
circulation of a face is the oriented sum of the link values around it, which `exterior_derivative` gives.

```python
import networkx as nx

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import UpdateLaw
from skcomplex.core.components.laws.utils import exterior_derivative, neighbors
from skcomplex.core.components.spaces import CellComplexSpace, CofaceNeighborhood, NetworkSpace
from skcomplex.core.systems import System

ring = nx.cycle_graph(4)
ring.add_edge(0, 2)
flows = Element(
    'flows', values={'a': [1, 0, 0, 0, 0], 'n': 0.0}, space=CellComplexSpace(NetworkSpace(ring)), on='edges'
)
circulation = lambda s: neighbors(exterior_derivative(s.target.a), over=CofaceNeighborhood(), reduce='sum')
around = Interaction('around', roles={'target': 'flows'}, laws=lambda s: [UpdateLaw(s.target.n, circulation(s))])
system = System(elements=[flows], interactions=[around]).prepare()
system.simulate(until=1).get_values('flows.n')[-1].tolist()
```

A unit of flow runs on the first edge, from node 0 to node 1, so only the triangle `(0, 1, 2)` circulates. Its three
edges, the first, third, and fourth, read 1, and the two edges of the other triangle read 0.

On a periodic square lattice of a gauge theory every link lies on two plaquettes. A sum of 1 over the cofaces counts
them.

```python continuation
from skcomplex.core.components.spaces import LatticeSpace

count_cofaces = lambda s: [UpdateLaw(s.target.n, neighbors(1.0, over=CofaceNeighborhood(), reduce='sum'))]
links = Element('links', values={'n': 0.0}, space=CellComplexSpace(LatticeSpace((3, 3))), on='edges')
count = Interaction('count', roles={'target': 'links'}, laws=count_cofaces)
counts = System(elements=[links], interactions=[count]).prepare().simulate(until=1)
set(counts.get_values('links.n')[-1].tolist())
```

Every one of the 18 links reads 2.

**How the engine runs it.** The cell complex stores `cofaces_`, the cofaces of every cell with their orientations.
The reduction runs over them, and inside it `exterior_derivative` is the oriented sum around the one coface. Every
coface counts with weight one.

**How it combines.** It needs a `CellComplexSpace`, and an element on its edges or sites. The Wilson action of a
lattice gauge field reads it, and so may an update or a rate law of any value on the links.

**Limits and errors.** It reads the cells of the next degree only, so the cells of the same degree need an
`AdjacencyNeighborhood`. Outside a cell complex the run raises a `ValidationError`.

```python continuation
from skcomplex.core.errors import ValidationError

nodes = Element('nodes', values={'n': 0.0}, space=NetworkSpace(ring))
count = Interaction('count', roles={'target': 'nodes'}, laws=count_cofaces)
try:
    System(elements=[nodes], interactions=[count]).prepare().simulate(until=1)
except ValidationError as error:
    message = str(error)
message
```

The message says that the reduction over `CofaceNeighborhood()` needs a `CellComplexSpace`, but the group lives on a
`NetworkSpace`. The fix is to wrap the network in a `CellComplexSpace`.

**A field realization.** A field names the cofaces it reads, such as the plaquettes around a link.

```python continuation
class PlaquetteNeighborhood(CofaceNeighborhood):
    def __init__(self, *, label='plaquettes'):
        super().__init__(label=label)


PlaquetteNeighborhood().prepare().label
```

See its [fact sheet](../../../components/spaces.md#cofaceneighborhood)
and its [API entry][skcomplex.core.components.spaces.CofaceNeighborhood].

## RadiusNeighborhood

A `RadiusNeighborhood` takes the members within a distance of each member under a metric. In a continuous space they are
the members whose positions lie within the radius, so the number of neighbours varies. In the Vicsek model birds align
with the flockmates they see within a range. Here each bird counts the birds within one unit.

```python
from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import UpdateLaw
from skcomplex.core.components.laws.utils import neighbors
from skcomplex.core.components.spaces import ContinuousSpace, RadiusNeighborhood
from skcomplex.core.systems import System


def counting(neighborhood):
    return lambda s: [UpdateLaw(s.target.seen, neighbors(1.0, over=neighborhood))]


birds = {'x': [1.0, 1.5, 2.0, 6.0, 9.8], 'y': [1.0, 1.0, 1.2, 6.0, 1.0], 'seen': 0.0}
sky = ContinuousSpace(extent=(10.0, 10.0))
flock = Element('flock', values=birds, size=5, space=sky)
sight = Interaction('sight', roles={'target': 'flock'}, laws=counting(RadiusNeighborhood(1.0)))
system = System(elements=[flock], interactions=[sight]).prepare()
system.simulate(until=1).get_values('flock.seen')[-1].tolist()
```

The middle bird of the three on the left sees both others, the outer two see only the middle one, and the last two
birds see nobody.

In physics atoms interact with every atom within a cutoff, and in a periodic box the distance is that of the nearest
periodic image. With a cutoff of 1.5 the atom at `x = 9.8` reaches the atom at `x = 1.0` across the wall.

```python continuation
crystal = ContinuousSpace(extent=(10.0, 10.0), boundary='periodic')
atoms = Element('flock', values=birds, size=5, space=crystal)
cutoff = Interaction('cutoff', roles={'target': 'flock'}, laws=counting(RadiusNeighborhood(1.5)))
system = System(elements=[atoms], interactions=[cutoff]).prepare()
system.simulate(until=1).get_values('flock.seen')[-1].tolist()
```

The first atom now reaches three others, the atom at `x = 9.8` among them, and that atom reaches the first.

On a lattice it takes the offsets within the radius under the metric, in as many dimensions as the lattice has. The
Manhattan metric of radius 1 gives the four sites of the von Neumann neighbourhood, and the Chebyshev metric the eight
of the Moore neighbourhood. In a sandpile a site with four grains or more topples, sends one grain to each of its four
neighbours, and grains that leave the grid are lost.

```python
from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import UpdateLaw
from skcomplex.core.components.spaces import LatticeSpace, RadiusNeighborhood
from skcomplex.core.systems import System


def topple(height, *neighbours):
    return height - 4 * (height >= 4) + sum(neighbour >= 4 for neighbour in neighbours)


board = LatticeSpace((3, 3), boundary='fixed')
pile = Element('pile', values={'h': [0, 0, 0, 0, 4, 0, 0, 0, 0]}, space=board)
cross = RadiusNeighborhood(1, metric='manhattan', center=True)
toppling = Interaction(
    'toppling', roles={'target': 'pile'}, laws=lambda s: [UpdateLaw(s.target.h, topple, neighborhood=cross)]
)
sandpile = System(elements=[pile], interactions=[toppling]).prepare()
sandpile.simulate(until=1).get_values('pile.h')[-1].reshape(3, 3).tolist()
```

With `center=True` the rule receives the site itself before its four neighbours. The centre topples, and each of
its four neighbours gains one grain.

In a model of opinions each person reads the eight people around them on a grid, as in the voter model. Under a
majority rule a person takes the opinion of more than half of them, and keeps their own on a tie.

```python continuation
import numpy as np


def majority(opinion, *neighbours):
    votes = sum(neighbours)
    return np.where(votes > 4, 1.0, np.where(votes < 4, 0.0, opinion))


camps = [1, 1, 0, 0] * 4
camps[5] = 0
people = Element('people', values={'o': camps}, space=LatticeSpace((4, 4)))
moore = RadiusNeighborhood(1, metric='chebyshev', center=True)
vote = Interaction(
    'vote', roles={'target': 'people'}, laws=lambda s: [UpdateLaw(s.target.o, majority, neighborhood=moore)]
)
opinions = System(elements=[people], interactions=[vote]).prepare().simulate(until=1)
opinions.get_values('people.o')[-1].reshape(4, 4).astype(int).tolist()
```

The person at site 5, alone in their camp, joins the camp of the left half after one step.


**How the engine runs it.** In a continuous space, at every evaluation the engine finds the pairs of members within the
radius from their current position states, so the neighbours follow the members as they move. An `angle` below `pi`
keeps only the neighbours inside a cone around the direction of each member, a field of view. On a lattice the engine
asks for the offsets in the dimensions of the lattice, in lexicographic order for the Euclidean, Manhattan, and
Chebyshev metrics and ring by ring for the hexagonal one, with the site first when `center` is set. The lattice turns
them into a table of the neighbour of every site in every slot, and a rule receives the values in that order. The
sandpile rule above receives the site first, at `(0, 0)`, and its four neighbours in lexicographic order.

**How it combines.** It is the neighbourhood of local rules, of the `neighbors` reduction, of the pairs of an
interaction with `within=`, and of the hyperedges of a continuous space. Its partners are groups with positions on a
`ContinuousSpace`, or elements on the sites of a `LatticeSpace`. A neighbourhood
of one direction, or of an order of its own, such as a cell and its two neighbours from left to right, needs a
`LatticeNeighborhood`.

**Limits and errors.** The radius is positive, and `prepare` raises a `ValidationError` otherwise.

```python continuation
from skcomplex.core.errors import ValidationError

try:
    RadiusNeighborhood(0.0).prepare()
except ValidationError as error:
    message = str(error)
message
```

The message asks for a positive radius. The fix is a radius above zero, such as `1.0`.

**A field realization.** A field names the range, such as the sight of a bird with its field of view.

```python continuation
import math


class SightNeighborhood(RadiusNeighborhood):
    def __init__(self, reach=1.0, *, label='sight'):
        super().__init__(reach, angle=3 * math.pi / 4, label=label)


round(SightNeighborhood().angle, 3)
```

The half-angle is 2.356, three quarters of `pi`, so the bird sees all but a cone behind it.

See its [fact sheet](../../../components/spaces.md#radiusneighborhood)
and its [API entry][skcomplex.core.components.spaces.RadiusNeighborhood].

## NearestNeighborhood

A `NearestNeighborhood` takes the `k` members of a continuous space closest to each member, whatever their distance.
Starlings in a flock follow a fixed number of nearest birds. Here each bird takes the mean heading of its nearest
bird.

```python
from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import UpdateLaw
from skcomplex.core.components.laws.utils import neighbors
from skcomplex.core.components.spaces import ContinuousSpace, NearestNeighborhood
from skcomplex.core.systems import System


def following(k):
    return lambda s: [
        UpdateLaw(s.target.heading, neighbors(s.target.other.heading, over=NearestNeighborhood(k), reduce='mean'))
    ]


starlings = {'x': [1.0, 1.5, 2.0, 6.0, 9.8], 'y': [1.0, 1.0, 1.2, 6.0, 1.0], 'heading': [0.0, 0.5, 1.0, 2.0, 3.0]}
sky = ContinuousSpace(extent=(10.0, 10.0))
flock = Element('flock', values=starlings, size=5, space=sky)
follow = Interaction('follow', roles={'target': 'flock'}, laws=following(1))
system = System(elements=[flock], interactions=[follow]).prepare()
system.simulate(until=1).get_values('flock.heading')[-1].tolist()
```

Unlike a radius, the nearest bird always exists, so the lone birds on the right also follow one: the bird at
`(6, 6)` follows the bird at `(2, 1.2)`, and the bird at `(9.8, 1)` follows the bird at `(6, 6)`.

Robots of a swarm align with their two nearest robots, which here means the mean heading of two neighbours.

```python continuation
swarm = Interaction('swarm', roles={'target': 'flock'}, laws=following(2))
system = System(elements=[flock], interactions=[swarm]).prepare()
system.simulate(until=1).get_values('flock.heading')[-1].tolist()
```

The headings become 0.75, 0.5, 0.25, 2.0, and 1.5. Each of the two lone birds takes the mean of the other lone bird
and the bird at `(2, 1.2)`.

**How the engine runs it.** At every evaluation the engine sorts the other members of each member by distance from
their current position states, by the nearest periodic image in a periodic box, and keeps the first `k`. A group of
at most `k` other members gives each member all the others.

**How it combines.** It is the neighbourhood of the `neighbors` reduction, on a group with positions in a
`ContinuousSpace`.

**Limits and errors.** The number of neighbours is a whole number of at least 1, and `prepare` raises a
`ValidationError` otherwise.

```python continuation
from skcomplex.core.errors import ValidationError

try:
    NearestNeighborhood(2.5).prepare()
except ValidationError as error:
    message = str(error)
message
```

The message reads "A nearest neighbourhood needs a whole number of neighbours of at least 1, but got 2.5." The fix is
`NearestNeighborhood(2)`.

**A field realization.** A field fixes the number, such as the seven nearest birds that a starling follows.

```python continuation
class StarlingNeighborhood(NearestNeighborhood):
    def __init__(self, *, label='starlings'):
        super().__init__(7, label=label)


StarlingNeighborhood().k
```

See its [fact sheet](../../../components/spaces.md#nearestneighborhood)
and its [API entry][skcomplex.core.components.spaces.NearestNeighborhood].

## DelaunayNeighborhood

A `DelaunayNeighborhood` joins two members when their Voronoi cells touch, the regions of the plane closer to each of
them than to any other member. In a tissue each cell pushes on the cells that touch it. Here four cells sit at the
corners of a square and one at its centre.

```python
from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import UpdateLaw
from skcomplex.core.components.laws.utils import neighbors
from skcomplex.core.components.spaces import ContinuousSpace, DelaunayNeighborhood, LatticeSpace
from skcomplex.core.systems import System


def touching(neighborhood):
    return lambda s: [UpdateLaw(s.target.touching, neighbors(1.0, over=neighborhood))]


cells = {'x': [0.5, 1.5, 1.5, 0.5, 1.0], 'y': [0.5, 0.5, 1.5, 1.5, 1.0], 'touching': 0.0}
dish = ContinuousSpace(extent=(2.0, 2.0))
tissue = Element('tissue', values=cells, size=5, space=dish)
touch = Interaction('touch', roles={'target': 'tissue'}, laws=touching(DelaunayNeighborhood()))
system = System(elements=[tissue], interactions=[touch]).prepare()
system.simulate(until=1).get_values('tissue.touching')[-1].tolist()
```

Each corner touches its two neighbouring corners and the centre, and the centre touches all four corners.

Bubbles of a foam share walls in the same way. With `center=True` each bubble also counts itself, as a mean over a
bubble and its neighbours needs.

```python continuation
foam = Interaction('foam', roles={'target': 'tissue'}, laws=touching(DelaunayNeighborhood(center=True)))
system = System(elements=[tissue], interactions=[foam]).prepare()
system.simulate(until=1).get_values('tissue.touching')[-1].tolist()
```

Each count grows by one: 4 for every corner and 5 for the centre.

**How the engine runs it.** At every evaluation the engine triangulates the current positions with SciPy, so the
neighbours follow moving members. In a periodic box the triangulation covers the periodic images. The neighbourhood
runs on the NumPy backend only.

**How it combines.** It is the neighbourhood of the `neighbors` reduction, on a group with positions in a
`ContinuousSpace`.

**Limits and errors.** A group with too few members to triangulate gives each member all the others. On a lattice or
a network the run raises a `ValidationError`, since members there have no positions to triangulate.

```python continuation
from skcomplex.core.errors import ValidationError

row = Element('row', values={'touching': 0.0}, space=LatticeSpace(4))
touch = Interaction('touch', roles={'target': 'row'}, laws=touching(DelaunayNeighborhood()))
try:
    System(elements=[row], interactions=[touch]).prepare().simulate(until=1)
except ValidationError as error:
    message = str(error)
message
```

The message says that the reduction over `DelaunayNeighborhood()` needs a `ContinuousSpace`, but the group lives on a
`LatticeSpace`. The fix is a continuous space, or `neighborhood=None` for the neighbours of the lattice.

**A field realization.** A field names the contact, such as the cells of a tissue that share a wall.

```python continuation
class ContactNeighborhood(DelaunayNeighborhood):
    def __init__(self, *, label='contact'):
        super().__init__(label=label)


ContactNeighborhood().prepare().label
```

See its [fact sheet](../../../components/spaces.md#delaunayneighborhood)
and its [API entry][skcomplex.core.components.spaces.DelaunayNeighborhood].

## BaseNeighborhood

`BaseNeighborhood` is the base of every neighbourhood. A new rule for choosing the members around a member derives
from it, and the space that reads it decides which kinds it accepts. A neighbourhood of fixed slots, such as the words
of the memory set `MemoryNeighborhood` of the automata package above, gives them with `_get_slots` and names the
spaces it is read on with `_get_spaces`.

```python
from skcomplex.core.components.spaces import BaseNeighborhood, LatticeNeighborhood, RadiusNeighborhood

issubclass(LatticeNeighborhood, BaseNeighborhood), issubclass(RadiusNeighborhood, BaseNeighborhood)
```

See its [fact sheet](../../../components/spaces.md#baseneighborhood)
and its [API entry][skcomplex.core.components.spaces.BaseNeighborhood].

## ContinuousTime

A `ContinuousTime` lets states change at every instant, by rates, noise, or random jumps. Here an epidemic grows
logistically in a town of 1000 people, from 10 infected people at a rate of 0.3 a day.

```python
from skcomplex.core.components.elements import DeclaredElement
from skcomplex.core.components.interactions import RateInteraction
from skcomplex.core.components.spaces import ContinuousTime
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.specs import DeclaredVariableSpec, StateSpec

infected = DeclaredElement('infected', variables={'x': DeclaredVariableSpec(10.0, StateSpec())})
logistic = lambda s: 0.3 * s.target.x * (1 - s.target.x / 1000)
spread = RateInteraction('spread', target='infected', state='x', rate=logistic)
epidemic = CompositeSystem(elements=[infected], interactions=[spread], time=ContinuousTime())
simulate(epidemic, until=20, dt=10.0).get_values('infected.x').round(1).tolist()
```

The infected number 168.7 after ten days and 803.0 after twenty.

In finance a stock price follows a stochastic differential equation, a rate with a random term. Here the price drifts
at 5 percent a year with a volatility of 20 percent, and a fixed `random_state` makes the path repeatable.

```python continuation
stock = DeclaredElement('stock', variables={'x': DeclaredVariableSpec(100.0, StateSpec())})
drift, volatility = lambda s: 0.05 * s.target.x, lambda s: 0.2 * s.target.x
price = RateInteraction('price', target='stock', state='x', rate=drift, noise=volatility)
market = CompositeSystem(elements=[stock], interactions=[price], time=ContinuousTime())
simulate(market, until=1, dt=0.5, random_state=0).get_values('stock.x').round(1).tolist()
```

On this path the price is 108.3 after half a year and 126.2 after a year.

**How the engine runs it.** The time gives the frames of a run, every `dt` up to `until`, and a solver integrates the
laws between them. Rate laws alone go to `ScipyODESolver`, rates with noise to `EulerMaruyamaSolver`, and jumps to
`GillespieSolver`.

**How it combines.** A system without a time takes the time its components agree on, so a system of rate laws runs in
continuous time without naming it. Its laws come from `RateInteraction`, `JumpInteraction`, `CrossingInteraction`, and
elements with rate laws of their own.

```python continuation
sorted(CompositeSystem(elements=[infected], interactions=[spread]).get_tags().time)
```

The tags of the system without a time hold `'continuous'`.

**Limits and errors.** It accepts no update laws, which set the next value of a state at a step. A
`MapInteraction` in continuous time makes the run raise a `ValidationError`.

```python continuation
from skcomplex.core.components.interactions import MapInteraction
from skcomplex.core.errors import ValidationError

census = MapInteraction('census', target='infected', state='x', function=lambda s: 2 * s.target.x)
try:
    simulate(CompositeSystem(elements=[infected], interactions=[census], time=ContinuousTime()), until=1)
except ValidationError as error:
    message = str(error)
message
```

The message says that the update law of `census` needs `DiscreteTime`, but its system declares `ContinuousTime`. The
fix is `DiscreteTime()`, or a `RateInteraction` in place of the map.

**A field realization.** A field names its clock, such as the calendar of an epidemic that runs in days.

```python continuation
class DaysTime(ContinuousTime):
    def __init__(self, *, label='days'):
        super().__init__(label=label)


DaysTime().prepare().label
```

See its [fact sheet](../../../components/spaces.md#continuoustime)
and its [API entry][skcomplex.core.components.spaces.ContinuousTime].

## DiscreteTime

A `DiscreteTime` advances the states by whole steps, by maps, local rules, and agent rules. Here a population of
beetles grows from one generation to the next by the map of Beverton and Holt, `x' = 2 x / (1 + x / 100)`.

```python
from skcomplex.core.components.elements import DeclaredElement
from skcomplex.core.components.interactions import MapInteraction
from skcomplex.core.components.spaces import DiscreteTime
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.specs import DeclaredVariableSpec, StateSpec

beetles = DeclaredElement('beetles', variables={'x': DeclaredVariableSpec(50.0, StateSpec())})
offspring = lambda s: 2.0 * s.target.x / (1 + s.target.x / 100)
breeding = MapInteraction('breeding', target='beetles', state='x', function=offspring)
system = CompositeSystem(elements=[beetles], interactions=[breeding], time=DiscreteTime())
simulate(system, until=3).get_values('beetles.x').round(1).tolist()
```

The population rises from 50 to 66.7, 80.0, and 88.9 toward its equilibrium of 100.

A bank account compounds interest of 6 percent a year once a month. The frames here are every six steps.

```python continuation
account = DeclaredElement('account', variables={'x': DeclaredVariableSpec(1000.0, StateSpec())})
interest = MapInteraction('interest', target='account', state='x', function=lambda s: s.target.x * (1 + 0.06 / 12))
system = CompositeSystem(elements=[account], interactions=[interest], time=DiscreteTime())
simulate(system, until=12, dt=6).get_values('account.x').round(2).tolist()
```

The balance reaches 1061.68 after a year, which is `1000 * 1.005^12`.

**How the engine runs it.** Each step applies every update law once. With the schedule `'synchronous'` every update
reads the states of the start of the step, and with `'random-sequential'` members update one after another in a new
random order. A `ScheduleSpec` orders the laws of a step in phases, as the phases of agents do. `IterationSolver` runs
the maps.

**How it combines.** Its laws come from `MapInteraction`, `LocalRuleInteraction`, `TransitionInteraction`,
`SamplingInteraction`, and agent rules.

**Limits and errors.** It accepts no rate laws. The schedule is one of two names or a `ScheduleSpec`, and `prepare`
raises a `ValidationError` for any other.

```python continuation
from skcomplex.core.errors import ValidationError

try:
    DiscreteTime('parallel').prepare()
except ValidationError as error:
    message = str(error)
message
```

The message reads "Unknown schedule 'parallel'. Use one of synchronous, random-sequential, or a ScheduleSpec." The
fix is `'synchronous'`, which updates all members at once.

**A field realization.** A field names its steps, such as the generations of a population.

```python continuation
class GenerationsTime(DiscreteTime):
    def __init__(self, *, label='generations'):
        super().__init__('synchronous', label=label)


GenerationsTime().prepare().schedule_
```

See its [fact sheet](../../../components/spaces.md#discretetime)
and its [API entry][skcomplex.core.components.spaces.DiscreteTime].

## StaticTime

A `StaticTime` has no dynamics. The system holds equations, constraints, and objectives, and a run solves them once.
In economics two firms of a Cournot duopoly each choose the quantity that maximizes their profit given the other's
quantity, and the equilibrium solves both conditions together.

```python
from typing import Annotated

from skcomplex.core.components.elements import Element
from skcomplex.core.components.laws import ConstraintLaw
from skcomplex.core.components.spaces import StaticTime
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.specs import ParameterSpec, StateSpec


class DuopolyElement(Element):
    law_classes = (ConstraintLaw,)

    def __init__(
        self,
        label,
        *,
        q1: Annotated[float, StateSpec()] = 0.0,
        q2: Annotated[float, StateSpec()] = 0.0,
        a: Annotated[float, ParameterSpec()] = 10.0,
        c: Annotated[float, ParameterSpec()] = 1.0,
    ):
        super().__init__(label)
        self.q1 = q1
        self.q2 = q2
        self.a = a
        self.c = c

    def _get_laws(self, namespace):
        return [
            ConstraintLaw(namespace.a - 2 * namespace.q1 - namespace.q2 == namespace.c, target=namespace.q1),
            ConstraintLaw(namespace.a - 2 * namespace.q2 - namespace.q1 == namespace.c, target=namespace.q2),
        ]


market = simulate(CompositeSystem(elements=[DuopolyElement('market')], time=StaticTime()), until=1)
market.get_values('market.q1').tolist(), market.get_values('market.q2').tolist()
```

Each firm produces 3 units, `(a - c) / 3`, and the trajectory has one frame.

In electronics a circuit finds its operating point. A source of 9 volts feeds two resistors of 1000 and 2000 ohms in
series, and the voltage between them balances the currents through both.

```python continuation
class DividerElement(Element):
    law_classes = (ConstraintLaw,)

    def __init__(
        self,
        label,
        *,
        v: Annotated[float, StateSpec()] = 0.0,
        source: Annotated[float, ParameterSpec()] = 9.0,
        r1: Annotated[float, ParameterSpec()] = 1000.0,
        r2: Annotated[float, ParameterSpec()] = 2000.0,
    ):
        super().__init__(label)
        self.v = v
        self.source = source
        self.r1 = r1
        self.r2 = r2

    def _get_laws(self, namespace):
        current = (namespace.source - namespace.v) / namespace.r1
        return [ConstraintLaw(current == namespace.v / namespace.r2, target=namespace.v)]


simulate(CompositeSystem(elements=[DividerElement('node')], time=StaticTime()), until=1).get_values('node.v').tolist()
```

The node sits at 6 volts, two thirds of the source.

**How the engine runs it.** The time has one frame, at time 0. The compiler gathers the equations of every target
into one problem, solves a linear problem in closed form where it can, and passes the others to a root solver.

**How it combines.** Its laws are `ConstraintLaw` and the constraints and objectives of linear programs, as in flux
balance analysis of a metabolic network.

**Limits and errors.** It accepts no rates or updates, since nothing changes in it. A `RateInteraction` makes the run
raise a `ValidationError`.

```python continuation
from skcomplex.core.components.interactions import RateInteraction
from skcomplex.core.errors import ValidationError

drift = RateInteraction('drift', target='node', state='v', rate=1.0)
try:
    simulate(CompositeSystem(elements=[DividerElement('node')], interactions=[drift], time=StaticTime()), until=1)
except ValidationError as error:
    message = str(error)
message
```

The message says that the rate law of `drift` needs `ContinuousTime`, but its system declares `StaticTime`. The fix
is `ContinuousTime()` for a voltage that changes, or an equation in place of the rate.

**A field realization.** A field names its kind of solution, such as the equilibrium of a market.

```python continuation
class EquilibriumTime(StaticTime):
    def __init__(self, *, label='equilibrium'):
        super().__init__(label=label)


EquilibriumTime().prepare().schedule_ is None
```

See its [fact sheet](../../../components/spaces.md#statictime)
and its [API entry][skcomplex.core.components.spaces.StaticTime].

## BaseTime

`BaseTime` is the base of every time. A system that advances in a new way, beyond flowing, stepping, and solving
once, derives a time from it and gives the frames of a run.

```python
from skcomplex.core.components.spaces import BaseTime, ContinuousTime, DiscreteTime

issubclass(ContinuousTime, BaseTime), issubclass(DiscreteTime, BaseTime)
```

See its [fact sheet](../../../components/spaces.md#basetime)
and its [API entry][skcomplex.core.components.spaces.BaseTime].
