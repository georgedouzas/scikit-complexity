# Spaces, neighbourhoods, and times

A space places the members of a group and says who neighbours whom. A neighbourhood picks, around one member, the
members that a law reads. A time says whether the states flow, step, or are solved once. Elements hold the states,
interactions return the laws that change them, and a system composes all of these parts into one model.
[How the parts fit](index.md) shows the families together, and the [spaces parts page](../../../parts/spaces.md)
compares every class of this page and holds its fact sheet. All of them live in `skcomplex.core.components.spaces`.
The page takes the spaces first, then the neighbourhoods, then the times, and each family ends with its base class.

## LatticeSpace

A `LatticeSpace` is a box of a regular grid, such as a row of cells or a square of sites, with a boundary. Here a ring
of five cells conducts heat. The first cell starts at 10 degrees, and each cell gains the difference between each
neighbour and itself.

```python
from skcomplex.core.components.elements import DeclaredElement, GroupElement
from skcomplex.core.components.interactions import CouplingInteraction
from skcomplex.core.components.spaces import LatticeSpace
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.specs import DeclaredVariableSpec, StateSpec

heat = {'variables': {'u': DeclaredVariableSpec(0.0, StateSpec())}, 'u': [10.0, 0.0, 0.0, 0.0, 0.0]}
ring = LatticeSpace(5, boundary='periodic')
rod = GroupElement('rod', element=DeclaredElement, space=ring, on='sites', values=heat)
conduction = CouplingInteraction('conduction', target='rod', read='u', write='u')
system = CompositeSystem(elements=[rod], interactions=[conduction])
simulate(system, until=4, dt=2.0).get_values('rod.u').round(2).tolist()
```

The heat spreads around the ring. After four time units every cell is within 0.02 of the mean, 2 degrees.

The same lattice holds the cells of a cellular automaton in computer science. In rule 90 of Wolfram a cell becomes
live when exactly one of its two neighbours is live, and a single live cell grows into the Sierpinski triangle.

```python continuation
from skcomplex.core.components.interactions import LocalRuleInteraction
from skcomplex.core.components.spaces import BallNeighborhood, DiscreteTime
from skcomplex.core.specs import TotalisticSpec

start = {'variables': {'c': DeclaredVariableSpec(0.0, StateSpec())}, 'c': [0, 0, 0, 1, 0, 0, 0]}
cells = GroupElement('cells', element=DeclaredElement, space=LatticeSpace(7), on='sites', values=start)
pair = BallNeighborhood(metric='manhattan', n_dim=1)
rule = LocalRuleInteraction('rule', target='cells', state='c', rule=TotalisticSpec({1: 1}), neighborhood=pair)
automaton = CompositeSystem(elements=[cells], interactions=[rule], time=DiscreteTime())
simulate(automaton, until=2).get_values('cells.c').astype(int).tolist()
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

**How it combines.** A `GroupElement` with `on='sites'` holds one member per site, and a `FieldElement` adds the
operators of space, such as the Laplacian. `LocalRuleInteraction` reads a `BallNeighborhood` or a
`LatticeNeighborhood` around each site, and `CouplingInteraction` and `SamplingInteraction` read the nearest
neighbours. `CellComplexSpace`, `PhaseSpace`, and `SimplexSpace` build on a lattice.

**Limits and errors.** Every site has the same pattern of neighbours, so an irregular pattern of links needs a
`NetworkSpace`, and an irregular domain needs a `MeshSpace`. The boundary is one of three names, and `prepare` raises a
`ValidationError` for any other.

```python continuation
from skcomplex.core.errors import ValidationError

try:
    LatticeSpace((4, 4), boundary='open').prepare()
except ValidationError as error:
    message = str(error)
message
```

The message reads "Unknown lattice boundary 'open'. Use one of periodic, fixed, reflecting." The fix is
`boundary='fixed'` for edges that read the value `fill`.

**A field realization.** A field fixes the shape and the boundary of its lattice, such as the grid of a forest fire
whose edges do not burn.

```python continuation
class ForestSpace(LatticeSpace):
    def __init__(self, rows=10, columns=10, *, label='forest'):
        super().__init__((rows, columns), boundary='fixed', label=label)


ForestSpace(4, 5).prepare().size_
```

See its [fact sheet](../../../parts/spaces.md#latticespace)
and its [API entry][skcomplex.core.components.spaces.LatticeSpace].

## NetworkSpace

A `NetworkSpace` places the members on the nodes of a graph, so each member has the neighbours its edges give. Here
an epidemic spreads on a contact network shaped as a star. The person at the hub is infected, and each person is
infected at a rate of 0.5 times their own susceptibility and the infection of each contact.

```python
import networkx as nx

from skcomplex.core.components.elements import DeclaredElement, GroupElement
from skcomplex.core.components.interactions import CouplingInteraction
from skcomplex.core.components.spaces import NetworkSpace
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.specs import DeclaredVariableSpec, StateSpec

contacts = NetworkSpace(nx.star_graph(3))
sick = {'variables': {'x': DeclaredVariableSpec(0.0, StateSpec())}, 'x': [1.0, 0.0, 0.0, 0.0]}
people = GroupElement('people', element=DeclaredElement, space=contacts, on='sites', values=sick)
infection = CouplingInteraction(
    'infection', target='people', read='x', write='x', function=lambda s: 0.5 * (1 - s.member.x) * s.other.x
)
system = CompositeSystem(elements=[people], interactions=[infection])
simulate(system, until=4, dt=2.0).get_values('people.x').round(2).tolist()
```

The three leaves catch the infection from the hub, and their share of infection reaches 0.63 after two days and 0.86
after four, which is `1 - exp(-0.5 t)`.

The same class holds a power grid of four generators wired in a ring. Each generator pulls its phase toward those of
the generators it is wired to by the sine of their difference, as in the model of Kuramoto.

```python continuation
from sage.all__sagemath_symbolics import sin

grid = NetworkSpace(nx.cycle_graph(4))
phases = {'variables': {'theta': DeclaredVariableSpec(0.0, StateSpec())}, 'theta': [0.0, 0.4, 0.8, 1.2]}
generators = GroupElement('generators', element=DeclaredElement, space=grid, on='sites', values=phases)
pull = lambda s: sin(s.other.theta - s.member.theta)
sync = CouplingInteraction('sync', target='generators', read='theta', write='theta', function=pull)
system = CompositeSystem(elements=[generators], interactions=[sync])
simulate(system, until=6, dt=3.0).get_values('generators.theta').round(2).tolist()
```

The generators lock to the common phase 0.6, the mean of their initial phases.

**How the engine runs it.** `prepare` numbers the nodes in the order of `graph.nodes` and stores the neighbours of
every node in compressed sparse row form, the attribute `adjacency_`. A coupling over edges writes its laws once per
edge, and each law reads the states of both ends. A local rule reads the neighbours of every node padded to the
largest degree, the largest number of edges at a node.

**How it combines.** Groups sit on its nodes with `on='sites'`, on its edges with `on='edges'`, and on its
hyperedges, links of more than two nodes, with `on='hyperedges'`. A `LayerSpec` adds a named relation over the same
nodes, such as the colleagues of each person at work, which an `AdjacencyNeighborhood` or a coupling reads by its
label. A `TopologyInteraction` adds and removes edges during a run.

**Limits and errors.** The nodes are fixed for a run, and a node has no position of its own beyond the layout of the
graph. Every layer holds the nodes of the graph, and `prepare` raises a `ValidationError` for a layer with other
nodes.

```python continuation
from skcomplex.core.errors import ValidationError
from skcomplex.core.specs import LayerSpec

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

See its [fact sheet](../../../parts/spaces.md#networkspace)
and its [API entry][skcomplex.core.components.spaces.NetworkSpace].

## ContinuousSpace

A `ContinuousSpace` is a box of real positions, or the whole of `R^d`, through which members move. The positions are
states of the members, so the space has no sites. Here two atoms of a gas move in a periodic unit box, one toward
the right wall and one upward.

```python
from skcomplex.core.components.elements import DeclaredElement, GroupElement
from skcomplex.core.components.spaces import ContinuousSpace
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.specs import DeclaredVariableSpec, StateSpec

position = DeclaredVariableSpec(0.0, StateSpec(vector=True, order=2, role='position'))
start = {'variables': {'x': position}, 'x': [[0.9, 0.5], [0.2, 0.5]], 'x_d1': [[1.0, 0.0], [0.0, 1.0]]}
atoms = GroupElement('atoms', element=DeclaredElement, size=2, values=start)
box = ContinuousSpace(extent=(1.0, 1.0), boundary='periodic')
simulate(CompositeSystem(elements=[atoms], space=box), until=0.2, dt=0.2).get_values('atoms.x')[-1].round(2).tolist()
```

The position `x` is a vector state of order 2, so it has a velocity `x_d1`. The first atom leaves through the right
wall at 1.1 and enters again at 0.1, and the second moves from 0.5 to 0.7.

In geography the same space holds the districts of a city as polygons, called regions, which agents ask about.

```python continuation
from shapely import Point
from shapely import box as rectangle

districts = [rectangle(0.0, 0.0, 2.0, 2.0), rectangle(2.0, 0.0, 4.0, 2.0)]
city = ContinuousSpace(extent=(4.0, 2.0), regions={'districts': districts}).prepare()
city.regions_['districts'].query(Point(3.0, 1.0), predicate='intersects').tolist()
```

The point `(3, 1)` lies in district 1, the eastern one.

**How the engine runs it.** The space returns no laws. After every step the solver applies the boundary to every
position state: a periodic box wraps it, and a reflecting box mirrors it and reverses the velocity across the wall.
Pair distances in a periodic box use the nearest periodic image.

```python continuation
walls = ContinuousSpace(extent=(1.0, 1.0), boundary='reflecting')
trajectory = simulate(CompositeSystem(elements=[atoms], space=walls), until=0.2, dt=0.2)
trajectory.get_values('atoms.x')[-1].round(2).tolist(), trajectory.get_values('atoms.x_d1')[-1].round(2).tolist()
```

In the reflecting box the first atom bounces back to 0.9, and its velocity turns from 1 to -1.

**How it combines.** The members are agents or particles whose positions are states. `PairInteraction` reads pairs
within a cutoff, and a `CouplingInteraction` reads a `RadiusNeighborhood`, a `NearestNeighborhood`, or a
`DelaunayNeighborhood`.

**Limits and errors.** The space has no sites, so it holds no `FieldElement`, and regions need two dimensions. A
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

See its [fact sheet](../../../parts/spaces.md#continuousspace)
and its [API entry][skcomplex.core.components.spaces.ContinuousSpace].

## CellComplexSpace

A `CellComplexSpace` adds the links between the sites of a lattice or a network and the faces that the links bound,
each with an orientation. Cells of degree 0 are sites, of degree 1 links, and of degree 2 faces. In a lattice gauge
theory of physics the states live on the links of a lattice.

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

**How the engine runs it.** The space is a network whose edges are the oriented links and whose hyperedges are the
oriented faces. The law utilities `exterior_derivative`, `codifferential`, and `hodge_star` read its incidence
matrices and its Hodge stars, the ratios of dual to primal cell volumes, so the laws of discrete exterior calculus
compile to sparse products.

**How it combines.** A `GroupElement` or a `FieldElement` with `on='edges'` holds a state per link, and one with
`on='hyperedges'` a state per face. A `CofaceNeighborhood` reads the faces around each link, and
`SamplingInteraction` samples the links of a gauge field.

**Limits and errors.** It derives from a square lattice or a network, and it has no layers. Cells of degree 3 take
part in incidence only, since a group lives on sites, edges, or hyperedges. A triangular lattice raises a
`SpaceError` in `prepare`.

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

See its [fact sheet](../../../parts/spaces.md#cellcomplexspace)
and its [API entry][skcomplex.core.components.spaces.CellComplexSpace].

## PhaseSpace

A `PhaseSpace` pairs every position of a lattice with a value on one or more extra axes, called class axes, such as
velocity. A class axis is a range cut into equal cells. In the kinetic theory of gases a density varies over the
position and the velocity of the molecules.

```python
from skcomplex.core.components.spaces import LatticeSpace, PhaseSpace
from skcomplex.core.specs import AxisSpec

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

**How the engine runs it.** The position axes come first, so the sites of one position come one after the other. A
`FieldElement` on the space holds one density per site, and its laws read `coordinate('v')`, the velocity of the site,
for streaming. Moments such as the density at each position, `moment(u, 0, axes=('v',))`, reduce over the class axes.

**How it combines.** The usual partners are a `FieldElement` with the finite-volume discretization, a
`ConservationInteraction` for streaming and forces, and a `RateInteraction` for collisions. `SimplexSpace` is a phase
space whose class axes are frequencies.

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

See its [fact sheet](../../../parts/spaces.md#phasespace)
and its [API entry][skcomplex.core.components.spaces.PhaseSpace].

## SimplexSpace

A `SimplexSpace` is a phase space whose class axes are frequencies that are nonnegative and sum to at most one. The
last frequency is one minus the others, so two axes hold the shares of three types. In evolutionary game theory a
population is spread over the shares of three strategies.

```python
from skcomplex.core.components.spaces import LatticeSpace, SimplexSpace
from skcomplex.core.specs import AxisSpec

shares = [AxisSpec('rock', 4, 0.0, 1.0), AxisSpec('paper', 4, 0.0, 1.0)]
strategies = SimplexSpace(LatticeSpace((1,)), shares).prepare()
strategies.volumes_[0].tolist()
```

Each entry is the fraction of a cell inside the simplex. Cells below the slanted face are whole, cells along it are
cut in half, and cells beyond it are empty.

In population genetics the frequencies of three alleles of one gene, two of them free, fill the same triangle. With
three free frequencies the simplex is a tetrahedron of four alleles.

```python continuation
alleles = [AxisSpec(name, 4, 0.0, 1.0) for name in ('a', 'b', 'c')]
tetrahedron = SimplexSpace(LatticeSpace((1,)), alleles).prepare()
tetrahedron.shape_, round(float(tetrahedron.volumes_.sum()), 3)
```

The 64 cells hold a volume of 10.667 cells, one sixth of the cube, which is the volume of the tetrahedron.

**How the engine runs it.** Finite-volume fluxes pass only through the open part of each face, given by `apertures_`,
and change a cell by its volume, so the total is conserved and no density enters the empty cells. Moments and other
reductions over the class axes weight each cell by its volume.

**How it combines.** A `FieldElement` with the finite-volume discretization holds the density, and a
`ConservationInteraction` moves it over the frequencies, as replicator dynamics do.

**Limits and errors.** Local operators such as `laplacian` do not apply on a simplex. The class axes run over `[0, 1]`
with the same number of cells, and `prepare` raises a `ValidationError` otherwise.

```python continuation
from skcomplex.core.errors import ValidationError

try:
    SimplexSpace(LatticeSpace((1,)), [AxisSpec('x', 4, 0.0, 1.0), AxisSpec('y', 5, 0.0, 1.0)]).prepare()
except ValidationError as error:
    message = str(error)
message
```

The message says that the class axes of a simplex space are intervals on `[0, 1]` with the same number of cells, and
lists the axes it was given. The fix is 4 cells, or 5, on both axes.

**A field realization.** A field names the frequencies, such as the shares of the three strategies of a game at one
place.

```python continuation
class StrategySpace(SimplexSpace):
    def __init__(self, cells=20, *, label='strategies'):
        axes = [AxisSpec('rock', cells, 0.0, 1.0), AxisSpec('paper', cells, 0.0, 1.0)]
        super().__init__(LatticeSpace((1,)), axes, label=label)


StrategySpace(10).prepare().shape_
```

See its [fact sheet](../../../parts/spaces.md#simplexspace)
and its [API entry][skcomplex.core.components.spaces.SimplexSpace].

## MeshSpace

A `MeshSpace` is an unstructured mesh of triangles or tetrahedra, built with scikit-fem, whose nodes hold the members
of a field. Here heat leaves an L-shaped machine part whose walls are held at zero.

```python
from typing import Annotated

import skfem

from skcomplex.core.components.elements import Element, FieldElement
from skcomplex.core.components.laws import RateLaw
from skcomplex.core.components.laws.utils import laplacian
from skcomplex.core.components.spaces import MeshSpace
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.specs import ParameterSpec, StateSpec


class DiffusionElement(Element):
    law_classes = (RateLaw,)

    def __init__(
        self,
        label,
        visual=None,
        *,
        u: Annotated[float, StateSpec()] = 0.0,
        D: Annotated[float, ParameterSpec()] = 1.0,
    ):
        super().__init__(label, visual=visual)
        self.u = u
        self.D = D

    def _get_laws(self, namespace):
        return [RateLaw(namespace.u, namespace.D * laplacian(namespace.u))]


shape = skfem.MeshTri.init_lshaped().refined(1)
walls = {'u': ('fixed', 0.0)}
part = FieldElement(
    'part', element=DiffusionElement, space=MeshSpace(shape.p, shape.t), values={'u': 1.0}, boundary=walls,
    discretization='finite-element',
)
simulate(CompositeSystem(elements=[part]), until=1, dt=0.5).get_values('part.u').max(axis=1).round(3).tolist()
```

The hottest node falls from 1 to 0.013 after half a time unit, and after one unit it rounds to 0.

In hydrology the same class holds a block of an aquifer cut into tetrahedra, through which groundwater seeps.

```python continuation
import numpy as np

block = skfem.MeshTet.init_tensor(*[np.linspace(0.0, 1.0, 3)] * 3)
aquifer = MeshSpace(block.p, block.t).prepare()
aquifer.size_, aquifer.n_dim_
```

The block has 27 nodes in three dimensions.

**How the engine runs it.** The field takes the finite-element discretization. Its stiffness and mass matrices come
from scikit-fem, so `laplacian(u)` becomes a sparse product, and the boundary values hold at the nodes of
`boundary_nodes_`. The implicit methods of `ScipyODESolver` integrate the stiff system.

**How it combines.** A `FieldElement` with `discretization='finite-element'` is its usual partner, and the laws are
the same as on a lattice.

**Limits and errors.** Only fields live on a mesh, and the mesh does not change during a run. A boundary condition
is a name or a pair `('fixed', value)`, and the field raises a `ValidationError` in `prepare` for a bare number.

```python continuation
from skcomplex.core.errors import ValidationError

try:
    FieldElement(
        'part', element=DiffusionElement, space=MeshSpace(shape.p, shape.t), boundary={'u': 0.0},
        discretization='finite-element',
    ).prepare()
except ValidationError as error:
    message = str(error)
message
```

The message reads "Unknown boundary '0.0' of part.u. Use 'periodic', 'zero-flux', or ('fixed', value)." The fix is
`('fixed', 0.0)`.

**A field realization.** A field builds the mesh of its domain, such as an L-shaped plate refined a given number of
times.

```python continuation
class PlateSpace(MeshSpace):
    def __init__(self, refinements=2, *, label='plate'):
        plate = skfem.MeshTri.init_lshaped().refined(refinements)
        super().__init__(plate.p, plate.t, label=label)


PlateSpace(1).prepare().size_
```

See its [fact sheet](../../../parts/spaces.md#meshspace)
and its [API entry][skcomplex.core.components.spaces.MeshSpace].

## CayleySpace

A `CayleySpace` has one site for every element of a group, and joins each site `g` to the sites `g s` for every
generator `s`. The group is a Sage or GAP group. The symmetric group of three items, the six shuffles of three cards,
is generated by a swap and a cycle.

```python
from sage.all__sagemath_gap import SymmetricGroup

from skcomplex.core.components.spaces import CayleySpace

shuffles = CayleySpace(SymmetricGroup(3)).prepare()
shuffles.size_, shuffles.words_, shuffles.ball_sizes_
```

Each site carries its shortest word in the generators `a` and `b`. One site is the identity `1`, 4 sites lie within
one step of it, and all 6 within two.

In symbolic dynamics the free group on two generators has a tree as its Cayley graph. It is infinite, so the space
takes a ball of words up to a length.

```python continuation
from sage.all__sagemath_gap import libgap

tree = CayleySpace(libgap.FreeGroup(['a', 'b']), ball=2).prepare()
tree.size_, tree.ball_sizes_
```

The ball of radius 2 holds 17 sites: the identity, its 4 neighbours, and 12 words of length two.

**How the engine runs it.** `prepare` lists the elements by word length and builds `generator_table_`, the site of
`g s` for every site and generator. A local rule reads the sites of a memory set from it, and a neighbour outside the
ball reads the value `fill`.

**How it combines.** A `GroupElement` on its sites holds one cell per element. A `LocalRuleInteraction` reads a
`MemoryNeighborhood` for a cellular automaton on the group, or a `StateTranslationNeighborhood` for a map that reads
the cell named by the value of each cell.

**Limits and errors.** An infinite group needs a ball, and the argument `max_sites` bounds the size. A free group
without a ball raises a `SpaceTooLargeError` in `prepare`.

```python continuation
from skcomplex.core.errors import SpaceTooLargeError

try:
    CayleySpace(libgap.FreeGroup(['a', 'b'])).prepare()
except SpaceTooLargeError as error:
    message = str(error)
message
```

The message reads "The group space has more than 100000 sites. Ask for a smaller ball or a finite quotient, or raise
max_sites." The fix is `ball=2`, or a finite quotient of the group.

**A field realization.** A field fixes its group, such as the free group of a given rank in symbolic dynamics.

```python continuation
class FreeGroupSpace(CayleySpace):
    def __init__(self, rank=2, ball=3, *, label='tree'):
        letters = [chr(ord('a') + index) for index in range(rank)]
        super().__init__(libgap.FreeGroup(letters), ball=ball, label=label)


FreeGroupSpace(rank=2, ball=1).prepare().size_
```

See its [fact sheet](../../../parts/spaces.md#cayleyspace)
and its [API entry][skcomplex.core.components.spaces.CayleySpace].

## BaseSpace

`BaseSpace` is the base of every space. A new kind of structure that members sit on, beyond lattices, networks,
continuous boxes, meshes, and groups, derives from it and answers the hooks the engine reads, such as the neighbours
of a site.

```python
from skcomplex.core.components.spaces import BaseSpace, ContinuousSpace, LatticeSpace

issubclass(LatticeSpace, BaseSpace), issubclass(ContinuousSpace, BaseSpace)
```

See its [fact sheet](../../../parts/spaces.md#basespace)
and its [API entry][skcomplex.core.components.spaces.BaseSpace].

## BallNeighborhood

A `BallNeighborhood` takes the lattice offsets within a radius under a metric. The Manhattan metric of radius 1 gives
the four sites of the von Neumann neighbourhood, and the Chebyshev metric the eight of the Moore neighbourhood. In a
sandpile a site with four grains or more topples, sends one grain to each of its four neighbours, and grains that
leave the grid are lost.

```python
from skcomplex.core.components.elements import DeclaredElement, GroupElement
from skcomplex.core.components.interactions import LocalRuleInteraction
from skcomplex.core.components.spaces import BallNeighborhood, DiscreteTime, LatticeSpace
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.specs import DeclaredVariableSpec, StateSpec


def topple(height, *neighbours):
    return height - 4 * (height >= 4) + sum(neighbour >= 4 for neighbour in neighbours)


grains = {'variables': {'h': DeclaredVariableSpec(0.0, StateSpec())}, 'h': [0, 0, 0, 0, 4, 0, 0, 0, 0]}
board = LatticeSpace((3, 3), boundary='fixed')
pile = GroupElement('pile', element=DeclaredElement, space=board, on='sites', values=grains)
cross = BallNeighborhood(metric='manhattan', center='first')
toppling = LocalRuleInteraction('toppling', target='pile', state='h', rule=topple, neighborhood=cross)
sandpile = CompositeSystem(elements=[pile], interactions=[toppling], time=DiscreteTime())
simulate(sandpile, until=1).get_values('pile.h')[-1].reshape(3, 3).tolist()
```

With `center='first'` the rule receives the site itself before its four neighbours. The centre topples, and each of
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
voters = {'variables': {'o': DeclaredVariableSpec(0.0, StateSpec())}, 'o': camps}
people = GroupElement('people', element=DeclaredElement, space=LatticeSpace((4, 4)), on='sites', values=voters)
moore = BallNeighborhood(center='first')
vote = LocalRuleInteraction('vote', target='people', state='o', rule=majority, neighborhood=moore)
opinions = simulate(CompositeSystem(elements=[people], interactions=[vote], time=DiscreteTime()), until=1)
opinions.get_values('people.o')[-1].reshape(4, 4).astype(int).tolist()
```

The person at site 5, alone in their camp, joins the camp of the left half after one step.

**How the engine runs it.** `prepare` lists the offsets in `offsets_`, in lexicographic order for the Manhattan and
Chebyshev metrics and ring by ring for the hexagonal one. The space turns them into a table of the neighbour of every
site in every slot, and the rule receives the values in that order.

```python continuation
cross.prepare().offsets_
```

The site comes first, at `(0, 0)`, and its four neighbours follow in lexicographic order.

**How it combines.** It is the neighbourhood of `LocalRuleInteraction`, of `SamplingInteraction`, and of the
`neighbors` reduction on a `LatticeSpace`. Its offsets have the dimension of the lattice.

**Limits and errors.** A ball is symmetric, so a neighbourhood of one direction needs a `LatticeNeighborhood`. Its
`n_dim` must match the lattice, and the run raises a `ValidationError` when it does not.

```python continuation
from skcomplex.core.errors import ValidationError

line = GroupElement('line', element=DeclaredElement, space=LatticeSpace(5), on='sites', values=grains | {'h': 0.0})
toppling = LocalRuleInteraction('toppling', target='line', state='h', rule=topple, neighborhood=BallNeighborhood())
try:
    simulate(CompositeSystem(elements=[line], interactions=[toppling], time=DiscreteTime()), until=1)
except ValidationError as error:
    message = str(error)
message
```

The message says that the neighbourhood has offsets of dimension 2, but the lattice has 1 dimension. The fix is
`BallNeighborhood(n_dim=1)`.

**A field realization.** A field names the ball it reads, such as the von Neumann neighbourhood of a sandpile.

```python continuation
class VonNeumannNeighborhood(BallNeighborhood):
    def __init__(self, *, label='von_neumann'):
        super().__init__(1, metric='manhattan', label=label)


len(VonNeumannNeighborhood().prepare().offsets_)
```

See its [fact sheet](../../../parts/spaces.md#ballneighborhood)
and its [API entry][skcomplex.core.components.spaces.BallNeighborhood].

## LatticeNeighborhood

A `LatticeNeighborhood` takes any list of lattice offsets, in the order a rule receives their values. Rule 90 of an
elementary cellular automaton reads the left and the right neighbour of each cell and returns their sum modulo 2.

```python
from skcomplex.core.components.elements import DeclaredElement, GroupElement
from skcomplex.core.components.interactions import LocalRuleInteraction
from skcomplex.core.components.spaces import DiscreteTime, LatticeNeighborhood, LatticeSpace
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.specs import DeclaredVariableSpec, StateSpec

start = {'variables': {'c': DeclaredVariableSpec(0.0, StateSpec())}, 'c': [0, 0, 0, 1, 0, 0, 0]}
cells = GroupElement('cells', element=DeclaredElement, space=LatticeSpace(7), on='sites', values=start)
sides = LatticeNeighborhood([(-1,), (1,)])
parity = lambda left, right: (left + right) % 2
rule = LocalRuleInteraction('rule', target='cells', state='c', rule=parity, neighborhood=sides)
system = CompositeSystem(elements=[cells], interactions=[rule], time=DiscreteTime())
simulate(system, until=2).get_values('cells.c').astype(int).tolist()
```

The rows are those of rule 90 in the `LatticeSpace` section, written here with the two offsets in a chosen order.

In a model of advection a pulse of pollutant moves one cell downstream at every step. The rule reads only the upwind
cell, the cell at offset -1.

```python continuation
pulse = {'variables': {'c': DeclaredVariableSpec(0.0, StateSpec())}, 'c': [0, 1, 1, 0, 0, 0]}
river = GroupElement('river', element=DeclaredElement, space=LatticeSpace(6), on='sites', values=pulse)
upwind = LatticeNeighborhood([(-1,)])
flow = LocalRuleInteraction('flow', target='river', state='c', rule=lambda upstream: upstream, neighborhood=upwind)
system = CompositeSystem(elements=[river], interactions=[flow], time=DiscreteTime())
simulate(system, until=2).get_values('river.c').astype(int).tolist()
```

The pulse of two cells moves one cell to the right at each step.

**How the engine runs it.** `prepare` stores the offsets as tuples of integers in `offsets_`. The space builds the
table of the neighbour of every site at every offset, following its boundary, and the rule receives one array per
offset.

**How it combines.** It fills the same places as `BallNeighborhood`: `LocalRuleInteraction`, `SamplingInteraction`,
and the `neighbors` reduction on a `LatticeSpace`.

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

See its [fact sheet](../../../parts/spaces.md#latticeneighborhood)
and its [API entry][skcomplex.core.components.spaces.LatticeNeighborhood].

## AdjacencyNeighborhood

An `AdjacencyNeighborhood` takes the nodes joined to a node by the edges of a network, so each node may have a
different number of neighbours. Here people in a line of contacts catch a disease when any contact is infected.

```python
import networkx as nx
import numpy as np

from skcomplex.core.components.elements import DeclaredElement, GroupElement
from skcomplex.core.components.interactions import LocalRuleInteraction
from skcomplex.core.components.spaces import AdjacencyNeighborhood, DiscreteTime, NetworkSpace
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.specs import DeclaredVariableSpec, StateSpec


def catch(infected, *contacts):
    return np.maximum(infected, sum(contacts) > 0)


sick = {'variables': {'x': DeclaredVariableSpec(0.0, StateSpec())}, 'x': [1, 0, 0, 0, 0]}
people = GroupElement('people', element=DeclaredElement, space=NetworkSpace(nx.path_graph(5)), on='sites', values=sick)
spread = LocalRuleInteraction(
    'spread', target='people', state='x', rule=catch, neighborhood=AdjacencyNeighborhood(center=True)
)
system = CompositeSystem(elements=[people], interactions=[spread], time=DiscreteTime())
simulate(system, until=2).get_values('people.x').astype(int).tolist()
```

With `center=True` the rule receives the node itself first. The infection moves one person along the line at every
step.

At work the same people have other contacts. A `LayerSpec` adds the network of colleagues over the same nodes, and
the neighbourhood reads it by its label.

```python continuation
from skcomplex.core.specs import LayerSpec

office = nx.empty_graph(5)
office.add_edges_from([(0, 4), (4, 3)])
town = NetworkSpace(nx.path_graph(5), layers=[LayerSpec('work', office)])
colleagues = GroupElement('people', element=DeclaredElement, space=town, on='sites', values=sick)
at_work = AdjacencyNeighborhood(center=True, layer='work')
spread = LocalRuleInteraction('spread', target='people', state='x', rule=catch, neighborhood=at_work)
system = CompositeSystem(elements=[colleagues], interactions=[spread], time=DiscreteTime())
simulate(system, until=2).get_values('people.x').astype(int).tolist()
```

At work the infection goes from person 0 to person 4 and then to person 3, along the edges of the layer.

**How the engine runs it.** The space gives the neighbours of every node from its adjacency in compressed sparse row
form. A local rule receives their values padded to the largest degree with the value `fill` of the network. The
default fill is 0, so a sum is not changed by the padding.

**How it combines.** It is the neighbourhood of `LocalRuleInteraction` and `TransitionInteraction` on a
`NetworkSpace`, and the default of the `neighbors` reduction on a network.

**Limits and errors.** It reads a network, not a lattice, and a local rule on a lattice raises a `ValidationError`
when it is given one.

```python continuation
from skcomplex.core.components.spaces import LatticeSpace
from skcomplex.core.errors import ValidationError

row = GroupElement('people', element=DeclaredElement, space=LatticeSpace(5), on='sites', values=sick)
spread = LocalRuleInteraction('spread', target='people', state='x', rule=catch, neighborhood=AdjacencyNeighborhood())
try:
    simulate(CompositeSystem(elements=[row], interactions=[spread], time=DiscreteTime()), until=1)
except ValidationError as error:
    message = str(error)
message
```

The message says that a lattice reads a lattice neighbourhood of offsets, but was given an `AdjacencyNeighborhood`.
The fix is a `BallNeighborhood` on a lattice, or a `NetworkSpace` for the adjacency.

**A field realization.** A field names the layer it reads, such as the household of each person in an epidemic.

```python continuation
class HouseholdNeighborhood(AdjacencyNeighborhood):
    def __init__(self, *, label='household'):
        super().__init__(center=True, layer='household', label=label)


HouseholdNeighborhood().layer
```

See its [fact sheet](../../../parts/spaces.md#adjacencyneighborhood)
and its [API entry][skcomplex.core.components.spaces.AdjacencyNeighborhood].

## CofaceNeighborhood

A `CofaceNeighborhood` takes the cells of the next degree that contain a cell: the faces around a link, or the links
at a site. Here a state on the links of a network sums the circulation around the faces that contain each link. The
circulation of a face is the oriented sum of the link values around it, which `exterior_derivative` gives.

```python
import networkx as nx

from skcomplex.core.components.elements import DeclaredElement, GroupElement
from skcomplex.core.components.interactions import MapInteraction
from skcomplex.core.components.laws.utils import exterior_derivative, neighbors
from skcomplex.core.components.spaces import CellComplexSpace, CofaceNeighborhood, DiscreteTime, NetworkSpace
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.specs import DeclaredVariableSpec, StateSpec

ring = nx.cycle_graph(4)
ring.add_edge(0, 2)
complex_ = CellComplexSpace(NetworkSpace(ring))
variables = {'a': DeclaredVariableSpec(0.0, StateSpec()), 'n': DeclaredVariableSpec(0.0, StateSpec())}
unit = {'variables': variables, 'a': [1, 0, 0, 0, 0]}
flows = GroupElement('flows', element=DeclaredElement, space=complex_, on='edges', values=unit)
circulation = lambda s: neighbors(exterior_derivative(s.target.a), over=CofaceNeighborhood(), reduce='sum')
around = MapInteraction('around', target='flows', state='n', function=circulation)
system = CompositeSystem(elements=[flows], interactions=[around], time=DiscreteTime())
simulate(system, until=1).get_values('flows.n')[-1].tolist()
```

A unit of flow runs on the first edge, from node 0 to node 1, so only the triangle `(0, 1, 2)` circulates. Its three
edges, the first, third, and fourth, read 1, and the two edges of the other triangle read 0.

On a periodic square lattice of a gauge theory every link lies on two plaquettes. A sum of 1 over the cofaces counts
them.

```python continuation
from skcomplex.core.components.spaces import LatticeSpace

plaquettes = CellComplexSpace(LatticeSpace((3, 3)))
links = GroupElement('links', element=DeclaredElement, space=plaquettes, on='edges', values={'variables': variables})
count = MapInteraction(
    'count', target='links', state='n', function=lambda s: neighbors(1.0, over=CofaceNeighborhood(), reduce='sum')
)
counts = simulate(CompositeSystem(elements=[links], interactions=[count], time=DiscreteTime()), until=1)
set(counts.get_values('links.n')[-1].tolist())
```

Every one of the 18 links reads 2.

**How the engine runs it.** The cell complex stores `cofaces_`, the cofaces of every cell with their orientations.
The reduction runs over them, and inside it `exterior_derivative` is the oriented sum around the one coface. Every
coface counts with weight one.

**How it combines.** It needs a `CellComplexSpace`, and a group on its edges or sites. `SamplingInteraction` reads it
in the Wilson action of a lattice gauge field, and `MapInteraction` or `RateInteraction` in any law on the links.

**Limits and errors.** It reads the cells of the next degree only, so the cells of the same degree need an
`AdjacencyNeighborhood`. Outside a cell complex the run raises a `ValidationError`.

```python continuation
from skcomplex.core.errors import ValidationError

plain = NetworkSpace(ring)
nodes = GroupElement('nodes', element=DeclaredElement, space=plain, on='sites', values={'variables': variables})
count = MapInteraction(
    'count', target='nodes', state='n', function=lambda s: neighbors(1.0, over=CofaceNeighborhood(), reduce='sum')
)
try:
    simulate(CompositeSystem(elements=[nodes], interactions=[count], time=DiscreteTime()), until=1)
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

See its [fact sheet](../../../parts/spaces.md#cofaceneighborhood)
and its [API entry][skcomplex.core.components.spaces.CofaceNeighborhood].

## RadiusNeighborhood

A `RadiusNeighborhood` takes the members of a continuous space within a distance of each member, so the number of
neighbours varies. In the Vicsek model birds align with the flockmates they see within a range. Here each bird counts
the birds within one unit.

```python
from skcomplex.core.components.elements import DeclaredElement, GroupElement
from skcomplex.core.components.interactions import CouplingInteraction
from skcomplex.core.components.spaces import ContinuousSpace, DiscreteTime, RadiusNeighborhood
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.specs import DeclaredVariableSpec, StateSpec

variables = {
    'x': DeclaredVariableSpec(0.0, StateSpec(vector=True, role='position')),
    'seen': DeclaredVariableSpec(0.0, StateSpec()),
}
positions = [[1.0, 1.0], [1.5, 1.0], [2.0, 1.2], [6.0, 6.0], [9.8, 1.0]]
flock = GroupElement('flock', element=DeclaredElement, size=5, values={'variables': variables, 'x': positions})
sight = CouplingInteraction(
    'sight', target='flock', write='seen', function=lambda s: 1.0, neighborhood=RadiusNeighborhood(1.0), form='map'
)
sky = ContinuousSpace(extent=(10.0, 10.0))
system = CompositeSystem(elements=[flock], interactions=[sight], space=sky, time=DiscreteTime())
simulate(system, until=1).get_values('flock.seen')[-1].tolist()
```

The middle bird of the three on the left sees both others, the outer two see only the middle one, and the last two
birds see nobody.

In physics atoms interact with every atom within a cutoff, and in a periodic box the distance is that of the nearest
periodic image. With a cutoff of 1.5 the atom at `x = 9.8` reaches the atom at `x = 1.0` across the wall.

```python continuation
cutoff = CouplingInteraction(
    'cutoff', target='flock', write='seen', function=lambda s: 1.0, neighborhood=RadiusNeighborhood(1.5), form='map'
)
crystal = ContinuousSpace(extent=(10.0, 10.0), boundary='periodic')
system = CompositeSystem(elements=[flock], interactions=[cutoff], space=crystal, time=DiscreteTime())
simulate(system, until=1).get_values('flock.seen')[-1].tolist()
```

The first atom now reaches three others, the atom at `x = 9.8` among them, and that atom reaches the first.

**How the engine runs it.** At every evaluation the engine finds the pairs of members within the radius from their
current position states, so the neighbours follow the members as they move. An `angle` below `pi` keeps only the
neighbours inside a cone around the direction of each member, a field of view.

**How it combines.** It is the neighbourhood of `CouplingInteraction`, of the `neighbors` reduction, and of the
hyperedges of a continuous space. Its partners are groups with a vector position state on a `ContinuousSpace`.

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

See its [fact sheet](../../../parts/spaces.md#radiusneighborhood)
and its [API entry][skcomplex.core.components.spaces.RadiusNeighborhood].

## NearestNeighborhood

A `NearestNeighborhood` takes the `k` members of a continuous space closest to each member, whatever their distance.
Starlings in a flock follow a fixed number of nearest birds. Here each bird takes the mean heading of its nearest
bird.

```python
from skcomplex.core.components.elements import DeclaredElement, GroupElement
from skcomplex.core.components.interactions import CouplingInteraction
from skcomplex.core.components.spaces import ContinuousSpace, DiscreteTime, NearestNeighborhood
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.specs import DeclaredVariableSpec, StateSpec

variables = {
    'x': DeclaredVariableSpec(0.0, StateSpec(vector=True, role='position')),
    'heading': DeclaredVariableSpec(0.0, StateSpec()),
}
positions = [[1.0, 1.0], [1.5, 1.0], [2.0, 1.2], [6.0, 6.0], [9.8, 1.0]]
starlings = {'variables': variables, 'x': positions, 'heading': [0.0, 0.5, 1.0, 2.0, 3.0]}
flock = GroupElement('flock', element=DeclaredElement, size=5, values=starlings)
follow = CouplingInteraction(
    'follow', target='flock', write='heading', function=lambda s: s.other.heading, reduce='mean',
    neighborhood=NearestNeighborhood(1), form='map',
)
sky = ContinuousSpace(extent=(10.0, 10.0))
system = CompositeSystem(elements=[flock], interactions=[follow], space=sky, time=DiscreteTime())
simulate(system, until=1).get_values('flock.heading')[-1].tolist()
```

Unlike a radius, the nearest bird always exists, so the lone birds on the right also follow one: the bird at
`(6, 6)` follows the bird at `(2, 1.2)`, and the bird at `(9.8, 1)` follows the bird at `(6, 6)`.

Robots of a swarm align with their two nearest robots, which here means the mean heading of two neighbours.

```python continuation
swarm = CouplingInteraction(
    'swarm', target='flock', write='heading', function=lambda s: s.other.heading, reduce='mean',
    neighborhood=NearestNeighborhood(2), form='map',
)
system = CompositeSystem(elements=[flock], interactions=[swarm], space=sky, time=DiscreteTime())
simulate(system, until=1).get_values('flock.heading')[-1].tolist()
```

The headings become 0.75, 0.5, 0.25, 2.0, and 1.5. Each of the two lone birds takes the mean of the other lone bird
and the bird at `(2, 1.2)`.

**How the engine runs it.** At every evaluation the engine sorts the other members of each member by distance from
their current position states, by the nearest periodic image in a periodic box, and keeps the first `k`. A group of
at most `k` other members gives each member all the others.

**How it combines.** It is the neighbourhood of `CouplingInteraction` and of the `neighbors` reduction, on a group
with a vector position state in a `ContinuousSpace`.

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

See its [fact sheet](../../../parts/spaces.md#nearestneighborhood)
and its [API entry][skcomplex.core.components.spaces.NearestNeighborhood].

## DelaunayNeighborhood

A `DelaunayNeighborhood` joins two members when their Voronoi cells touch, the regions of the plane closer to each of
them than to any other member. In a tissue each cell pushes on the cells that touch it. Here four cells sit at the
corners of a square and one at its centre.

```python
from skcomplex.core.components.elements import DeclaredElement, GroupElement
from skcomplex.core.components.interactions import CouplingInteraction
from skcomplex.core.components.spaces import ContinuousSpace, DelaunayNeighborhood, DiscreteTime
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.specs import DeclaredVariableSpec, StateSpec

variables = {
    'x': DeclaredVariableSpec(0.0, StateSpec(vector=True, role='position')),
    'touching': DeclaredVariableSpec(0.0, StateSpec()),
}
positions = [[0.5, 0.5], [1.5, 0.5], [1.5, 1.5], [0.5, 1.5], [1.0, 1.0]]
tissue = GroupElement('tissue', element=DeclaredElement, size=5, values={'variables': variables, 'x': positions})
touch = CouplingInteraction(
    'touch', target='tissue', write='touching', function=lambda s: 1.0, neighborhood=DelaunayNeighborhood(), form='map'
)
dish = ContinuousSpace(extent=(2.0, 2.0))
system = CompositeSystem(elements=[tissue], interactions=[touch], space=dish, time=DiscreteTime())
simulate(system, until=1).get_values('tissue.touching')[-1].tolist()
```

Each corner touches its two neighbouring corners and the centre, and the centre touches all four corners.

Bubbles of a foam share walls in the same way. With `center=True` each bubble also counts itself, as a mean over a
bubble and its neighbours needs.

```python continuation
foam = CouplingInteraction(
    'foam', target='tissue', write='touching', function=lambda s: 1.0,
    neighborhood=DelaunayNeighborhood(center=True), form='map',
)
system = CompositeSystem(elements=[tissue], interactions=[foam], space=dish, time=DiscreteTime())
simulate(system, until=1).get_values('tissue.touching')[-1].tolist()
```

Each count grows by one: 4 for every corner and 5 for the centre.

**How the engine runs it.** At every evaluation the engine triangulates the current positions with SciPy, so the
neighbours follow moving members. In a periodic box the triangulation covers the periodic images. The neighbourhood
runs on the NumPy backend only.

**How it combines.** It is the neighbourhood of `CouplingInteraction` and of the `neighbors` reduction, on a group
with a vector position state in a `ContinuousSpace`.

**Limits and errors.** A group with too few members to triangulate gives each member all the others. On a lattice or
a network the run raises a `ValidationError`, since members there have no positions to triangulate.

```python continuation
from skcomplex.core.components.spaces import LatticeSpace
from skcomplex.core.errors import ValidationError

counts = {'variables': {'touching': DeclaredVariableSpec(0.0, StateSpec())}}
row = GroupElement('row', element=DeclaredElement, space=LatticeSpace(4), on='sites', values=counts)
touch = CouplingInteraction(
    'touch', target='row', write='touching', function=lambda s: 1.0, neighborhood=DelaunayNeighborhood(), form='map'
)
try:
    simulate(CompositeSystem(elements=[row], interactions=[touch], time=DiscreteTime()), until=1)
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

See its [fact sheet](../../../parts/spaces.md#delaunayneighborhood)
and its [API entry][skcomplex.core.components.spaces.DelaunayNeighborhood].

## MemoryNeighborhood

A `MemoryNeighborhood` is the memory set of a cellular automaton on a group: words in the generators that a cell at
`g` reads at `g s`. Here three cells sit on the cyclic group of order three, and each cell copies the cell one step
along the generator `a`.

```python
from sage.all__sagemath_gap import PermutationGroup

from skcomplex.core.components.elements import DeclaredElement, GroupElement
from skcomplex.core.components.interactions import LocalRuleInteraction
from skcomplex.core.components.spaces import CayleySpace, DiscreteTime, MemoryNeighborhood
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.specs import DeclaredVariableSpec, StateSpec

cycle = CayleySpace(PermutationGroup([[(1, 2, 3)]]))
start = {'variables': {'c': DeclaredVariableSpec(0.0, StateSpec())}, 'c': [1, 0, 0]}
cells = GroupElement('cells', element=DeclaredElement, space=cycle, on='sites', values=start)
ahead = MemoryNeighborhood(['a'])
shift = LocalRuleInteraction('shift', target='cells', state='c', rule=lambda value: value, neighborhood=ahead)
system = CompositeSystem(elements=[cells], interactions=[shift], time=DiscreteTime())
simulate(system, until=3).get_values('cells.c').astype(int).tolist()
```

The live cell moves around the three sites and returns after three steps.

On the tree of the free group a cell becomes live when it or a cell one generator away is live, as a signal spreads
through a network without loops.

```python continuation
from sage.all__sagemath_gap import libgap

from skcomplex.core.specs import TotalisticSpec

tree = CayleySpace(libgap.FreeGroup(['a', 'b']), ball=2)
seed = {'variables': {'c': DeclaredVariableSpec(0.0, StateSpec())}, 'c': [1] + [0] * 16}
nodes = GroupElement('nodes', element=DeclaredElement, space=tree, on='sites', values=seed)
star = MemoryNeighborhood(['1', 'a', 'a^-1', 'b', 'b^-1'])
any_live = TotalisticSpec({1: 1, 2: 1, 3: 1, 4: 1, 5: 1})
spread = LocalRuleInteraction('spread', target='nodes', state='c', rule=any_live, neighborhood=star)
system = CompositeSystem(elements=[nodes], interactions=[spread], time=DiscreteTime())
simulate(system, until=2).get_values('nodes.c').sum(axis=1).tolist()
```

The live cells number 1, then 5, then 17, the sizes of the balls of radius 0, 1, and 2.

**How the engine runs it.** `prepare` stores the words in `elements_`. The `CayleySpace` multiplies every site by
every word with its table of generators, and the rule receives the values in the order of the words, so the
neighbourhood keeps its orientation.

**How it combines.** It needs a `CayleySpace`, and it is the neighbourhood of `LocalRuleInteraction`.

**Limits and errors.** Each factor of a word is a generator of the group with an optional exponent, and the run raises
a `ValidationError` for any other.

```python continuation
from skcomplex.core.errors import ValidationError

unknown = MemoryNeighborhood(['c'])
shift = LocalRuleInteraction('shift', target='cells', state='c', rule=lambda value: value, neighborhood=unknown)
try:
    simulate(CompositeSystem(elements=[cells], interactions=[shift], time=DiscreteTime()), until=1)
except ValidationError as error:
    message = str(error)
message
```

The message says that the factor `c` of the word `c` is not a generator of the group. The fix is a word in `a`, the
one generator of this group, such as `'a^-1'`.

**A field realization.** A field names its memory set, such as a cell and its four neighbours on the tree of the free
group of rank two.

```python continuation
class TreeNeighborhood(MemoryNeighborhood):
    def __init__(self, *, label='tree'):
        super().__init__(['1', 'a', 'a^-1', 'b', 'b^-1'], label=label)


TreeNeighborhood().prepare().elements_
```

See its [fact sheet](../../../parts/spaces.md#memoryneighborhood)
and its [API entry][skcomplex.core.components.spaces.MemoryNeighborhood].

## StateTranslationNeighborhood

A `StateTranslationNeighborhood` reads cells that move with the configuration. A cell at `g` holds a group element
`x(g)`, stored as the number of its site, and reads the cell at `g x(g) s` for every word `s`. The map
`x(g x(g))` of Ceccherini-Silberstein and Coornaert (2023, Example 1.8.2) is continuous and equivariant but is no
cellular automaton. Here it runs on the symmetric group of three items.

```python
from sage.all__sagemath_gap import SymmetricGroup
from sage.all__sagemath_symbolics import ZZ

from skcomplex.core.components.elements import DeclaredElement, GroupElement
from skcomplex.core.components.interactions import LocalRuleInteraction
from skcomplex.core.components.spaces import CayleySpace, DiscreteTime, StateTranslationNeighborhood
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.specs import DeclaredVariableSpec, StateSpec

shuffles = CayleySpace(SymmetricGroup(3))
start = {'variables': {'c': DeclaredVariableSpec(0, StateSpec(domain=ZZ))}, 'c': [1, 2, 3, 4, 5, 0]}
cells = GroupElement('cells', element=DeclaredElement, space=shuffles, on='sites', values=start)
translate = LocalRuleInteraction(
    'translate', target='cells', state='c', rule=lambda value: value, neighborhood=StateTranslationNeighborhood()
)
system = CompositeSystem(elements=[cells], interactions=[translate], time=DiscreteTime())
simulate(system, until=1).get_values('cells.c').tolist()
```

Each cell takes the value of the cell at the product of its site and its value, so the cells hold 2, 1, 0, 3, 3,
and 0 after one step.

A word after the translation shifts every read by a generator, here `a`, which gives a second map of the same kind.

```python continuation
shifted = LocalRuleInteraction(
    'translate', target='cells', state='c', rule=lambda value: value, neighborhood=StateTranslationNeighborhood(['a'])
)
system = CompositeSystem(elements=[cells], interactions=[shifted], time=DiscreteTime())
simulate(system, until=1).get_values('cells.c')[-1].tolist()
```

The cells now hold 3, 2, 5, 1, 1, and 5, a configuration other than that of the first map.

**How the engine runs it.** The `CayleySpace` builds the table of the products of every site with every site, and
the rule reads the cell at the product of each site and its current value. On an infinite group the run truncates the
map to a ball, and a product outside it reads the value `fill` of the space.

**How it combines.** It needs a `CayleySpace` and a state whose values name group elements, and it is the
neighbourhood of `LocalRuleInteraction`.

**Limits and errors.** The cells read move with the configuration, so it has no finite memory set on an infinite
group. A lattice is no group space, and the run raises a `ValidationError` there.

```python continuation
from skcomplex.core.components.spaces import LatticeSpace
from skcomplex.core.errors import ValidationError

row = GroupElement('cells', element=DeclaredElement, space=LatticeSpace(6), on='sites', values=start)
try:
    simulate(CompositeSystem(elements=[row], interactions=[translate], time=DiscreteTime()), until=1)
except ValidationError as error:
    message = str(error)
message
```

The message says that `LatticeSpace` is no group space, so it reads no memory translated by a group element. The fix
is a `CayleySpace`, such as that of the cyclic group of order six in place of the ring of six cells.

**A field realization.** The neighbourhood belongs to symbolic dynamics on groups, and a subclass names its words,
such as the read one generator past the translated site.

```python continuation
class ShiftedTranslationNeighborhood(StateTranslationNeighborhood):
    def __init__(self, *, label='shifted'):
        super().__init__(['a'], label=label)


ShiftedTranslationNeighborhood().prepare().elements_
```

See its [fact sheet](../../../parts/spaces.md#statetranslationneighborhood)
and its [API entry][skcomplex.core.components.spaces.StateTranslationNeighborhood].

## BaseNeighborhood

`BaseNeighborhood` is the base of every neighbourhood. A new rule for choosing the members around a member derives
from it, and the space that reads it decides which kinds it accepts.

```python
from skcomplex.core.components.spaces import BallNeighborhood, BaseNeighborhood, RadiusNeighborhood

issubclass(BallNeighborhood, BaseNeighborhood), issubclass(RadiusNeighborhood, BaseNeighborhood)
```

See its [fact sheet](../../../parts/spaces.md#baseneighborhood)
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

**How it combines.** A system without a time takes the time its parts agree on, so a system of rate laws runs in
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

See its [fact sheet](../../../parts/spaces.md#continuoustime)
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

See its [fact sheet](../../../parts/spaces.md#discretetime)
and its [API entry][skcomplex.core.components.spaces.DiscreteTime].

## StaticTime

A `StaticTime` has no dynamics. The system holds equations, constraints, and objectives, and a run solves them once.
In economics two firms of a Cournot duopoly each choose the quantity that maximizes their profit given the other's
quantity, and the equilibrium solves both conditions together.

```python
from typing import Annotated

from skcomplex.core.components.elements import Element
from skcomplex.core.components.laws import EquationLaw
from skcomplex.core.components.spaces import StaticTime
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.specs import ParameterSpec, StateSpec


class DuopolyElement(Element):
    law_classes = (EquationLaw,)

    def __init__(
        self,
        label,
        visual=None,
        *,
        q1: Annotated[float, StateSpec()] = 0.0,
        q2: Annotated[float, StateSpec()] = 0.0,
        a: Annotated[float, ParameterSpec()] = 10.0,
        c: Annotated[float, ParameterSpec()] = 1.0,
    ):
        super().__init__(label, visual=visual)
        self.q1 = q1
        self.q2 = q2
        self.a = a
        self.c = c

    def _get_laws(self, namespace):
        return [
            EquationLaw(namespace.a - 2 * namespace.q1 - namespace.q2 == namespace.c, target=namespace.q1),
            EquationLaw(namespace.a - 2 * namespace.q2 - namespace.q1 == namespace.c, target=namespace.q2),
        ]


market = simulate(CompositeSystem(elements=[DuopolyElement('market')], time=StaticTime()), until=1)
market.get_values('market.q1').tolist(), market.get_values('market.q2').tolist()
```

Each firm produces 3 units, `(a - c) / 3`, and the trajectory has one frame.

In electronics a circuit finds its operating point. A source of 9 volts feeds two resistors of 1000 and 2000 ohms in
series, and the voltage between them balances the currents through both.

```python continuation
class DividerElement(Element):
    law_classes = (EquationLaw,)

    def __init__(
        self,
        label,
        visual=None,
        *,
        v: Annotated[float, StateSpec()] = 0.0,
        source: Annotated[float, ParameterSpec()] = 9.0,
        r1: Annotated[float, ParameterSpec()] = 1000.0,
        r2: Annotated[float, ParameterSpec()] = 2000.0,
    ):
        super().__init__(label, visual=visual)
        self.v = v
        self.source = source
        self.r1 = r1
        self.r2 = r2

    def _get_laws(self, namespace):
        current = (namespace.source - namespace.v) / namespace.r1
        return [EquationLaw(current == namespace.v / namespace.r2, target=namespace.v)]


simulate(CompositeSystem(elements=[DividerElement('node')], time=StaticTime()), until=1).get_values('node.v').tolist()
```

The node sits at 6 volts, two thirds of the source.

**How the engine runs it.** The time has one frame, at time 0. The compiler gathers the equations of every target
into one problem, solves a linear problem in closed form where it can, and passes the others to a root solver.

**How it combines.** Its laws are `EquationLaw` and the constraints and objectives of linear programs, as in flux
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

See its [fact sheet](../../../parts/spaces.md#statictime)
and its [API entry][skcomplex.core.components.spaces.StaticTime].

## BaseTime

`BaseTime` is the base of every time. A system that advances in a new way, beyond flowing, stepping, and solving
once, derives a time from it and gives the frames of a run.

```python
from skcomplex.core.components.spaces import BaseTime, ContinuousTime, DiscreteTime

issubclass(ContinuousTime, BaseTime), issubclass(DiscreteTime, BaseTime)
```

See its [fact sheet](../../../parts/spaces.md#basetime)
and its [API entry][skcomplex.core.components.spaces.BaseTime].
