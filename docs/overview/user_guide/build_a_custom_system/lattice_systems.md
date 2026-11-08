# Lattice systems

Use a lattice for systems that live on a grid of cells, such as cellular automata, sandpiles, and spin systems. You
write the laws for one cell, and the library runs them on every site.

```python
from skcomplex.core.components.spaces import LatticeSpace

space = LatticeSpace((4, 3)).prepare()
space.size_, space.shape_, space.periodic_, space.positions_[4].tolist()
```

- `LatticeSpace(shape, boundary=...)` is a box of `Z^d` with `'periodic'`, `'fixed'`, or `'reflecting'` edges.
- `LatticeElement(label, element=..., space=space, values={...})` puts one member of an element type on every site,
  with values given as arrays of the lattice shape.

A space is a part of its system and speaks the same verbs. `space.prepare()` checks its arguments and derives its
facts, whose names end in `_`:

- `size_` is the number of sites, and `n_dim_` the dimension.
- `shape_` is the shape of the arrays of its sites, and `positions_` the position of every site.
- `periodic_` says whether it wraps, and `default_visual_` is the visual of its groups.

Compiling a system prepares its spaces, so you call `prepare()` on a space only when you read its facts before that.

## Local rules

Use a local rule when a cell's next value depends on the values of its neighbours. Conway's Game of Life is one, and
a glider moves one cell diagonally every four steps:

```python
from typing import Annotated

import numpy as np
from sage.sets.finite_enumerated_set import FiniteEnumeratedSet

from skcomplex.core.components.elements import Element, LatticeElement
from skcomplex.core.components.laws import LocalRuleLaw
from skcomplex.core.components.spaces import LatticeSpace, MooreNeighborhood
from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.specs import OuterTotalisticSpec, StateSpec


class LifeElement(Element):
    def __init__(self, label, *, alive: Annotated[float, StateSpec(domain=FiniteEnumeratedSet([0, 1]))] = 0):
        super().__init__(label)
        self.alive = alive

    def _get_laws(self, s):
        return [LocalRuleLaw(s.alive, MooreNeighborhood(), OuterTotalisticSpec({(0, 3): 1, (1, 2): 1, (1, 3): 1}))]


glider = np.zeros((20, 20))
glider[1, 2] = glider[2, 3] = glider[3, 1] = glider[3, 2] = glider[3, 3] = 1
space = LatticeSpace((20, 20))
cells = System(elements=[LatticeElement('cells', element=LifeElement, space=space, values={'alive': glider})])
trajectory = simulate(cells, until=4, dt=1)
np.array_equal(trajectory.lattice('cells.alive')[-1], np.roll(glider, (1, 1), axis=(0, 1)))
```

`LocalRuleLaw(targets, over, rule)` computes a cell's next value from the ordered values of its neighbourhood.

- `MooreNeighborhood`, `VonNeumannNeighborhood`, `BoxNeighborhood`, or `LatticeNeighborhood(offsets)` give the
  offsets of the neighbours.
- A rule is a `TableSpec` of patterns, a `TotalisticSpec` or `OuterTotalisticSpec` rule, a rule code whose `table()`
  returns a table, or a function of NumPy arrays, one per neighbour. A function also receives the parameters it names
  and, for stochastic rules, uniform random numbers.
- Tables and totalistic rules run as lookups, fused into one compiled loop on the Numba backend.
- Rules update synchronously by default, every cell from the previous configuration. `schedule='random-sequential'`,
  `'random-site'`, `'checkerboard'`, or `'colored'` update cells one at a time, or one class of sites at a time.
- One unit of time is one application of every rule, so `dt` is an integer. `trajectory.lattice(name)` returns the
  states in lattice shape.
- `DriveLaw(target, function)` changes a whole lattice at once, such as dropping a grain on a random site.

## Cascades

Use a cascade when a slow drive builds up stress and a fast relaxation releases it in avalanches, as in
self-organised criticality. The ready-made `SandpileSystem` records the size of every avalanche:

```python
from skcomplex.core.engine.runtime import simulate
from skcomplex.systems.physics.criticality import SandpileSystem

sandpile = SandpileSystem(shape=(20, 20))
trajectory = simulate(sandpile, until=2000, dt=100, random_state=0)
max(event.size for event in trajectory.events)
```

`CascadeLaw(drive, relax, stable)` separates the slow drive from the fast cascade. Every step applies the drive laws
once, and the relax laws until `stable` holds. Each cascade is recorded among the trajectory's events with its size,
duration, and area. `activity` marks the sites that count as updates, such as the toppling sites of a sandpile.

## Monte Carlo

Use a sampling law to draw the configurations of a lattice at a temperature, such as an Ising magnet, whose spins
align below the critical temperature:

```python
from typing import Annotated

from sage.all__sagemath_symbolics import ZZ

from skcomplex.core.components.elements import Element, LatticeElement
from skcomplex.core.components.laws import SampleLaw
from skcomplex.core.components.laws.utils import neighbors
from skcomplex.core.components.spaces import LatticeSpace, VonNeumannNeighborhood
from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.specs import MoveSpec, ObjectiveSpec, ParameterSpec, SamplerSpec, StateSpec

nearest = VonNeumannNeighborhood()


class SpinElement(Element):
    def __init__(
        self,
        label,
        *,
        spin: Annotated[float, StateSpec(domain=ZZ)] = 1.0,
        J: Annotated[float, ParameterSpec()] = 1.0,
        temperature: Annotated[float, ParameterSpec()] = 1.5,
    ):
        super().__init__(label)
        self.spin = spin
        self.J = J
        self.temperature = temperature

    def _get_laws(self, s):
        energy = -s.J * s.spin * sum(neighbors(s.spin, nearest, reduce='each'))
        sampler = SamplerSpec('metropolis', 1 / s.temperature)
        return [SampleLaw(ObjectiveSpec(energy, nearest), MoveSpec(s.spin, 'flip'), sampler)]


spins = System(elements=[LatticeElement('spins', element=SpinElement, space=LatticeSpace((32, 32)))])
abs(simulate(spins, until=200, dt=50, random_state=0).states[-1].mean()) > 0.9
```

A `SampleLaw(objective, move, sampler)` samples states.

- `ObjectiveSpec(expression, over)` is the local objective of a site, every term of the total objective that involves
  it. Neighbour values are read through `neighbors`.
- `MoveSpec(target, kind)` proposes changes, with the kinds `'flip'`, `'random-value'`, and `'rotate'`.
- `SamplerSpec('metropolis' | 'glauber' | 'heat-bath', intensity)` accepts them.

The change of the objective is derived symbolically, and sweeps update the two sublattices in turn. The intensity is
an expression of parameters, so the `schedule` of `simulate` can change it during a run.

## Triangular lattices

Use a triangular lattice when every site needs six neighbours, as hexagonal cells and lattice gases do. A fire spreads
from one site to its six neighbours at every step:

```python
from typing import Annotated

import numpy as np

from skcomplex.core.components.elements import Element, LatticeElement
from skcomplex.core.components.laws import LocalRuleLaw
from skcomplex.core.components.spaces import LatticeSpace, TriangularNeighborhood
from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.specs import StateSpec


def spread(*values):
    return np.max(values, axis=0)


class BurnElement(Element):
    def __init__(self, label, *, fire: Annotated[float, StateSpec()] = 0.0):
        super().__init__(label)
        self.fire = fire

    def _get_laws(self, s):
        return [LocalRuleLaw(s.fire, TriangularNeighborhood(center=True), spread)]


fire = np.zeros((40, 40))
fire[20, 20] = 1
space = LatticeSpace((40, 40), geometry='triangular')
forest = System(elements=[LatticeElement('forest', element=BurnElement, space=space, values={'fire': fire})])
burning = simulate(forest, until=3, dt=1).lattice('forest.fire')
[int(item.sum()) for item in burning]
```

- `LatticeSpace(shape, geometry='triangular')` keeps integer coordinates on a skewed grid, so every boundary works as
  on a square lattice.
- It places site `(i, j)` at `(j + i / 2, i sqrt(3) / 2)` in the plane, given by the fitted `positions_`.
- `TriangularNeighborhood()` holds the six neighbours counterclockwise from 0 degrees, so direction `k + 3` modulo 6 is
  opposite to direction `k`, as lattice gases need.
- The cells of the sites are the hexagons of a hexagonal tiling.

## Lattices on graphs

Use a graph as the lattice when the sites are the nodes of a network, such as an Ising magnet on a random regular
graph:

```python
from typing import Annotated

import networkx as nx
from sage.all__sagemath_symbolics import ZZ

from skcomplex.core.components.elements import Element, LatticeElement
from skcomplex.core.components.laws import SampleLaw
from skcomplex.core.components.laws.utils import neighbors
from skcomplex.core.components.spaces import AdjacencyNeighborhood, NetworkSpace
from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.specs import MoveSpec, ObjectiveSpec, ParameterSpec, SamplerSpec, StateSpec


class GraphSpinElement(Element):
    def __init__(
        self,
        label,
        *,
        spin: Annotated[float, StateSpec(domain=ZZ)] = 1.0,
        J: Annotated[float, ParameterSpec()] = 1.0,
        temperature: Annotated[float, ParameterSpec()] = 1.0,
    ):
        super().__init__(label)
        self.spin = spin
        self.J = J
        self.temperature = temperature

    def _get_laws(self, s):
        energy = -s.J * s.spin * neighbors(s.other.spin, reduce='sum')
        sampler = SamplerSpec('metropolis', 1 / s.temperature, schedule='colored')
        return [SampleLaw(ObjectiveSpec(energy, AdjacencyNeighborhood()), MoveSpec(s.spin, 'flip'), sampler)]


graph = nx.random_regular_graph(4, 200, seed=0)
spins = System(elements=[LatticeElement('spins', element=GraphSpinElement, space=NetworkSpace(graph))])
abs(simulate(spins, until=200, dt=50, random_state=0).states[-1].mean()) > 0.9
```

`LatticeElement` also takes a `NetworkSpace`, with one site per node in the order of `graph.nodes`. Initial values are
arrays in node order, or mappings from node to value.

- `AdjacencyNeighborhood(center=False)` is the neighbourhood of a node, its neighbours along its edges, padded to the
  largest degree with the network's `fill` value.
- Local rules, drives, relaxations, Monte Carlo laws, and extremal laws run on graphs as on lattices.
- Lookup rules on graphs take the totalistic forms. A function rule that names the keyword `degree` receives the
  number of real neighbours of every site.
- The padded table grows with the largest degree. Graphs with hubs of very high degree, beyond scale-free graphs of
  about 10^5 nodes, therefore use much memory.

## Label copies and regions

Use label copies when the sites of one label form a region whose total size matters, such as the cells of the cellular
Potts model. The example grows one region of 9 sites in a medium to a volume of 30:

```python
from typing import Annotated

import numpy as np

from skcomplex.core.components.elements import Element, LatticeElement
from skcomplex.core.components.interactions import SamplingInteraction
from skcomplex.core.components.spaces import DiscreteTime, LatticeSpace
from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.specs import RegionScoreSpec, StateSpec


class SiteElement(Element):
    def __init__(self, label, region: Annotated[float, StateSpec()] = 0.0):
        super().__init__(label)
        self.region = region


def contact(first, second):
    return np.ones(np.broadcast(first, second).shape)


def volume(labels):
    return np.where(labels == 0, 0.0, 1.0), np.full(labels.shape, 30.0)


regions = np.zeros((12, 12))
regions[4:7, 4:7] = 1
copies = SamplingInteraction(
    'copies', target='sites', state='region', objective=RegionScoreSpec(contact, volume), move='copy'
)
sites = LatticeElement('sites', element=SiteElement, space=LatticeSpace((12, 12)), values={'region': regions})
system = System(elements=[sites], interactions=[copies], time=DiscreteTime())
labels = simulate(system, until=200, random_state=0).values('sites.region')
abs((labels[50:].reshape(-1, 144) == 1).sum(axis=1).mean() - 30) < 2
```

`SamplingInteraction` takes a `RegionScoreSpec(contact, volume=None, surface=None, hits=1)` as its objective, with the
move `'copy'`. The state holds integer labels from 0, and the objective is
`sum_<i,j> w_ij contact(l_i, l_j) [l_i != l_j] + sum_l a_l (V_l - A_l)^2 + b_l (S_l - B_l)^2`.

- `contact(first, second)` is the energy per unit contact between two labels, evaluated once per sweep on all pairs
  of labels.
- `volume(labels)` and `surface(labels)` return the strength and the target of the constraint of every label. A
  strength of 0 leaves a label free, such as the medium.
- The volume of a region sums the volumes of its sites, and its surface the weights of its contacts with other labels.
  The sampler keeps both per label and updates them on every accepted copy, so a copy costs the size of the
  neighbourhood, whatever the number of regions.
- Every sweep proposes `hits` copies per site, each of the label of a random neighbour into a random site, accepted by
  the Metropolis or Glauber rule.
- The neighbourhood is that of the interaction, or the default of the space: the Moore neighbourhood of a square
  lattice, the six neighbours of a triangular one, or the edges of a network. On a `NetworkSpace` the edge attribute
  `'weight'` weighs a contact and the node attribute `'volume'` the volume of a site, so a Voronoi tessellation of
  irregular points is a disordered lattice.

The ready-made `CellularPottsSystem` of `skcomplex.systems.biology.physiology` sorts cells of two kinds by differential
adhesion on square, triangular, and disordered lattices.

## Reads that follow the state, and series in the cells

Use a `StateTranslationNeighborhood` when the cell a rule reads depends on the configuration, such as the map
`τ(x)(g) = x(g · x(g))` whose alphabet is the group itself. A cell of a `CayleySpace` holds a group element as the
number of its site, and the rule reads the cells at `g x(g) s` for the words `s` of the neighbourhood. Use a
`PowerSeriesSpec` as the domain of a state whose values are formal power series, such as the automaton
`x(n) - t x(n + 1)` over `K[[t]]`:

```python
from typing import Annotated

import numpy as np
from sage.all__sagemath_gap import SymmetricGroup
from sage.all__sagemath_symbolics import ZZ

from skcomplex.core.components.elements import Element, LatticeElement
from skcomplex.core.components.interactions import LocalRuleInteraction
from skcomplex.core.components.spaces import (
    CayleySpace,
    LatticeNeighborhood,
    LatticeSpace,
    StateTranslationNeighborhood,
)
from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.specs import PowerSeriesSpec, StateSpec

RING = PowerSeriesSpec(order=4, field='QQ')


class GroupValuedElement(Element):
    def __init__(self, label, *, value: Annotated[float, StateSpec(domain=ZZ)] = 0):
        super().__init__(label)
        self.value = value


class SeriesElement(Element):
    def __init__(self, label, *, value: Annotated[float, StateSpec(domain=RING)] = 0):
        super().__init__(label)
        self.value = value


group = CayleySpace(SymmetricGroup(3))
shift = System(
    elements=[LatticeElement('cells', element=GroupValuedElement, space=group, values={'value': [1, 2, 3, 4, 5, 0]})],
    interactions=[
        LocalRuleInteraction(
            'translation',
            target='cells',
            state='value',
            rule=lambda value: value,
            neighborhood=StateTranslationNeighborhood(),
        ),
    ],
)
simulate(shift, until=1).values('cells.value')[-1].tolist()

start = np.zeros((6, 4))
start[3, 0] = 1
series = System(
    elements=[LatticeElement('cells', element=SeriesElement, space=LatticeSpace(6), values={'value': start})],
    interactions=[
        LocalRuleInteraction(
            'difference',
            target='cells',
            state='value',
            rule=lambda here, right: here - RING.multiply(RING.coefficients_of('t'), right),
            neighborhood=LatticeNeighborhood(((0,), (1,))),
        ),
    ],
)
final = simulate(series, until=3).values('cells.value')[-1]
RING.series_of(final[2])
```

- The first state a rule reads holds the group elements. On a finite group every product is a site. On a ball a
  product outside it, or reached through a site outside it, reads the fill value of the space, as does a value that
  names no site, so a run truncates a map of unbounded memory on an infinite group to a stated boundary.
- The products of every pair of sites are tabulated once, so balls of a few thousand sites fit. Translated reads run
  on the synchronous and random schedules, and refuse colorings and `degree`, which need a fixed table.
- A state whose domain is a `PowerSeriesSpec` holds `order` coefficients per cell. A function rule receives one row of
  coefficients per cell for every neighbour and returns rows, and the spec multiplies series by the truncated Cauchy
  product, reduces coefficients over a prime field `'GF(p)'`, and converts them to Sage's power series ring with
  `series_of` and `coefficients_of`.
- Function rules read and set any state of several components this way. Tables and totalistic rules stay scalar.

## Neighbour sums and colored schedules

The graph example above uses two tools that work on every lattice.

- `neighbors(expression, reduce='sum')` sums an expression of a site and one neighbour, read through `s.other`, over
  the real neighbours of the site. Local energies are therefore exact on sites of any degree and outside fixed edges.
- `schedule='colored'` on local rules and samplers updates the classes of a proper coloring of the neighbour graph in
  turn. NetworkX computes the coloring once per law. The checkerboard schedule stays the two-color case of square
  lattices.

## Extremal updates

Use an extremal update when each step acts on the worst or best site, such as the least fit species of the Bak-Sneppen
system of evolution, which is replaced together with its neighbours:

```python
from typing import Annotated

import numpy as np

from skcomplex.core.components.elements import Element, LatticeElement
from skcomplex.core.components.laws import ExtremalLaw
from skcomplex.core.components.spaces import LatticeNeighborhood, LatticeSpace
from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.specs import StateSpec


def renew(fitness, random):
    return random


class SpeciesElement(Element):
    def __init__(self, label, *, fitness: Annotated[float, StateSpec()] = 0.5):
        super().__init__(label)
        self.fitness = fitness

    def _get_laws(self, s):
        return [ExtremalLaw(s.fitness, LatticeNeighborhood(((-1,), (1,))), renew, repeat=100, label='extinction')]


fitness = np.random.default_rng(0).random(100)
space = LatticeSpace(100)
ring = System(elements=[LatticeElement('ring', element=SpeciesElement, space=space, values={'fitness': fitness})])
trajectory = simulate(ring, until=100, dt=10, random_state=0)
len(trajectory.events), trajectory.events[0].label
```

`ExtremalLaw(targets, over, rule, key=None, extremum='min', repeat=1)` finds the site with the smallest or largest
value of a key state. It breaks ties with the random stream of the run, and applies
`rule(*values, random, **parameters)` to the site and its real neighbours, `repeat` times per step.

- Every application is recorded among the trajectory's events as an `ExtremalRecord` with its time, site, and
  extremal value.
- On the Numba backend, the applications of a step run in one compiled loop when Numba compiles the rule. The rule
  then takes its arguments by position or keyword, with no keyword-only arguments.
- Both backends give the same events.

## Viewing lattices and ready-made lattice systems

Square lattice groups stream to Rerun as images colored by value. Triangular lattices and graph lattices stream as
sites with `SiteVisual`: hexagonal cells at the site positions, or nodes colored by state with their edges logged once.

`all_systems` of `skcomplex.systems` lists the ready-made lattice systems by their topic `lattice`, optionally of one
field or subfield:

```python
from skcomplex.systems import all_systems

[name for name, _ in all_systems(field='physics/criticality', topics='lattice')]
```

Among them, by subfield of `skcomplex.systems`:

- `computer_science.automata`: `ElementaryAutomatonSystem`, `GameOfLifeSystem`, and `LangtonsAntSystem`.
- `physics.statistical_mechanics`: `IsingSystem`, `GraphIsingSystem`, `PottsSystem`, and `XYSystem`.
- `physics.criticality`: `SandpileSystem`, `ForestFireSystem`, and `HexagonalForestFireSystem`.
- `physics.seismology`: `EarthquakesSystem`, and `physics.fluids`: `FHPLatticeGasSystem`.
- `social.opinion_dynamics`: `MajorityRuleSystem`, `engineering.traffic`: `NagelSchreckenbergSystem`,
  `chemistry.kinetics`: `HodgepodgeSystem`, and `biology.evolution`: `BakSneppenSystem`.
