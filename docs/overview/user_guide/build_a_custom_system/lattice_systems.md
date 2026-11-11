# Lattice systems

This page builds systems that live on a grid of cells, such as cellular automata, sandpiles, and spin systems. You
write the laws of one cell, and the library runs them on every site of the grid.

## The lattice

This section builds a lattice and reads the facts it derives. A
[`LatticeSpace`](../core_components/spaces.md#latticespace) is a box of the integer grid `Z^d` whose edges are
`'periodic'`, `'fixed'`, or `'reflecting'`. A periodic edge wraps around to the opposite side, a fixed edge reads a
constant value beyond the box, and a reflecting edge mirrors the cells inside. Here a periodic box has 4 rows and 3
columns.

```python
from skcomplex.core.components.spaces import LatticeSpace

space = LatticeSpace((4, 3)).prepare()
space.size_, space.shape_, space.periodic_, space.positions_[4].tolist()
```

The lattice has 12 sites in an array of shape `(4, 3)`, it wraps, and its fifth site, the second of the second row,
sits at the position `[1.0, 1.0]`.

A space is a component of its system and has the same main public methods. `prepare()` checks the arguments and derives
the facts, whose names end in `_`.

- `size_` is the number of sites, and `n_dim_` the dimension.
- `shape_` is the shape of the arrays of the sites, and `positions_` the position of every site.
- `periodic_` says whether the lattice wraps, and `default_visual_` is the visual of its groups.

Compiling a system prepares its spaces, so a space needs its own call to `prepare()` only when you read its facts
before that. A [`GroupElement`](../core_components/elements.md#groupelement) with `on='sites'` puts one member of an
element class on every site, and takes its initial values as arrays of the shape of the lattice. Every section below
builds one.

## Local rules

This section computes the next value of a cell from the values of its neighbours, the rule of a cellular automaton.
Conway's Game of Life is one. A dead cell with three live neighbours comes alive, and a live cell with two or three
live neighbours stays alive. A [`LocalRuleLaw`](../core_components/laws.md#localrulelaw) holds the state it changes,
the neighbourhood it reads, and the rule. Here the neighbourhood is the
[`BallNeighborhood`](../core_components/spaces.md#ballneighborhood) of the eight surrounding cells, and the rule is an
outer totalistic rule, which reads the cell and the sum of its neighbours. A glider of five live cells starts in the
corner of a periodic grid of 20 by 20.

```python
from typing import Annotated

import numpy as np
from sage.sets.finite_enumerated_set import FiniteEnumeratedSet

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.laws import LocalRuleLaw
from skcomplex.core.components.spaces import BallNeighborhood, LatticeSpace
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.specs import OuterTotalisticSpec, StateSpec


class LifeElement(Element):
    def __init__(self, label, *, alive: Annotated[float, StateSpec(domain=FiniteEnumeratedSet([0, 1]))] = 0):
        super().__init__(label)
        self.alive = alive

    def _get_laws(self, s):
        return [LocalRuleLaw(s.alive, BallNeighborhood(), OuterTotalisticSpec({(0, 3): 1, (1, 2): 1, (1, 3): 1}))]


glider = np.zeros((20, 20))
glider[1, 2] = glider[2, 3] = glider[3, 1] = glider[3, 2] = glider[3, 3] = 1
space = LatticeSpace((20, 20))
cells = CompositeSystem(elements=[GroupElement('cells', element=LifeElement, space=space, values={'alive': glider}, on='sites')])
trajectory = simulate(cells, until=4, dt=1)
np.array_equal(trajectory.get_lattice('cells.alive')[-1], np.roll(glider, (1, 1), axis=(0, 1)))
```

After four steps the configuration equals the glider moved one cell down and one cell right, so the block returns
`True`.

`LocalRuleLaw(targets, over, rule)` reads the values of the neighbourhood in a fixed order.

- `BallNeighborhood(radius, metric=...)`, the Chebyshev, Manhattan, or hexagonal ball, or
  [`LatticeNeighborhood(offsets)`](../core_components/spaces.md#latticeneighborhood) gives the offsets of the
  neighbours. `center` leaves the site out, puts it first, or keeps it `'inline'`, in its lexicographic place, which is
  the order of the rule numbers of Wolfram.
- A rule is a `TableSpec` of patterns, a `TotalisticSpec` or `OuterTotalisticSpec` rule, a rule code whose `table()`
  returns a table, or a function of NumPy arrays, one per neighbour. A function also receives the parameters it names
  and, for stochastic rules, uniform random numbers.
- Tables and totalistic rules run as lookups, fused into one compiled loop on the Numba backend.
- Rules update synchronously by default, every cell from the previous configuration. `schedule='random-sequential'`,
  `'random-site'`, `'checkerboard'`, or `'colored'` update cells one at a time, or one class of sites at a time.
- One unit of time is one application of every rule, so `dt` is an integer. `trajectory.get_lattice(name)` returns the
  states in the shape of the lattice.
- [`ArrayUpdateLaw(target, function)`](../core_components/laws.md#arrayupdatelaw) changes a whole lattice at once, such
  as dropping a grain on a random site.

## Cascades

This section drives a lattice slowly and lets it relax fast, in avalanches. Self-organised criticality is the state
that such systems reach, where avalanches of every size occur. In the sandpile of Bak, Tang, and Wiesenfeld, a grain
falls on a random site, and a site with four grains or more topples and gives one grain to each neighbour. The
catalog class `SandpileSystem` records the size of every avalanche.

```python
from skcomplex.core.engine.runtime import simulate
from skcomplex.systems.physics.criticality import SandpileSystem

sandpile = SandpileSystem(shape=(20, 20))
trajectory = simulate(sandpile, until=2000, dt=100, random_state=0)
max(event.size for event in trajectory.events)
```

Over 2000 steps of the drive on a grid of 20 by 20, the largest avalanche has the size 613.

A [`CascadeLaw(drive, relax, stable)`](../core_components/laws.md#cascadelaw) separates the slow drive from the fast
cascade. Every step applies the drive laws once, and the relax laws until `stable` holds. Each cascade is recorded
among the events of the [trajectory](../glossary.md#trajectory) with its size, duration, and area. `activity` marks the
sites that count as updates, such as the toppling sites of a sandpile.

## Monte Carlo

This section draws the configurations of a lattice at a temperature. In the Ising magnet each site holds a spin of
plus or minus one, and the energy falls when neighbouring spins align. Below the critical temperature, about 2.27 on the
square lattice, most spins align. A [`SampleLaw`](../core_components/laws.md#samplelaw) proposes changes of a state and
accepts them by their change of energy. Here the spins of a periodic grid of 32 by 32 start aligned and evolve at the
temperature 1.5.

```python
from typing import Annotated

from sage.all__sagemath_symbolics import ZZ

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.laws import SampleLaw
from skcomplex.core.components.laws.utils import neighbors
from skcomplex.core.components.spaces import BallNeighborhood, LatticeSpace
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.specs import MoveSpec, ObjectiveSpec, ParameterSpec, SamplerSpec, StateSpec

nearest = BallNeighborhood(metric='manhattan')


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


spins = CompositeSystem(elements=[GroupElement('spins', element=SpinElement, space=LatticeSpace((32, 32)), on='sites')])
abs(simulate(spins, until=200, dt=50, random_state=0).states[-1].mean()) > 0.9
```

After 200 units of time the mean spin stays above 0.9 in absolute value, so the magnet stays ordered.

`SampleLaw(objective, move, sampler)` has three parts.

- `ObjectiveSpec(expression, over)` is the local objective of a site, every term of the total objective that involves
  it. The law reads the values of the neighbours through `neighbors`.
- `MoveSpec(target, kind)` proposes changes, with the kinds `'flip'`, `'random-value'`, and `'rotate'`.
- `SamplerSpec(kind, intensity)` accepts them, with the kinds `'metropolis'`, `'glauber'`, and `'heat-bath'`.

The change of the objective is derived symbolically, and sweeps update the two sublattices in turn. The intensity is an
expression of parameters, so the `schedule` of `simulate` can change it during a run.

## Triangular lattices

This section gives every site six neighbours, as hexagonal cells and lattice gases need. A fire spreads from one site
to its six neighbours at every step, and the rule `spread` takes the largest value of the neighbourhood.

```python
from typing import Annotated

import numpy as np

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.laws import LocalRuleLaw
from skcomplex.core.components.spaces import BallNeighborhood, LatticeSpace
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.specs import StateSpec


def spread(*values):
    return np.max(values, axis=0)


class BurnElement(Element):
    def __init__(self, label, *, fire: Annotated[float, StateSpec()] = 0.0):
        super().__init__(label)
        self.fire = fire

    def _get_laws(self, s):
        return [LocalRuleLaw(s.fire, BallNeighborhood(metric='hexagonal', center='first'), spread)]


fire = np.zeros((40, 40))
fire[20, 20] = 1
space = LatticeSpace((40, 40), geometry='triangular')
forest = CompositeSystem(elements=[GroupElement('forest', element=BurnElement, space=space, values={'fire': fire}, on='sites')])
burning = simulate(forest, until=3, dt=1).get_lattice('forest.fire')
[int(item.sum()) for item in burning]
```

The burning sites number 1, 7, 19, and 37 after zero to three steps, the hexagonal numbers `3k(k + 1) + 1`.

- `LatticeSpace(shape, geometry='triangular')` keeps integer coordinates on a skewed grid, so every boundary works as
  on a square lattice.
- It places site `(i, j)` at `(j + i / 2, i sqrt(3) / 2)` in the plane, which the fitted `positions_` give.
- `BallNeighborhood(metric='hexagonal')` holds the six neighbours counterclockwise from 0 degrees, so direction `k + 3`
  modulo 6 is opposite to direction `k`, as lattice gases need.
- The cells of the sites are the hexagons of a hexagonal tiling.

## Lattices on graphs

This section uses the nodes of a network as the sites. A `GroupElement` on `'sites'` also takes a
[`NetworkSpace`](../core_components/spaces.md#networkspace), with one site per node in the order of `graph.nodes`, and
initial values are arrays in node order or mappings from node to value. The
[`AdjacencyNeighborhood`](../core_components/spaces.md#adjacencyneighborhood) of a node holds its neighbours along its
edges. Here an Ising magnet lives on a random regular graph of 200 nodes, each with four neighbours.

```python
from typing import Annotated

import networkx as nx
from sage.all__sagemath_symbolics import ZZ

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.laws import SampleLaw
from skcomplex.core.components.laws.utils import neighbors
from skcomplex.core.components.spaces import AdjacencyNeighborhood, NetworkSpace
from skcomplex.core.components.systems import CompositeSystem
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
spins = CompositeSystem(elements=[GroupElement('spins', element=GraphSpinElement, space=NetworkSpace(graph), on='sites')])
abs(simulate(spins, until=200, dt=50, random_state=0).states[-1].mean()) > 0.9
```

At the temperature 1.0 the mean spin stays above 0.9 in absolute value after 200 units of time, so the magnet on the
graph stays ordered too.

- `AdjacencyNeighborhood(center=False)` pads the neighbours of every node to the largest degree with the `fill` value
  of the network.
- Local rules, drives, relaxations, Monte Carlo laws, and extremal laws run on graphs as on lattices.
- Lookup rules on graphs take the totalistic forms. A function rule that names the keyword `degree` receives the
  number of real neighbours of every site.
- The padded table grows with the largest degree. Graphs with hubs of very high degree, beyond scale-free graphs of
  about 10^5 nodes, therefore use much memory.

## Label copies and regions

This section names the recipe for sites that copy labels, so that the sites of one label form a region whose total size
matters. The cellular Potts model of biology works this way, with one label per biological cell.
`RegionSamplingInteraction` of `skcomplex.systems.biology.physiology` subclasses
[`SamplingInteraction`](../core_components/interactions.md#samplinginteraction) with the objective
`sum_<i,j> w_ij contact(l_i, l_j) [l_i != l_j] + sum_l a_l (V_l - A_l)^2 + b_l (S_l - B_l)^2` over integer labels. It
keeps the volume `V_l` and the surface `S_l` of every region up to date on every accepted copy. The catalog class
`CellularPottsSystem` of the same module sorts cells of two kinds by differential adhesion on square, triangular, and
disordered lattices.

## Reads that follow the state, and series in the cells

This section builds rules whose reads depend on the configuration and states whose values are formal power series. A
[`StateTranslationNeighborhood`](../core_components/spaces.md#statetranslationneighborhood) chooses the cell a rule
reads from the state itself, such as the map `τ(x)(g) = x(g · x(g))` whose alphabet is the group itself. A cell of a
[`CayleySpace`](../core_components/spaces.md#cayleyspace) holds an element of a group as the number of its site, and the
rule reads the cells at `g x(g) s` for the words `s` of the neighbourhood. Here a
[`LocalRuleInteraction`](../core_components/interactions.md#localruleinteraction) applies the map once on the
symmetric group of three letters, whose six elements are the sites.

```python
from typing import Annotated

import numpy as np
from sage.all__sagemath_gap import SymmetricGroup
from sage.all__sagemath_symbolics import ZZ

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.interactions import LocalRuleInteraction
from skcomplex.core.components.spaces import (
    CayleySpace,
    LatticeNeighborhood,
    LatticeSpace,
    StateTranslationNeighborhood,
)
from skcomplex.core.components.systems import CompositeSystem
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
shift = CompositeSystem(
    elements=[
        GroupElement(
            'cells', element=GroupValuedElement, space=group, on='sites', values={'value': [1, 2, 3, 4, 5, 0]}
        ),
    ],
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
simulate(shift, until=1).get_values('cells.value')[-1].tolist()
```

The map sends the values `[1, 2, 3, 4, 5, 0]` to `[2.0, 1.0, 0.0, 3.0, 3.0, 0.0]`.

- The first state a rule reads holds the group elements. On a finite group every product is a site. On a ball a
  product outside it, or reached through a site outside it, reads the fill value of the space, as does a value that
  names no site, so a run truncates a map of unbounded memory on an infinite group to a stated boundary.
- The products of every pair of sites are tabulated once, so balls of a few thousand sites fit. Translated reads run
  on the synchronous and random schedules, and refuse colorings and `degree`, which need a fixed table.

A `PowerSeriesSpec` as the domain of a state gives it values in a ring of formal power series, truncated at an order.
Here the automaton `x(n) - t x(n + 1)` over `K[[t]]` runs three steps on a line of six cells whose fourth cell holds the
series 1.

```python continuation
start = np.zeros((6, 4))
start[3, 0] = 1
series = CompositeSystem(
    elements=[GroupElement('cells', element=SeriesElement, space=LatticeSpace(6), values={'value': start}, on='sites')],
    interactions=[
        LocalRuleInteraction(
            'difference',
            target='cells',
            state='value',
            rule=lambda here, right: here - RING.multiply(RING.get_coefficients_of('t'), right),
            neighborhood=LatticeNeighborhood(((0,), (1,))),
        ),
    ],
)
final = simulate(series, until=3).get_values('cells.value')[-1]
RING.get_series_of(final[2])
```

The third cell holds `-3*t + O(t^4)`, the coefficient of one shift in `(1 - t S)^3`, where `S` reads the cell to the
right.

- A state whose domain is a `PowerSeriesSpec` holds `order` coefficients per cell. A function rule receives one row of
  coefficients per cell for every neighbour and returns rows. The spec multiplies series by the truncated Cauchy
  product, reduces coefficients over a prime field `'GF(p)'`, and converts them to the power series ring of Sage with
  `get_series_of` and `get_coefficients_of`.
- Function rules read and set any state of several components this way. Tables and totalistic rules stay scalar.

## Neighbour sums and colored schedules

This section names two tools of the graph example that work on every lattice.

- `neighbors(expression, reduce='sum')` sums an expression of a site and one neighbour, read through `s.other`, over
  the real neighbours of the site. Local energies are therefore exact on sites of any degree and beside fixed edges.
- `schedule='colored'` on local rules and samplers updates the classes of a proper coloring of the neighbour graph in
  turn. A proper coloring gives neighbours different colors, so the sites of one class update independently. NetworkX
  computes the coloring once per law. The checkerboard schedule is the case of two colors on square lattices.

## Extremal updates

This section acts at each step on the site with the worst or best value. In the Bak-Sneppen model of evolution, the
least fit species of a ring is replaced together with its two neighbours, and each gets a new random fitness. An
[`ExtremalLaw`](../core_components/laws.md#extremallaw) finds the site and applies the rule `renew` to it and its
neighbours, here 100 times per step for 100 steps.

```python
from typing import Annotated

import numpy as np

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.laws import ExtremalLaw
from skcomplex.core.components.spaces import LatticeNeighborhood, LatticeSpace
from skcomplex.core.components.systems import CompositeSystem
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
members = GroupElement('ring', element=SpeciesElement, space=space, on='sites', values={'fitness': fitness})
ring = CompositeSystem(elements=[members])
trajectory = simulate(ring, until=100, dt=10, random_state=0)
len(trajectory.events), trajectory.events[0].label
```

The run records 10000 events, one per application, each with the label `'extinction'`.

`ExtremalLaw(targets, over, rule, key=None, extremum='min', repeat=1)` finds the site with the smallest or largest
value of a key state. It breaks ties with the random stream of the run, and applies
`rule(*values, random, **parameters)` to the site and its real neighbours, `repeat` times per step.

- Every application is recorded among the events of the trajectory as an `ExtremalRecord` with its time, site, and
  extremal value.
- On the Numba backend, the applications of a step run in one compiled loop when Numba compiles the rule. The rule
  then takes its arguments by position or keyword, with no keyword-only arguments.
- Both backends give the same events.

## Viewing lattices and ready-made lattice systems

This section shows lattices in Rerun and finds the lattice systems of the catalog. Square lattice groups stream to
Rerun as images colored by value. Triangular lattices and graph lattices stream as sites with `SiteVisual`, hexagonal
cells at the site positions or nodes colored by state with their edges logged once.

`all_systems` of `skcomplex.systems` lists the lattice systems of the catalog by their topic `lattice`, optionally of
one field or subfield.

```python
from skcomplex.systems import all_systems

[name for name, _ in all_systems(field='physics/criticality', topics='lattice')]
```

In `physics/criticality` the list holds `FixedEnergySandpileSystem`, `ForestFireSystem`, `HexagonalForestFireSystem`,
`MannaSandpileSystem`, `SandpileSystem`, and `ZhangSandpileSystem`.

Other lattice systems, by subfield of `skcomplex.systems`, include these.

- `computer_science.automata`: `ElementaryAutomatonSystem`, `GameOfLifeSystem`, and `LangtonsAntSystem`.
- `physics.statistical_mechanics`: `IsingSystem`, `GraphIsingSystem`, `PottsSystem`, and `XYSystem`.
- `physics.seismology`: `EarthquakesSystem`, and `physics.fluids`: `FHPLatticeGasSystem`.
- `social.opinion_dynamics`: `MajorityRuleSystem`, `engineering.traffic`: `NagelSchreckenbergSystem`,
  `chemistry.kinetics`: `HodgepodgeSystem`, and `biology.evolution`: `BakSneppenSystem`.
