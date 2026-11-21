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
before that. An [`Element`](../core_components/elements.md#element) with `space=` puts one member on every site, and
takes its initial values as arrays of the shape of the lattice. Every section below
builds one.

## Local rules

This section computes the next value of a cell from the values of its neighbours, the rule of a cellular automaton.
Conway's Game of Life is one. A dead cell with three live neighbours comes alive, and a live cell with two or three
live neighbours stays alive. An [`UpdateLaw`](../core_components/laws.md#updatelaw) with a `neighborhood` is such a
rule: it holds the state it changes, the rule, and the neighbours the rule reads. Here the neighbourhood is the
[`RadiusNeighborhood`](../core_components/spaces.md#radiusneighborhood) of the eight surrounding cells, and the rule is an
outer totalistic rule, which reads the cell and the sum of its neighbours. An interaction sets the law in `_prepare`,
and a glider of five live cells starts in the corner of a periodic grid of 20 by 20.

```python
from types import SimpleNamespace

import numpy as np

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import OuterTotalisticSpec, UpdateLaw
from skcomplex.core.components.spaces import LatticeSpace, RadiusNeighborhood
from skcomplex.core.types import Role
from skcomplex.core.systems import System

MOORE = RadiusNeighborhood(1, metric='chebyshev')
LIFE = OuterTotalisticSpec({(0, 3): 1, (1, 2): 1, (1, 3): 1})


class LifeInteraction(Interaction):
    def __init__(self, label, *, target: Role = 'cells'):
        super().__init__(label)
        self.target = target

    def _prepare(self, context):
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            self.laws_ = [UpdateLaw(context.target.alive, LIFE, neighborhood=MOORE)]


glider = np.zeros((20, 20))
glider[1, 2] = glider[2, 3] = glider[3, 1] = glider[3, 2] = glider[3, 3] = 1
cells = Element('cells', values={'alive': glider.ravel()}, space=LatticeSpace((20, 20)))
system = System(elements=[cells], interactions=[LifeInteraction('life')]).prepare()
trajectory = system.simulate(until=4, dt=1)
np.array_equal(trajectory.get_lattice('cells.alive')[-1], np.roll(glider, (1, 1), axis=(0, 1)))
```

After four steps the configuration equals the glider moved one cell down and one cell right, so the block returns
`True`. The equations name the rule by its interaction and the neighbours it reads.

```python continuation
system.equations_
```

They are `{'cells.alive[i]': "life(neighbors(cells.alive, reduce='each'))"}`.

`UpdateLaw(target, rule, neighborhood=...)` reads the values of the neighbourhood in a fixed order.

- `RadiusNeighborhood(radius, metric=...)`, the Euclidean, Chebyshev, Manhattan, or hexagonal ball in the dimensions
  of the lattice, or [`LatticeNeighborhood(offsets)`](../core_components/spaces.md#latticeneighborhood) gives the
  offsets of the neighbours. `center=True` puts the site first. A neighbourhood in an order of its own, such as the
  left neighbour, the cell, and the right neighbour of the rule numbers of Wolfram, lists its offsets in a
  `LatticeNeighborhood`.
- A rule is a `TableSpec` of patterns, a `TotalisticSpec` or `OuterTotalisticSpec` rule, a rule code whose
  `get_table()` returns a table, or a function of NumPy arrays, one per neighbour. A function also receives the
  parameters of the cells it names, `random`, one uniform number per cell, when it names it, and `degree`, the number
  of neighbours of each cell, when it names it.
- A tuple of states as the target, such as `(s.target.color, s.target.ant)` for Langton's ant, sets them together from
  one function that returns one array per state, and `reads=` names other states whose neighbours the function reads.
- Tables and totalistic rules run as lookups, fused into one compiled loop on the Numba backend, which a system picks
  for lattices of 1000 cells or more.
- Rules update synchronously by default, every cell from the previous configuration. `schedule='random-sequential'`,
  `'random-site'`, `'checkerboard'`, or `'colored'` update cells one at a time, or one class of sites at a time.
- Several rules of one state run one after the other in every step, such as the acceleration and the movement of the
  cars of `NagelSchreckenbergSystem`.
- One unit of time is one application of every rule, so `dt` is an integer. `trajectory.get_lattice(name)` returns the
  states in the shape of the lattice.
- A rule of the whole system, `UpdateLaw(s, rule)` with `def rule(system): ...`, runs once per step and changes a
  whole lattice at once, such as dropping a grain on a random site.

## Cascades

This section drives a lattice slowly and lets it relax fast, in avalanches. Self-organised criticality is the state
that such systems reach, where avalanches of every size occur. In the sandpile of Bak, Tang, and Wiesenfeld, a grain
falls on a random site, and a site with four grains or more topples and gives one grain to each neighbour. The
catalog class `SandpileSystem` records the size of every avalanche.

```python
from skcomplex.systems.physics.criticality import SandpileSystem

sandpile = SandpileSystem(shape=(20, 20))
trajectory = sandpile.prepare().simulate(until=2000, dt=100, random_state=0)
max(event.size for event in trajectory.events)
```

Over 2000 steps of the drive on a grid of 20 by 20, the largest avalanche has the size 613.

An `UpdateLaw` whose value is [`relax(state, rules, stable, drive=...)`](../core_components/laws.md#relax) separates
the slow drive from the fast cascade. Every step applies the drive laws once, and the rules until `stable` holds. Each cascade is recorded
among the events of the [trajectory](../glossary.md#trajectory) with its size, duration, and area. `activity` marks the
sites that count as updates, such as the toppling sites of a sandpile.

## Monte Carlo

This section draws the configurations of a lattice at a temperature. In the Ising magnet each site holds a spin of
plus or minus one, and the energy falls when neighbouring spins align. Below the critical temperature, about 2.27 on the
square lattice, most spins align. An interaction gives the energy of every site over the temperature as an
[`ObjectiveLaw`](../core_components/laws.md#objectivelaw), and a
[`MetropolisSolver`](../core_components/laws.md#objectivelaw) proposes changes of a state and accepts them by
their change of energy. Here the spins of a periodic grid of 32 by 32 start aligned and evolve at the temperature 1.5.

```python
import numpy as np

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import ObjectiveLaw
from skcomplex.core.components.laws.utils import neighbors
from skcomplex.core.components.spaces import LatticeNeighborhood, LatticeSpace
from skcomplex.core.engine.solvers import MetropolisSolver
from skcomplex.core.systems import System

nearest = LatticeNeighborhood(((-1, 0), (0, -1), (0, 1), (1, 0)))


def ising(s):
    spin = s.target.spin
    energy = -s.target.J * spin * sum(neighbors(spin, nearest, reduce='each'))
    return [ObjectiveLaw(energy / s.target.temperature, spin)]


spins = Element('spins', values={'spin': np.ones(1024), 'J': 1.0, 'temperature': 1.5}, space=LatticeSpace((32, 32)))
magnet = Interaction('ising', roles={'target': 'spins'}, laws=ising)
solver = MetropolisSolver(moves={'spins.spin': 'flip'}, neighborhood=nearest)
system = System(elements=[spins], interactions=[magnet], solver=solver).prepare()
abs(system.simulate(until=200, dt=50, random_state=0).get_values('spins.spin')[-1].mean()) > 0.9
```

After 200 units of time the mean spin stays above 0.9 in absolute value, so the magnet stays ordered.

The solver lowers the objective to a sweep of the lattice with three parts.

- The local objective of a site is every term of the total objective that involves it. It reads the values of the
  neighbours through `neighbors`, over the `neighborhood` of the solver.
- The `moves` of the solver propose changes, with the kinds `'flip'`, `'random-value'`, `'rotate'`, `'gaussian'`, and
  `'cluster'`.
- The `rule` accepts them, `'metropolis'`, `'glauber'`, or `'heat-bath'`.

The change of the objective is derived symbolically, and sweeps update the two sublattices in turn. The temperature is
a value of the system, so the `schedule` of `simulate` can change it during a run.

```python continuation
heating = {'spins.temperature': lambda t: 1.5 if t < 100 else 5.0}
hot = system.simulate(until=200, dt=50, random_state=0, schedule=heating)
round(float(abs(hot.get_values('spins.spin')[-1].mean())), 2)
```

Heated to the temperature 5 halfway through the run, the magnet loses its order, with a mean spin of 0.04 in size.

## Triangular lattices

This section gives every site six neighbours, as hexagonal cells and lattice gases need. A fire spreads from one site
to its six neighbours at every step, and the rule `spread` takes the largest value of the neighbourhood.

```python
from types import SimpleNamespace

import numpy as np

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import UpdateLaw
from skcomplex.core.components.spaces import LatticeSpace, RadiusNeighborhood
from skcomplex.core.types import Role
from skcomplex.core.systems import System


def spread(*values):
    return np.max(values, axis=0)


class SpreadInteraction(Interaction):
    def __init__(self, label, *, target: Role = 'forest'):
        super().__init__(label)
        self.target = target

    def _prepare(self, context):
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            hexagonal = RadiusNeighborhood(1, metric='hexagonal', center=True)
            self.laws_ = [UpdateLaw(context.target.fire, spread, neighborhood=hexagonal)]


fire = np.zeros((40, 40))
fire[20, 20] = 1
forest = Element('forest', values={'fire': fire.ravel()}, space=LatticeSpace((40, 40), geometry='triangular'))
system = System(elements=[forest], interactions=[SpreadInteraction('spread')]).prepare()
burning = system.simulate(until=3, dt=1).get_lattice('forest.fire')
[int(item.sum()) for item in burning]
```

The burning sites number 1, 7, 19, and 37 after zero to three steps, the hexagonal numbers `3k(k + 1) + 1`.

- `LatticeSpace(shape, geometry='triangular')` keeps integer coordinates on a skewed grid, so every boundary works as
  on a square lattice.
- It places site `(i, j)` at `(j + i / 2, i sqrt(3) / 2)` in the plane, which the fitted `positions_` give.
- `RadiusNeighborhood(1, metric='hexagonal')` holds the six neighbours counterclockwise from 0 degrees, so direction `k + 3`
  modulo 6 is opposite to direction `k`, as lattice gases need.
- The cells of the sites are the hexagons of a hexagonal tiling.

## Lattices on graphs

This section uses the nodes of a network as the sites. An element also takes a
[`NetworkSpace`](../core_components/spaces.md#networkspace), with one member per node in the order of `graph.nodes`,
and initial values are arrays in node order. The
[`AdjacencyNeighborhood`](../core_components/spaces.md#adjacencyneighborhood) of a node holds its neighbours along its
edges. Here an Ising magnet lives on a random regular graph of 200 nodes, each with four neighbours. Its local energy
over the temperature is an `ObjectiveLaw` of each spin, and a `MetropolisSolver` flips the spins in colored sweeps.

```python
import networkx as nx

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import ObjectiveLaw
from skcomplex.core.components.laws.utils import neighbors
from skcomplex.core.components.spaces import AdjacencyNeighborhood, NetworkSpace
from skcomplex.core.engine.solvers import MetropolisSolver
from skcomplex.core.systems import System


def energy(s):
    spins = s.target
    return [ObjectiveLaw(-spins.J * spins.spin * neighbors(spins.other.spin) / spins.temperature, spins.spin)]


graph = nx.random_regular_graph(4, 200, seed=0)
spins = Element('spins', values={'spin': 1.0, 'J': 1.0, 'temperature': 1.0}, space=NetworkSpace(graph))
solver = MetropolisSolver(moves={'spins.spin': 'flip'}, neighborhood=AdjacencyNeighborhood(), schedule='colored')
magnet = System(elements=[spins], interactions=[Interaction('energy', roles={'target': 'spins'}, laws=energy)])
values = magnet.set_params(solver=solver).prepare().simulate(until=200, dt=50, random_state=0).get_values('spins.spin')
abs(values[-1].mean()) > 0.9
```

At the temperature 1.0 the mean spin stays above 0.9 in absolute value after 200 units of time, so the magnet on the
graph stays ordered too.

- `AdjacencyNeighborhood(center=False)` pads the neighbours of every node to the largest degree with the `fill` value
  of the network.
- Local rules, drives, relaxations, and Monte Carlo laws run on graphs as on lattices.
- Lookup rules on graphs take the totalistic forms. A function rule that names the keyword `degree` receives the
  number of real neighbours of every site.
- The padded table grows with the largest degree. Graphs with hubs of very high degree, beyond scale-free graphs of
  about 10^5 nodes, therefore use much memory.

## Label copies and regions

This section names the recipe for sites that copy labels, so that the sites of one label form a region whose total size
matters. The cellular Potts model of biology works this way, with one label per biological cell.
`RegionSamplingInteraction` of `skcomplex.systems.biology.physiology` returns its own sweep of label copies with the
objective
`sum_<i,j> w_ij contact(l_i, l_j) [l_i != l_j] + sum_l a_l (V_l - A_l)^2 + b_l (S_l - B_l)^2` over integer labels. It
keeps the volume `V_l` and the surface `S_l` of every region up to date on every accepted copy. The catalog class
`CellularPottsSystem` of the same module sorts cells of two kinds by differential adhesion on square, triangular, and
disordered lattices.

## Reads that follow the state, and series in the cells

This section builds rules whose reads depend on the configuration and states whose values are formal power series. The
neighbourhood `StateTranslationNeighborhood` of
[`skcomplex.systems.computer_science.common.automata`][skcomplex.systems.computer_science.common.automata] chooses the
cell a rule reads from the state itself, such as the map `τ(x)(g) = x(g · x(g))` whose alphabet is the group itself. A
cell of a `CayleySpace` of the same package holds an element of a group as the number of its site, and the rule reads
the cells at `g x(g) s` for the words `s` of the neighbourhood. `SelfIndexedShiftSystem` of
`skcomplex.systems.computer_science.automata` writes the map as an `UpdateLaw` with this neighbourhood. Here it runs
once on the symmetric group of three letters, whose six elements are the sites.

```python
from typing import Annotated

import numpy as np
from sage.all__sagemath_gap import SymmetricGroup

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import UpdateLaw
from skcomplex.core.components.spaces import LatticeNeighborhood, LatticeSpace
from skcomplex.core.components import PowerSeriesSpec
from skcomplex.core.systems import System
from skcomplex.systems.computer_science.automata import SelfIndexedShiftSystem

shift = SelfIndexedShiftSystem(SymmetricGroup(3), radius=None, initial=[1, 2, 3, 4, 5, 0])
shift.prepare().simulate(until=1).get_values('cells.value')[-1].tolist()
```

The map sends the values `[1, 2, 3, 4, 5, 0]` to `[2.0, 1.0, 0.0, 3.0, 3.0, 0.0]`.

- The first state a rule reads holds the group elements. On a finite group every product is a site. On a ball a
  product outside it, or reached through a site outside it, reads the fill value of the space, as does a value that
  names no site, so a run truncates a map of unbounded memory on an infinite group to a stated boundary.
- The products of every pair of sites are tabulated once, so balls of a few thousand sites fit. Translated reads run
  on the synchronous and random schedules, and refuse colorings and `degree`, which need a fixed table.

A `PowerSeriesSpec` in `Annotated` gives a value its values in a ring of formal power series, truncated at an order.
Here the automaton `x(n) - t x(n + 1)` over `K[[t]]` runs three steps on a line of six cells whose fourth cell holds the
series 1.

```python continuation
RING = PowerSeriesSpec(order=4, field='QQ')


class Series(Element):
    def __init__(self, label, value: Annotated[object, RING] = 0, space=None):
        super().__init__(label, space=space)
        self.value = value


def difference(here, right):
    return here - RING.multiply(RING.get_coefficients_of('t'), right)


def shift_laws(s):
    here_and_right = LatticeNeighborhood(((0,), (1,)))
    return [UpdateLaw(s.target.value, difference, neighborhood=here_and_right)]


start = np.zeros((6, 4))
start[3, 0] = 1
cells = Series('cells', value=start, space=LatticeSpace(6))
series = System(elements=[cells], interactions=[Interaction('difference', roles={'target': 'cells'}, laws=shift_laws)])
final = series.prepare().simulate(until=3).get_values('cells.value')[-1]
RING.get_series_of(final[2])
```

The third cell holds `-3*t + O(t^4)`, the coefficient of one shift in `(1 - t S)^3`, where `S` reads the cell to the
right.

- A value annotated with a `PowerSeriesSpec` holds `order` coefficients per cell. A function rule receives one row of
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
`UpdateLaw` writes it with two reductions: a species renews when the smallest fitness of the ring is its own or that of
a neighbour, and keeps its fitness otherwise. Every step is one extinction.

```python
import numpy as np

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import UpdateLaw
from skcomplex.core.components.laws.utils import minimum, neighbors, uniform, where
from skcomplex.core.components.spaces import LatticeNeighborhood, LatticeSpace
from skcomplex.core.systems import System

BESIDE = LatticeNeighborhood(((-1,), (1,)))


def extinction(s):
    fitness = s.target.fitness
    around = neighbors(s.target.other.fitness, over=BESIDE, reduce='min')
    renewed = minimum(fitness, around) == minimum(fitness)
    return [UpdateLaw(fitness, where(renewed, uniform('fitness'), fitness), label='extinction')]


fitness = np.random.default_rng(0).random(100)
ring = Element('ring', values={'fitness': fitness}, space=LatticeSpace(100))
system = System(elements=[ring], interactions=[Interaction('evolution', roles={'target': 'ring'}, laws=extinction)])
trajectory = system.prepare().simulate(until=1, dt=1, random_state=0)
changed = np.flatnonzero(trajectory.states[1] != trajectory.states[0])
changed.tolist(), int(np.argmin(fitness))
```

The first step changes three sites, the least fit species, at the site the second value names, and its two
neighbours.

- `minimum(s.target.fitness)` reduces over the whole group, and
  `neighbors(s.target.other.fitness, over=BESIDE, reduce='min')` over the neighbours of every member. Comparing the
  two finds the members whose neighbourhood holds the extremum, so the update needs no index of the extremal site. A
  largest value takes `maximum` and `reduce='max'`.
- On a network, `AdjacencyNeighborhood()` gives the neighbours of every node, and an isolated node renews only when it
  holds the extremum itself.
- Ties of the extremum renew every tied member with its neighbours. Uniform draws make them vanishingly rare.
- One extinction per step makes time count the extinctions, and `dt` sets how many of them a recorded frame skips.
  `BakSneppenSystem` of `skcomplex.systems.biology.evolution` runs the ready-made model on a ring or a graph.

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
