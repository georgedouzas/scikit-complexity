# General realizations

`skcomplex.systems.general` holds the recipes that systems of several fields share, such as a nonnegative amount, a
reaction, an agent that moves, or a market that matches buyers with sellers. Each recipe is written with the public
parts of the core only, so it is also an example of how to build a part of your own. The package has one subpackage per
concept, `elements`, `flows`, `agents`, `matching`, `kinetics`, `reactions`, and `dynamics`, and you import every recipe
from its subpackage.
The core never imports them, and the fields import them freely.

## Where a part goes

One ordered test, the placement rule, decides where any part goes, now and in the future. `tests/test_sources.py`
checks it mechanically.

1. A structural base, such as a `Base` class, the composite system, or a base where laws run per member, edge,
   hyperedge, or pair, goes to the core.
2. A part that the engine handles by type, through a compile pass, a kernel, a solver, a backend lowering, or an
   analysis that checks the class or its spec, goes to the core, and so does an operator generated from one function.
   It is named by its mathematics and must be cross-domain in principle.
3. Every other part is a recipe, expressible with core parts. Its users place it. Several fields:
   `skcomplex.systems.general.<concept>`. One field with several systems: `systems/<field>/common/<family>`. One system:
   that system's module. A field with its own term names it, and `general` uses neutral names.
4. A recipe that needs something the core cannot express reveals a missing core primitive, which enters by step 2.
   Engine support never lives inside a recipe.
5. The core never imports `skcomplex.systems`.

One example for each step:

1. `HyperedgeInteraction` runs its laws once per hyperedge, and `CompositeSystem` holds the parts of a system, so both
   are structural bases of the core.
2. `GroupingInteraction` stays in the core, since a compile pass lowers its symbolic form by type to kernels over
   random groups redrawn every step. Its name is the mathematics it does, and economics and social both form groups.
3. `MovementInteraction` returns agent rules, which the core already runs, and systems of biology, economics,
   engineering, mathematics, and social use it, so it lives in `skcomplex.systems.general.agents`. The toppling of a
   sandpile serves several systems of physics only, so it lives in a family of the physics `common` package under the
   physics name `TopplingInteraction`. The satisfiability of logical constraints serves one system, so
   `SatisfiabilityInteraction` lives in the module of `PerceptronSatisfiabilitySystem`.
4. The satisfiability once had a check of its own in the compiler, which is engine support inside a recipe. The
   check that a role accepts only Boolean states is a primitive of the core, `RoleSpec(domains=...)`, so the recipe
   declares its role with it and the compiler keeps no code for the class.
5. `sparse_identification` of the core returns a `CompositeSystem` of core parts, never a class of
   `skcomplex.systems`, so the core runs without the catalog.

## Elements

`NonnegativeElement` and `PointElement` of `skcomplex.systems.general.elements` are the two elements that most fields
share.

```python
from skcomplex.systems.general.elements import NonnegativeElement, PointElement
from skcomplex.core.components.spaces import ContinuousSpace
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate

system = CompositeSystem(
    elements=[NonnegativeElement('stock', x=5.0), PointElement('point', x_d1=[1.0, 0.0])],
    space=ContinuousSpace(n_dim=2),
)
trajectory = simulate(system, until=2.0, dt=1.0)
trajectory.values('stock.x')[-1], trajectory.values('point.x')[-1].tolist()
```

- `NonnegativeElement` holds one nonnegative state `x`, such as a stock, a species, a queue, or the density of one
  activity class of a kinetic system. A field names its own quantity by a thin subclass, such as the biology
  `CompartmentElement`.
- `PointElement` is a member with a vector state `x` of order 2 in a continuous space. Its first derivative is `x_d1`,
  so a backend such as LAMMPS reads the velocity of a point from its declared states.


## Flows

`LorenzElement` of `skcomplex.systems.general.flows` is the Lorenz flow, the convection model of mathematics and the
response that a Rössler oscillator drives in the chaotic synchronization of physics.

```python
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.systems.general.flows import LorenzElement

trajectory = simulate(CompositeSystem(elements=[LorenzElement('lorenz', rho=10.0)]), until=2.0, dt=1.0)
trajectory.values('lorenz.z')
```

## Reactions

`ReactionInteraction` of `skcomplex.systems.general.reactions` writes a reaction between species at a product
propensity, and `ReactionNetworkSystem` builds a whole network from a table.

Use `ReactionInteraction` for reactions between species at a product propensity, such as an epidemic written as two
reactions. The total population stays the same:

```python
from skcomplex.systems.general.elements import NonnegativeElement
from skcomplex.systems.general.reactions import ReactionInteraction
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate

infection = ReactionInteraction('infection', reactants={('S', 'x'): 1, ('I', 'x'): 1}, products={('I', 'x'): 2}, k=5e-4)
recovery = ReactionInteraction('recovery', reactants={('I', 'x'): 1}, products={('R', 'x'): 1}, k=0.1)
sir = CompositeSystem(
    elements=[NonnegativeElement('S', x=990), NonnegativeElement('I', x=10), NonnegativeElement('R')],
    interactions=[infection, recovery],
)
round(float(simulate(sir, until=50, dt=1.0).states[-1].sum()), 6)
```

`ReactionInteraction(target=None, reactants, products, k, propensity=None, modifiers=None, mode='rate')` runs as
rates, as jumps (`mode='jump'`), or with Langevin noise. A species is a pair of a path and a state, or a state name of
the `target` alone. A reaction with a group or field target therefore reacts in every member.

### Reaction networks

Write a compartment model of any field without writing an element or an interaction. `ReactionNetworkSystem` takes the
initial amount of every species and a table of reactions. Every reaction maps its label to its reactants and products,
as coefficients by species name, and to its rate constant.

```python
from skcomplex.systems.general.reactions import ReactionNetworkSystem
from skcomplex.core.engine.runtime import simulate

isomerization = ReactionNetworkSystem(
    species={'A': 10.0, 'B': 0.0},
    reactions={'forth': ({'A': 1}, {'B': 1}, 0.3), 'back': ({'B': 1}, {'A': 1}, 0.1)},
)
round(float(simulate(isomerization, until=50.0, dt=1.0).values('A.x')[-1]), 3)
isomerization.set_params(mode='jump').get_tags().randomness
```

Every species is a `NonnegativeElement` named after it, whose state `x` holds its amount, and every reaction is a
`ReactionInteraction`. The `mode` writes the reactions as rate equations, `'rate'`, as random events of single
molecules, `'jump'`, or as chemical Langevin equations, `'langevin'`. The amount of `A` relaxes to
`0.1 / (0.3 + 0.1)` of the total.

### Realizations of a reaction network

A field realization takes the parameters of its model under its own names, and gives the general system its parts
through hooks. `SIRSystem` subclasses `ReactionNetworkSystem`. It takes `susceptible`, `infected`, `beta`, and `gamma`,
returns its compartments from `_get_species`, and its infection and recovery from `_get_interactions`. Epidemics of
susceptible, infected, and recovered individuals and enzyme kinetics are both reaction networks with a product
propensity, which epidemiology calls homogeneous mixing and chemistry the law of mass action, so they share the general
system and differ in their realizations.

The hooks of `ReactionNetworkSystem` are these:

- `_get_species` returns the initial amount of every species by name,
- `_get_reactions` returns the reaction table,
- `_get_mode` returns the mode,
- `_get_interactions` returns the reactions as interactions, for a realization whose reactions have roles of their own
  names, such as the `susceptible` and `infected` roles of an infection.

## Dynamical systems

`skcomplex.systems.general.dynamics` holds the general systems of states and their evolution. Each takes its data by
name, and a field system subclasses it, takes the parameters of its model under its own names, and returns its data
from hooks. The [parts page of systems](../../../parts/systems.md) lists every general system with the catalog
systems that subclass it.

### DynamicalSystem

`DynamicalSystem` is a set of states and the equations that evolve them, rates in continuous time or the next values
in discrete time. The Wilson-Cowan populations of neuroscience are rates, and the multiplier-accelerator of economics
is a map whose equations read the states at the start of the step:

```python
from sage.all__sagemath_symbolics import exp

from skcomplex.core.engine.runtime import simulate
from skcomplex.systems.general.dynamics import DynamicalSystem

populations = DynamicalSystem(
    states={'e': 0.1, 'i': 0.05},
    equations={
        'e': lambda s: -s.target.e + 1 / (1 + exp(-(2 * s.target.e - 3 * s.target.i))),
        'i': lambda s: -s.target.i + 1 / (1 + exp(-(2 * s.target.e - s.target.i))),
    },
    element='populations',
)
economy = DynamicalSystem(
    states={'y': 1.0, 'previous': 1.0},
    equations={
        'y': lambda s: s.target.g + s.target.c * (1 + s.target.v) * s.target.y - s.target.c * s.target.v * s.target.previous,
        'previous': lambda s: s.target.y,
    },
    parameters={'g': 1.0, 'c': 0.5, 'v': 0.5},
    time='discrete',
    element='economy',
)
simulate(populations, until=20.0, dt=1.0).values('populations.e')[-1], simulate(economy, until=3).values('economy.y')
```

Its hooks are `_get_states`, `_get_equations`, `_get_parameters`, `_get_noises`, `_get_time`, `_get_element_label`, and
`_get_variables`, which gives the specs of the states and the parameters, such as a domain or the order of a Caputo
derivative. A noise amplitude, read in the Itô or Stratonovich convention, needs continuous time.

### NetworkDynamicalSystem

`NetworkDynamicalSystem` puts the states on every node of a graph, and adds to the rate of every coupled state the sum
over the neighbours `sum_j A_ij g(x_i, x_j)`. Diffusion on a graph in mathematics and the exchange of gap junctions
between cells in physiology are both of this shape:

```python
import networkx as nx

from skcomplex.core.engine.runtime import simulate
from skcomplex.systems.general.dynamics import NetworkDynamicalSystem

cells = NetworkDynamicalSystem(
    nx.cycle_graph(4),
    states={'v': [1.0, 0.0, 0.0, 0.0]},
    equations={'v': lambda s: -s.target.v / s.target.tau},
    coupling={'v': lambda s: s.member.g * (s.other.v - s.member.v)},
    parameters={'tau': 10.0, 'g': 0.5},
    element='cells',
)
simulate(cells, until=5.0, dt=1.0).values('cells.v')[-1]
```

It adds the hooks `_get_graph` and `_get_coupling`. A local equation may also read the graph Laplacian, as network
diffusion of mathematics does.

### ReactionDiffusionSystem

`ReactionDiffusionSystem` puts the states as fields on the sites of a lattice, with local kinetics and the diffusion
`D laplacian(u)` of every field that has a diffusivity. The Fisher-Kolmogorov front of ecology invades an empty line:

```python
import numpy as np

from skcomplex.core.engine.runtime import simulate
from skcomplex.systems.general.dynamics import ReactionDiffusionSystem

initial = np.zeros(50)
initial[:5] = 1.0
front = ReactionDiffusionSystem(
    states={'u': initial},
    equations={'u': lambda s: s.target.r * s.target.u * (1 - s.target.u)},
    diffusivities={'u': 1.0},
    shape=50,
    boundary='reflecting',
    parameters={'r': 1.0},
)
simulate(front, until=10.0, dt=5.0).values('medium.u')[-1][:25]
```

It adds the hooks `_get_diffusivities`, `_get_space`, and `_get_boundaries`. A chemistry system whose kinetics are
reactions returns them from `_get_equation_interactions`, as the Gray-Scott system of pattern formation does.

### StockFlowSystem

`StockFlowSystem` is a set of stocks that change only through named flows, each from a stock or a source to a stock or
a sink, and the trajectory reports every flow. The bathtub of engineering and the arms race of social organizations are
stock-flow systems, as are the files of system dynamics that the translator reads:

```python
from skcomplex.core.engine.runtime import simulate
from skcomplex.systems.general.dynamics import StockFlowSystem

race = StockFlowSystem(
    stocks={'a': 1.0, 'b': 1.0},
    flows={
        'arming_a': (None, 'a', lambda s: s.source.k * s.source.b),
        'arming_b': (None, 'b', lambda s: s.source.k * s.source.a),
    },
    parameters={'k': 0.1},
    element='nations',
)
trajectory = simulate(race, until=10.0, dt=5.0)
trajectory.values('nations.a'), trajectory.values('nations.arming_a')
```

`None` stands for the environment. A labelled `Source('suppliers')` or `Sink('consumers')` of
`skcomplex.core.components.laws` in its place names it. Its hooks are `_get_stocks`, `_get_flows`, `_get_parameters`,
`_get_element_label`, and `_get_variables`.

## Encounters of a kinetic system

Use `EncounterInteraction` for the bilinear encounters of a kinetic system, whose elements are densities on activity
classes:

```python
import numpy as np

from skcomplex.core.components.elements import GroupElement
from skcomplex.systems.general.elements import NonnegativeElement
from skcomplex.systems.general.kinetics import EncounterInteraction
from skcomplex.core.components.spaces import LatticeSpace
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate

kernel = np.zeros((3, 1, 3))
kernel[:, :, 0] = 1.0
gather = EncounterInteraction(
    'gather', candidate='candidates', field='field', rate=lambda s: 1.0, kernel=lambda s: kernel
)
system = CompositeSystem(
    elements=[
        GroupElement('candidates', element=NonnegativeElement, size=3, on='sites', values={'x': [0.2, 0.3, 0.5]}),
        GroupElement('field', element=NonnegativeElement, size=1, values={'x': 1.0}),
    ],
    interactions=[gather],
    space=LatticeSpace(3, boundary='fixed'),
)
densities = simulate(system, until=4.0, dt=0.5).values('candidates.x')
np.round(densities[-1], 2).tolist()
```

`EncounterInteraction(candidate, field, rate, kernel, outcome)` gives encounters that are conservative, destructive,
or proliferative.

## Agent rules

The recipes of `skcomplex.systems.general.agents` move agents, create and remove them, let them choose and imitate, and
evolve their strategies. They return agent rules, or the symbolic laws of the core where an operation has a symbolic
form.

### Movement

Use `MovementInteraction` to move agents. Here two walkers each take one step to a neighbouring cell:

```python
from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.systems.general.agents import MovementInteraction
from skcomplex.core.components.spaces import LatticeSpace
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate

walkers = GroupElement('walkers', element=Element, size=2, location=[(2, 2), (2, 2)])
walk = MovementInteraction('walk', target='walkers', step='neighbor')
system = CompositeSystem(elements=[walkers], interactions=[walk], space=LatticeSpace((5, 5), metric='manhattan'))
trajectory = simulate(system, until=1, random_state=0)
[int(abs(cell - 2).sum()) for cell in trajectory.get_agents('walkers', 1).locations]
```

`MovementInteraction(target, step='neighbor', ...)` moves agents by a neighbour, Gaussian, Lévy, best-cell, or user
step, with a condition, an acceptance, exclusion, and a cost.

### Agents in three dimensions and on region layers

Agents can live in a box of three dimensions, or move between named regions such as census tracts.

```python
from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.systems.general.agents import MovementInteraction
from skcomplex.core.components.spaces import ContinuousSpace
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate

divers = GroupElement('divers', element=Element, size=4)
swim = MovementInteraction('swim', target='divers', step='gaussian', scale=0.1)
space = ContinuousSpace(extent=(10.0, 5.0, 2.0), boundary='periodic')
system = CompositeSystem(elements=[divers], interactions=[swim], space=space)
trajectory = simulate(system, until=5, random_state=0)
trajectory.get_agents('divers', 5).locations.shape
```

`MesaSolver` runs agents in a `ContinuousSpace` of two or three sides. Placement, `MovementInteraction` steps, radius
queries, the minimum image of a periodic box, and the locations of a trajectory use every side, and
`system.dimensions` returns every side.

- The solver rejects a continuous space with any other number of sides, or without an extent, before the first frame.
- The compile report holds the issue `location-dimension` for an initial location whose length differs from the
  dimension of its space.
- `AgentsVisual` draws agents in three dimensions as `Points3D`.

To place agents in regions, give the space layers of polygons:

```python
from shapely import Point, box

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.systems.general.agents import MovementInteraction
from skcomplex.core.components.spaces import ContinuousSpace
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate

census = [box(0.0, 0.0, 4.0, 4.0), box(5.0, 0.0, 9.0, 4.0)]
places = [box(1.0, 6.0, 3.0, 8.0)]
people = GroupElement('people', element=Element, size=3, location=[census[0], census[0], census[1]])
outing = MovementInteraction('outing', target='people', step=lambda agent, system: system.get_region('places', 0))
space = ContinuousSpace(extent=(10.0, 10.0), regions={'census': census, 'places': places})
system = CompositeSystem(elements=[people], interactions=[outing], space=space)
trajectory = simulate(system, until=1, random_state=0)
all(places[0].contains(Point(point)) for point in trajectory.get_agents('people', 1).locations)
```

`ContinuousSpace(regions=...)` holds layers of regions in a space of two sides. It is a mapping from the name of a
layer to a sequence of Shapely polygons in the coordinates of the space. `prepare` keeps one `STRtree` per layer in
`regions_`. For a region that is not a valid polygon, it raises `SpaceError`, which names the layer and the region.

- A polygon is a location in `GroupElement(location=...)`, `SystemView.create`, and `AgentView.move_to`. The solver
  draws a uniform point inside it from the generator of the run. It chooses a triangle of the constrained
  Delaunay triangulation of the polygon with a probability proportional to its area.
- `AgentView.get_regions_here(layer)` returns the indices of the regions of a layer that contain the agent.
- `SystemView.get_region(layer, index)` returns the polygon of a region, and `SystemView.get_regions(layer)` the number of
  regions of a layer.
- A `MovementInteraction` step that returns the polygon of a region moves an agent into it.
- `AgentsVisual` draws every layer once as static `LineStrips2D` under the agents.

**Real maps.** The polygons of a space are in the coordinates of the space, and the library handles no coordinate
reference system. Census shapefiles usually come in longitude and latitude, where every distance, such as a radius, is
in degrees. Reproject them to a metric system first. With GeoPandas,
`geopandas.read_file(path).to_crs(epsg=32604).geometry` gives the polygons of a shapefile in metres for a Hawaiian
island. Then shift them so that no coordinate is negative, since a box runs from 0 to its sides. The next example does
the same with a plain local projection of one census polygon.

```python
import math

import numpy as np
import shapely
from shapely import Polygon
from shapely.affinity import translate

EARTH_RADIUS = 6371000.0
tract = Polygon([(-159.6, 21.9), (-159.4, 21.9), (-159.4, 22.1), (-159.6, 22.1)])


def project(points):
    scale = math.cos(math.radians(22.0))
    return EARTH_RADIUS * np.radians(points) * np.array([scale, 1.0])


projected = shapely.transform(tract, project)
low_x, low_y, _, _ = projected.bounds
tract_in_metres = translate(projected, -low_x, -low_y)
[round(value, -3) for value in tract_in_metres.bounds]
```

`KauaiPopulationSystem` of `skcomplex.systems.social.organizations` places one agent per resident of the census
regions of the synthetic island of its package, and `Boids3dSystem` of `skcomplex.systems.biology.collective_behaviour`
flies in a periodic cube.

```python
from skcomplex.systems.biology.collective_behaviour import Boids3dSystem
from skcomplex.systems.social.organizations import KauaiPopulationSystem

system = KauaiPopulationSystem(populations=(10, 5, 0)).prepare().system_
system.elements[0].size, list(system.space.regions), Boids3dSystem(n_boids=10).prepare().system_.space.extent
```


### Choice and imitation

Use `ChoiceInteraction` when every member picks an option by its utility, such as members that prefer to agree with the
mean:

```python
from typing import Annotated

import numpy as np

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.systems.general.agents import ChoiceInteraction
from skcomplex.core.components.laws.utils import mean
from skcomplex.core.components.spaces import DiscreteTime
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.specs import StateSpec


class ChooserElement(Element):
    def __init__(self, label, *, option: Annotated[float, StateSpec()] = 1.0):
        super().__init__(label)
        self.option = option


utility = lambda s, option: 2.0 * option * mean(s.target.option)
choice = ChoiceInteraction('choice', target='members', state='option', options=(-1, 1), utility=utility)
members = GroupElement('members', element=ChooserElement, size=500)
system = CompositeSystem(elements=[members], interactions=[choice], time=DiscreteTime())
values = simulate(system, until=10, random_state=0).values('members.option')
round(float(np.mean(values[-1])), 1)
```

`ChoiceInteraction(target, state, options, utility, rule='logit', intensity=1.0)` draws an option for every member by
the logit, Metropolis, Fermi, or best rule from `utility(s, option)`, or `utility(agent, system, option)` in the agent
form.

Use `ImitationInteraction` when members copy their neighbours, such as voters that follow the local majority:

```python
from typing import Annotated

import numpy as np

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.systems.general.agents import ImitationInteraction
from skcomplex.core.components.spaces import DiscreteTime, LatticeSpace
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.specs import StateSpec


class VoterElement(Element):
    def __init__(self, label, *, state: Annotated[float, StateSpec()] = 0.0):
        super().__init__(label)
        self.state = state


initial = np.array([1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0], dtype=float)
majority = ImitationInteraction('majority', target='voters', rule='majority', form='probability')
voters = GroupElement('voters', element=VoterElement, space=LatticeSpace(12), values={'state': initial}, on='sites')
system = CompositeSystem(elements=[voters], interactions=[majority], time=DiscreteTime())
simulate(system, until=1, random_state=0).values('voters.state')[-1].tolist()
```

`ImitationInteraction(target, state, payoff, options, rate, rule='random', form='rate')` copies the state of a
neighbour at random, by majority, from the best, from a random better neighbour, or by the Fermi rule. It runs as
jumps, as updates with a probability per step (`form='probability'`), or as an agent rule.

### Evolution

Use `EvolutionInteraction` for a genetic algorithm or evolutionary dynamics over strategies. In the example, the
fitness of a bit string is its number of ones, and the mean rises over the generations:

```python
from typing import Annotated

import numpy as np

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.systems.general.agents import EvolutionInteraction
from skcomplex.core.components.laws import RuleLaw
from skcomplex.core.components.spaces import DiscreteTime
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.specs import PhaseSpec, PhasesScheduleSpec, StateSpec


class BitStringElement(Element):
    def __init__(
        self,
        label,
        *,
        strategy: Annotated[object, StateSpec()] = 0.0,
        fitness: Annotated[float, StateSpec()] = 0.0,
    ):
        super().__init__(label)
        self.strategy = strategy
        self.fitness = fitness

    def _get_laws(self, s):
        return [RuleLaw(self.score, phase='scoring')]

    def score(self, agent, system):
        agent.fitness = float(np.sum(agent.strategy))


initial = np.random.default_rng(0).integers(0, 2, size=(40, 20)).astype(float)
agents = GroupElement('agents', element=BitStringElement, size=40, values={'strategy': initial})
evolution = EvolutionInteraction('evolution', target='agents', phase='evolution')
schedule = PhasesScheduleSpec([PhaseSpec('scoring'), PhaseSpec('evolution')])
system = CompositeSystem(elements=[agents], interactions=[evolution], time=DiscreteTime(schedule=schedule))
trajectory = simulate(system, until=30, random_state=0)
final = np.array(trajectory.get_agents('agents', 30).values['strategy'].tolist())
final.sum(axis=1).mean() > initial.sum(axis=1).mean() + 4
```

`EvolutionInteraction(target, strategy, fitness, selection='tournament', replacement='generational', crossover,
mutation, ...)` replaces strategies by selection, crossover, and mutation every `period` steps.

- The selection is by tournament, roulette, proportional, or rank.
- The replacement replaces the whole generation, or replaces by Moran, death-birth, truncation, or the best mutant.


### Birth and death

Use `BirthDeathInteraction` to create and remove agents. Here a cell of size 8 divides twice, and each division splits
the size between parent and offspring:

```python
from typing import Annotated

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.systems.general.agents import BirthDeathInteraction
from skcomplex.core.components.spaces import LatticeSpace
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.specs import StateSpec


class CellElement(Element):
    def __init__(self, label, *, size: Annotated[float, StateSpec()] = 0.0):
        super().__init__(label)
        self.size = size


cells = GroupElement('cells', element=CellElement, size=1, values={'size': 8.0}, variable=True)
division = BirthDeathInteraction('division', target='cells', birth=1.0, split=('size',))
system = CompositeSystem(elements=[cells], interactions=[division], space=LatticeSpace((3, 3)))
simulate(system, until=2, random_state=0).get_agents('cells', 2).values['size'].tolist()
```

`BirthDeathInteraction(target, birth, death, rate=None, offspring=1, placement='here', inherit=(), split=(), ...)`
creates offspring by a condition, a probability, or a rate. Offspring values are inherited, given, split, or mutated.
It also removes or renews agents.

## Matching

Use `MatchingInteraction` of `skcomplex.systems.general.matching` to pair requesters with providers, such as buyers
and sellers in a market:

```python
from skcomplex.systems.general.matching import MatchingInteraction, QueueSpec, StableMatchingSpec

queue = MatchingInteraction('queue', requesters='buyers', providers='sellers', rule=QueueSpec())
stable = MatchingInteraction('stable', requesters='buyers', providers='sellers', rule=StableMatchingSpec())
queue.reports, type(stable.rule).__name__
```

`MatchingInteraction(requesters, providers, rule, limit, quantity, budget, transfers, on_match)` pairs requesters
with providers by the algorithm its rule names, then moves the states of `transfers`. The rules are frozen specs of
`skcomplex.systems.general.matching` that hold only the settings of their algorithm:

- `RandomSearchSpec(tries, value)`, the default.
- `PartnerMatchingSpec(candidates, choice, fallback, capacity, score, value)`.
- `PartnerSwitchSpec(sample, intensity, partner)`.
- `QueueSpec(order)`.
- `IterativeMatchingSpec(adjustment, iterations, tolerance)`.
- `ContinuousMatchingSpec(arrival)`.
- `BatchMatchingSpec()`.
- `StableMatchingSpec(requester_preference, provider_preference, value)`.

The matching runs each algorithm as its own method, chosen by the type of its rule, so a new algorithm is a new spec
and a method. The next section runs a batch auction.

### A market in two phases

Traders quote prices in one phase and trade in a batch auction in the next, while a clerk sets a fee from the volume
traded. The phases and the clerk are parts of the core, which the section
[Phases, institutions, and learning](../build_a_custom_system/agent_based_systems.md#phases-institutions-and-learning)
explains:

```python
from typing import Annotated, ClassVar

import numpy as np

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import RuleLaw
from skcomplex.core.components.spaces import DiscreteTime
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.specs import ParameterSpec, PhaseSpec, PhasesScheduleSpec, StateSpec
from skcomplex.systems.general.matching import BatchMatchingSpec, MatchingInteraction


class TraderElement(Element):
    def __init__(
        self,
        label,
        *,
        value: Annotated[float, StateSpec()] = 0.0,
        bid: Annotated[float, StateSpec()] = 0.0,
        ask: Annotated[float, StateSpec()] = 0.0,
        demand: Annotated[float, StateSpec()] = 0.0,
        supply: Annotated[float, StateSpec()] = 0.0,
        money: Annotated[float, StateSpec()] = 0.0,
        goods: Annotated[float, StateSpec()] = 0.0,
    ):
        super().__init__(label)
        self.value = value
        self.bid = bid
        self.ask = ask
        self.demand = demand
        self.supply = supply
        self.money = money
        self.goods = goods

    def _get_laws(self, s):
        return [RuleLaw(self.quote, phase='quote')]

    def quote(self, agent, system):
        agent.bid = system.rng.uniform(0, agent.value) if agent.demand else 0.0
        agent.ask = system.rng.uniform(agent.value, 2.0) if agent.supply else 2.0


class ClerkInteraction(Interaction):
    law_classes: ClassVar[tuple[type, ...]] = (RuleLaw,)
    reports: ClassVar[tuple[str, ...]] = ('traded',)

    def __init__(self, label, *, fee: Annotated[float, ParameterSpec()] = 0.0):
        super().__init__(label)
        self.fee = fee

    def _get_laws(self, s):
        return [RuleLaw(self.act, phase='trade')]

    def act(self, system):
        traded = sum(agent.goods for agent in system.get_agents('buyers'))
        system.report('traded', traded)
        system.set_parameter('fee', 0.01 * traded)


rng = np.random.default_rng(0)
buyers = {'value': rng.uniform(0.5, 2, 20), 'demand': 1.0, 'money': 2.0}
sellers = {'value': rng.uniform(0, 1.5, 20), 'supply': 1.0}
auction = CompositeSystem(
    elements=[
        GroupElement('buyers', element=TraderElement, size=20, values=buyers),
        GroupElement('sellers', element=TraderElement, size=20, values=sellers),
    ],
    interactions=[
        MatchingInteraction(
            'auction',
            requesters='buyers',
            providers='sellers',
            rule=BatchMatchingSpec(),
            reserve=('bid', 'ask'),
            quantity=('demand', 'supply'),
            cap='money',
            transfers=(('money', -1), ('goods', 1)),
            phase='trade',
        ),
        ClerkInteraction('clerk'),
    ],
    time=DiscreteTime(schedule=PhasesScheduleSpec([PhaseSpec('quote'), PhaseSpec('trade')])),
)
trajectory = simulate(auction, until=5, random_state=0)
prices = [round(float(price), 3) for price in trajectory.values('auction.value')[1:]]
prices, float(trajectory.values('clerk.traded')[-1]), round(float(trajectory.values('clerk.fee')[-1]), 3)
trajectory.values('auction.volume').tolist()
```

`MatchingInteraction(label, requesters=..., providers=..., rule=...)` matches two groups.

- Requesters state the highest clearing value they accept in their `reserve` state, the quantity they want in
  `quantity`, and the most value they can give in `cap`. Providers state their lowest clearing value and the
  quantity they offer. A missing state counts as no reserve, one unit, and no cap.
- A name given as a pair, such as `reserve=('bid', 'ask')`, names the state of the requesters and of the providers.
- The rule is a frozen spec of `skcomplex.systems.general.matching` that names the matching algorithm and holds only
  its settings:
  - `RandomSearchSpec(tries, value)`, the default,
  - `PartnerSwitchSpec(sample, intensity, partner)`, partner choice with switching,
  - `QueueSpec(order)`,
  - `IterativeMatchingSpec(adjustment, iterations, tolerance)`, a clearing value adjusted until requests and offers
    balance,
  - `ContinuousMatchingSpec(arrival)`, where arriving offers match resting ones by limit and arrival,
  - `BatchMatchingSpec()`, one clearing value for all offers,
  - `StableMatchingSpec(requester_preference, provider_preference, value)`, stable matching by the `matching`
    library,
  - `PartnerMatchingSpec(candidates, choice, fallback, capacity, score, value)`, where requesters in random order
    pick among the candidates that still offer a quantity at a limit they accept.
- The rule may also be a function `rule(requesters, providers, rng, capacity)` that returns pairs, which match at the
  midpoint of both limits.
- The matching runs the algorithm of its rule on the offers of every round. Each match is a `MatchSpec`, whose `value`
  is the clearing value per unit.
- Every match lowers the quantities of both sides and moves the states of `transfers`. With `('money', -1)` the value
  times the quantity moves from the requester to the provider, and with `('goods', 1)` the quantity moves back, so the
  totals stay the same. `on_match(match, system)` adds any other effect.
- The matching reports `<label>.value`, the quantity-weighted clearing value, `<label>.volume`, and
  `<label>.unmatched`.

## Recipes of one field

A recipe that only one field uses lives in that field, under the name the field gives it. Its system classes are the
examples to read, and the gallery example of each system runs it.

- `TopplingInteraction` drives the members of a lattice and relaxes every member at or above a threshold, as in
  `SandpileSystem` of `skcomplex.systems.physics.criticality`. The section
  [Cascades](../build_a_custom_system/lattice_systems.md#cascades) writes a cascade with the core `CascadeLaw`.
- `PolicyRuleInteraction` sets parameters once per round of its phase from what happened in the round, as in
  `BrockHommesSystem` of `skcomplex.systems.economics.markets`. A plain `Interaction` that returns
  `RuleLaw(self.act, phase=...)` does the same, as the clerk of the market above shows.
- `SpacetimeSpace` and `GeodesicInteraction` move a particle along the geodesics of a metric, as in
  `SchwarzschildPrecessionSystem` of `skcomplex.systems.physics.celestial_mechanics`. They need the extra
  `relativity`, which installs EinsteinPy.
- `ZoneCouplingInteraction` couples agents by zones of repulsion, alignment, and attraction, as in `BoidsSystem` of
  `skcomplex.systems.biology.collective_behaviour`.
- `CellAdhesionInteraction` samples the regions of a cellular Potts model, as in `CellularPottsSystem` of
  `skcomplex.systems.biology.physiology`. `SequenceSamplingInteraction` samples the rows of sequences, as in
  `RNANeutralNetworksSystem` of `skcomplex.systems.biology.evolution`. `WilsonSamplingInteraction` samples the matrix
  links of a lattice gauge field, as in `LatticeYangMillsSystem` of `skcomplex.systems.physics.waves`.
- `OptimizationInteraction` moves a population of candidates through pymoo, as in `EvolutionaryOptimisersSystem` of
  `skcomplex.systems.computer_science.evolutionary_computation`.
- `SatisfiabilityInteraction` reveals logical constraints step by step to a PySAT solver, as in
  `PerceptronSatisfiabilitySystem` of `skcomplex.systems.computer_science.neural_networks`.
