# Elements

An element holds the states of a system, the numbers that change as the system runs. An interaction reads these states
and returns laws that change them, a space places the members of a group, and a system composes elements,
interactions, and a space into one model. [How the components fit](index.md) shows the families together, and the
[elements page](../../../components/elements.md) compares every element class and holds its fact sheet.

## DeclaredElement

A `DeclaredElement` takes its states and parameters as data, so it needs no class of its own. Each variable is a
`DeclaredVariableSpec` with an initial value and a spec. Here one compartment holds 100 infected people, and a
`RateInteraction` makes them recover at a rate of 0.1 a day.

```python
from skcomplex.core.components.elements import DeclaredElement
from skcomplex.core.components.interactions import RateInteraction
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.specs import DeclaredVariableSpec, StateSpec

infected = DeclaredElement('infected', variables={'x': DeclaredVariableSpec(100.0, StateSpec())})
recovery = RateInteraction('recovery', target='infected', state='x', rate=lambda s: -0.1 * s.target.x)
epidemic = CompositeSystem(elements=[infected], interactions=[recovery])
simulate(epidemic, until=10, dt=5.0).get_values('infected.x').round(1).tolist()
```

The infected fall from 100 to 60.7 after five days and to 36.8 after ten, the exponential decay `100 exp(-0.1 t)`.

The same class holds the balance of a bank account. Interest at 5 percent a year grows it continuously.

```python continuation
account = DeclaredElement('account', variables={'x': DeclaredVariableSpec(250.0, StateSpec())})
interest = RateInteraction('interest', target='account', state='x', rate=lambda s: 0.05 * s.target.x)
savings = CompositeSystem(elements=[account], interactions=[interest])
simulate(savings, until=1, dt=1.0).get_values('account.x').round(2).tolist()
```

The balance reaches 262.82 after one year, which is `250 exp(0.05)`.

**How the engine runs it.** The element returns no laws of its own. The compiler reads each declared variable as it
reads a state that a subclass annotates, so the rate laws of the interactions above form an ordinary differential
equation that `ScipyODESolver` integrates. A state of order 2 also gets a companion state for its first derivative,
named `x_d1`, which `get_value` reads and `set_params` changes.

```python continuation
mass = DeclaredElement('mass', variables={'x': DeclaredVariableSpec(1.0, StateSpec(order=2))})
mass.get_value('x'), mass.set_params(x_d1=0.5).get_value('x_d1')
```

**How it combines.** It fills any role whose domains include those of its states. Its usual partners are
`RateInteraction` for flows, `TransferInteraction` for amounts that move between compartments, and `JumpInteraction`
for random events. Systems that learn their variables while they build, such as a network read from a table of
reactions, build one `DeclaredElement` per entry.

**Limits and errors.** The element holds one value per variable, not a member per site, so a lattice or a network
needs a `GroupElement`. Every name must be an identifier that no argument of the constructor takes, and `prepare`
raises a `ValidationError` otherwise.

```python continuation
from skcomplex.core.errors import ValidationError

try:
    DeclaredElement('cell', variables={'2x': DeclaredVariableSpec(1.0, StateSpec())}).prepare()
except ValidationError as error:
    message = str(error)
message
```

The message reads "Element 'cell' declares '2x'. Use an identifier that no argument takes." The fix is a name such
as `x2`.

**A field realization.** A field names the element in its own words and derives the variables from its arguments.
The subclass overrides the hook `_declare_variables`, which `DeclaredElement` reads in place of `variables`.

```python continuation
from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet


class CompartmentElement(DeclaredElement):
    def __init__(self, label, *, people=0.0):
        super().__init__(label)
        self.people = people

    def _declare_variables(self):
        return {'x': DeclaredVariableSpec(self.people, StateSpec(domain=RealSet.closed_open(0, oo)))}


CompartmentElement('susceptible', people=990.0).get_value('x')
```

See its [fact sheet](../../../components/elements.md#declaredelement)
and its [API entry][skcomplex.core.components.elements.DeclaredElement].

## Element

An `Element` is the base of an element class of your own. The class declares each state and parameter as an argument
of its `__init__`, annotated with `StateSpec` or `ParameterSpec`, and may return laws of its own from `_get_laws`.
Here a population grows at a rate proportional to its size.

```python
from typing import Annotated

from skcomplex.core.components.elements import Element
from skcomplex.core.components.laws import RateLaw
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.specs import ParameterSpec, StateSpec


class GrowthElement(Element):
    law_classes = (RateLaw,)

    def __init__(
        self,
        label,
        *,
        x: Annotated[float, StateSpec()] = 1.0,
        r: Annotated[float, ParameterSpec()] = 0.5,
    ):
        super().__init__(label)
        self.x = x
        self.r = r

    def _get_laws(self, namespace):
        return [RateLaw(namespace.x, namespace.r * namespace.x)]


rabbits = GrowthElement('rabbits', x=40.0, r=0.5)
simulate(CompositeSystem(elements=[rabbits]), until=1, dt=1.0).get_values('rabbits.x').round(1).tolist()
```

The same class is a bank deposit that earns interest at 5 percent.

```python continuation
deposit = GrowthElement('deposit', x=250.0, r=0.05)
simulate(CompositeSystem(elements=[deposit]), until=1, dt=1.0).get_values('deposit.x').round(2).tolist()
```

**How the engine runs it.** `prepare` reads the annotations to find the states, parameters, and their specs. At
compile time the element returns its laws for a namespace whose attributes are the symbols of its variables, and the
class attribute `law_classes` names their kinds, so `get_tags` knows them before a run.

```python continuation
sorted(rabbits.get_tags().laws), sorted(rabbits.get_tags().domains)
```

**How it combines.** An element with laws of its own runs alone, and interactions add laws to its states. As the
member class of a `GroupElement` or a `FieldElement` it runs on every site of a space. An element whose laws are agent
rules is an agent, and one that declares `ports` joins others through a `ConnectionInteraction`.

**Limits and errors.** A spec must be valid for its state. An order of 0, for example, is neither a whole number of
at least 1 nor a fraction in `(0, 1)`, and `prepare` raises a `ValidationError`.

```python continuation
from skcomplex.core.errors import ValidationError


class BrokenElement(Element):
    def __init__(self, label, *, x: Annotated[float, StateSpec(order=0)] = 0.0):
        super().__init__(label)
        self.x = x


try:
    BrokenElement('broken').prepare()
except ValidationError as error:
    message = str(error)
message
```

The message says that the order 0 is neither a whole number of at least 1 nor a real number in `(0, 1)`. The fix is
`StateSpec(order=1)`, the default, for a state with a first derivative.

**A field realization.** The catalog writes its elements as subclasses that rename the states for one field, such as
a capacitor whose state is a voltage.

```python continuation
class CapacitorElement(Element):
    def __init__(
        self,
        label,
        *,
        v: Annotated[float, StateSpec()] = 0.0,
        C: Annotated[float, ParameterSpec()] = 1e-3,
    ):
        super().__init__(label)
        self.v = v
        self.C = C


CapacitorElement('c1', v=5.0).get_params()
```

**Members on the links of a network.** `space=` puts one member of an element on every node of a network, and
`on='edges'` puts one on every link instead, such as the width of every road between three cities. An interaction that
runs `along` the edges reads the member of each link as `s.edge`, beside the two ends `s.source` and `s.target`.

```python
import networkx as nx

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import RateLaw
from skcomplex.core.components.spaces import NetworkSpace
from skcomplex.core.specs import Role
from skcomplex.core.systems import System


class Exchange(Interaction):
    def __init__(self, label, *, among: Role = 'cities', along=True):
        super().__init__(label, among=among, along=along)

    def _prepare(self, s):
        super()._prepare(s)
        flow = s.edge.w * (s.source.x - s.target.x)
        self.laws_ = [RateLaw(s.target.x.diff(s.t), flow), RateLaw(s.source.x.diff(s.t), -flow)]


roads = NetworkSpace(nx.path_graph(3))
cities = Element('cities', values={'x': [90.0, 0.0, 0.0]}, space=roads)
links = Element('links', values={'w': [2.0, 0.5]}, space=roads, on='edges')
system = System(elements=[cities, links], interactions=[Exchange('exchange')]).prepare()
system.simulate(until=1.0, dt=0.5).get_values('cities.x')[-1].round(1).tolist()
```

After one unit of time the three cities hold 41.3, 36.4, and 12.3, since the wide first road carries four times the
flow of the second. `on='hyperedges'` puts one member on every hyperedge of a network, which an interaction with
`along='hyperedges'` reads as `s.hyperedge`, beside the members `s.member` and `s.others` of the hyperedge.

**Groups that grow.** A group has no capacity to give. When rules or topology requests add members, the engine
doubles the rows of the group as a Python list grows and goes on from the step where the group filled, so a network
may start with one node and the trajectory holds one unbroken run. Here a network grows to 500 nodes by preferential
attachment, each new node linking to a node drawn by its degree plus one.

```python continuation
from skcomplex.core.components.laws import TopologyRequestSpec, UpdateLaw
from skcomplex.core.components.laws.utils import neighbors


def degree(node):
    return neighbors(1) + 1


class Attach(Interaction):
    def __init__(self, label, *, among: Role = 'nodes', along=None):
        super().__init__(label, among=among, along=along)

    def _prepare(self, s):
        super()._prepare(s)
        self.laws_ = [UpdateLaw(TopologyRequestSpec('add-node', edges=1, one=True, toward=degree), 1.0)]


nodes = Element('nodes', values={'k': 0.0}, space=NetworkSpace(nx.empty_graph(1)))
web = System(elements=[nodes], interactions=[Attach('attach')]).prepare()
grown = web.simulate(until=499, random_state=0).get_network(-1).get_graph()
grown.number_of_nodes(), max(degree for _, degree in grown.degree())
```

The network holds 500 nodes, and its largest hub has 27 links, far more than a network whose new nodes link uniformly
would give.

See its [fact sheet](../../../components/elements.md#element)
and its [API entry][skcomplex.core.components.elements.Element].

## GroupElement

A `GroupElement` holds many members of one element class, with one row per member in every state. Here three herds
of grazing animals start at different sizes.

```python
from typing import Annotated

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.interactions import RateInteraction
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.specs import ParameterSpec, StateSpec


class StockElement(Element):
    def __init__(
        self,
        label,
        *,
        x: Annotated[float, StateSpec()] = 0.0,
        r: Annotated[float, ParameterSpec()] = 0.1,
    ):
        super().__init__(label)
        self.x = x
        self.r = r


herds = GroupElement('herds', element=StockElement, size=3, values={'x': [40.0, 25.0, 60.0]})
herds.prepare().size_
```

The same class holds the assets of a portfolio, each with its own yearly return.

```python continuation
holdings = {'x': [100.0, 50.0], 'r': [0.05, 0.02]}
assets = GroupElement('assets', element=StockElement, size=2, values=holdings)
returns = RateInteraction('returns', target='assets', state='x', rate=lambda s: s.target.r * s.target.x)
portfolio = CompositeSystem(elements=[assets], interactions=[returns])
simulate(portfolio, until=1, dt=1.0).get_values('assets.x').round(2).tolist()
```

**How the engine runs it.** The laws of the member class and of the interactions are written for one member. The
compiler runs them for every member at once as arrays, so a group of a thousand members costs one vectorised call per
evaluation.

**How it combines.** With `on='sites'` the members sit on the sites of a lattice or the nodes of a network, and with
`on='edges'` or `on='hyperedges'` on the links of a network. Neighbour interactions, such as `CouplingInteraction`,
then read the members around each member. With `variable=True`, laws add and remove members during a run.

```python continuation
import networkx as nx

from skcomplex.core.components.spaces import NetworkSpace

lines = GroupElement('lines', element=StockElement, space=NetworkSpace(nx.cycle_graph(5)), on='edges')
lines.prepare().size_
```

**Limits and errors.** Every member shares the class of the group. Each value holds one row per member or one value
for all, and `prepare` raises a `ValidationError` for any other count.

```python continuation
from skcomplex.core.errors import ValidationError

try:
    GroupElement('herds', element=StockElement, size=3, values={'x': [40.0, 25.0]}).prepare()
except ValidationError as error:
    message = str(error)
message
```

The message says that the group has 2 values of `x` for 3 members. The fix is three values, or one value shared by
every member.

**A field realization.** A field fixes the member class and the space of its group in a subclass, such as the
sites of a forest.

```python continuation
class ForestElement(GroupElement):
    def __init__(self, label, *, size=100):
        super().__init__(label, element=StockElement, size=size)


ForestElement('forest', size=4).prepare().size_
```

See its [fact sheet](../../../components/elements.md#groupelement)
and its [API entry][skcomplex.core.components.elements.GroupElement].

## FieldElement

A `FieldElement` holds one member on every site of a space and gives its states the operators of space, such as the
Laplacian, and boundary conditions. Here heat spreads through a plate of 10 by 20 sites with insulated edges.

```python
from typing import Annotated

from skcomplex.core.components.elements import Element, FieldElement
from skcomplex.core.components.laws import RateLaw
from skcomplex.core.components.laws.utils import laplacian
from skcomplex.core.components.spaces import LatticeSpace
from skcomplex.core.specs import ParameterSpec, StateSpec


class DiffusionElement(Element):
    law_classes = (RateLaw,)

    def __init__(
        self,
        label,
        *,
        u: Annotated[float, StateSpec()] = 0.0,
        D: Annotated[float, ParameterSpec()] = 1.0,
    ):
        super().__init__(label)
        self.u = u
        self.D = D

    def _get_laws(self, namespace):
        return [RateLaw(namespace.u, namespace.D * laplacian(namespace.u))]


walls = {'u': 'zero-flux'}
plate = FieldElement('plate', element=DiffusionElement, space=LatticeSpace((10, 20)), boundary=walls)
plate.prepare().size_
```

The same law spreads a pollutant along a channel of three reaches until each reach holds the same amount.

```python continuation
import networkx as nx

from skcomplex.core.components.spaces import NetworkSpace
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate

channel = NetworkSpace(nx.path_graph(3))
river = FieldElement('river', element=DiffusionElement, space=channel, values={'u': [9.0, 0.0, 0.0]})
simulate(CompositeSystem(elements=[river]), until=10, dt=10.0).get_values('river.u').round(2).tolist()
```

On a field with `discretization='finite-volume'`, a rate `-divergence(flux)` is a conservation law. The density of
cars on a road has the flux `ρ (1 - ρ)` of Lighthill, Whitham, and Richards. Here light traffic at density 0.2 runs into
a jam at density 0.9.

```python continuation
import numpy as np

from skcomplex.core.components.spaces import FiniteVolumeSpec
from skcomplex.core.components.laws.utils import divergence
from skcomplex.core.engine.solvers import SSPRKSolver


class TrafficElement(Element):
    law_classes = (RateLaw,)

    def __init__(self, label, *, rho: Annotated[float, StateSpec()] = 0.0):
        super().__init__(label)
        self.rho = rho

    def _get_laws(self, namespace):
        return [RateLaw(namespace.rho, -divergence([namespace.rho * (1 - namespace.rho)]))]


density = np.where(np.arange(50) < 25, 0.2, 0.9)
road = FieldElement(
    'road',
    element=TrafficElement,
    space=LatticeSpace(50, spacing=0.1),
    values={'rho': density},
    discretization='finite-volume',
    scheme=FiniteVolumeSpec(riemann='godunov'),
)
cars = simulate(CompositeSystem(elements=[road]), until=1, dt=1.0, solver=SSPRKSolver()).get_values('road.rho')
cars.sum(axis=1).round(6).tolist(), int(np.argmax(np.diff(cars[-1])))
```

The number of cars stays 27.5, and the back of the jam moves from cell 24 to cell 23. The shock travels backwards at
the speed `1 - 0.2 - 0.9 = -0.1`, one cell of length 0.1 in one unit of time. The `FiniteVolumeSpec` of the field
chooses the scheme: the slope limiter `limiter` of the reconstruction in each cell, the numerical flux `riemann`
between cells, here the exact Godunov flux, and an optional wave speed `speed`. Without a scheme the field takes the
minmod limiter and the Rusanov flux, and the wave speeds come from the derivative of the flux.

**How the engine runs it.** The compiler turns each operator of space into a stencil of the space: the Laplacian of a
lattice reads the neighbours of each site, and that of a network reads the edges. On a finite-volume field the
divergence of a flux becomes the differences of the numerical flux between cells, which `SSPRKSolver` steps by the CFL
condition while it keeps the total. The boundary conditions replace the
values beyond the edges of a lattice. The result is one large system of ordinary differential equations, which a
solver for stiff fields can integrate.

**How it combines.** It is a `GroupElement` with one member per site, so every interaction of groups applies.
`ConservationInteraction` and `GradientInteraction` read its operators directly.

**Limits and errors.** A field needs a space with sites, and it cannot add or remove members. Its values follow the
rules of a group, one per site or one for all.

```python continuation
from skcomplex.core.errors import ValidationError

try:
    FieldElement('river', element=DiffusionElement, space=channel, values={'u': [9.0, 0.0]}).prepare()
except ValidationError as error:
    message = str(error)
message
```

The message says that the group has 2 values of `u` for 3 members. The fix is three values, one per reach.

**A field realization.** A field names the field and fixes its member class, such as the temperature of a rod.

```python continuation
class RodElement(FieldElement):
    def __init__(self, label, *, sites=50):
        super().__init__(label, element=DiffusionElement, space=LatticeSpace((sites,)))


RodElement('rod', sites=8).prepare().size_
```

See its [fact sheet](../../../components/elements.md#fieldelement)
and its [API entry][skcomplex.core.components.elements.FieldElement].

## BaseElement

`BaseElement` is the abstract base of every element. A new kind of element that stores its members in a way of its
own, as `GroupElement` does, derives from it. Every other element derives from `Element`.

```python
from skcomplex.core.components.elements import BaseElement, Element, GroupElement

issubclass(Element, BaseElement), issubclass(GroupElement, BaseElement)
```

See its [fact sheet](../../../components/elements.md#baseelement)
and its [API entry][skcomplex.core.components.elements.BaseElement].
