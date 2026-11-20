# Elements

An element holds the states of a system, the numbers that change as the system runs. An interaction reads these states
and returns laws that change them, a space places the members of a group, and a system composes elements,
interactions, and a space into one model. [How the components fit](index.md) shows the families together, and the
[elements page](../../../components/elements.md) compares every element class and holds its fact sheet.

## Element

An `Element` holds values. A class of your own is a plain Python class whose arguments are its values, stored
unmodified, and `Element('label', values={...})` holds values without a class. The element returns no laws: a value
that some law changes is a state, and any other value is a parameter. Here a population of rabbits grows through an
interaction at the rate `r x`.

```python
from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import RateLaw
from skcomplex.core.systems import System
from skcomplex.core.types import Role


class Population(Element):
    def __init__(self, label, *, x=1.0):
        super().__init__(label)
        self.x = x


class Growth(Interaction):
    def __init__(self, label, *, target: Role, r=0.5):
        super().__init__(label)
        self.target = target
        self.r = r

    def _prepare(self, s):
        super()._prepare(s)
        x = s.target.x
        self.laws_ = [RateLaw(x.diff(s.t), s.r * x)]


rabbits = System(elements=[Population('rabbits', x=40.0)], interactions=[Growth('growth', target='rabbits')])
rabbits.prepare().simulate(until=1, dt=1.0).get_values('rabbits.x').round(1).tolist()
```

The rabbits grow from 40 to 65.9 in one unit of time, which is `40 exp(0.5)`. The same classes are a bank deposit that
earns interest at 5 percent.

```python continuation
deposit = System(elements=[Population('deposit', x=250.0)], interactions=[Growth('interest', target='deposit', r=0.05)])
deposit.prepare().simulate(until=1, dt=1.0).get_values('deposit.x').round(2).tolist()
```

**How the engine runs it.** `prepare` collects the laws of the interactions and reads which values they change. Those
values are the states, and the others are the parameters. The kind of a value comes from its Python type hint or its
value, so `1.0` is real, `1` an integer, and `'S'` a category.

```python continuation
deposit.states_, deposit.parameters_
```

The balance `deposit.x` is the state, and the rate `interest.r` is a parameter.

**How it combines.** Interactions add laws to its values, and the laws of several interactions on one value add up.
`size=N` makes `N` members of one element, and `space=` puts one member on every site of a space, so one class covers
one member, a group, and a field. An element bound to an interaction that gives it a rule is an agent.

**Limits and errors.** The allowed values of a value are written in `Annotated`, such as a size that is never
negative, and `prepare` raises a `ValidationError` that names a value outside them.

```python continuation
from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from skcomplex.core.errors import ValidationError


class Herd(Element):
    def __init__(self, label, *, x: Annotated[float, RealSet.closed_open(0, oo)] = 1.0):
        super().__init__(label)
        self.x = x


try:
    System(elements=[Herd('herd', x=-5.0)], interactions=[Growth('growth', target='herd')]).prepare()
except ValidationError as error:
    message = str(error)
message
```

The message reads "herd.x = -5.0 is not in its domain [0, +oo)." The fix is a starting size of zero or more.

**A field realization.** The catalog writes its elements as classes that name the values of one field, such as a
capacitor whose value is a voltage and whose capacitance belongs to it alone.

```python continuation
class Capacitor(Element):
    def __init__(self, label, *, v=0.0, C=1e-3):
        super().__init__(label)
        self.v = v
        self.C = C


Capacitor('c1', v=5.0).get_params()
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
from skcomplex.core.systems import System
from skcomplex.core.types import Role


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

## Quantity

A `Quantity` says what a value holds, such as money, water, or energy. It is written in `Annotated` beside the
allowed values of the value. The total of a quantity changes only through flows in and out of the system, so the
conservation monitor checks the total at every frame, and a plain rate law that changes the value is rejected. A flow
is a labelled `RateLaw` whose target maps each value to the sign of its change, and a flow out of the system names
only the value it leaves. Here a tank drains into the ground, and the monitor follows the water that is left.

```python
from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from skcomplex.core.components.elements import Element, Quantity
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import RateLaw
from skcomplex.core.engine.runtime import ConservationObserver
from skcomplex.core.systems import System
from skcomplex.core.types import Role


class Tank(Element):
    def __init__(self, label, level: Annotated[float, RealSet.closed_open(0, oo), Quantity('water')] = 10.0):
        super().__init__(label)
        self.level = level


class Drain(Interaction):
    def __init__(self, label, *, target: Role = 'tank', k: float = 0.5):
        super().__init__(label)
        self.target = target
        self.k = k

    def _prepare(self, s):
        super()._prepare(s)
        level = s.target.level
        self.laws_ = [RateLaw({level: -1}, s.k * level, label='drained')]


monitor = ConservationObserver()
tank = System(elements=[Tank('tank')], interactions=[Drain('drain')]).prepare()
tank.simulate(until=2.0, dt=1.0, observers=[monitor])
round(monitor.totals_['water'], 3)
```

About 3.679 of the 10 units of water are left after two units of time. A value given through `values=` has no
quantity, since it has no annotation.

See its [API entry][skcomplex.core.components.elements.Quantity].

## Period

A `Period` says that a value wraps around, such as an angle or a phase. It is written in `Annotated`, as
`theta: Annotated[float, Period(2 * pi)]`, and the value is brought back within one period around zero after every
step. The analyses read a periodic value as a phase, or as a heading when its element moves in a space, so no further
setting is needed. Here a clock turns at one radian per unit of time, and its hand stays on the circle.

```python
import math
from typing import Annotated

from skcomplex.core.components.elements import Element, Period
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import RateLaw
from skcomplex.core.systems import System
from skcomplex.core.types import Role


class Clock(Element):
    def __init__(self, label, theta: Annotated[float, Period(2 * math.pi)] = 0.0, omega: float = 1.0):
        super().__init__(label)
        self.theta = theta
        self.omega = omega


class Turn(Interaction):
    def __init__(self, label, *, target: Role = 'clock'):
        super().__init__(label)
        self.target = target

    def _prepare(self, s):
        super()._prepare(s)
        self.laws_ = [RateLaw(s.target.theta.diff(s.t), s.target.omega)]


clock = System(elements=[Clock('clock')], interactions=[Turn('turn')]).prepare()
clock.simulate(until=5.0, dt=1.0).get_values('clock.theta').round(3).tolist()
```

The angle grows to 3.0, then goes on at -2.283 instead of 4.0, the same point of the circle. Two values of an angle
that differ by a whole turn are one value, so the alignment of a group of such clocks reads them as phases.

```python continuation
from skcomplex.core.analysis import alignment

clocks = System(elements=[Clock('clock')], interactions=[Turn('turn')]).prepare()
alignment(clocks.simulate(until=5.0, dt=1.0), kind='mean-phase').phase.round(3).tolist()
```

The mean phase follows the hand around the circle. A period must be a positive number, so `Period(0)` is rejected
when the system is prepared.

See its [API entry][skcomplex.core.components.elements.Period].

## Member

A `Member` says that a value names a member of another group, such as the firm a worker works for or the bank of a
customer. It is written in `Annotated`, as `employer: Annotated[int, Member('firms')] = -1`. The value is the index of
the member in its group when the system starts, and -1 names none. During the run the value holds the id of that
member, and a rule reads the member with `agent.get_referenced('employer')`. When a member of the group is removed,
every value that names it becomes -1. Here four workers name their firms, and the first firm closes in the first step.

```python
from typing import Annotated

from skcomplex.core.components.elements import Element, Member
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import UpdateLaw
from skcomplex.core.systems import System
from skcomplex.core.types import Role


class Worker(Element):
    def __init__(self, label, employer: Annotated[object, Member('firms')] = -1, size=None):
        super().__init__(label, size=size)
        self.employer = employer


class Closing(Interaction):
    def __init__(self, label, *, target: Role = 'workers', firms: Role = 'firms'):
        super().__init__(label)
        self.target = target
        self.firms = firms

    def _prepare(self, s):
        super()._prepare(s)
        self.laws_ = [UpdateLaw(s, self.close)]

    def close(self, system):
        if system.count('firms') == 2:
            system.get_agents('firms')[0].remove()


workers = Worker('workers', employer=[0, 1, 0, -1], size=4)
firms = Element('firms', values={'wage': [1.0, 1.2]}, size=2)
trajectory = System(elements=[workers, firms], interactions=[Closing('closing')]).prepare().simulate(until=1)
[trajectory.get_agents('workers', step).values['employer'].tolist() for step in (0, 1)]
```

The firms have the ids 5 and 6, so the workers start with the employers 5, 6, 5, and -1. After the first firm closes,
its two workers have the employer -1, and the worker of the second firm keeps it.

See its [API entry][skcomplex.core.components.elements.Member].

## Tensor

A `Tensor` says that a value holds the entries of a square matrix of one symmetry, such as the order tensor of a liquid
crystal. It is written in `Annotated`, as `Q: Annotated[object, Tensor('traceless')] = 0.0`. The value holds only the
independent entries, `Q_xx` and `Q_xy` for a traceless matrix in two dimensions, and the matrix has one row per
dimension of the space of the element. Laws read the value as a Sage matrix. Here the order of every site of a small
lattice relaxes until `tr(Q^2)` is one half.

```python
from typing import Annotated

import numpy as np

from skcomplex.core.components.elements import Element, Tensor
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import RateLaw
from skcomplex.core.components.spaces import LatticeSpace
from skcomplex.core.systems import System
from skcomplex.core.types import Role


class Crystal(Element):
    def __init__(self, label, Q: Annotated[object, Tensor('traceless')] = 0.0, space=None):
        super().__init__(label, space=space)
        self.Q = Q


class Ordering(Interaction):
    def __init__(self, label, *, target: Role = 'crystal'):
        super().__init__(label)
        self.target = target

    def _prepare(self, s):
        super()._prepare(s)
        Q = s.target.Q
        self.laws_ = [RateLaw(Q, -((Q * Q).trace() - 0.5) * Q)]


crystal = Crystal('crystal', Q=np.tile([0.2, 0.1], (4, 1)), space=LatticeSpace((2, 2)))
trajectory = System(elements=[crystal], interactions=[Ordering('ordering')]).prepare().simulate(until=20.0, dt=10.0)
trajectory.get_values('crystal.Q')[-1].round(3).tolist()
```

Every site keeps its axis, the ratio of its two entries, and reaches the order where `2 (Q_xx^2 + Q_xy^2)` is one
half. A symmetry other than `'full'`, `'symmetric'`, `'antisymmetric'`, or `'traceless'` is rejected when the system
is prepared.

See its [API entry][skcomplex.core.components.elements.Tensor].

## BaseElement

`BaseElement` is the abstract base of every element. A new kind of element that stores its members in a way of its
own derives from it. Every other element derives from `Element`.

```python
from skcomplex.core.components.elements import BaseElement, Element

issubclass(Element, BaseElement)
```

See its [fact sheet](../../../components/elements.md#baseelement)
and its [API entry][skcomplex.core.components.elements.BaseElement].
