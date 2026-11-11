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
    def __init__(self, label, visual=None, *, people=0.0):
        super().__init__(label, visual=visual)
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
        visual=None,
        *,
        x: Annotated[float, StateSpec()] = 1.0,
        r: Annotated[float, ParameterSpec()] = 0.5,
    ):
        super().__init__(label, visual=visual)
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
    def __init__(self, label, visual=None, *, x: Annotated[float, StateSpec(order=0)] = 0.0):
        super().__init__(label, visual=visual)
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
        visual=None,
        *,
        v: Annotated[float, StateSpec()] = 0.0,
        C: Annotated[float, ParameterSpec()] = 1e-3,
    ):
        super().__init__(label, visual=visual)
        self.v = v
        self.C = C


CapacitorElement('c1', v=5.0).get_params()
```

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
        visual=None,
        *,
        x: Annotated[float, StateSpec()] = 0.0,
        r: Annotated[float, ParameterSpec()] = 0.1,
    ):
        super().__init__(label, visual=visual)
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
    def __init__(self, label, visual=None, *, size=100):
        super().__init__(label, element=StockElement, size=size, visual=visual)


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

**How the engine runs it.** The compiler turns each operator of space into a stencil of the space: the Laplacian of a
lattice reads the neighbours of each site, and that of a network reads the edges. The boundary conditions replace the
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
    def __init__(self, label, visual=None, *, sites=50):
        super().__init__(label, element=DiffusionElement, space=LatticeSpace((sites,)), visual=visual)


RodElement('rod', sites=8).prepare().size_
```

See its [fact sheet](../../../components/elements.md#fieldelement)
and its [API entry][skcomplex.core.components.elements.FieldElement].

## WordElement

A `WordElement` holds a word of symbols that productions rewrite at every generation. In the algae of Lindenmayer `A`
becomes `AB` and `B` becomes `A`, so the length of the word runs through the Fibonacci numbers.

```python
from skcomplex.core.components.elements import WordElement
from skcomplex.core.components.laws import ProductionLaw
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate


class AlgaeElement(WordElement):
    law_classes = (ProductionLaw,)

    def _get_laws(self, namespace):
        return [ProductionLaw('A', 'AB'), ProductionLaw('B', 'A')]


trajectory = simulate(CompositeSystem(elements=[AlgaeElement('algae', axiom='A')]), until=5)
trajectory.get_values('algae.size').tolist()
```

The Thue and Morse sequence of symbolic dynamics comes from the same class, with `0` becoming `01` and `1` becoming
`10`.

```python continuation
class MorseElement(WordElement):
    law_classes = (ProductionLaw,)

    def _get_laws(self, namespace):
        return [ProductionLaw('0', '01'), ProductionLaw('1', '10')]


simulate(CompositeSystem(elements=[MorseElement('morse', axiom='0')]), until=3).get_values('morse.size').tolist()
```

**How the engine runs it.** Each generation applies every production to every matching symbol at once. The
rewriting solver advances one generation per unit of time and records the size of the word as the state `size`.

**How it combines.** Its productions are its own laws, so it usually runs alone. A `geometry` draws the word as line
segments, as a turtle does, for plants and fractal curves.

**Limits and errors.** The word grows at every generation, so `max_size` bounds it. A symbol outside `alphabet`, or a
word that would grow past `max_size`, raises a `RewritingError` during the run.

```python continuation
from skcomplex.core.errors import RewritingError

try:
    simulate(CompositeSystem(elements=[AlgaeElement('algae', axiom='A', max_size=4)]), until=5)
except RewritingError as error:
    message = str(error)
message
```

The message says that the word would reach 5 symbols at generation 3, above its maximum size 4. The fix is a larger
`max_size` or fewer generations.

**A field realization.** The subclasses above are already field realizations, the algae of biology and the sequence
of symbolic dynamics.

See its [fact sheet](../../../components/elements.md#wordelement)
and its [API entry][skcomplex.core.components.elements.WordElement].

## PatchElement

A `PatchElement` holds tiles, placed copies of basic shapes, that substitutions replace by smaller pieces at every
generation. Keeping the outer thirds of every interval builds the Cantor set.

```python
from skcomplex.core.components.elements import PatchElement
from skcomplex.core.components.laws import SubstitutionLaw
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.specs import AffineSpec, PrototileSpec


class CantorElement(PatchElement):
    law_classes = (SubstitutionLaw,)

    def _get_laws(self, namespace):
        thirds = [AffineSpec(scale=1 / 3), AffineSpec(scale=1 / 3, translation=(2 / 3, 0))]
        return [SubstitutionLaw('interval', [('interval', third) for third in thirds])]


cantor = CantorElement('cantor', prototiles=[PrototileSpec('interval', [(0, 0), (1, 0)])])
simulate(CompositeSystem(elements=[cantor]), until=3).get_values('cantor.size').tolist()
```

Replacing every triangle by its three corner triangles builds the Sierpinski triangle of geometry.

```python continuation
class SierpinskiElement(PatchElement):
    law_classes = (SubstitutionLaw,)

    def _get_laws(self, namespace):
        corners = [(0, 0), (1 / 2, 0), (1 / 4, 1 / 2)]
        pieces = [('triangle', AffineSpec(scale=1 / 2, translation=corner)) for corner in corners]
        return [SubstitutionLaw('triangle', pieces)]


triangle = SierpinskiElement('sierpinski', prototiles=[PrototileSpec('triangle', [(0, 0), (1, 0), (1 / 2, 1)])])
simulate(CompositeSystem(elements=[triangle]), until=3).get_values('sierpinski.size').tolist()
```

**How the engine runs it.** Each generation applies every substitution to every tile of its prototile at once, and
the rewriting solver records the number of tiles as the state `size`.

**How it combines.** Its substitutions are its own laws, so it usually runs alone, and its tiles are drawn as polygons.

**Limits and errors.** Every tile must name a prototile of the patch, and `prepare` raises a `RewritingError`
otherwise.

```python continuation
from skcomplex.core.errors import RewritingError

try:
    interval = PrototileSpec('interval', [(0, 0), (1, 0)])
    PatchElement('patch', tiles=[('square', None)], prototiles=[interval]).prepare()
except RewritingError as error:
    message = str(error)
message
```

The message names the unknown prototile `square`. The fix is a tile of the prototile `interval`.

**A field realization.** The two subclasses above are realizations of mathematics, and a tiling of physics, such as
the Penrose tiling of a quasicrystal, is written the same way.

See its [fact sheet](../../../components/elements.md#patchelement)
and its [API entry][skcomplex.core.components.elements.PatchElement].

## BaseElement

`BaseElement` is the abstract base of every element. A new kind of element that stores its members in a way of its
own, as `GroupElement` does, derives from it. Every other element derives from `Element`.

```python
from skcomplex.core.components.elements import BaseElement, Element, GroupElement

issubclass(Element, BaseElement), issubclass(GroupElement, BaseElement)
```

See its [fact sheet](../../../components/elements.md#baseelement)
and its [API entry][skcomplex.core.components.elements.BaseElement].
