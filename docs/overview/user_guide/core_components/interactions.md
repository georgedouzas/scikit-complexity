# Interactions

An interaction changes the states of elements. It binds elements through roles, such as `target` and `source`, reads
their states, and returns laws, the equations or rules that a solver runs. It holds no state of its own. An element
holds the states, a space says which members neighbour each other, and a system composes elements, interactions, and a
space into one model. [How the parts fit](index.md) shows the families together, and the
[interactions parts page](../../../parts/interactions.md) compares every interaction class and holds its fact sheet.

## Interaction

An `Interaction` is the base of an interaction class of your own. The class declares each role and parameter as an
argument of its `__init__`, annotated with `RoleSpec` or `ParameterSpec`, and returns its laws from `_get_laws`. Here
susceptible people become infected at a rate proportional to the product of the two compartments, the mass action of
epidemiology.

```python
from typing import Annotated

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import RateLaw
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.specs import ParameterSpec, RoleSpec, StateSpec


class AmountElement(Element):
    def __init__(self, label, visual=None, *, x: Annotated[float, StateSpec()] = 0.0):
        super().__init__(label, visual=visual)
        self.x = x


class ConversionInteraction(Interaction):
    law_classes = (RateLaw,)

    def __init__(
        self,
        label,
        *,
        target: Annotated[str, RoleSpec()],
        source: Annotated[str, RoleSpec()],
        a: Annotated[float, ParameterSpec()] = 0.01,
    ):
        super().__init__(label)
        self.target = target
        self.source = source
        self.a = a

    def _get_laws(self, namespace):
        flow = namespace.a * namespace.target.x * namespace.source.x
        return [RateLaw(namespace.target.x, -flow), RateLaw(namespace.source.x, flow)]


people = [AmountElement('susceptible', x=99.0), AmountElement('infected', x=1.0)]
infection = ConversionInteraction('infection', target='susceptible', source='infected', a=0.01)
epidemic = CompositeSystem(elements=people, interactions=[infection])
simulate(epidemic, until=10, dt=5.0).get_values('infected.x').round(1).tolist()
```

The infected grow from 1 to 60.0 after five days and to 99.6 after ten. This is the logistic curve of a simple
epidemic, in which every person ends infected.

The same class describes autocatalysis in chemistry, where a molecule of `B` turns a molecule of `A` into a second
`B`, as in the reaction `A + B -> 2B`.

```python continuation
species = [AmountElement('A', x=2.0), AmountElement('B', x=0.1)]
autocatalysis = ConversionInteraction('autocatalysis', target='A', source='B', a=1.0)
reactor = CompositeSystem(elements=species, interactions=[autocatalysis])
simulate(reactor, until=4, dt=2.0).get_values('B.x').round(3).tolist()
```

The concentration of `B` rises from 0.1 to 1.615 and then to 2.091, close to the total of 2.1 that the reaction
conserves.

**How the engine runs it.** `prepare` reads the annotations to find the roles and parameters. At compile time the
interaction returns its laws for a namespace whose attribute `target` holds the symbols of the states of the element
that role binds, and whose attribute `a` is the symbol of the parameter. The class attribute `law_classes` names the
kinds of its laws, so `get_tags` knows them before a run. Rate laws on one state add up, and the system of ordinary
differential equations goes to `ScipyODESolver`.

```python continuation
sorted(infection.get_tags().laws), sorted(infection.get_requirements().roles)
```

The interaction emits rate laws and binds the roles `source` and `target`.

**How it combines.** A role binds an element, a group, or a member of a group. When it binds a group, the laws run
once for every member of the group, as arrays. An interaction of this kind sits beside the others of a system, and
its laws add to the rates they give to the same state.

**Limits and errors.** The laws run once per bound element or member, so an interaction that runs along the edges of
a network or between close members derives from `NeighborInteraction` or `PairInteraction`. Every role must name an
element or a group of the system, and `simulate` raises a `RoleError` otherwise.

```python continuation
from skcomplex.core.errors import RoleError

wrong = ConversionInteraction('infection', target='susceptible', source='sick')
try:
    simulate(CompositeSystem(elements=people, interactions=[wrong]), until=1)
except RoleError as error:
    message = str(error)
message
```

The message says that the role `source` is bound to `'sick'`, which is not the path of an element or a group. The fix
is the path `'infected'`.

**A field realization.** A field names the roles in its own words, such as foxes that eat rabbits and turn a part of
them into new foxes.

```python continuation
class PredationInteraction(Interaction):
    law_classes = (RateLaw,)

    def __init__(
        self,
        label,
        *,
        prey: Annotated[str, RoleSpec()],
        predator: Annotated[str, RoleSpec()],
        a: Annotated[float, ParameterSpec()] = 0.02,
        b: Annotated[float, ParameterSpec()] = 0.01,
    ):
        super().__init__(label)
        self.prey = prey
        self.predator = predator
        self.a = a
        self.b = b

    def _get_laws(self, namespace):
        meetings = namespace.prey.x * namespace.predator.x
        return [
            RateLaw(namespace.prey.x, -namespace.a * meetings),
            RateLaw(namespace.predator.x, namespace.b * meetings),
        ]


PredationInteraction('hunting', prey='rabbits', predator='foxes').get_params()
```

Its parameters are the two roles, the label, and the rates `a` and `b`.

See its [fact sheet](../../../parts/interactions.md#interaction)
and its [API entry][skcomplex.core.components.interactions.Interaction].

## RateInteraction

A `RateInteraction` adds a rate, and optionally a noise, to the time derivative of one state, so it needs no class of
its own. Here a capacitor of time constant 2 seconds charges through a resistor toward a source of 5 volts.

```python
from typing import Annotated

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import RateInteraction
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.specs import StateSpec


class AmountElement(Element):
    def __init__(self, label, visual=None, *, x: Annotated[float, StateSpec()] = 0.0):
        super().__init__(label, visual=visual)
        self.x = x


charging = RateInteraction('charging', target='capacitor', rate=lambda s: (5.0 - s.target.x) / 2.0)
circuit = CompositeSystem(elements=[AmountElement('capacitor')], interactions=[charging])
simulate(circuit, until=4, dt=2.0).get_values('capacitor.x').round(2).tolist()
```

The voltage reaches 3.16 after 2 seconds and 4.32 after 4, which is `5 (1 - exp(-t / 2))`.

The same class with a noise gives the geometric Brownian motion of a stock price, with a drift of 5 percent and a
volatility of 20 percent a year. A noise is a random term `noise(namespace) dW`, where `dW` is the increment of a
Wiener process, the random walk of continuous time.

```python continuation
price = RateInteraction('price', target='stock', rate=lambda s: 0.05 * s.target.x, noise=lambda s: 0.2 * s.target.x)
market = CompositeSystem(elements=[AmountElement('stock', x=100.0)], interactions=[price])
round(float(simulate(market, until=1, dt=1.0, random_state=0).get_values('stock.x')[-1]), 1)
```

The price ends the year at 126.2 in this run. Another seed draws another path, and a study averages many of them.

**How the engine runs it.** The interaction returns a rate law, and a noise law when it has a noise. Without noise the
laws form ordinary differential equations, which `ScipyODESolver` integrates. With noise they form stochastic
differential equations, which `EulerMaruyamaSolver` integrates. On a state of order 2 the rate adds to the second
derivative, so it acts as an acceleration.

```python continuation
sorted(charging.get_tags().laws), sorted(price.get_tags().laws)
```

The first interaction emits a rate law, and the second a rate law and a noise law.

**How it combines.** The role `target` binds an element or a group, and `sources` names the elements the rate reads
without changing them. Rates on one state add, so a birth rate and a death rate can be two interactions. A `divisor`,
such as a capacitance or a mass, divides the rate and the noise.

**Limits and errors.** A rate reads the target and the sources only, with no neighbours, so a coupling between members
of a group needs `CouplingInteraction` or `NeighborInteraction`. The stochastic calculus of the noise must be `'ito'`,
`'stratonovich'`, or `'anti-ito'`, and `prepare` raises a `ValidationError` for any other name.

```python continuation
from skcomplex.core.errors import ValidationError

try:
    RateInteraction('price', target='stock', noise=0.2, convention='itoh').prepare()
except ValidationError as error:
    message = str(error)
message
```

The message reads "Unknown noise convention 'itoh'. Use one of ito, stratonovich, anti-ito." The fix is
`convention='ito'`, the default.

**A field realization.** A field fixes the rate and names its parameter, such as the recovery of infected people at a
rate `gamma`. The rate reads the parameter from the namespace, so `set_params` changes it.

```python continuation
from skcomplex.core.specs import ParameterSpec


class RecoveryInteraction(RateInteraction):
    def __init__(self, label, *, target, gamma: Annotated[float, ParameterSpec()] = 0.1):
        super().__init__(label, target=target, rate=lambda s: -s.gamma * s.target.x)
        self.gamma = gamma


recovery = RecoveryInteraction('recovery', target='infected', gamma=0.2)
ward = CompositeSystem(elements=[AmountElement('infected', x=100.0)], interactions=[recovery])
simulate(ward, until=5, dt=5.0).get_values('infected.x').round(1).tolist()
```

The infected fall from 100 to 36.8 in five days, which is `100 exp(-0.2 t)`.

See its [fact sheet](../../../parts/interactions.md#rateinteraction)
and its [API entry][skcomplex.core.components.interactions.RateInteraction].

## TransferInteraction

A `TransferInteraction` moves an amount out of a state of one holder and into a state of another, so what one loses
the other gains. Here a tenant pays a rent of 100 a month to a landlord.

```python
from typing import Annotated

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import TransferInteraction
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.specs import StateSpec


class AccountElement(Element):
    def __init__(self, label, visual=None, *, cash: Annotated[float, StateSpec()] = 0.0):
        super().__init__(label, visual=visual)
        self.cash = cash


rent = TransferInteraction('rent', source='tenant', target='landlord', state='cash', amount=100.0)
accounts = [AccountElement('tenant', cash=500.0), AccountElement('landlord')]
trajectory = simulate(CompositeSystem(elements=accounts, interactions=[rent]), until=3, dt=1.0)
trajectory.get_values('tenant.cash').round(1).tolist(), trajectory.get_values('landlord.cash').round(1).tolist()
```

The tenant falls from 500 to 200 in three months, and the landlord rises from 0 to 300.

The same class lets water flow from an upper reservoir into a lower one, at a rate of half the upper level.

```python continuation
class ReservoirElement(Element):
    def __init__(self, label, visual=None, *, level: Annotated[float, StateSpec()] = 0.0):
        super().__init__(label, visual=visual)
        self.level = level


half = lambda s: s.source.level / 2
outflow = TransferInteraction('outflow', source='upper', target='lower', state='level', amount=half)
reservoirs = [ReservoirElement('upper', level=4.0), ReservoirElement('lower')]
trajectory = simulate(CompositeSystem(elements=reservoirs, interactions=[outflow]), until=2, dt=2.0)
trajectory.get_values('upper.level').round(3).tolist(), trajectory.get_values('lower.level').round(3).tolist()
```

After two hours the upper level falls to 1.472, which is `4 exp(-1)`, and the lower rises to 2.528, so their sum stays
4.

**How the engine runs it.** In its symbolic form the interaction returns one transfer law per leg. A transfer law is
one outflow of the source and one inflow of the target, and its amount is a rate in continuous time. The trajectory
records each flow under its name, such as `rent.flow_0`.

```python continuation
trajectory.names
```

The names hold the two levels and the flow `outflow.flow_0`.

**How it combines.** The roles `source` and `target` bind two elements or two groups, or one holder twice for a
transfer between its own states, such as susceptible to infected. `legs` move further amounts in the same event, and
`Source()` and `Sink()` stand for the environment outside the system. With a `rate`, the amount moves in random jumps.
In the agent form, `pairing` pairs the members of two groups, as in a market.

**Limits and errors.** A transfer conserves the amount it moves, so a quantity that is created or destroyed needs a
leg from `Source()` or into `Sink()`, or a `RateInteraction`. The form must be `'symbolic'` or `'agent'`, and `prepare`
raises a `ValidationError` for any other name.

```python continuation
from skcomplex.core.errors import ValidationError

try:
    TransferInteraction('rent', source='tenant', target='landlord', state='cash', amount=100.0, form='agents').prepare()
except ValidationError as error:
    message = str(error)
message
```

The message reads "Unknown form 'agents' of 'rent'. Use one of symbolic, agent." The fix is `form='agent'`.

**A field realization.** A field fixes the state and names the amount, such as a payment of a fixed sum from a payer
to a payee.

```python continuation
class PaymentInteraction(TransferInteraction):
    def __init__(self, label, *, payer, payee, value=100.0):
        super().__init__(label, source=payer, target=payee, state='cash', amount=value)
        self.payer = payer
        self.payee = payee
        self.value = value


PaymentInteraction('wages', payer='firm', payee='worker', value=2000.0).amount
```

The wages flow at 2000.0 per unit of time.

See its [fact sheet](../../../parts/interactions.md#transferinteraction)
and its [API entry][skcomplex.core.components.interactions.TransferInteraction].

## JumpInteraction

A `JumpInteraction` is a random event that changes states by whole amounts. Here births and deaths change a population
of 20 one animal at a time, at rates of 0.5 and 0.4 per animal a year.

```python
from typing import Annotated

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import JumpInteraction
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.specs import StateSpec


class AmountElement(Element):
    def __init__(self, label, visual=None, *, x: Annotated[float, StateSpec()] = 0.0):
        super().__init__(label, visual=visual)
        self.x = x


births = JumpInteraction('births', target='population', changes={'x': 1}, rate=lambda s: 0.5 * s.target.x)
deaths = JumpInteraction('deaths', target='population', changes={'x': -1}, rate=lambda s: 0.4 * s.target.x)
population = CompositeSystem(elements=[AmountElement('population', x=20.0)], interactions=[births, deaths])
simulate(population, until=10, dt=5.0, random_state=0).get_values('population.x').tolist()
```

The population falls to 17 after five years and to 13 after ten in this run. Its mean grows as `20 exp(0.1 t)`, but
a small population can drift down, and another seed draws another path.

The same class describes customers who join a queue twice a minute and each leave after one minute on average.

```python continuation
join = JumpInteraction('join', target='queue', changes={'x': 1}, rate=lambda s: 2.0)
leave = JumpInteraction('leave', target='queue', changes={'x': -1}, rate=lambda s: s.target.x)
shop = CompositeSystem(elements=[AmountElement('queue')], interactions=[join, leave])
lengths = simulate(shop, until=200, dt=1.0, random_state=0).get_values('queue.x')
bool((lengths >= 0).all()), round(float(lengths.mean()), 1)
```

The length never falls below 0, and its mean over 200 minutes is 1.9, close to the value 2 of queueing theory.

**How the engine runs it.** The interaction returns a jump law, whose propensity is the rate. `GillespieSolver` runs
the stochastic simulation algorithm of Gillespie: it draws the time of the next event from the total propensity,
picks the event in proportion to its propensity, and applies its changes.

```python continuation
sorted(join.get_tags().laws)
```

The interaction emits a jump law.

**How it combines.** The role `target` binds an element or a group. Several jump interactions on one target form
one jump process. With `probability` in place of `rate` the event fires once per step of discrete time, and with
`waiting` it fires after a drawn waiting time, as in a renewal process.

**Limits and errors.** The changes are fixed amounts, so a change that depends on the states needs a `MapInteraction`
or a `TransitionInteraction`. An event has exactly one clock, and `prepare` raises a `ValidationError` for none or
several.

```python continuation
from skcomplex.core.errors import ValidationError

try:
    JumpInteraction('join', target='queue', changes={'x': 1}, rate=lambda s: 2.0, probability=lambda s: 0.1).prepare()
except ValidationError as error:
    message = str(error)
message
```

The message says that the jump `join` has the clocks `rate` and `probability`. The fix is one of them.

**A field realization.** A field fixes the change and names the rate, such as births at a rate `b` per individual.

```python continuation
from skcomplex.core.specs import ParameterSpec


class BirthInteraction(JumpInteraction):
    def __init__(self, label, *, target, b: Annotated[float, ParameterSpec()] = 0.5):
        super().__init__(label, target=target, changes={'x': 1}, rate=lambda s: s.b * s.target.x)
        self.b = b


BirthInteraction('births', target='population').get_params()
```

Its parameters are the label, the target, and `b`.

See its [fact sheet](../../../parts/interactions.md#jumpinteraction)
and its [API entry][skcomplex.core.components.interactions.JumpInteraction].

## MapInteraction

A `MapInteraction` sets the next value of a state at every step of discrete time. Here the logistic map follows a
population, as a fraction of its carrying capacity, from one year to the next.

```python
from typing import Annotated

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import MapInteraction
from skcomplex.core.components.spaces import DiscreteTime
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.specs import StateSpec


class AmountElement(Element):
    def __init__(self, label, visual=None, *, x: Annotated[float, StateSpec()] = 0.0):
        super().__init__(label, visual=visual)
        self.x = x


logistic = lambda s: 2.5 * s.target.x * (1 - s.target.x)
growth = MapInteraction('growth', target='population', state='x', function=logistic)
census = CompositeSystem(elements=[AmountElement('population', x=0.1)], interactions=[growth], time=DiscreteTime())
simulate(census, until=3).get_values('population.x').round(4).tolist()
```

The fraction goes from 0.1 to 0.225, 0.4359, and 0.6147, on its way to the fixed point 0.6 of the map.

The same class compounds the interest of a bank account once a month. With `mode='increment'` the function gives the
change, not the next value.

```python continuation
monthly = lambda s: 0.01 * s.target.x
interest = MapInteraction('interest', target='account', state='x', function=monthly, mode='increment')
bank = CompositeSystem(elements=[AmountElement('account', x=1000.0)], interactions=[interest], time=DiscreteTime())
round(float(simulate(bank, until=12).get_values('account.x')[-1]), 2)
```

After twelve months at 1 percent a month the balance is 1126.83, which is `1000 * 1.01 ** 12`.

**How the engine runs it.** The symbolic form returns update laws, and `IterationSolver` applies them once per step.
The members of a group update all at once by default, from the states at the start of the step. The `schedule`
`'sequential'` or `'random-sequential'` updates them one after another. In the agent form, with `form='agent'`, the
function takes the agent and a view of the system and returns its next value.

```python continuation
sorted(interest.get_tags().laws)
```

The interaction emits an update law, the next value of a state.

**How it combines.** The role `target` binds an element or a group, and `sources` names the elements or members the
function reads. Maps on different states of one system run in the same step, and the maps of the agent form run in
phases of the step.

**Limits and errors.** A map gives the next value in discrete time, so a state that flows needs a `RateInteraction`.
The function must be callable, and `prepare` raises a `ValidationError` otherwise.

```python continuation
from skcomplex.core.errors import ValidationError

try:
    MapInteraction('interest', target='account', state='x', function=0.01, mode='increment').prepare()
except ValidationError as error:
    message = str(error)
message
```

The message reads "The function of map 'interest' must be callable, but it is 0.01." The fix is
`function=lambda s: 0.01 * s.target.x`.

**A field realization.** A field fixes the function and names its parameter, such as the growth rate `r` of the
logistic map.

```python continuation
from skcomplex.core.specs import ParameterSpec


class LogisticMapInteraction(MapInteraction):
    def __init__(self, label, *, target, r: Annotated[float, ParameterSpec()] = 3.0):
        super().__init__(label, target=target, state='x', function=lambda s: s.r * s.target.x * (1 - s.target.x))
        self.r = r


chaos = LogisticMapInteraction('growth', target='population', r=4.0)
census = CompositeSystem(elements=[AmountElement('population', x=0.1)], interactions=[chaos], time=DiscreteTime())
simulate(census, until=3).get_values('population.x').round(4).tolist()
```

With `r = 4` the map is chaotic, and the fraction jumps to 0.36, 0.9216, and 0.289.

See its [fact sheet](../../../parts/interactions.md#mapinteraction)
and its [API entry][skcomplex.core.components.interactions.MapInteraction].

## TransitionInteraction

A `TransitionInteraction` moves a discrete state, one of a few values, from one value to another. Here an infection
spreads along a chain of five people, where 0 is healthy and 1 is infected, at a rate of 1 per infected neighbour.

```python
from typing import Annotated

import networkx as nx
from sage.all__sagemath_symbolics import NN

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.interactions import TransitionInteraction
from skcomplex.core.components.spaces import NetworkSpace
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.specs import StateSpec


class HolderElement(Element):
    def __init__(self, label, visual=None, *, state: Annotated[int, StateSpec(domain=NN)] = 0):
        super().__init__(label, visual=visual)
        self.state = state


chain = NetworkSpace(nx.path_graph(5))
people = GroupElement('people', element=HolderElement, space=chain, on='sites', values={'state': [1, 0, 0, 0, 0]})
infection = TransitionInteraction(
    'infection', target='people', state='state', source=0, destination=1, rate=lambda s: 1.0, over='neighbors'
)
outbreak = CompositeSystem(elements=[people], interactions=[infection])
simulate(outbreak, until=10, dt=5.0, random_state=0).get_values('people.state').tolist()
```

In this run every person is infected by the fifth day. The domain `NN`, the natural numbers, marks the state as
discrete.

The same class describes customers who switch between two brands. A transition matrix gives, for each brand, the
probabilities of the brand of the next month, as in a Markov chain.

```python continuation
from skcomplex.core.components.spaces import DiscreteTime

customers = GroupElement('customers', element=HolderElement, size=500)
switching = TransitionInteraction(
    'switching', target='customers', state='state', matrix=lambda s: [[0.9, 0.1], [0.2, 0.8]]
)
market = CompositeSystem(elements=[customers], interactions=[switching], time=DiscreteTime())
round(float(simulate(market, until=30, random_state=0).get_values('customers.state')[-1].mean()), 2)
```

After 30 months a share of 0.32 buys the second brand, close to the stationary share `0.1 / (0.1 + 0.2) = 1/3`.

**How the engine runs it.** The interaction returns a jump law whose event moves the state from its source value
to its destination. At a `rate`, `GillespieSolver` runs the jumps in continuous time. A `probability`, a `condition`,
or a `matrix` moves the state once per step of discrete time, with a draw for every member.

```python continuation
sorted(infection.get_tags().laws), sorted(switching.get_tags().laws)
```

Both interactions emit jump laws.

**How it combines.** The role `target` binds an element or a group. With `over='neighbors'` the rate is that of one
neighbour, and the rates of the infected neighbours add, so the space of the group decides who infects whom. Several
transitions on one state build a compartment model, such as an infection from 0 to 1 and a recovery from 1 to 2.

**Limits and errors.** The state moves between named values, so an amount that grows by steps needs a
`JumpInteraction`. A transition has exactly one kind among `rate`, `probability`, `condition`, and `matrix`, and its
source and destination differ. `prepare` raises a `ValidationError` otherwise.

```python continuation
from skcomplex.core.errors import ValidationError

try:
    TransitionInteraction('recovery', target='people', source=1, destination=1, rate=lambda s: 0.5).prepare()
except ValidationError as error:
    message = str(error)
message
```

The message reads "The transition 'recovery' moves from 1 to itself. Give distinct values." The fix is a destination
of 0, or of 2 for a removed state.

**A field realization.** A field fixes the values and names the rate, such as the recovery of an infected person at a
rate `gamma`.

```python continuation
from skcomplex.core.specs import ParameterSpec


class RecoveryInteraction(TransitionInteraction):
    def __init__(self, label, *, target, gamma: Annotated[float, ParameterSpec()] = 0.2):
        super().__init__(label, target=target, state='state', source=1, destination=0, rate=lambda s: s.gamma)
        self.gamma = gamma


RecoveryInteraction('recovery', target='people').get_params()
```

Its parameters are the label, the target, and `gamma`.

See its [fact sheet](../../../parts/interactions.md#transitioninteraction)
and its [API entry][skcomplex.core.components.interactions.TransitionInteraction].

## CrossingInteraction

A `CrossingInteraction` fires an event at the exact time a guard, a function of the states, crosses zero. At the
event it sets states to new values or ends the run. Here a ball dropped from 1 metre bounces on the floor and keeps
80 percent of its speed.

```python
from typing import Annotated

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import CrossingInteraction, RateInteraction
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.specs import StateSpec


class BodyElement(Element):
    def __init__(self, label, visual=None, *, x: Annotated[float, StateSpec(order=2)] = 0.0, x_d1: float = 0.0):
        super().__init__(label, visual=visual)
        self.x = x
        self.x_d1 = x_d1


gravity = RateInteraction('gravity', target='ball', rate=-9.81)
bounce = CrossingInteraction(
    'bounce', target='ball', guard=lambda s: s.target.x, reset={'x_d1': lambda s: -0.8 * s.target.x_d1}, direction=-1
)
fall = CompositeSystem(elements=[BodyElement('ball', x=1.0)], interactions=[gravity, bounce])
[round(event.time, 3) for event in simulate(fall, until=2, dt=0.5).events]
```

The ball hits the floor at 0.452 seconds, which is `sqrt(2 / 9.81)`, and again at 1.174 and 1.752. The height `x` is
a state of order 2, so the rate of gravity adds to its acceleration, and `x_d1` is its velocity.

The same class makes a neuron fire. Its potential rises toward 2, and each time it reaches the threshold 1 it fires
and resets to 0, as in the leaky integrate-and-fire model of neuroscience.

```python continuation
class AmountElement(Element):
    def __init__(self, label, visual=None, *, x: Annotated[float, StateSpec()] = 0.0):
        super().__init__(label, visual=visual)
        self.x = x


current = RateInteraction('current', target='neuron', rate=lambda s: 2.0 - s.target.x)
spike = CrossingInteraction(
    'spike', target='neuron', guard=lambda s: s.target.x - 1.0, reset={'x': lambda s: 0.0}, direction=1
)
cell = CompositeSystem(elements=[AmountElement('neuron')], interactions=[current, spike])
[round(event.time, 3) for event in simulate(cell, until=2, dt=0.5).events]
```

The neuron fires at 0.693 and 1.386 seconds, once every `log 2`.

**How the engine runs it.** The interaction returns a crossing law. The solver watches the sign of the guard, locates
each crossing within its tolerance, applies the reset, and restarts the integration from the new states. The
trajectory records every crossing among its `events`, with the states before and after.

```python continuation
[law.__name__ for law in spike.law_classes]
```

The interaction emits a crossing law.

**How it combines.** The role `target` binds an element or a group, and a guard that reads the members of a group
gives each member its own event. Rate interactions move the states between the events. With `terminal=True` the
first crossing ends the run, as when a projectile lands.

**Limits and errors.** The event fires at a condition on the states, never at random times, so a random event needs
a `JumpInteraction`. The direction is 1 for a rise through zero, -1 for a fall, or 0 for both, and `prepare` raises a
`ValidationError` for any other value.

```python continuation
from skcomplex.core.errors import ValidationError

try:
    CrossingInteraction('spike', target='neuron', guard=lambda s: s.target.x - 1.0, direction=2).prepare()
except ValidationError as error:
    message = str(error)
message
```

The message reads "The direction of crossing 'spike' is 2. Use -1, 0, or 1." The fix is `direction=1`.

**A field realization.** A field names the threshold, such as a neuron that fires at a potential `theta`.

```python continuation
from skcomplex.core.specs import ParameterSpec


class FiringInteraction(CrossingInteraction):
    def __init__(self, label, *, target, theta: Annotated[float, ParameterSpec()] = 1.0):
        super().__init__(
            label, target=target, guard=lambda s: s.target.x - s.theta, reset={'x': lambda s: 0.0}, direction=1
        )
        self.theta = theta


FiringInteraction('spike', target='neuron', theta=1.5).get_params()
```

Its parameters are the label, the target, and `theta`.

See its [fact sheet](../../../parts/interactions.md#crossinginteraction)
and its [API entry][skcomplex.core.components.interactions.CrossingInteraction].

## GradientInteraction

A `GradientInteraction` takes a potential, one scalar function of the states, and moves every state down its
gradient. Here a particle in the double well `U = (x^2 - 1)^2 / 4` slides from 0.5 to the bottom of the well at 1.

```python
from typing import Annotated

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import GradientInteraction
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.specs import StateSpec


class AmountElement(Element):
    def __init__(self, label, visual=None, *, x: Annotated[float, StateSpec()] = 0.0):
        super().__init__(label, visual=visual)
        self.x = x


well = GradientInteraction('well', target='particle', potential=lambda s: (s.target.x**2 - 1) ** 2 / 4)
landscape = CompositeSystem(elements=[AmountElement('particle', x=0.5)], interactions=[well])
simulate(landscape, until=10, dt=5.0).get_values('particle.x').round(3).tolist()
```

The particle reaches 1.0 by the fifth unit of time and stays there.

The same class fits the two weights of a model by descending its loss, the squared distance to the values 1 and -2.

```python continuation
class ModelElement(Element):
    def __init__(
        self, label, visual=None, *, a: Annotated[float, StateSpec()] = 0.0, b: Annotated[float, StateSpec()] = 0.0
    ):
        super().__init__(label, visual=visual)
        self.a = a
        self.b = b


loss = lambda s: (s.target.a - 1) ** 2 + (s.target.b + 2) ** 2
descent = GradientInteraction('descent', target='model', states=('a', 'b'), potential=loss)
trajectory = simulate(CompositeSystem(elements=[ModelElement('model')], interactions=[descent]), until=5, dt=5.0)
round(float(trajectory.get_values('model.a')[-1]), 3), round(float(trajectory.get_values('model.b')[-1]), 3)
```

The weights reach 1.0 and -2.0, the minimum of the loss.

**How the engine runs it.** The interaction differentiates the potential symbolically and returns one rate law per
moving state, `-dU/dx` divided by the divisor. On a state of order 1 this is a gradient flow, and on a state of order
2 it is Newton's law with the force `-dU/dx`. `ScipyODESolver` integrates the rates.

```python continuation
[law.__name__ for law in descent.law_classes]
```

The interaction emits rate laws.

**How it combines.** The role `target` binds an element or a group, and `sources` names the elements the potential
reads without moving them, such as the position of a fixed charge. Several gradient interactions on one state add
their forces. On a `FieldElement`, with `variational=True`, the potential is the density of a functional and the field
follows the Allen-Cahn equation, and `conserved=True` gives the Cahn-Hilliard equation.

**Limits and errors.** One potential gives every rate, so a force that is not the gradient of a potential, such as
friction, needs a `RateInteraction`. The potential must be a function, and `prepare` raises a `ValidationError`
otherwise.

```python continuation
from skcomplex.core.errors import ValidationError

try:
    GradientInteraction('well', target='particle', potential=1.0).prepare()
except ValidationError as error:
    message = str(error)
message
```

The message reads "The potential of 'well' is 1.0. Give a function of the namespace." The fix is a function, such as
`lambda s: s.target.x**2 / 2`.

**A field realization.** A field fixes the potential and names its parameters, such as a spring of stiffness `k`.

```python continuation
from skcomplex.core.specs import ParameterSpec


class SpringInteraction(GradientInteraction):
    def __init__(self, label, *, target, k: Annotated[float, ParameterSpec()] = 1.0):
        super().__init__(label, target=target, potential=lambda s: s.k * s.target.x**2 / 2)
        self.k = k


spring = SpringInteraction('spring', target='particle', k=2.0)
relaxation = CompositeSystem(elements=[AmountElement('particle', x=1.0)], interactions=[spring])
round(float(simulate(relaxation, until=1, dt=1.0).get_values('particle.x')[-1]), 3)
```

The particle relaxes from 1 to 0.135 in one unit of time, which is `exp(-2)`.

See its [fact sheet](../../../parts/interactions.md#gradientinteraction)
and its [API entry][skcomplex.core.components.interactions.GradientInteraction].

## EulerLagrangeInteraction

An `EulerLagrangeInteraction` takes a Lagrangian, the kinetic minus the potential energy, and moves the coordinates by
its equations of motion. Here a pendulum swings from an angle of 0.1 radians, with the Lagrangian
`q'^2 / 2 + cos q` in units where gravity and the length are 1.

```python
from typing import Annotated

from sage.all__sagemath_symbolics import cos

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import EulerLagrangeInteraction
from skcomplex.core.components.laws.utils import derivative
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.specs import StateSpec


class CoordinateElement(Element):
    def __init__(self, label, visual=None, *, q: Annotated[float, StateSpec(order=2)] = 0.1, q_d1: float = 0.0):
        super().__init__(label, visual=visual)
        self.q = q
        self.q_d1 = q_d1


lagrangian = lambda s: derivative(s.target.q, 1) ** 2 / 2 + cos(s.target.q)
swing = EulerLagrangeInteraction('swing', target='bob', states='q', lagrangian=lagrangian)
pendulum = CompositeSystem(elements=[CoordinateElement('bob')], interactions=[swing])
simulate(pendulum, until=3.2, dt=1.6).get_values('bob.q').round(3).tolist()
```

The angle passes through -0.003 at 1.6 seconds and reaches -0.1 at 3.2, half of the period `2 pi` of small swings.
`derivative(q, 1)` is the velocity of the coordinate.

The same class gives an electric circuit of a coil and a capacitor, written by its energy. The charge `q` is the
coordinate, the coil of inductance 2 stores `L q'^2 / 2`, and the capacitor of capacitance 0.5 stores `q^2 / (2 C)`.

```python continuation
energy = lambda s: 2.0 * derivative(s.target.q, 1) ** 2 / 2 - s.target.q**2 / (2 * 0.5)
oscillation = EulerLagrangeInteraction('oscillation', target='circuit', states='q', lagrangian=energy)
circuit = CompositeSystem(elements=[CoordinateElement('circuit', q=1.0)], interactions=[oscillation])
simulate(circuit, until=3.14159, dt=3.14159 / 2).get_values('circuit.q').round(3).tolist()
```

The charge goes from 1 to 0 and to -1 in half a period, since the frequency `1 / sqrt(L C)` is 1.

**How the engine runs it.** The interaction differentiates the Lagrangian and returns one equation per coordinate,
`d/dt dL/dq' - dL/dq = Q - dR/dq'`, with the generalized forces `Q` and the dissipation `R`. The compiler solves the
equations together for the accelerations, so a Lagrangian with a mass matrix, such as that of a double pendulum,
needs no inversion by hand.

```python continuation
[law.__name__ for law in swing.law_classes]
```

The interaction emits equation laws, which hold at every time.

**How it combines.** The role `target` binds an element or a group, and every role that binds one element adds its
coordinates. Equations of one coordinate from several interactions add, so a spring between two pendulums is an
interaction of its own whose Lagrangian holds only the coupling energy. `dissipation` and `forcing` add friction and
external forces.

**Limits and errors.** The coordinates are states of order 2, and the Lagrangian must be a function. `prepare` raises
a `ValidationError` for a Lagrangian that is not one.

```python continuation
from skcomplex.core.errors import ValidationError

try:
    EulerLagrangeInteraction('swing', target='bob', states='q', lagrangian=0.5).prepare()
except ValidationError as error:
    message = str(error)
message
```

The message reads "The lagrangian of 'swing' is 0.5. Give a function of the namespace." The fix is a function of
the namespace, as above.

**A field realization.** A field fixes the Lagrangian and names its parameters, such as a pendulum of length `l` under
gravity `g`.

```python continuation
from skcomplex.core.specs import ParameterSpec


class PendulumInteraction(EulerLagrangeInteraction):
    def __init__(
        self,
        label,
        *,
        target,
        g: Annotated[float, ParameterSpec()] = 9.81,
        l: Annotated[float, ParameterSpec()] = 1.0,
    ):
        lagrangian = lambda s: s.l**2 * derivative(s.target.q, 1) ** 2 / 2 + s.g * s.l * cos(s.target.q)
        super().__init__(label, target=target, states='q', lagrangian=lagrangian)
        self.g = g
        self.l = l


PendulumInteraction('swing', target='bob', l=2.0).get_params()
```

Its parameters are the label, the target, `g`, and `l`.

See its [fact sheet](../../../parts/interactions.md#eulerlagrangeinteraction)
and its [API entry][skcomplex.core.components.interactions.EulerLagrangeInteraction].

## HamiltonianInteraction

A `HamiltonianInteraction` takes a Hamiltonian, the total energy as a function of coordinates `q` and momenta `p`, and
moves them by Hamilton's equations. Here the bond of a molecule vibrates as a harmonic oscillator with the energy
`(p^2 + q^2) / 2`.

```python
from typing import Annotated

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import HamiltonianInteraction
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.specs import StateSpec


class PhaseElement(Element):
    def __init__(
        self, label, visual=None, *, q: Annotated[float, StateSpec()] = 1.0, p: Annotated[float, StateSpec()] = 0.0
    ):
        super().__init__(label, visual=visual)
        self.q = q
        self.p = p


energy = lambda s: (s.target.p**2 + s.target.q**2) / 2
vibration = HamiltonianInteraction('vibration', target='bond', hamiltonian=energy)
molecule = CompositeSystem(elements=[PhaseElement('bond')], interactions=[vibration])
simulate(molecule, until=3.14159, dt=3.14159 / 2).get_values('bond.q').round(3).tolist()
```

The stretch of the bond goes from 1 to 0 and to -1 in half a period.

The same class moves a planet around a star. The coordinates and momenta are vectors in a plane, and the
Hamiltonian is `|p|^2 / 2 - 1 / |q|` in units where the mass of the star and the constant of gravity are 1.

```python continuation
from sage.all__sagemath_symbolics import sqrt

from skcomplex.core.components.spaces import ContinuousSpace


class PlanetElement(Element):
    def __init__(
        self,
        label,
        visual=None,
        *,
        q: Annotated[object, StateSpec(vector=True)] = (1.0, 0.0),
        p: Annotated[object, StateSpec(vector=True)] = (0.0, 1.0),
    ):
        super().__init__(label, visual=visual)
        self.q = q
        self.p = p


kepler = lambda s: (s.target.p[0] ** 2 + s.target.p[1] ** 2) / 2 - 1 / sqrt(s.target.q[0] ** 2 + s.target.q[1] ** 2)
orbit = HamiltonianInteraction('orbit', target='planet', hamiltonian=kepler)
sky = CompositeSystem(elements=[PlanetElement('planet')], interactions=[orbit], space=ContinuousSpace(n_dim=2))
simulate(sky, until=3.14159, dt=3.14159).get_values('planet.q').round(3).tolist()
```

The planet starts at `(1, 0)` and reaches `(-1, 0)` after half of its circular orbit of period `2 pi`. A vector state
needs a space, here the plane `ContinuousSpace(n_dim=2)`.

**How the engine runs it.** The interaction differentiates the Hamiltonian and returns the rates `q' = dH/dp` and
`p' = -dH/dq`, one per coordinate and per momentum. A symplectic solver, such as `Yoshida4Solver`, keeps the
energy of a separable Hamiltonian bounded over long runs.

```python continuation
[law.__name__ for law in orbit.law_classes]
```

The interaction emits rate laws.

**How it combines.** The role `target` binds an element or a group, whose states `q` and `p` pair in order by default.
The rates of one state add, so a coupling between two oscillators is an interaction of its own whose Hamiltonian
holds only the coupling energy. Dissipation or forcing is a `RateInteraction` on the momenta.

**Limits and errors.** Each coordinate needs one momentum of the same size, both states of order 1. `prepare` raises a
`ValidationError` when the counts differ.

```python continuation
from skcomplex.core.errors import ValidationError

try:
    HamiltonianInteraction('orbit', target='planet', coordinates=('q', 'r'), momenta='p', hamiltonian=kepler).prepare()
except ValidationError as error:
    message = str(error)
message
```

The message says that the coordinate `r` has no partner and asks for one momentum per coordinate. The fix is a second
momentum, such as `momenta=('p', 's')`.

**A field realization.** A field fixes the Hamiltonian and names its parameters, such as a harmonic bond of stiffness
`k` and reduced mass `m`.

```python continuation
from skcomplex.core.specs import ParameterSpec


class BondInteraction(HamiltonianInteraction):
    def __init__(
        self, label, *, target, k: Annotated[float, ParameterSpec()] = 1.0, m: Annotated[float, ParameterSpec()] = 1.0
    ):
        energy = lambda s: s.target.p**2 / (2 * s.m) + s.k * s.target.q**2 / 2
        super().__init__(label, target=target, hamiltonian=energy)
        self.k = k
        self.m = m


BondInteraction('vibration', target='bond', k=4.0).get_params()
```

Its parameters are the label, the target, `k`, and `m`.

See its [fact sheet](../../../parts/interactions.md#hamiltonianinteraction)
and its [API entry][skcomplex.core.components.interactions.HamiltonianInteraction].

## ConservationInteraction

A `ConservationInteraction` changes a field by the divergence of a flux, so the amount it moves through space is
neither created nor destroyed. It gives the conservation law `du/dt + divergence(flux) = source`. Here cars on a ring
road of 5 kilometres follow the traffic model of Lighthill and Whitham, whose flux is the density times the speed
`1 - u`.

```python
from typing import Annotated

import numpy as np

from skcomplex.core.components.elements import Element, FieldElement
from skcomplex.core.components.interactions import ConservationInteraction
from skcomplex.core.components.spaces import LatticeSpace
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.specs import StateSpec


class CellElement(Element):
    def __init__(self, label, visual=None, *, u: Annotated[float, StateSpec()] = 0.0):
        super().__init__(label, visual=visual)
        self.u = u


ring = LatticeSpace((50,), spacing=0.1, boundary='periodic')
density = 0.2 + 0.6 * (np.arange(50) < 10)
road = FieldElement('road', element=CellElement, space=ring, values={'u': density}, discretization='finite-volume')
traffic = ConservationInteraction('traffic', target='road', state='u', flux=lambda s: s.target.u * (1 - s.target.u))
cars = simulate(CompositeSystem(elements=[road], interactions=[traffic]), until=1.0, dt=0.5).get_values('road.u')
round(float(cars[0].sum()), 6), round(float(cars[-1].sum()), 6), round(float(cars[-1].max()), 3)
```

The total density stays 16 while the jam of density 0.8 spreads, and its peak falls to 0.794.

The same class carries a pollutant down a river at a speed of 1.5, with the flux `1.5 u`.

```python continuation
reaches = LatticeSpace((40,), spacing=0.25, boundary='periodic')
spill = np.exp(-((np.arange(40) * 0.25 - 2.0) ** 2))
river = FieldElement('river', element=CellElement, space=reaches, values={'u': spill}, discretization='finite-volume')
advection = ConservationInteraction('advection', target='river', state='u', flux=lambda s: 1.5 * s.target.u)
flow = CompositeSystem(elements=[river], interactions=[advection])
pollutant = simulate(flow, until=2.0, dt=2.0).get_values('river.u')
int(np.argmax(pollutant[0])), int(np.argmax(pollutant[-1])), round(float(pollutant[-1].sum()), 4)
```

The peak moves from cell 8 to cell 20, a distance of 3 in 2 units of time, and the total stays 7.0809.

**How the engine runs it.** On a finite-volume field the interaction lowers to a flux law. The scheme reconstructs the
state in each cell with a slope limiter, computes the flux between cells with a numerical flux such as that of
Rusanov, and gives every cell the rate of what enters minus what leaves. On a field of another discretization it
lowers to the rate `-divergence(flux)`, which the field discretizes by its own operators.

```python continuation
[law.__name__ for law in advection.law_classes]
```

The interaction emits a flux law on a finite-volume field and a rate law on any other.

**How it combines.** The role `target` binds a `FieldElement`, and `sources` names other elements the flux reads,
such as a velocity field. Several conservation laws on one field form one system, such as the Euler equations of a
gas, and their numerical flux reads the wave speeds of all of them. A `RateInteraction` on the same state adds a
reaction or a decay.

**Limits and errors.** The interaction needs a field on a lattice, and it conserves the total only on a periodic field
or with no flux through the boundaries. The limiter must be `'minmod'`, `'van-leer'`, or `None`, and `prepare` raises
a `ValidationError` otherwise.

```python continuation
from skcomplex.core.errors import ValidationError

try:
    ConservationInteraction('traffic', target='road', flux=lambda s: s.target.u, limiter='superbee').prepare()
except ValidationError as error:
    message = str(error)
message
```

The message reads "Unknown limiter 'superbee' of 'traffic'. Use minmod, van-leer, or None." The fix is
`limiter='van-leer'` or the default `'minmod'`.

**A field realization.** A field fixes the flux and names its parameter, such as a road on which cars drive at most at
the speed `v` on an empty road.

```python continuation
from skcomplex.core.specs import ParameterSpec


class TrafficInteraction(ConservationInteraction):
    def __init__(self, label, *, target, v: Annotated[float, ParameterSpec()] = 1.0):
        super().__init__(label, target=target, state='u', flux=lambda s: s.v * s.target.u * (1 - s.target.u))
        self.v = v


TrafficInteraction('traffic', target='road', v=1.2).get_params()
```

Its parameters are the label, the target, and `v`.

See its [fact sheet](../../../parts/interactions.md#conservationinteraction)
and its [API entry][skcomplex.core.components.interactions.ConservationInteraction].

## NeighborInteraction

A `NeighborInteraction` is the base of an interaction whose laws run once for every edge of a lattice or a network.
Each law reads `namespace.source` and `namespace.target`, the members at the two ends of one edge. Here heat flows
between the neighbouring cells of a ring of four, in proportion to their difference in temperature.

```python
from typing import Annotated

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.interactions import NeighborInteraction
from skcomplex.core.components.laws import RateLaw
from skcomplex.core.components.spaces import LatticeSpace
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.specs import ParameterSpec, RoleSpec, StateSpec


class CellElement(Element):
    def __init__(self, label, visual=None, *, u: Annotated[float, StateSpec()] = 0.0):
        super().__init__(label, visual=visual)
        self.u = u


class ExchangeInteraction(NeighborInteraction):
    law_classes = (RateLaw,)

    def __init__(
        self, label, *, cells: Annotated[str, RoleSpec()], D: Annotated[float, ParameterSpec()] = 1.0, layer=None
    ):
        super().__init__(label, layer=layer)
        self.cells = cells
        self.D = D

    def _get_laws(self, namespace):
        flow = namespace.D * (namespace.target.u - namespace.source.u)
        return [RateLaw(namespace.source.u, flow), RateLaw(namespace.target.u, -flow)]


ring = GroupElement('ring', element=CellElement, space=LatticeSpace(4), on='sites', values={'u': [1.0, 0.0, 0.0, 0.0]})
heat = CompositeSystem(elements=[ring], interactions=[ExchangeInteraction('heat', cells='ring')])
simulate(heat, until=5, dt=5.0).get_values('ring.u').round(3).tolist()
```

After 5 seconds every cell holds 0.25, the mean of the initial temperatures.

The same class moves people between towns along the roads of a network, here a hub with three outer towns, one of
which starts with 9 thousand people.

```python continuation
import networkx as nx

from skcomplex.core.components.spaces import NetworkSpace

star = NetworkSpace(nx.star_graph(3))
towns = GroupElement('towns', element=CellElement, space=star, on='sites', values={'u': [0.0, 0.0, 0.0, 9.0]})
migration = CompositeSystem(elements=[towns], interactions=[ExchangeInteraction('migration', cells='towns', D=0.5)])
simulate(migration, until=10, dt=10.0).get_values('towns.u').round(2).tolist()
```

After 10 years the towns hold 2.25, 2.23, 2.23, and 2.29 thousand, close to the even share of 2.25.

**How the engine runs it.** The laws are written for one edge. The compiler evaluates them for every edge at once, as
arrays over the lists of sources and targets of the space, and adds the result at each end. On a lattice the edges
are the bonds between neighbouring sites. A law that reads only one end outside its reductions runs once per member.

**How it combines.** A first role binds a group on the sites of a lattice or the nodes of a network. A second role
may bind a group on the edges of the same network, read as `namespace.edge`, such as the conductance of each line of a
power grid. `layer` picks one layer of a network with several kinds of links.

**Limits and errors.** The edges are fixed by the space, so members that move and meet those close by need a
`PairInteraction`, and links that change need a `TopologyInteraction`. Its group must live on a space with edges, and
`simulate` raises a `ValidationError` otherwise.

```python continuation
from skcomplex.core.errors import ValidationError

loose = GroupElement('ring', element=CellElement, size=4)
try:
    simulate(CompositeSystem(elements=[loose], interactions=[ExchangeInteraction('heat', cells='ring')]), until=1)
except ValidationError as error:
    message = str(error)
message
```

The message says that the interaction has laws per edge but its members live on a space without edges. The fix is a
lattice or a network space for the group, as above.

**A field realization.** The subclass above is already a field realization, the exchange of heat or of people
between neighbours. A field writes its own laws for one edge in the same way, such as an infection along each contact.

See its [fact sheet](../../../parts/interactions.md#neighborinteraction)
and its [API entry][skcomplex.core.components.interactions.NeighborInteraction].

## CouplingInteraction

A `CouplingInteraction` gives every member of a group a sum, a mean, or another reduction of a function of itself and
the members it is coupled to. Here two fireflies, as in the Kuramoto model of synchrony, pull each other by the sine of
their phase difference until they flash in step.

```python
from sage.all__sagemath_symbolics import sin

from skcomplex.core.components.elements import DeclaredElement, GroupElement
from skcomplex.core.components.interactions import CouplingInteraction
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.specs import DeclaredVariableSpec, StateSpec

phases = {'variables': {'theta': DeclaredVariableSpec(0.0, StateSpec())}, 'theta': [0.0, 1.0]}
fireflies = GroupElement('fireflies', element=DeclaredElement, size=2, values=phases)
sine = lambda s: sin(s.other.theta - s.member.theta)
pull = CouplingInteraction('pull', target='fireflies', read='theta', write='theta', function=sine, mode='group')
swarm = CompositeSystem(elements=[fireflies], interactions=[pull])
simulate(swarm, until=4, dt=2.0).get_values('fireflies.theta').round(3).tolist()
```

The phases 0 and 1 meet at 0.5, and the fireflies flash together. The namespace reads the member as `member` and the
other member as `other`.

The same class gives consensus of opinions on a network. Three people on a line move their opinions toward those of
their neighbours, with the default function, the difference `x_j - x_i`, summed over the edges.

```python continuation
import networkx as nx

from skcomplex.core.components.spaces import NetworkSpace

opinions = {'variables': {'x': DeclaredVariableSpec(0.0, StateSpec())}, 'x': [0.0, 3.0, 9.0]}
line = NetworkSpace(nx.path_graph(3))
people = GroupElement('people', element=DeclaredElement, space=line, on='sites', values=opinions)
consensus = CouplingInteraction('consensus', target='people')
society = CompositeSystem(elements=[people], interactions=[consensus])
simulate(society, until=10, dt=10.0).get_values('people.x').round(2).tolist()
```

The opinions 0, 3, and 9 agree on their mean 4 after 10 days.

**How the engine runs it.** In the form `'rate'` the reduction adds to the rate of the state `write`. A sum or a mean
over the edges of a lattice or network runs once per edge in both directions, and every other coupling runs once per
member. On a `FieldElement`, a constant times the difference of the two ends becomes that constant times the
Laplacian of the field. In the form `'map'` the reduction sets the next value at every step of discrete time.

```python continuation
[law.__name__ for law in consensus.law_classes]
```

The coupling may emit rate laws or update laws, one kind for each form.

**How it combines.** The argument `mode` chooses the members it couples: `'edges'` for the neighbours along a lattice,
a network, or a `neighborhood` in continuous space, `'group'` for every other member, `'next'` for a ring, and
`'hyperedges'` for the members of each group link. `weights` weighs each term, such as by the weight of an edge, and
`reduce` combines the terms, such as `'circular-mean'` for the headings of a flock.

**Limits and errors.** The coupling reduces over its neighbours, so a law that needs both ends of an edge at once,
such as a flow recorded on the edge, needs a `NeighborInteraction`. A function must be callable, and `prepare` raises a
`ValidationError` otherwise.

```python continuation
from skcomplex.core.errors import ValidationError

try:
    CouplingInteraction('consensus', target='people', function=0.5).prepare()
except ValidationError as error:
    message = str(error)
message
```

The message reads "The function of coupling must be callable or None, but it is 0.5." The fix is a function of the
namespace, such as `lambda s: 0.5 * (s.other.x - s.member.x)`.

**A field realization.** A field fixes the function and names its strength, such as the coupling `K` of the Kuramoto
model.

```python continuation
from typing import Annotated

from skcomplex.core.specs import ParameterSpec


class KuramotoInteraction(CouplingInteraction):
    def __init__(self, label, *, target, K: Annotated[float, ParameterSpec()] = 1.0):
        function = lambda s: s.K * sin(s.other.theta - s.member.theta)
        super().__init__(label, target=target, read='theta', write='theta', function=function, mode='group')
        self.K = K


weak = CompositeSystem(elements=[fireflies], interactions=[KuramotoInteraction('pull', target='fireflies', K=0.5)])
simulate(weak, until=4, dt=2.0).get_values('fireflies.theta').round(3).tolist()
```

With half the strength, the phases are 0.426 and 0.574 after 2 seconds and 0.49 and 0.51 after 4, a slower approach.

See its [fact sheet](../../../parts/interactions.md#couplinginteraction)
and its [API entry][skcomplex.core.components.interactions.CouplingInteraction].

## PairInteraction

A `PairInteraction` runs its laws for every pair of members whose positions lie closer than a cutoff, so it finds its
pairs by distance as the members move. With a `potential`, a function of the distance `r`, each member moves down the
gradient of the potential. Here two atoms of a gas, one unit apart, repel under the Lennard-Jones potential
`4 (r^-12 - r^-6)`.

```python
from typing import Annotated

import numpy as np

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.interactions import PairInteraction
from skcomplex.core.components.spaces import ContinuousSpace
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.specs import StateSpec


class AtomElement(Element):
    def __init__(
        self,
        label,
        visual=None,
        *,
        x: Annotated[object, StateSpec(vector=True, order=2, role='position')] = 0.0,
        x_d1: object = 0.0,
    ):
        super().__init__(label, visual=visual)
        self.x = x
        self.x_d1 = x_d1


atoms = GroupElement('gas', element=AtomElement, size=2, values={'x': [[0.0, 0.0], [1.0, 0.0]]})
lennard_jones = PairInteraction('lj', 'gas', 'gas', cutoff=2.5, potential=lambda s, r: 4 * (r**-12 - r**-6))
box = CompositeSystem(elements=[atoms], interactions=[lennard_jones], space=ContinuousSpace(n_dim=2))
positions = simulate(box, until=2, dt=0.5).get_values('gas.x')
np.linalg.norm(positions[:, 1] - positions[:, 0], axis=1).round(3).tolist()
```

The distance grows from 1.0 to 2.369 in 2 units of time, because the minimum of the potential lies at `2^(1/6)`,
about 1.12. The position is a vector state of order 2, so the force sets the acceleration.

The same class keeps pedestrians at a distance. Their positions are of order 1, so each moves with a velocity down the
gradient of the repulsion `exp(-r / 0.5)`.

```python continuation
from sage.all__sagemath_symbolics import exp


class WalkerElement(Element):
    def __init__(self, label, visual=None, *, x: Annotated[object, StateSpec(vector=True, role='position')] = 0.0):
        super().__init__(label, visual=visual)
        self.x = x


crowd = GroupElement('crowd', element=WalkerElement, size=2, values={'x': [[0.0, 0.0], [0.2, 0.0]]})
distance = PairInteraction('distance', 'crowd', 'crowd', cutoff=2.0, potential=lambda s, r: exp(-r / 0.5))
square = CompositeSystem(elements=[crowd], interactions=[distance], space=ContinuousSpace(n_dim=2))
positions = simulate(square, until=4, dt=2.0).get_values('crowd.x')
np.linalg.norm(positions[:, 1] - positions[:, 0], axis=1).round(3).tolist()
```

The two pedestrians move apart from 0.2 to 1.431 and then to 1.756, ever more slowly as the repulsion weakens.

**How the engine runs it.** The laws are written for one pair, with `namespace.first` the member that receives the
value and `namespace.second` the other. With a potential, the interaction differentiates it with respect to the
position of the first member and adds minus the gradient to the rate of `write`. At every evaluation the compiler
finds the pairs within the cutoff with a k-d tree, a search tree of the positions, and sums the values of each
member over its pairs.

```python continuation
[law.__name__ for law in distance.law_classes]
```

The interaction emits rate laws.

**How it combines.** The roles `first` and `second` bind groups, the same group for pairs within one population or
two groups for pairs between them. `function` gives a value of the pair in place of a potential, such as an alignment
of headings, and `divisor` divides the value by a mass. The space is a `ContinuousSpace`, whose period, if any, gives
the minimum image distance.

**Limits and errors.** The pairs come from distances, so members that interact along fixed links need a
`NeighborInteraction`. A pair interaction takes a function or a potential, not both, and `prepare` raises a
`ValidationError` when both are given.

```python continuation
from skcomplex.core.errors import ValidationError

try:
    PairInteraction('lj', 'gas', 'gas', potential=lambda s, r: r**-12, function=lambda s: 0.0).prepare()
except ValidationError as error:
    message = str(error)
message.endswith('Give at most one of them, as a callable.')
```

The message names the function and the potential of `lj` and asks for at most one of them. The fix is to drop one.

**A field realization.** A field fixes the potential and names its parameters, such as the depth `epsilon` and the
size `sigma` of the Lennard-Jones potential.

```python continuation
from skcomplex.core.specs import ParameterSpec


class LennardJonesInteraction(PairInteraction):
    def __init__(
        self,
        label,
        first,
        second,
        cutoff=2.5,
        *,
        epsilon: Annotated[float, ParameterSpec()] = 1.0,
        sigma: Annotated[float, ParameterSpec()] = 1.0,
    ):
        potential = lambda s, r: 4 * s.epsilon * ((s.sigma / r) ** 12 - (s.sigma / r) ** 6)
        super().__init__(label, first, second, cutoff=cutoff, potential=potential)
        self.epsilon = epsilon
        self.sigma = sigma


LennardJonesInteraction('lj', 'gas', 'gas', epsilon=0.5).get_params()
```

Its parameters are the two groups, the cutoff, the label, `epsilon`, and `sigma`.

See its [fact sheet](../../../parts/interactions.md#pairinteraction)
and its [API entry][skcomplex.core.components.interactions.PairInteraction].

## HyperedgeInteraction

A `HyperedgeInteraction` is the base of an interaction whose laws run once for every hyperedge of a network. A
hyperedge is a link that joins more than two nodes, such as a group of friends who talk together. Here two groups of
three share one person, and every person moves toward the mean opinion of each group they belong to.

```python
from typing import Annotated

import networkx as nx

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.interactions import HyperedgeInteraction
from skcomplex.core.components.laws import RateLaw
from skcomplex.core.components.laws.utils import mean
from skcomplex.core.components.spaces import NetworkSpace
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.specs import RoleSpec, StateSpec


class NodeElement(Element):
    def __init__(self, label, visual=None, *, x: Annotated[float, StateSpec()] = 0.0):
        super().__init__(label, visual=visual)
        self.x = x


class GroupMeanInteraction(HyperedgeInteraction):
    law_classes = (RateLaw,)

    def __init__(self, label, *, nodes: Annotated[str, RoleSpec()], size=None):
        super().__init__(label, size=size)
        self.nodes = nodes

    def _get_laws(self, namespace):
        return [RateLaw(namespace.member.x, mean(namespace.members.x, over=namespace.members) - namespace.member.x)]


groups = NetworkSpace(nx.empty_graph(5), hyperedges=[[0, 1, 2], [2, 3, 4]])
people = GroupElement('people', element=NodeElement, space=groups, on='sites', values={'x': [0.0, 0.0, 3.0, 6.0, 6.0]})
talks = CompositeSystem(elements=[people], interactions=[GroupMeanInteraction('talks', nodes='people')])
simulate(talks, until=10, dt=10.0).get_values('people.x').round(2).tolist()
```

After 10 days the opinions are 2.89, 2.89, 3.0, 3.11, and 3.11, close to the overall mean 3. The person in both groups
carries the opinion from one group to the other.

The same class couples three oscillators at once, as in higher-order models of synchrony, where the term
`sin(theta_j + theta_k - 2 theta_i)` reads both other members of a triangle.

```python continuation
from sage.all__sagemath_symbolics import sin


class TriadInteraction(HyperedgeInteraction):
    law_classes = (RateLaw,)

    def __init__(self, label, *, nodes: Annotated[str, RoleSpec()]):
        super().__init__(label, size=3)
        self.nodes = nodes

    def _get_laws(self, namespace):
        first, second = namespace.others
        return [RateLaw(namespace.member.x, sin(first.x + second.x - 2 * namespace.member.x))]


triangle = NetworkSpace(nx.empty_graph(3), hyperedges=[[0, 1, 2]])
phases = GroupElement('phases', element=NodeElement, space=triangle, on='sites', values={'x': [0.0, 0.5, 1.3]})
rhythm = CompositeSystem(elements=[phases], interactions=[TriadInteraction('triad', nodes='phases')])
simulate(rhythm, until=10, dt=10.0).get_values('phases.x').round(3).tolist()
```

The three phases lock at 0.626.

**How the engine runs it.** A law of the focal member, `namespace.member`, runs once for each member of every
hyperedge, with `namespace.others` the other members in the order of the hyperedge. The reductions `total`, `mean`,
`count`, `maximum`, and `minimum` with `over=namespace.members` run over all members of one hyperedge. The compiler
evaluates the laws for every hyperedge at once, as arrays.

**How it combines.** A first role binds a group on the nodes of a network with hyperedges. A second role may bind a
group on the hyperedges, read as `namespace.hyperedge`, such as the activity of a group. `size` restricts the laws to
hyperedges of one size, such as 3 for triangles. A `CouplingInteraction` with `mode='hyperedges'` covers the common
case of a reduction over the other members without a class of its own.

**Limits and errors.** A hyperedge joins at least two members, and `prepare` raises a `ValidationError` for a size
below two.

```python continuation
from skcomplex.core.errors import ValidationError

try:
    GroupMeanInteraction('talks', nodes='people', size=1).prepare()
except ValidationError as error:
    message = str(error)
message
```

The message reads "The size of the hyperedges of 'talks' is 1. Give a whole number of 2 or more." The fix is a size
of 2 or more, or `None` for every size.

**A field realization.** The subclasses above are already field realizations, the opinions of groups in social
science and the triads of physics.

See its [fact sheet](../../../parts/interactions.md#hyperedgeinteraction)
and its [API entry][skcomplex.core.components.interactions.HyperedgeInteraction].

## GroupingInteraction

A `GroupingInteraction` splits the members of a group into new small groups at every step and acts on each of them.
Here six traders meet in random pairs every day, and each pair splits its joint wealth evenly.

```python
from typing import Annotated

import numpy as np

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.interactions import GroupingInteraction
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.specs import StateSpec


class TraderElement(Element):
    def __init__(self, label, visual=None, *, w: Annotated[float, StateSpec()] = 0.0):
        super().__init__(label, visual=visual)
        self.w = w


def share(members, system):
    total = sum(member.w for member in members)
    for member in members:
        member.w = total / len(members)


traders = GroupElement('traders', element=TraderElement, size=6, values={'w': [0.0, 4.0, 8.0, 12.0, 16.0, 20.0]})
trades = GroupingInteraction('trades', target='traders', groups='pairs', act=share)
market = CompositeSystem(elements=[traders], interactions=[trades])
wealth = simulate(market, until=5, random_state=0).get_agents('traders', 5).values['w']
sorted(np.round(wealth, 1).tolist()), float(wealth.sum())
```

After five days in this run the wealth lies between 9.2 and 10.4, near the mean 10, and the total stays 60. The
function `act` receives the agents of one group and a view of the system, and sets their states.

The same class runs the majority model of Galam, a model of opinion dynamics. Nine voters, five of them in favour,
discuss in random groups of three, and each group adopts the opinion of its majority. This symbolic form sets the
state `write` to a function of the group.

```python continuation
from skcomplex.core.components.laws.utils import mean, where

voters = GroupElement('voters', element=TraderElement, size=9, values={'w': [1.0] * 5 + [0.0] * 4})
majority = lambda g: where(mean(g.members.w, over=g.members) > 1 / 2, 1, 0)
galam = GroupingInteraction(
    'galam', target='voters', groups='partition', size=3, form='symbolic', write='w', function=majority
)
debate = CompositeSystem(elements=[voters], interactions=[galam])
simulate(debate, until=3, random_state=0).get_values('voters.w').mean(axis=1).round(3).tolist()
```

In this run the share in favour falls from 0.556 to 0.333 after the first round and stays there. A majority of the
voters lost, because the random groups left it a minority in two of the three groups.

**How the engine runs it.** In the agent form the interaction returns a rule that, at every step, draws the groups
and calls `act` on each group in random order. In the symbolic form it returns an update law of the state `write`,
and a member in no group keeps its value. Both run in discrete time.

```python continuation
[law.__name__ for law in galam.law_classes]
```

The interaction emits a match law, a rule law, or an update law, depending on its form.

**How it combines.** The role `target` binds a group. The argument `groups` chooses how the groups form: `'pairs'`,
`'partition'` into groups of `size`, `'neighborhood'` for each member and the members around it in space, or
`'state'` for the members that share a value of the state `by`. Several groupings run in phases of one step, each
reading the states that earlier phases set.

**Limits and errors.** The groups are drawn anew at every step, so members that interact along fixed links need a
`NeighborInteraction` or a `CouplingInteraction`. A group holds at least two members, and `prepare` raises a
`ValidationError` for a smaller size.

```python continuation
from skcomplex.core.errors import ValidationError

try:
    GroupingInteraction('trades', target='traders', groups='partition', size=1, act=share).prepare()
except ValidationError as error:
    message = str(error)
message
```

The message reads "The size of the groups of 'trades' is 1. Give a whole number of at least 2." The fix is a size of
2 or more.

**A field realization.** A field writes the action as the method `act_on` of a subclass, such as an exchange in which
each pair splits its wealth evenly.

```python continuation
class ExchangeInteraction(GroupingInteraction):
    def __init__(self, label, *, target):
        super().__init__(label, target=target, groups='pairs')

    def act_on(self, members, system):
        total = sum(member.w for member in members)
        for member in members:
            member.w = total / len(members)


exchange = CompositeSystem(elements=[traders], interactions=[ExchangeInteraction('trades', target='traders')])
float(simulate(exchange, until=5, random_state=0).get_agents('traders', 5).values['w'].sum())
```

The total wealth stays 60.

See its [fact sheet](../../../parts/interactions.md#groupinginteraction)
and its [API entry][skcomplex.core.components.interactions.GroupingInteraction].

## LocalRuleInteraction

A `LocalRuleInteraction` sets the next value of every cell of a lattice or network from the values in its
neighbourhood, as a cellular automaton does. Here a blinker of the Game of Life of Conway turns from a row of three
live cells into a column and back. The `OuterTotalisticSpec` maps the value of a cell and the sum of its eight
neighbours to the next value.

```python
from typing import Annotated

import numpy as np

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.interactions import LocalRuleInteraction
from skcomplex.core.components.spaces import BallNeighborhood, DiscreteTime, LatticeSpace
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.specs import OuterTotalisticSpec, StateSpec


class CellElement(Element):
    def __init__(self, label, visual=None, *, c: Annotated[float, StateSpec()] = 0.0):
        super().__init__(label, visual=visual)
        self.c = c


board = np.zeros((5, 5))
board[2, 1:4] = 1.0
cells = GroupElement('board', element=CellElement, space=LatticeSpace((5, 5)), on='sites', values={'c': board.ravel()})
life = OuterTotalisticSpec({(0, 3): 1, (1, 2): 1, (1, 3): 1}, default=0)
rule = LocalRuleInteraction('life', target='board', state='c', rule=life, neighborhood=BallNeighborhood())
game = CompositeSystem(elements=[cells], interactions=[rule], time=DiscreteTime())
[np.flatnonzero(frame).tolist() for frame in simulate(game, until=2).get_values('board.c')]
```

The live cells are 11, 12, and 13 in the middle row, then 7, 12, and 17 in the middle column, then the row again.
A dead cell with three live neighbours is born, and a live cell with two or three survives.

The same class moves cars along a ring road by the elementary rule 184. A car moves one cell ahead when that cell is
empty. The `TableSpec` maps each pattern of the left neighbour, the cell, and the right neighbour to the next value.

```python continuation
from skcomplex.core.specs import TableSpec

moves = {(1, 1, 1): 1, (1, 1, 0): 0, (1, 0, 1): 1, (1, 0, 0): 1}
patterns = moves | {(0, 1, 1): 1, (0, 1, 0): 0, (0, 0, 1): 0, (0, 0, 0): 0}
cars = {'c': [1, 1, 0, 1, 0, 0, 0, 0]}
road = GroupElement('road', element=CellElement, space=LatticeSpace(8), on='sites', values=cars)
around = BallNeighborhood(metric='manhattan', n_dim=1, center='inline')
traffic = LocalRuleInteraction('traffic', target='road', state='c', rule=TableSpec(patterns), neighborhood=around)
ring = CompositeSystem(elements=[road], interactions=[traffic], time=DiscreteTime())
simulate(ring, until=3).get_values('road.c').astype(int).tolist()
```

The jam of the first two cars dissolves. After three steps the three cars stand at cells 2, 4, and 6, each with a free
cell ahead, and drive on.

**How the engine runs it.** The interaction returns a local rule law. At every step the solver gathers the values of
the neighbourhood of every cell in its order, looks up or computes the next values, and writes them all at once under
the default schedule `'synchronous'`. The schedules `'random-sequential'`, `'checkerboard'`, and `'colored'` update
the cells in turns.

```python continuation
[law.__name__ for law in traffic.law_classes]
```

The interaction emits a local rule law.

**How it combines.** The role `target` binds a group on the sites of a lattice or the nodes of a network, and the
`neighborhood`, such as `BallNeighborhood`, gives the ordered neighbours. A rule can be a `TableSpec`, a
`TotalisticSpec` of the sum of the neighbourhood, an `OuterTotalisticSpec`, or a function of arrays that sets several
states together.

**Limits and errors.** The next value depends on the neighbourhood alone, in discrete time, so a state that flows needs
a `CouplingInteraction` or a field. The neighbourhood must be a neighbourhood object, not a name, and `prepare` raises
a `ValidationError` otherwise.

```python continuation
from skcomplex.core.errors import ValidationError

try:
    LocalRuleInteraction('life', target='board', state='c', rule=life, neighborhood='moore').prepare()
except ValidationError as error:
    message = str(error)
message
```

The message says that the neighbourhood of the local rule `life` is `'moore'`. The fix is `BallNeighborhood()`, the
Moore neighbourhood of the eight surrounding cells.

**A field realization.** A field fixes the rule and the neighbourhood, such as the Game of Life.

```python continuation
class LifeInteraction(LocalRuleInteraction):
    def __init__(self, label, *, target):
        super().__init__(label, target=target, state='c', rule=life, neighborhood=BallNeighborhood())


LifeInteraction('life', target='board').get_params()
```

Its parameters are the label and the target.

See its [fact sheet](../../../parts/interactions.md#localruleinteraction)
and its [API entry][skcomplex.core.components.interactions.LocalRuleInteraction].

## SamplingInteraction

A `SamplingInteraction` samples the states of the members by Monte Carlo moves. Every sweep proposes a new value for
each member and accepts it with a probability that depends on the change of an objective, such as an energy, scaled
by `intensity`, the inverse of a temperature. Here the spins of a magnet in the Ising model align on a lattice of 8 by
8 sites from a random start.

```python
from typing import Annotated

import numpy as np

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.interactions import SamplingInteraction
from skcomplex.core.components.laws.utils import neighbors
from skcomplex.core.components.spaces import DiscreteTime, LatticeSpace
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.specs import StateSpec


class SpinElement(Element):
    def __init__(self, label, visual=None, *, s: Annotated[float, StateSpec()] = 1.0):
        super().__init__(label, visual=visual)
        self.s = s


start = np.random.default_rng(0).choice([-1.0, 1.0], 64)
spins = GroupElement('spins', element=SpinElement, space=LatticeSpace((8, 8)), on='sites', values={'s': start})
energy = lambda n: -n.target.s * neighbors(n.target.other.s)
ising = SamplingInteraction('ising', target='spins', state='s', objective=energy, intensity=1.0)
magnet = CompositeSystem(elements=[spins], interactions=[ising], time=DiscreteTime())
values = simulate(magnet, until=20, random_state=0).get_values('spins.s')
round(float(values[0].mean()), 2), round(abs(float(values[-1].mean())), 2)
```

The magnetization, the mean spin, is 0.16 at the start, and its size is 1.0 after 20 sweeps in this run, since the
temperature lies below the critical temperature of the model. `neighbors` sums the spins of the neighbours.

The same class lets voters of a social network align with their friends. The network is the karate club of Zachary,
and `AdjacencyNeighborhood` reads the neighbours along its edges.

```python continuation
import networkx as nx

from skcomplex.core.components.spaces import AdjacencyNeighborhood, NetworkSpace

club = nx.karate_club_graph()
opinions = np.random.default_rng(0).choice([-1.0, 1.0], 34)
members = GroupElement('members', element=SpinElement, space=NetworkSpace(club), on='sites', values={'s': opinions})
alignment = SamplingInteraction(
    'alignment',
    target='members',
    state='s',
    objective=energy,
    intensity=2.0,
    neighborhood=AdjacencyNeighborhood(),
    schedule='random-sequential',
)
society = CompositeSystem(elements=[members], interactions=[alignment], time=DiscreteTime())
values = simulate(society, until=20, random_state=0).get_values('members.s')
ends = np.array(club.edges())
[round(float((frame[ends[:, 0]] == frame[ends[:, 1]]).mean()), 2) for frame in (values[0], values[-1])]
```

The share of friendships whose two ends agree rises from 0.51 to 0.95 in this run.

**How the engine runs it.** The interaction returns a sample law. The solver makes one sweep per unit of time. Under
the rule `'metropolis'` it accepts a move with the probability `min(1, exp(-intensity * change))`, and the rules
`'glauber'` and `'heat-bath'` use other acceptances. The schedule `'checkerboard'` updates the sites of one colour of
a lattice at once, since they share no neighbours, and `'random-sequential'` visits the members one by one.

```python continuation
[law.__name__ for law in alignment.law_classes]
```

The interaction emits a sample law.

**How it combines.** The role `target` binds a group on a lattice, a network, or the edges of a cell complex. The
argument `move` proposes the new values, by a flip, a random value, a rotation, a Gaussian step, or a flip of a whole
cluster by the algorithm of Wolff. `cooling` lowers the temperature over time, as in simulated annealing.

**Limits and errors.** The states follow the stationary distribution of the objective, not a dynamics in time, so a
sweep is not a physical time step. The move must be one of the names above or a function, and `prepare` raises a
`ValidationError` otherwise.

```python continuation
from skcomplex.core.errors import ValidationError

try:
    SamplingInteraction('ising', target='spins', state='s', objective=energy, move='swap').prepare()
except ValidationError as error:
    message = str(error)
message
```

The message reads "Unknown move 'swap' of 'ising'. Use one of 'flip', 'random-value', 'rotate', 'gaussian', 'cluster',
a function." The fix is a known move, such as `'flip'` for spins.

**A field realization.** A field fixes the objective and names the temperature, such as the Ising model at a
temperature `T`.

```python continuation
class IsingInteraction(SamplingInteraction):
    def __init__(self, label, *, target, T=1.0):
        super().__init__(label, target=target, state='s', objective=energy, intensity=1 / T)
        self.T = T


heated = IsingInteraction('ising', target='spins', T=10.0)
hot = CompositeSystem(elements=[spins], interactions=[heated], time=DiscreteTime())
values = simulate(hot, until=20, random_state=0).get_values('spins.s')
round(abs(float(values[-1].mean())), 2)
```

At the high temperature 10 the spins stay disordered, with a magnetization of 0.06 in size.

See its [fact sheet](../../../parts/interactions.md#samplinginteraction)
and its [API entry][skcomplex.core.components.interactions.SamplingInteraction].

## TopologyInteraction

A `TopologyInteraction` adds, removes, or rewires the nodes and edges of a network during a run. The groups on the
nodes and the edges are variable, so their members come and go. Here a network grows from one link, and at every step
a new node joins one node drawn at random, as pages join the web.

```python
import networkx as nx

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.interactions import TopologyInteraction
from skcomplex.core.components.spaces import DiscreteTime, NetworkSpace
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate

nodes = GroupElement('nodes', element=Element, on='sites', variable=True, capacity=30)
links = GroupElement('links', element=Element, on='edges', variable=True, capacity=30)
growth = TopologyInteraction('growth', action='add-node', edges=1)
seed = NetworkSpace(nx.path_graph(2))
web = CompositeSystem(elements=[nodes, links], interactions=[growth], space=seed, time=DiscreteTime())
trajectory = simulate(web, until=10, random_state=0)
[(len(trajectory.get_network(frame).nodes), len(trajectory.get_network(frame).sources)) for frame in (0, 10)]
```

The network grows from 2 nodes and 1 edge to 12 nodes and 11 edges, a tree. `capacity` bounds the number of
members of a variable group.

The same class lets people cut their contacts to avoid an infection. Each contact of a ring of 20 people ends with a
probability of 0.3 per day.

```python continuation
distancing = TopologyInteraction('distancing', target='nodes', action='remove-edge', probability=0.3)
contacts = CompositeSystem(
    elements=[nodes, links], interactions=[distancing], space=NetworkSpace(nx.cycle_graph(20)), time=DiscreteTime()
)
trajectory = simulate(contacts, until=4, random_state=0)
[len(trajectory.get_network(frame).sources) for frame in range(5)]
```

The contacts fall from 20 to 14, 10, 7, and 4 in this run, about 70 percent of them left each day.

**How the engine runs it.** The interaction returns laws that request changes of the network, by jumps at a `rate`
or by draws with a `probability` per step. The solver applies the requests, adds or removes the members of the
variable groups, and rebuilds the edges that other interactions read. `get_network` returns the network of a frame.

```python continuation
[law.__name__ for law in distancing.law_classes]
```

The interaction emits jump laws for changes at a rate and update laws for changes per step.

**How it combines.** The role `target` binds the group on the nodes, or nothing for a growth from outside.
`condition` restricts the change to where it holds, `toward` weighs the candidates of a new end, and `values` gives
the states of a new node or edge. Interactions on the nodes, such as a `TransitionInteraction` of an infection, run
on the network as it changes, so structure and states change together.

**Limits and errors.** A change fires at a rate or with a probability, not both, and `prepare` raises a
`ValidationError` when both are given.

```python continuation
from skcomplex.core.errors import ValidationError

try:
    TopologyInteraction('distancing', target='nodes', action='remove-edge', rate=0.1, probability=0.3).prepare()
except ValidationError as error:
    message = str(error)
message
```

The message reads "'distancing' fires at a rate and with a probability. Give one of them." The fix is one of them.

**A field realization.** A field fixes the action and names its number of edges, such as the growth of a network of
citations in which every new paper cites `m` earlier ones.

```python continuation
class CitationInteraction(TopologyInteraction):
    def __init__(self, label, *, m=2):
        super().__init__(label, action='add-node', edges=m)
        self.m = m


CitationInteraction('citations', m=3).get_params()
```

Its parameters are the label and `m`.

See its [fact sheet](../../../parts/interactions.md#topologyinteraction)
and its [API entry][skcomplex.core.components.interactions.TopologyInteraction].

## ConnectionInteraction

A `ConnectionInteraction` joins the ports of elements at a node, so that their potentials agree and their flows sum
to zero. A port is a named connector of an element, such as a pin of a resistor, with a potential and a flow. Here two
voltage sources of 1 and 3 volts, each behind a conductance of 1 siemens, meet at one node of an electric circuit.

```python
from typing import Annotated, ClassVar

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import ConnectionInteraction
from skcomplex.core.components.laws import EquationLaw
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.specs import ConnectorSpec, ParameterSpec, StateSpec

PIN = ConnectorSpec('pin', potential='v', flow='i')


class BranchElement(Element):
    ports: ClassVar = {'a': PIN}
    law_classes = (EquationLaw,)

    def __init__(
        self,
        label,
        visual=None,
        *,
        a_v: Annotated[float, StateSpec(algebraic=True)] = 0.0,
        a_i: Annotated[float, StateSpec(algebraic=True)] = 0.0,
        g: Annotated[float, ParameterSpec()] = 1.0,
        e: Annotated[float, ParameterSpec()] = 0.0,
    ):
        super().__init__(label, visual=visual)
        self.a_v = a_v
        self.a_i = a_i
        self.g = g
        self.e = e

    def _get_laws(self, namespace):
        return [EquationLaw(namespace.a_i, namespace.g * (namespace.a_v - namespace.e))]


node = ConnectionInteraction('node', members=['first', 'second'], ports=['a', 'a'])
circuit = CompositeSystem(elements=[BranchElement('first', e=1.0), BranchElement('second', e=3.0)], interactions=[node])
trajectory = simulate(circuit, until=1.0, dt=1.0)
float(trajectory.get_values('first.a_v')[-1]), float(trajectory.get_values('first.a_i')[-1])
```

The node settles at 2 volts, midway between the sources. The flow of the port of the first branch is 1 ampere, and
that of the second is -1, so they cancel. The states `a_v` and `a_i` are the potential and the flow of the port `a`,
and `algebraic=True` marks them as states that an equation fixes at every time.

The same class joins three pipes at a junction of a water network. Each pipe brings water from a reservoir at its own
pressure, through its own conductance, and the pressure at the junction makes the flows balance.

```python continuation
pipes = [BranchElement('north', e=10.0, g=2.0), BranchElement('east', e=4.0), BranchElement('south', e=0.0)]
junction = ConnectionInteraction('junction', members=['north', 'east', 'south'], ports=['a', 'a', 'a'])
trajectory = simulate(CompositeSystem(elements=pipes, interactions=[junction]), until=1.0, dt=1.0)
flows = [float(trajectory.get_values(f'{pipe}.a_i')[-1]) for pipe in ('north', 'east', 'south')]
float(trajectory.get_values('north.a_v')[-1]), flows
```

The pressure at the junction is 6, and the flows -8, 2, and 6 sum to zero, as the law of Kirchhoff for currents
requires.

**How the engine runs it.** The interaction returns algebraic equations: the potentials of the ports are equal and their
flows sum to zero. With the equations of the elements they form a system of differential algebraic equations, which the
solver solves for the algebraic states at every time.

```python continuation
[law.__name__ for law in junction.law_classes]
```

The interaction emits equation laws.

**How it combines.** The role `members` binds any number of elements, one per port in `ports`. An element declares
its ports in the class attribute `ports`, each a `ConnectorSpec` that names the potential and the flow. Elements with
rates of their own, such as a capacitor, join elements with equations, as in Modelica, a language of models of
components.

**Limits and errors.** A port that no connection names has a zero flow. The interaction needs one port per member,
and `prepare` raises a `ValidationError` otherwise.

```python continuation
from skcomplex.core.errors import ValidationError

try:
    ConnectionInteraction('node', members=['first', 'second'], ports=['a']).prepare()
except ValidationError as error:
    message = str(error)
message
```

The message reads "'node' connects 2 members through 1 ports. Name one port per member." The fix is
`ports=['a', 'a']`.

**A field realization.** A field fixes the name of the port, such as a node of a circuit that joins the pin `a` of
every member.

```python continuation
class PinNodeInteraction(ConnectionInteraction):
    def __init__(self, label, *, members):
        super().__init__(label, members=members, ports=['a'] * len(members))


PinNodeInteraction('ground', members=['first', 'second']).ports
```

The node names the port `a` of both members.

See its [fact sheet](../../../parts/interactions.md#connectioninteraction)
and its [API entry][skcomplex.core.components.interactions.ConnectionInteraction].

## TrainingInteraction

A `TrainingInteraction` fits the parameters of a nested system to data by gradient steps, as a neural network is
trained. The nested system runs through the data in its own time, and every step of the outer system is one epoch of
training. Here a model of a drug in the blood, which keeps a fraction `a` of the drug each hour, learns `a` from a
series of concentrations that fall to 0.8 of their value each hour.

```python
from typing import Annotated

import numpy as np

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import TrainingInteraction
from skcomplex.core.components.laws import UpdateLaw
from skcomplex.core.components.spaces import DiscreteTime
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.specs import DatasetSpec, ObservableSpec, ParameterSpec, StateSpec


class FactorElement(Element):
    def __init__(
        self,
        label,
        visual=None,
        *,
        x: Annotated[float, StateSpec()] = 1.0,
        a: Annotated[float, ParameterSpec(adaptive=True)] = 0.5,
        observed: Annotated[float, ParameterSpec()] = 0.0,
    ):
        super().__init__(label, visual=visual)
        self.x = x
        self.a = a
        self.observed = observed

    def _get_laws(self, namespace):
        return [UpdateLaw(namespace.x, namespace.a * namespace.x)]


squared_error = ObservableSpec('error', lambda s: (s.unit.x - s.unit.observed) ** 2)
series = 0.8 ** np.arange(1, 11)
blood = CompositeSystem('blood', elements=[FactorElement('unit')], time=DiscreteTime())
fit = TrainingInteraction(
    'fit',
    system='blood',
    trained=['unit.a'],
    loss=squared_error,
    data=DatasetSpec({'unit.observed': series[np.newaxis]}),
    learning_rate=0.05,
    batch_size=1,
)
fitting = CompositeSystem('fitting', systems=[blood], interactions=[fit], time=DiscreteTime())
trajectory = simulate(fitting, until=100, random_state=0)
round(float(trajectory.get_values('blood/unit.a')[-1]), 3), bool(trajectory.get_values('fit/loss')[-1] < 1e-6)
```

After 100 epochs the factor is 0.8 and the loss is below `1e-6`. The parameter `a` is declared
`ParameterSpec(adaptive=True)`, which marks it as trainable. `DatasetSpec` maps the parameter `unit.observed` to one
sequence of ten values, and the loss is the mean squared error over the steps.

The same class fits the monthly interest factor of a savings account from twelve monthly balances that grow by 1
percent a month.

```python continuation
balances = 1.01 ** np.arange(1, 13)
account = CompositeSystem('account', elements=[FactorElement('unit', a=1.0)], time=DiscreteTime())
fit = TrainingInteraction(
    'fit',
    system='account',
    trained=['unit.a'],
    loss=squared_error,
    data=DatasetSpec({'unit.observed': balances[np.newaxis]}),
    learning_rate=0.005,
    batch_size=1,
)
fitting = CompositeSystem('fitting', systems=[account], interactions=[fit], time=DiscreteTime())
round(float(simulate(fitting, until=100, random_state=0).get_values('account/unit.a')[-1]), 4)
```

The factor reaches 1.01. A real study would hold out part of the data and record a metric on it with `metrics`.

**How the engine runs it.** The interaction returns a gradient step law. `GradientDescentSolver` unrolls the nested
system over each batch of sequences with JAX, differentiates the loss with respect to the trained parameters, and
takes one step of the optimizer. The trajectory records the trained parameters under their paths in the nested
system, such as `account/unit.a`, and the mean loss of each epoch as `fit/loss`.

```python continuation
[law.__name__ for law in fit.law_classes]
```

The interaction emits a gradient step law.

**How it combines.** The role `system` binds a nested system, which runs, is analysed, and is drawn alone before or
after training. The hyperparameters of the optimizer, such as `learning_rate`, are parameters of the interaction, so a
schedule of `simulate` changes them between epochs.
[Learning by gradient steps](../build_a_custom_system/learning_by_gradient_steps.md) explains the data, the losses,
and the truncation of the gradient in full.

**Limits and errors.** The nested system must be differentiable, written with symbolic laws and without noise, jumps,
or agent rules. The optimizer is one of `'sgd'`, `'momentum'`, `'rmsprop'`, and `'adam'`, and `prepare` raises a
`ValidationError` for any other.

```python continuation
from skcomplex.core.errors import ValidationError

try:
    wrong = {'loss': squared_error, 'data': fit.data, 'optimizer': 'lbfgs'}
    TrainingInteraction('fit', system='account', trained=['unit.a'], **wrong).prepare()
except ValidationError as error:
    message = str(error)
message
```

The message says that the optimizer `'lbfgs'` is not supported and lists the four that are. The fix is one of them,
such as `optimizer='adam'`, the default.

**A field realization.** A field fixes the optimizer and the loss, such as the training of a network by stochastic
gradient descent on its squared error.

```python continuation
class SquaredErrorTrainingInteraction(TrainingInteraction):
    def __init__(self, label, *, system, trained, data, learning_rate=0.01):
        super().__init__(
            label,
            system=system,
            trained=trained,
            loss=squared_error,
            data=data,
            optimizer='sgd',
            learning_rate=learning_rate,
        )


SquaredErrorTrainingInteraction('fit', system='account', trained=['unit.a'], data=fit.data).optimizer
```

The training uses the optimizer `'sgd'`.

See its [fact sheet](../../../parts/interactions.md#traininginteraction)
and its [API entry][skcomplex.core.components.interactions.TrainingInteraction].

## BaseInteraction

`BaseInteraction` is the abstract base of every interaction. A new kind of interaction that runs its laws in a way of
its own, as `NeighborInteraction` runs them once per edge, derives from it. Every other interaction of your own
derives from `Interaction` or from one of the classes above.

```python
from skcomplex.core.components.interactions import BaseInteraction, Interaction, NeighborInteraction

issubclass(Interaction, BaseInteraction), issubclass(NeighborInteraction, BaseInteraction)
```

Both are subclasses of `BaseInteraction`.

See its [fact sheet](../../../parts/interactions.md#baseinteraction)
and its [API entry][skcomplex.core.components.interactions.BaseInteraction].
