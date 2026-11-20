# Interactions

An interaction changes the states of elements. It binds elements through roles, such as `target` and `source`, reads
their states, and returns laws, the equations or rules that a solver runs. It holds no state of its own. An element
holds the states, a space says which members neighbour each other, and a system composes elements, interactions, and a
space into one model. [How the components fit](index.md) shows the families together, and the [interactions
page](../../../components/interactions.md) compares every interaction class and holds its fact sheet.

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
    def __init__(self, label, *, x: Annotated[float, StateSpec()] = 0.0):
        super().__init__(label)
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
a network or between close members takes `along=` or `within=`. Every role must name an element or a group of the
system, and `simulate` raises a `RoleError` otherwise.

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

**A rule without a class.** A rule used once needs no class. `roles` binds each role to an element, and `laws` is a
function of the namespace that returns the laws. Here a ball falls under gravity.

```python continuation
from skcomplex.core.systems import System

gravity = Interaction(
    'gravity',
    roles={'target': 'ball'},
    laws=lambda s: [RateLaw(s.target.x.diff(s.t, 2), -9.81)],
)
fall = System(elements=[Element('ball', values={'x': 10.0})], interactions=[gravity]).prepare()
round(float(fall.simulate(until=1.0, dt=1.0).get_values('ball.x')[-1]), 3)
```

The ball falls from 10 to 5.095 in one second, as `x = 10 - 9.81 t^2 / 2` says. The namespace `s` holds every state as
a function of the time `s.t`, so `s.target.x.diff(s.t, 2)` is the acceleration of the ball.

**Random groups and neighbourhoods.** `along=RandomPartition(k)` binds random groups of `k` members drawn at every step,
which [RandomPartition](#randompartition) shows. `along='neighborhoods'` gives every member of a group on a lattice or a
network one group with the members around it, so a member belongs to its own group and to the groups of its neighbours,
and its updates from all of them add up. In the public goods game of Szabó and Hauert every player pays 1 into each
group it belongs to when it cooperates, and each pot, tripled, is shared among the members of its group.

```python continuation
from skcomplex.core.components.laws import UpdateLaw
from skcomplex.core.components.laws.utils import count, total
from skcomplex.core.components.spaces import LatticeSpace

play = Interaction(
    'play',
    among='players',
    along='neighborhoods',
    laws=lambda s: [
        UpdateLaw(
            s.member.payoff,
            3.0 * total(s.members.strategy, over=s.members) / count(over=s.members) - s.member.strategy,
        ),
    ],
)
players = Element(
    'players',
    values={'strategy': [1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0], 'payoff': 0.0},
    space=LatticeSpace((3, 3), metric='manhattan', boundary='fixed'),
)
game = System(elements=[players], interactions=[play]).prepare()
game.simulate(until=1).get_values('players.payoff')[-1].round(2).tolist()
```

The cooperator in the corner belongs to three groups and earns 1, 0.5, and 1.25 from them, 2.75 in all, while the
defector in the middle earns 7.35 from five groups without paying anything.

See its [fact sheet](../../../components/interactions.md#interaction)
and its [API entry][skcomplex.core.components.interactions.Interaction].

## RandomPartition

A `RandomPartition` binds an interaction to random groups. `along=RandomPartition(k)` shuffles the members of the
group `among` at every step and cuts them into groups of `k`, the members left over forming one smaller group. The laws
run once for every member of every group and read `s.member`, the member they change, `s.others`, the other members of
its group, and `s.members`, every member of the group, which reductions read with `over=s.members`. In Galam's model of
opinion, citizens discuss in groups of three and every group adopts the opinion of its majority.

```python continuation
from skcomplex.core.components.interactions import RandomPartition
from skcomplex.core.components.laws import UpdateLaw
from skcomplex.core.components.laws.utils import mean, where

talks = Interaction(
    'talks',
    among='citizens',
    along=RandomPartition(3),
    laws=lambda s: [UpdateLaw(s.member.opinion, where(mean(s.members.opinion, over=s.members) > 0.5, 1, 0))],
)
citizens = Element('citizens', values={'opinion': [1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0]}, size=9)
debate = System(elements=[citizens], interactions=[talks]).prepare()
debate.simulate(until=4, random_state=0).get_values('citizens.opinion').sum(axis=1).tolist()
```

The four citizens of opinion 1 lose one of their own in the first round of discussions, and three remain after the next
three rounds. The groups come from the random stream of the run, so one seed gives the same groups every time. Players
of a public goods game and animals that meet in random encounters form groups in the same way.

See its [API entry][skcomplex.core.components.interactions.RandomPartition].

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
from skcomplex.core.specs import ParameterSpec, StateSpec
from skcomplex.core.components.laws import DatasetSpec
from skcomplex.core.components import Observable


class FactorElement(Element):
    def __init__(
        self,
        label,
        *,
        x: Annotated[float, StateSpec()] = 1.0,
        a: Annotated[float, ParameterSpec(adaptive=True)] = 0.5,
        observed: Annotated[float, ParameterSpec()] = 0.0,
    ):
        super().__init__(label)
        self.x = x
        self.a = a
        self.observed = observed

    def _get_laws(self, namespace):
        return [UpdateLaw(namespace.x, namespace.a * namespace.x)]


squared_error = Observable('error', lambda s: (s.unit.x - s.unit.observed) ** 2)
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

**How the engine runs it.** The interaction states no laws of its own. The compiler finds it in the system, and
`GradientDescentSolver` reads its settings, unrolls the nested system over each batch of sequences with JAX,
differentiates the loss with respect to the trained parameters, and takes one step of the optimizer. The trajectory
records the trained parameters under their paths in the nested system, such as `account/unit.a`, and the mean loss of
each epoch as `fit/loss`.

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

See its [fact sheet](../../../components/interactions.md#traininginteraction)
and its [API entry][skcomplex.core.components.interactions.TrainingInteraction].

## BaseInteraction

`BaseInteraction` is the abstract base of every interaction. A new kind of interaction that runs its laws in a way of
its own derives from it. Every other interaction of your own derives from `Interaction` or from one of the classes
above.

```python
from skcomplex.core.components.interactions import BaseInteraction, Interaction, TrainingInteraction

issubclass(Interaction, BaseInteraction), issubclass(TrainingInteraction, BaseInteraction)
```

Both are subclasses of `BaseInteraction`.

See its [fact sheet](../../../components/interactions.md#baseinteraction)
and its [API entry][skcomplex.core.components.interactions.BaseInteraction].
