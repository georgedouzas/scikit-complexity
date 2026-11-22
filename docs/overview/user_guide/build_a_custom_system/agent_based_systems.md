# Agent-based systems

This page writes the behaviour of individual agents as plain Python. An agent is one member of a group element, such
as one walker of a crowd or one cell of a tissue. Its behaviour is a rule, a Python function that reads and changes one
agent at every step, and an interaction gives the rule to the members of the group it binds. Rules move agents, let
them give birth and die, and let them read each other on a grid, a network, or a continuous space.

## Agents with rules

This section lets a thousand walkers take random steps along a line. The walkers are one
[`Element`](../core_components/elements.md#element) with `size=1000`, each with its own position `x`. The interaction
`Walk` binds the walkers through its role `target`, and its `_prepare` sets the law
[`UpdateLaw`](../core_components/laws.md#updatelaw)`(s.target, self.move)`. An update law whose target is a group and
whose value is a function is a rule: every walker runs `move(agent, system)` once per step.

```python
from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import UpdateLaw
from skcomplex.core.systems import System
from skcomplex.core.types import Role


class Walk(Interaction):
    def __init__(self, label, *, target: Role = 'walkers', step=1.0):
        super().__init__(label)
        self.target = target
        self.step = step

    def _prepare(self, s):
        super()._prepare(s)
        self.laws_ = [UpdateLaw(s.target, self.move)]

    def move(self, agent, system):
        agent.x += self.step * system.rng.normal()


walkers = Element('walkers', values={'x': 0.0}, size=1000)
system = System(elements=[walkers], interactions=[Walk('walk')]).prepare()
x = system.simulate(until=100, random_state=0).get_agents('walkers', 100).values['x']
system.states_, system.form_, round(float(x.var()))
```

The states are `('walkers.x',)`, the form is an agent-based system, and the variance of the positions after 100 steps
is 98, close to the number of steps, as a random walk spreads.

- The rule reads and sets the values of its agent as attributes, such as `agent.x`. Since a rule may change any value,
  every value of a group with rules is a state, recorded in the trajectory. A value the element does not have raises
  Python's own `AttributeError`.
- The rule reads the settings of its interaction as `self.step`, and `system.rng` is the random generator of the run,
  seeded by `random_state`.
- `prepare` picks `MesaSolver` for a system with rules. Mesa is the Python library of agent-based modelling that runs
  the steps, one per unit of time.
- `trajectory.get_agents(group, frame)` returns the ids and values of the agents of a group at a frame, and the
  trajectory records the number of members of every group as `<group>.count`.
- An element without `size` whose interaction gives it a rule is a group of one agent.

## Births and deaths

A rule adds a member with `agent.create(**values)` and removes its own agent with `agent.remove()`. In a Galton-Watson
branching process every individual has a random number of offspring with mean `m` and then dies, so generation `n` has
`n_0 m^n` individuals on average. Here 20 individuals start, with `m = 1.5`.

```python
from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import UpdateLaw
from skcomplex.core.systems import System
from skcomplex.core.types import Role


class Branching(Interaction):
    def __init__(self, label, *, target: Role = 'population', m=1.0):
        super().__init__(label)
        self.target = target
        self.m = m

    def _prepare(self, s):
        super()._prepare(s)
        self.laws_ = [UpdateLaw(s.target, self.reproduce)]

    def reproduce(self, agent, system):
        for _ in range(system.rng.poisson(self.m)):
            agent.create(generation=agent.generation + 1)
        agent.remove()


population = Element('population', values={'generation': 0}, size=20)
system = System(elements=[population], interactions=[Branching('branching', m=1.5)]).prepare()
system.simulate(until=4, random_state=0).get_values('population.count').tolist()
```

In this run the generations have 20, 25, 36, 47, and 64 individuals, against the mean sizes 20, 30, 45, 68, and 101,
since one run of a branching process strays far from its mean. The offspring created in a step act from the next step
on, so each step is one generation.

Other births and deaths are a few lines of the same kind.

- An agent dies with the probability `p` when `system.rng.random() < p` leads to `agent.remove()`.
- A cell that divides and shares its mass halves `agent.mass` and creates `agent.create(mass=agent.mass)`.
- An offspring inherits a trait as `agent.create(trait=agent.trait)`, and a mutation adds a draw to it.
- `system.create(group)` adds a member of a group bound to a role of the interaction without a parent, such as an
  arrival.

## Agents in a space

Agents live on a [`LatticeSpace`](../core_components/spaces.md#latticespace), a
[`NetworkSpace`](../core_components/spaces.md#networkspace), or a
[`ContinuousSpace`](../core_components/spaces.md#continuousspace) given to the system. Here 50 foragers wander a
torus of 30 by 30 cells, gain a Poisson number of units of food with mean 0.8 per step, lose 1, split when they hold
more than 10, and starve when they hold none.

```python
from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import UpdateLaw
from skcomplex.core.components.spaces import LatticeSpace
from skcomplex.core.systems import System
from skcomplex.core.types import Role


class Forage(Interaction):
    def __init__(self, label, *, target: Role = 'foragers', hunger=1.0):
        super().__init__(label)
        self.target = target
        self.hunger = hunger

    def _prepare(self, s):
        super()._prepare(s)
        self.laws_ = [UpdateLaw(s.target, self.forage)]

    def forage(self, agent, system):
        cells = agent.get_neighbor_cells()
        agent.move_to(cells[int(system.rng.integers(len(cells)))])
        agent.food += system.rng.poisson(0.8) - self.hunger
        if agent.food <= 0:
            agent.remove()
        elif agent.food > 10:
            agent.food /= 2
            agent.create(food=agent.food)


system = System(
    elements=[Element('foragers', values={'food': 5.0}, size=50)],
    interactions=[Forage('forage')],
    space=LatticeSpace((30, 30), boundary='periodic'),
)
int(system.prepare().simulate(until=20, random_state=0).get_values('foragers.count')[-1])
```

Of the 50 foragers, 17 are alive after 20 steps, since most of them lose more food than they find.

An agent acts on its place through its own methods.

- `agent.get_neighbor_cells()`, `agent.get_neighbors()`, and `agent.get_agents_here()` read the cells and agents
  around it, and `agent.move_to(location)` moves it.
- `agent.get_displacement_to(other)` gives the vector to another agent or a location, the shortest one on a torus,
  and `agent.get_distance_to(other)` its length, or the number of hops on a network.
- `agent.location` is its cell, node, or position, and `agent.id` its unique id.

## Positions of agents

The values of an agent named after the axes of its space are its location. On a
[`ContinuousSpace`](../core_components/spaces.md#continuousspace) of two dimensions they are `x` and `y`, and on a
[`LatticeSpace`](../core_components/spaces.md#latticespace) they are the coordinates of a cell, `x` along the first
axis and `y` along the second. On a [`NetworkSpace`](../core_components/spaces.md#networkspace) the one axis is
`node`, so `values={'node': ['Paris', 'Rome']}` puts two agents at those nodes. The agent starts where its values say,
and a rule that changes them moves it. Here two
walkers start at given points of a box of 50 by 50 and drift one unit to the right per step.

```python
from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import UpdateLaw
from skcomplex.core.components.spaces import ContinuousSpace
from skcomplex.core.systems import System
from skcomplex.core.types import Role


class Drift(Interaction):
    def __init__(self, label, *, target: Role = 'walkers', step=1.0):
        super().__init__(label)
        self.target = target
        self.step = step

    def _prepare(self, s):
        super()._prepare(s)
        self.laws_ = [UpdateLaw(s.target, self.move)]

    def move(self, agent, system):
        agent.x += self.step


plane = ContinuousSpace(extent=[50.0, 50.0])
walkers = Element('walkers', values={'x': [22.7, 5.0], 'y': [1.5, 30.0]}, size=2, space=plane)
trajectory = System(elements=[walkers], interactions=[Drift('drift')]).prepare().simulate(until=3, random_state=0)
[trajectory.get_agents('walkers', step).locations.round(1).tolist() for step in (0, 3)]
```

The walkers start at `(22.7, 1.5)` and `(5.0, 30.0)` and end three units to the right, at `(25.7, 1.5)` and
`(8.0, 30.0)`.

- On a periodic space an agent that leaves one side enters the other, and its values wrap with it, so a car at cell 9
  of a ring of 10 cells that moves one cell is at cell 0 with `x = 0`.
- `agent.move_to(location)` moves an agent and sets its values to the new place.
- An agent whose group has no values named after the axes starts at a random place, and on a network with no more
  agents than nodes, at a node of its own.

## Rules of the whole system

A rule may also act on the whole system once per step, such as a matching of riders and drivers or a braking event
drawn for a whole road. Its law is [`UpdateLaw`](../core_components/laws.md#updatelaw)`(s, rule)`, whose target is the
namespace of the interaction itself, and the rule is a function of the system alone, `rule(system)`. It runs once per
step after the rules of the agents. Here three cars drive on a ring road of 30 cells, each speeding up by one cell per
step up to three, and with probability 0.3 a step brings every car of the road to a halt.

```python
from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import UpdateLaw
from skcomplex.core.components.spaces import LatticeSpace
from skcomplex.core.systems import System
from skcomplex.core.types import Role


class Drive(Interaction):
    def __init__(self, label, *, target: Role = 'cars'):
        super().__init__(label)
        self.target = target

    def _prepare(self, s):
        super()._prepare(s)
        self.laws_ = [UpdateLaw(s.target, self.drive)]

    def drive(self, car, system):
        car.x += car.v
        car.v = min(car.v + 1, 3)


class Braking(Interaction):
    def __init__(self, label, *, cars: Role = 'cars', p=0.3):
        super().__init__(label)
        self.cars = cars
        self.p = p

    def _prepare(self, s):
        super()._prepare(s)
        self.laws_ = [UpdateLaw(s, self.brake)]

    def brake(self, system):
        if system.rng.random() < self.p:
            for car in system.get_agents('cars'):
                car.v = 0


cars = Element('cars', values={'x': [0, 10, 20], 'v': 0}, size=3, space=LatticeSpace(30))
road = System(elements=[cars], interactions=[Drive('drive'), Braking('braking')]).prepare()
trajectory = road.simulate(until=6, random_state=0)
[trajectory.get_agents('cars', step).values['x'].tolist() for step in range(7)]
```

In this run the road brakes in the first, fourth, and fifth steps, so the cars, which start at the cells 0, 10, and
20, end three cells further on, at 3, 13, and 23. Every car moves alike, since the braking is one
event of the whole road and not a draw of each car.

## Activation and phases

How agents take turns is a setting of the solver, given to the system as `solver=MesaSolver(activation=...)`.

- `'random'`, the default, activates every agent once per step in a new random order.
- `'sequential'` activates them in their order of creation.
- `'simultaneous'` lets every agent read the values of the start of the step, and its changes apply at the end.
- `'by-type'` activates group after group, each in random order.

A step may also run in named phases, each a `PhaseSpec` with an activation of its own, and a rule joins a phase with
`UpdateLaw(s.target, rule, phase=name)`. Here 20 traders quote bids in the phase `quote`, and then a clerk, a group of
one agent, reads the best bid in the phase `settle` and sets a fee of one percent of it. The clerk reads the traders
through the role `traders` of its interaction, with `system.get_agents('traders')`.

```python
import numpy as np

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import UpdateLaw
from skcomplex.core.engine.compile import PhaseSpec
from skcomplex.core.engine.solvers import MesaSolver
from skcomplex.core.systems import System
from skcomplex.core.types import Role


class Quote(Interaction):
    def __init__(self, label, *, target: Role = 'traders'):
        super().__init__(label)
        self.target = target

    def _prepare(self, s):
        super()._prepare(s)
        self.laws_ = [UpdateLaw(s.target, self.quote, phase='quote')]

    def quote(self, trader, system):
        trader.bid = system.rng.uniform(0, trader.value)


class Settle(Interaction):
    def __init__(self, label, *, target: Role = 'clerk', traders: Role = 'traders'):
        super().__init__(label)
        self.target = target
        self.traders = traders

    def _prepare(self, s):
        super()._prepare(s)
        self.laws_ = [UpdateLaw(s.target, self.settle, phase='settle')]

    def settle(self, clerk, system):
        clerk.best = max(trader.bid for trader in system.get_agents('traders'))
        clerk.fee = 0.01 * clerk.best


traders = Element('traders', values={'value': np.random.default_rng(0).uniform(0.5, 2, 20), 'bid': 0.0}, size=20)
clerk = Element('clerk', values={'best': 0.0, 'fee': 0.0})
market = System(
    elements=[traders, clerk],
    interactions=[Quote('quote'), Settle('settle')],
    solver=MesaSolver(phases=[PhaseSpec('quote'), PhaseSpec('settle')]),
)
trajectory = market.prepare().simulate(until=3, random_state=0)
[round(float(trajectory.get_agents('clerk', step).values['best'][0]), 3) for step in (1, 2, 3)]
```

The best bids of the three steps are 1.537, 1.77, and 1.529, and the fee of every step is one percent of its best bid.

- A `PhaseSpec(name, activation, repeat)` runs its agents with its own activation, for `repeat` rounds, such as
  intraday rounds. Rules without a phase run in the phase `'default'`, which runs first.
- `system.get_agents(role)` and `system.count(role)` read the agents of a group bound to a role of the interaction of
  the rule, so a rule works unchanged inside a nested system.
- `pairs(left, right, system.rng)` of `skcomplex.core.components.laws.utils` pairs the agents of one or two groups
  inside a rule, such as buyers and sellers that a clerk matches.

## Porting Mesa models

A Mesa model is ported piece by piece. The attributes of its agent class become the values of a group element, its
`step` method becomes a rule, and the attributes of its `Model` become settings of the interaction. Here ten coins
flip ten times each with the bias 0.5.

```python
from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import UpdateLaw
from skcomplex.core.systems import System
from skcomplex.core.types import Role


class Flip(Interaction):
    def __init__(self, label, *, target: Role = 'coins', bias=0.5):
        super().__init__(label)
        self.target = target
        self.bias = bias

    def _prepare(self, s):
        super()._prepare(s)
        self.laws_ = [UpdateLaw(s.target, self.flip)]

    def flip(self, coin, system):
        coin.heads += int(system.rng.random() < self.bias)


coins = System(elements=[Element('coins', values={'heads': 0}, size=10)], interactions=[Flip('flip')]).prepare()
int(coins.simulate(until=10, random_state=0).get_agents('coins', 10).values['heads'].sum())
```

The ten coins show 54 heads in 100 flips. The bias is a parameter of the system, `flip__bias`, so
`coins.set_params(flip__bias=0.9)` changes it like any parameter before the system is prepared again.
