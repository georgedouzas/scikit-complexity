# Agent-based systems

Write the behaviour of individual agents as plain Python, and let them move, eat, reproduce, and die on a grid, a
network, or a continuous space. Foragers wander a torus, find food, split when they are full, and starve when they are
empty:

```python
from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.laws import RuleLaw
from skcomplex.core.components.spaces import LatticeSpace
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.specs import ParameterSpec, StateSpec, VisualSpec


class ForagerElement(Element):
    def __init__(
        self,
        label,
        visual=VisualSpec('agents', {'color_by': 'food'}),
        *,
        food: Annotated[float, StateSpec()] = 5.0,
        hunger: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
    ):
        super().__init__(label, visual=visual)
        self.food = food
        self.hunger = hunger

    def _get_laws(self, s):
        return [RuleLaw(self.forage)]

    def forage(self, agent, system):
        cells = agent.neighbor_cells()
        agent.move_to(cells[int(system.rng.integers(len(cells)))])
        agent.food += system.rng.poisson(0.8) - agent.hunger
        if agent.food <= 0:
            agent.remove()
        elif agent.food > 10:
            agent.food /= 2
            agent.create(food=agent.food)


foragers = CompositeSystem(
    elements=[GroupElement('foragers', element=ForagerElement, size=50, variable=True)],
    space=LatticeSpace((30, 30), boundary='periodic'),
)
trajectory = simulate(foragers, until=20, random_state=0)
int(trajectory.values('foragers.count')[-1]), len(trajectory.agents('foragers', 20).ids)
```

Agents are ordinary elements whose laws are rules written as code.

- An `Element` declares the states every agent holds, and the parameters its group shares, as annotated arguments of
  its `__init__`.
- Its `_get_laws` returns `RuleLaw`s, whose functions take the agent and the system.
- An element or a group whose laws include rules is an agent, and so is every member of a variable group.
- In a system with rules, a group whose element type has no laws is a group of passive agents that rules read and
  change, unless an interaction whose laws are not rules binds it.

A system with rules is agent-based. It runs in discrete time on `MesaSolver`, one Mesa step per unit of time, with
every agent activated in random order. `MesaSolver` declares what it runs like any solver, so
`simulate(system, solver=MesaSolver())` raises a `TagError` that names the laws, spaces, or times it does not support.

What an agent can do:

- An agent reads and sets its states as attributes, and reads its group's parameters, which change live like any
  parameter.
- `neighbors`, `neighbor_cells`, `agents_here`, `move_to`, `create`, and `remove` act on its place and its population.
  Only the members of a group declared `variable=True` are created and removed during a run.
- `agent.neighbors(group=path)` and `agent.agents_here(path)` read the agents of one group around the agent and at its
  place, so the agents of a variable group on the nodes of a network make a set of members per node that grows and
  that neighbours read, as the [networks](networks.md#sets-of-members-on-the-nodes) guide shows.
- `agent.displacement_to(other)` gives the vector to another agent or a location, the shortest one on a torus, and
  `agent.distance_to(other)` its length, or the number of hops on a network.
- `agent.group` is the path of the agent's group, and `agent.group_label` its label.

Where agents live:

- `GroupElement(label, element=..., size=..., location=...)` places the agents at cell coordinates, network nodes, or
  positions, one per agent or one shared. Agents without a location take random places.
- `LatticeSpace` gives agents Moore or von Neumann neighbourhoods and an optional cell capacity, `NetworkSpace` places
  agents on the nodes of a NetworkX graph, and `ContinuousSpace` finds neighbours within a radius.
- `system.dimensions` gives the extent of the space per axis, and `system.torus` whether it wraps.

What the system gives a rule:

- the random generator `system.rng`, the other agents with `system.agents(name)` and `system.count(name)`, and the
  empty cells.
- Every rule receives the system view scoped to the interaction or agent group that owns it.
  `system.parameter('name')` reads the owner's parameter. `system.agents('role')` reads the group bound to a role of the
  owner, then a group by its path relative to the owner's system, then by its path from the root. So a rule works
  unchanged in a subsystem.
- Interactions with agent rules bind any number of agent groups, of any sizes, through their roles.

What a run records: frames and trajectories hold every group's agents with their ids, states, and locations, through
`trajectory.agents(group, frame)`, and the count of every group as the reporter `<group>.count`.

The catalog has Schelling segregation as `SchellingSystem`, the voter system as `VoterSystem`, Boltzmann wealth exchange
as `BoltzmannWealthSystem`, and the wolf and sheep system as `WolfSheepSystem`.

An interaction can hold a rule too, and read the groups bound to its roles. Here walkers drift along a periodic road
inside a subsystem, and an interaction counts the pairs closer than its `reach`:

```python
from typing import Annotated

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import RuleLaw
from skcomplex.core.components.spaces import ContinuousSpace
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.specs import ParameterSpec, RoleSpec


class WalkerElement(Element):
    def _get_laws(self, s):
        return [RuleLaw(self.step)]

    def step(self, agent, system):
        width, height = system.dimensions
        x, y = agent.location
        agent.move_to(((x + system.rng.uniform(0, 2)) % width, y))


class SpreadInteraction(Interaction):
    def __init__(self, label, *, walkers: Annotated[str, RoleSpec()], reach: Annotated[float, ParameterSpec()] = 10.0):
        super().__init__(label)
        self.walkers = walkers
        self.reach = reach

    def _get_laws(self, s):
        return [RuleLaw(self.measure)]

    def measure(self, system):
        walkers = system.agents('walkers')
        close = [a for a in walkers for b in walkers if a != b and a.distance_to(b) < system.parameter('reach')]
        counts.append(len(close) // 2)


counts = []
places = [(10.0 * index, 0.5) for index in range(5)]
road = CompositeSystem(
    systems=[
        CompositeSystem(
            'road',
            elements=[GroupElement('walkers', element=WalkerElement, size=5, location=places)],
            interactions=[SpreadInteraction('spread', walkers='walkers')],
        ),
    ],
    space=ContinuousSpace(extent=(50.0, 1.0), boundary='periodic'),
)
_ = simulate(road, until=5, random_state=0)
counts
```

## Phases, markets, institutions, and learning

Order each step into phases, match buyers with sellers in a market, let an institution set policy, and let agents learn
from their past. Traders quote prices in one phase and trade in a batch auction in the next, while a clerk sets a fee
from the volume traded:

```python
from typing import Annotated, ClassVar

import numpy as np

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.interactions import FeedbackInteraction, MatchingInteraction
from skcomplex.core.components.laws import RuleLaw
from skcomplex.core.components.spaces import DiscreteTime
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.specs import BatchMatchingSpec, ParameterSpec, PhaseSpec, PhasesScheduleSpec, StateSpec


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


class ClerkInteraction(FeedbackInteraction):
    reports: ClassVar[tuple[str, ...]] = ('traded',)

    def __init__(self, label, *, rule=None, phase=None, fee: Annotated[float, ParameterSpec()] = 0.0):
        super().__init__(label, rule=rule, phase=phase)
        self.fee = fee

    def act(self, system):
        traded = sum(agent.goods for agent in system.agents('buyers'))
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
        ClerkInteraction('clerk', phase='trade'),
    ],
    time=DiscreteTime(schedule=PhasesScheduleSpec([PhaseSpec('quote'), PhaseSpec('trade')])),
)
trajectory = simulate(auction, until=5, random_state=0)
prices = [round(float(price), 3) for price in trajectory.values('auction.value')[1:]]
prices, float(trajectory.values('clerk.traded')[-1]), round(float(trajectory.values('clerk.fee')[-1]), 3)
trajectory.values('auction.volume').tolist()
```

**Phases.** `DiscreteTime(schedule=PhasesScheduleSpec([...]))` lists the phases of a step.

- A `PhaseSpec(name, activation, repeat)` runs its agents `'random'`ly, `'sequential'`ly in order of creation,
  `'simultaneous'`ly, or `'by-type'`, for `repeat` rounds, such as intraday rounds.
- In a simultaneous phase every rule reads the states of the start of the phase, and the writes apply at its end.
- The activation `'visits'` runs `visits` times as many visits as the agents of `groups` at the start of the phase.
  Each visit goes to an agent drawn with replacement, which runs its rules of the phase, its element's rules and then
  its member rules, at the time `k + i / n` of visit `i` of `n` in step `k`.
- `RuleLaw(function, phase=name)` puts a rule in a phase. Rules without a phase run in the phase `'default'`, which runs
  first unless the system lists it. So systems without phases keep one random activation per step.

**Markets.** `MatchingInteraction(label, requesters=..., providers=..., rule=...)`, from
`skcomplex.core.components.interactions`, matches two groups.

- Requesters state the highest clearing value they accept in their `reserve` state, the quantity they want in
  `quantity`, and the most value they can give in `cap`. Providers state their lowest clearing value and the
  quantity they offer. A missing state counts as no reserve, one unit, and no cap.
- A name given as a pair, such as `reserve=('bid', 'ask')`, names the state of the requesters and of the providers.
- The rule is a frozen spec of `skcomplex.core.specs` that names the matching algorithm and holds only its settings:
  - `RandomSearchSpec(tries, value)`, the default,
  - `PartnerSwitchSpec(sample, intensity, partner)`, partner choice with switching,
  - `QueueSpec(order)`,
  - `IterativeMatchingSpec(adjustment, iterations, tolerance)`, a clearing value adjusted until requests and offers
    balance,
  - `ContinuousMatchingSpec(arrival)`, where arriving offers match resting ones by limit and arrival,
  - `BatchMatchingSpec()`, one clearing value for all offers,
  - `StableMatchingSpec(requester_preference, provider_preference, value)`, stable matching by the `matching`
    library,
  - `PartnerSpec(candidates, choice, fallback, capacity, score, value)`, where requesters in random order pick among
    the candidates that still offer a quantity at a limit they accept.
- The rule may also be a function `rule(requesters, providers, rng, capacity)` that returns pairs, which match at the
  midpoint of both limits.
- The matching runs the algorithm of its rule on the offers of every round. Each match is a `MatchSpec`, whose `value`
  is the clearing value per unit.
- Every match lowers the quantities of both sides and moves the states of `transfers`. With `('money', -1)` the value
  times the quantity moves from the requester to the provider, and with `('goods', 1)` the quantity moves back, so the
  totals stay the same. `on_match(match, system)` adds any other effect.
- The matching reports `<label>.value`, the quantity-weighted clearing value, `<label>.volume`, and
  `<label>.unmatched`.

**Agent operations.** These interactions of `skcomplex.core.components.interactions` run as agent rules:

- `MovementInteraction` moves agents by neighbour, Gaussian, Lévy, best-cell, or user steps, with a condition, an
  acceptance, exclusion, and a cost.
- `TransferInteraction` moves amounts of states between members, the agents of one cell, a `Source()`, and a `Sink()`.
  It runs as transfer laws with `form='symbolic'`, or between paired agents with `form='agent'`.
- `BirthDeathInteraction` creates offspring by probability or rate, with inherited, split, and mutated states, and
  removes agents, the first or a random member of a group with `cull` and `pick`.
- `system.composition(group, state)` counts the agents of a group by the value of a state, kept up to date as agents
  change, so a birth probability can read the composition of the population.
- `GroupingInteraction` draws pairs, partitions, neighbourhoods, or groups by a state every step, and acts on each.
- `TopologyInteraction` adds and removes nodes and edges of a network, and rewires edges towards a weight.

**Institutions.** A `FeedbackInteraction`, from `skcomplex.core.components.interactions`, acts once per round of its
phase with its method `act`.

- It reads `system.observable(name)`, the reporters and the observables of the run at the end of earlier steps through
  `system.history(name, window)`, and the agents.
- It sets parameters of any element or interaction with `system.set_parameter(name, value)`. Rules see the change at
  once, and the trajectory records it.
- A subclass that declares its own parameters takes them as annotated arguments of its `__init__`, next to `rule` and
  `phase`.
- Elements and interactions declare the values they publish in the class variable `reports`, and publish them with
  `system.report(name, value)`. `system.storage` keeps private values of an element or an interaction across the steps
  of a run.

**Learning.** Agents learn from their own past and from each other.

- `agent.lags(state, window)`, a method of the agent view a rule receives, returns the past values of a state declared
  with lags, `StateSpec(lags=m)`. A rule writes adaptive, trend-following, anchoring, and fundamentalist forecasts from
  these values.
- In `skcomplex.core.components.interactions`, `ChoiceInteraction(form='agent')` draws an option by the logit,
  Metropolis, Fermi, or best rule, from a utility `utility(agent, system, option)` that can read `system.history` and
  `system.storage`.
- The economics `PayoffLearningInteraction` moves the scores of options towards their realized payoffs, or with
  `chosen` only the score of the option a member chose.
- Both run with `schedule='member'` in the activation of every member, such as one visit.
- `EvolutionInteraction` replaces strategies by selection, crossover, and mutation, with the system's random stream.
- `ImitationInteraction(rule='best')` copies the best neighbour.

## Porting Mesa systems

Bring an existing Mesa model over, and gain live, validated, recorded parameters. A coin-flipping Mesa model becomes:

```python
from typing import Annotated

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.laws import RuleLaw
from skcomplex.core.components.spaces import DiscreteTime
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.specs import ParameterSpec, StateSpec


class CoinElement(Element):
    def __init__(
        self,
        label,
        *,
        heads: Annotated[float, StateSpec()] = 0.0,
        bias: Annotated[float, ParameterSpec()] = 0.5,
    ):
        super().__init__(label)
        self.heads = heads
        self.bias = bias

    def _get_laws(self, s):
        return [RuleLaw(self.flip)]

    def flip(self, agent, system):
        agent.heads += float(system.rng.random() < agent.bias)


coins = CompositeSystem('coins', elements=[GroupElement('coins', element=CoinElement, size=10)], time=DiscreteTime())
int(simulate(coins, until=10, random_state=0).agents('coins', 10).values['heads'].sum())
```

A Mesa model is ported to a system piece by piece:

- its agent class becomes an element whose step is a `RuleLaw`,
- its agent attributes become states,
- the attributes of its `Model` become parameters, which then change live, are validated, and are recorded like those
  of any system.

Mesa still runs the steps, as the backend of agent-based systems.
