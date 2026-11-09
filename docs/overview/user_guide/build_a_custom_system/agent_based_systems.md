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
        cells = agent.get_neighbor_cells()
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
int(trajectory.get_values('foragers.count')[-1]), len(trajectory.get_agents('foragers', 20).ids)
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
- `agent.get_neighbors(group=path)` and `agent.get_agents_here(path)` read the agents of one group around the agent and at its
  place, so the agents of a variable group on the nodes of a network make a set of members per node that grows and
  that neighbours read, as the [networks](networks.md#sets-of-members-on-the-nodes) guide shows.
- `agent.get_displacement_to(other)` gives the vector to another agent or a location, the shortest one on a torus, and
  `agent.get_distance_to(other)` its length, or the number of hops on a network.
- `agent.group` is the path of the agent's group, and `agent.group_label` its label.

Where agents live:

- `GroupElement(label, element=..., size=..., location=...)` places the agents at cell coordinates, network nodes, or
  positions, one per agent or one shared. Agents without a location take random places.
- `LatticeSpace` gives agents Moore or von Neumann neighbourhoods and an optional cell capacity, `NetworkSpace` places
  agents on the nodes of a NetworkX graph, and `ContinuousSpace` finds neighbours within a radius.
- `system.dimensions` gives the extent of the space per axis, and `system.torus` whether it wraps.

What the system gives a rule:

- the random generator `system.rng`, the other agents with `system.get_agents(name)` and `system.count(name)`, and the
  empty cells.
- Every rule receives the system view scoped to the interaction or agent group that owns it.
  `system.get_parameter('name')` reads the owner's parameter. `system.get_agents('role')` reads the group bound to a role of the
  owner, then a group by its path relative to the owner's system, then by its path from the root. So a rule works
  unchanged in a subsystem.
- Interactions with agent rules bind any number of agent groups, of any sizes, through their roles.

What a run records: frames and trajectories hold every group's agents with their ids, states, and locations, through
`trajectory.get_agents(group, frame)`, and the count of every group as the reporter `<group>.count`.

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
        walkers = system.get_agents('walkers')
        close = [a for a in walkers for b in walkers if a != b and a.get_distance_to(b) < system.get_parameter('reach')]
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

## Phases, institutions, and learning

Order each step into phases, let an institution set policy, and let agents learn from their past. Traders quote bids in
one phase, and in the next a clerk reads the best bid and sets a fee from it:

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


class TraderElement(Element):
    def __init__(
        self,
        label,
        *,
        value: Annotated[float, StateSpec()] = 0.0,
        bid: Annotated[float, StateSpec()] = 0.0,
    ):
        super().__init__(label)
        self.value = value
        self.bid = bid

    def _get_laws(self, s):
        return [RuleLaw(self.quote, phase='quote')]

    def quote(self, agent, system):
        agent.bid = system.rng.uniform(0, agent.value)


class ClerkInteraction(Interaction):
    law_classes: ClassVar[tuple[type, ...]] = (RuleLaw,)
    reports: ClassVar[tuple[str, ...]] = ('best',)

    def __init__(self, label, *, fee: Annotated[float, ParameterSpec()] = 0.0):
        super().__init__(label)
        self.fee = fee

    def _get_laws(self, s):
        return [RuleLaw(self.act, phase='settle')]

    def act(self, system):
        best = max(agent.bid for agent in system.get_agents('traders'))
        system.report('best', best)
        system.set_parameter('fee', 0.01 * best)


values = {'value': np.random.default_rng(0).uniform(0.5, 2, 20)}
market = CompositeSystem(
    elements=[GroupElement('traders', element=TraderElement, size=20, values=values)],
    interactions=[ClerkInteraction('clerk')],
    time=DiscreteTime(schedule=PhasesScheduleSpec([PhaseSpec('quote'), PhaseSpec('settle')])),
)
trajectory = simulate(market, until=3, random_state=0)
[round(float(best), 3) for best in trajectory.get_values('clerk.best')[1:]], round(float(trajectory.get_values('clerk.fee')[-1]), 4)
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

**Markets.** A market that matches buyers with sellers is the recipe `MatchingInteraction` of
[General realizations](../core_components/general.md#matching), which runs a batch auction with phases and a clerk.

**Agent operations.** These interactions of `skcomplex.core.components.interactions` run as agent rules:

- `TransferInteraction` moves amounts of states between members, the agents of one cell, a `Source()`, and a `Sink()`.
  It runs as transfer laws with `form='symbolic'`, or between paired agents with `form='agent'`.
- `system.get_composition(group, state)` counts the agents of a group by the value of a state, kept up to date as agents
  change, so a birth probability can read the composition of the population.
- `GroupingInteraction` draws pairs, partitions, neighbourhoods, or groups by a state every step, and acts on each.
- `TopologyInteraction` adds and removes nodes and edges of a network, and rewires edges towards a weight.
- The movement of agents and their birth and death are the recipes `MovementInteraction` and `BirthDeathInteraction`
  of [General realizations](../core_components/general.md#agent-rules).

**Institutions.** An `Interaction` whose law is `RuleLaw(self.act, phase=...)` acts once per round of its phase, as
the clerk above does. Economics names such an institution `PolicyRuleInteraction`.

- It reads `system.get_observable(name)`, the reporters and the observables of the run at the end of earlier steps through
  `system.get_history(name, window)`, and the agents.
- It sets parameters of any element or interaction with `system.set_parameter(name, value)`. Rules see the change at
  once, and the trajectory records it.
- It declares its own parameters as annotated arguments of its `__init__`, as the clerk declares `fee`.
- Elements and interactions declare the values they publish in the class variable `reports`, and publish them with
  `system.report(name, value)`. `system.storage` keeps private values of an element or an interaction across the steps
  of a run.

**Learning.** Agents learn from their own past and from each other.

- `agent.get_lags(state, window)`, a method of the agent view a rule receives, returns the past values of a state declared
  with lags, `StateSpec(lags=m)`. A rule writes adaptive, trend-following, anchoring, and fundamentalist forecasts from
  these values.
- A `MapInteraction(form='agent', schedule='member')` moves a score towards a realized payoff in the activation of
  every member, such as one visit. The economics `PayoffLearningInteraction` names this learning.
- Choosing an option by its utility, copying the best neighbour, and evolving strategies are the recipes
  `ChoiceInteraction`, `ImitationInteraction`, and `EvolutionInteraction` of
  [General realizations](../core_components/general.md#choice-and-imitation).

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
int(simulate(coins, until=10, random_state=0).get_agents('coins', 10).values['heads'].sum())
```

A Mesa model is ported to a system piece by piece:

- its agent class becomes an element whose step is a `RuleLaw`,
- its agent attributes become states,
- the attributes of its `Model` become parameters, which then change live, are validated, and are recorded like those
  of any system.

Mesa still runs the steps, as the backend of agent-based systems.
