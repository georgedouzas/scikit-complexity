# Groups of many elements

This page builds systems of many identical elements, such as the molecules of a gas, the cars on a road, or the
neurons of a network. A group writes the laws once for one member and runs them over all members with array
operations.

Here a [`GroupElement`](../core_components/elements.md#groupelement) holds 1000 atoms of an
[`Element`](../core_components/elements.md#element) subclass with a two-dimensional position of order 2, so the
position has a velocity `x_d1`. The atoms start at random places with random velocities inside a unit box, a
[`ContinuousSpace`](../core_components/spaces.md#continuousspace) with reflecting walls, and a
[`CompositeSystem`](../core_components/systems.md#compositesystem) holds them.

```python
from typing import Annotated

import numpy as np

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.spaces import ContinuousSpace
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.specs import StateSpec


class AtomElement(Element):
    def __init__(
        self, label, *, x: Annotated[object, StateSpec(vector=True, order=2, role='position')] = 0.0, x_d1: object = 0.0
    ):
        super().__init__(label)
        self.x = x
        self.x_d1 = x_d1


rng = np.random.default_rng(0)
values = {'x': rng.uniform(0, 1, (1_000, 2)), 'x_d1': rng.normal(0, 1, (1_000, 2))}
gas = CompositeSystem(
    elements=[GroupElement('gas', element=AtomElement, size=1_000, values=values)],
    space=ContinuousSpace(extent=(1.0, 1.0), boundary='reflecting'),
)
trajectory = simulate(gas, until=0.5, dt=0.01)
trajectory.get_values('gas.x').shape
```

The positions hold 51 frames of 1000 atoms in two dimensions. A group holds N identical elements of one class as
arrays, with one row per member. It compiles once, whatever its size, and every law written for one member runs over
all members with array operations. A group follows these rules.

- A group is `GroupElement(label, element=..., size=..., values={...})`, with per-member values by state name. The
  states and parameters of agents cannot take the names of the agent view, such as `location` or `alive`.
- A value is an array whose first dimension is the size of the group, or one value shared by every member.
- An interaction whose [role](../glossary.md#role) binds to a group applies to every member. A single element bound next to the group
  receives the sum over the members, unless its law reads the group only through reductions.
- Group variables are named by the group path, such as `gas.x` with shape `(1000, 2)`.
- A group parameter takes one value for every member, or one value per member, in the `schedule` of `simulate`.

The ready-made `IdealGasSystem` builds such a gas of non-interacting molecules in a reflecting box, with the model
parameters `n_molecules`, `kT`, and `size`. `prepare` builds its components in `system_`.

```python
from skcomplex.systems.physics.statistical_mechanics import IdealGasSystem

IdealGasSystem(n_molecules=200, kT=2.0, size=1.0).prepare().system_.elements[0].size
```

The first element of the built system is a group of 200 molecules.

## Reductions over a group

This section lets the law of a member read a quantity of the whole group, such as its total, its mean, or the number
of members above a threshold. Such a quantity is a reduction. Here every unit of 50 relaxes toward the hyperbolic
tangent of the group total, through a [`RateLaw`](../core_components/laws.md#ratelaw).

```python
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import tanh

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.laws import RateLaw
from skcomplex.core.components.laws.utils import total
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.specs import StateSpec


class UnitElement(Element):
    def __init__(self, label, *, x: Annotated[float, StateSpec()] = 0.0):
        super().__init__(label)
        self.x = x

    def _get_laws(self, s):
        return [RateLaw(s.x, -s.x + tanh(total(s.x)))]


units = CompositeSystem(elements=[GroupElement('units', element=UnitElement, size=50, values={'x': np.linspace(-1, 1, 50)})])
simulate(units, until=2, dt=0.5).get_values('units.x')[-1, :3].round(3).tolist()
```

The first three units end at -0.861, -0.855, and -0.85 at time 2. The initial values sum to zero up to a rounding
error of about `4e-15`, but this balance is unstable. Each unit feeds the total back to all 50, so the error grows, the
total turns strongly negative, and every unit is pulled toward -1.

`total`, `mean`, `maximum`, `minimum`, `variance`, and `count` of `skcomplex.core.components.laws.utils` reduce an
expression over the members of a group inside any law. Inside the reduction, the symbols of a member stand for the
reduced member. Outside it, they stand for the reading member. So `RateLaw(s.x, -s.x + tanh(total(s.x)))` reads the
group total in every member's own law.

The engine evaluates a reduction once per evaluation of the laws, before the member laws, at a cost linear in the
size of the group, on NumPy and Numba.

Reductions also take weights and nest. Replicator dynamics, where the share of each strategy grows by how much its
payoff beats the mean payoff, needs both. Here an [`Interaction`](../core_components/interactions.md#interaction)
reads a population of two strategies, with a payoff matrix in which each strategy does better against the other.

```python continuation
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.specs import RoleSpec

payoff = np.array([[-0.5, 2.0], [0.0, 1.0]])


class StrategyElement(Element):
    def __init__(self, label, *, x: Annotated[float, StateSpec()] = 0.5):
        super().__init__(label)
        self.x = x


class ReplicationInteraction(Interaction):
    def __init__(self, label, *, population: Annotated[str, RoleSpec()]):
        super().__init__(label)
        self.population = population

    def _get_laws(self, s):
        x = s.population.x
        return [RateLaw(x, x * (total(x, weights=payoff) - total(x * total(x, weights=payoff))))]


game = CompositeSystem(
    elements=[GroupElement('population', element=StrategyElement, size=2, values={'x': [0.1, 0.9]})],
    interactions=[ReplicationInteraction('replication', population='population')],
)
simulate(game, until=40, dt=1.0).get_values('population.x')[-1].round(3).tolist()
```

The shares settle at 0.667 and 0.333, where the two payoffs `-0.5 x_1 + 2 x_2` and `x_2` are equal. A reduction takes
these arguments.

- `where` selects the members with a condition on the reduced member, such as `count(where=s.x > 0.5)`. A reduction
  that selects no member is 0.
- `weights` gives `total` and `mean` a constant matrix, a NumPy array or a SciPy sparse matrix, with one row per
  reading member and one column per reduced member. Member `i` then reads `sum_j W_ij a_j`, such as a payoff matrix.
  A vector with one weight per reduced member is a row that every reading member shares, such as the activities of a
  moment. A shaped parameter, such as `weights=s.W`, gives weights that a schedule, an experiment, or a training
  interaction changes, as [Shaped parameters](learning_by_gradient_steps.md#shaped-parameters) and
  [Coupling tensors](learning_by_gradient_steps.md#coupling-tensors) show.
- `over` names the group by its namespace, such as `over=s.neurons`, when the expression reads no symbol of it.

Reductions follow these rules.

- A reduction reads one group, and an argument that reads two groups raises an error.
- Reductions nest, so the mean fitness `total(s.x * total(s.x, weights=A))` of replicator dynamics evaluates the inner
  reduction first.
- An interaction reads groups member by member outside reductions, which needs groups of one size. It reads other
  groups through reductions.
- A law that sets a single element from reductions alone runs once, not once per member.
- Discrete maps read the reductions at step `n`, jump propensities after every event, and events at the state they
  receive.

**Element sets.** A role also binds to an element set, a list of single elements of one class. The laws of the
interaction apply to each element, the elements keep their own state names, and reductions over the role reduce over
the set. Here three named units grow logistically and inhibit each other with the strength `beta = 2`.

```python
from typing import Annotated

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import RateLaw
from skcomplex.core.components.laws.utils import total
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.specs import ParameterSpec, RoleSpec, StateSpec


class ActivityElement(Element):
    def __init__(self, label, *, x: Annotated[float, StateSpec()] = 0.1):
        super().__init__(label)
        self.x = x


class InhibitionInteraction(Interaction):
    def __init__(self, label, *, units: Annotated[str, RoleSpec()], beta: Annotated[float, ParameterSpec()] = 2.0):
        super().__init__(label)
        self.units = units
        self.beta = beta

    def _get_laws(self, s):
        x = s.units.x
        return [RateLaw(x, x * (1 - x) - s.beta * x * (total(x) - x))]


network = CompositeSystem(
    elements=[ActivityElement(f'unit_{index}', x=0.1 * (index + 1)) for index in range(3)],
    interactions=[InhibitionInteraction('inhibition', units=['unit_0', 'unit_1', 'unit_2'])],
)
(simulate(network, until=30, dt=1.0).states[-1].round(3) + 0.0).tolist()
```

The unit that starts highest, `unit_2`, ends at 1.0 and the other two at 0.0. Strong mutual inhibition lets one winner
take all.

## Variadic roles

This section gives one law any number of inputs, such as a node that sums weighted signals from a chosen list of
other nodes. A variadic role binds such a list. Here the node `out` sums two single nodes and the member 3 of a group,
with the weights 1, 1, and 10, through an [`UpdateLaw`](../core_components/laws.md#updatelaw). A schedule sets the
third weight to 0 from time 1.

```python
from typing import Annotated

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import UpdateLaw
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.specs import ParameterSpec, RoleSpec, StateSpec


class NodeElement(Element):
    def __init__(self, label, *, x: Annotated[float, StateSpec()] = 0.0):
        super().__init__(label)
        self.x = x


class WeightedSumInteraction(Interaction):
    def __init__(
        self,
        label,
        *,
        target: Annotated[str, RoleSpec()],
        inputs: Annotated[list[str], RoleSpec(variadic=True)],
        weights: Annotated[list[float], ParameterSpec(per='inputs')] = 1.0,
    ):
        super().__init__(label)
        self.target = target
        self.inputs = inputs
        self.weights = weights

    def _get_laws(self, s):
        total_input = sum(weight * item.x for weight, item in zip(s.weights, s.inputs, strict=True))
        return [UpdateLaw(s.target.x, total_input)]


summed = CompositeSystem(
    elements=[
        NodeElement('a', x=1.0),
        NodeElement('b', x=2.0),
        NodeElement('out'),
        GroupElement('nodes', element=NodeElement, size=5, values={'x': [0.0, 1.0, 2.0, 3.0, 4.0]}),
    ],
    interactions=[WeightedSumInteraction('sum', target='out', inputs=['a', 'b', 'nodes[3]'], weights=[1.0, 1.0, 10.0])],
)
trajectory = simulate(summed, until=2, dt=1, schedule={'sum.weights[2]': lambda t: 10.0 if t < 1 else 0.0})
trajectory.get_values('out.x').tolist(), trajectory.get_values('sum.weights')[-1].tolist()
```

The node `out` reads 0.0, then `1 + 2 + 10 * 3 = 33.0`, then 3.0 once the third weight is 0. The weights end at
`[1.0, 1.0, 0.0]`. The role and the parameter follow these rules.

- A role declared with `RoleSpec(variadic=True)` binds a list of element paths and group member references
  `<group>[<i>]`. Laws read it as a tuple of namespaces in the given order.
- A parameter declared with `ParameterSpec(per='<role>')` holds one value per entry, and laws read it as a tuple of
  symbols. Each value is the parameter component `<path>.<name>[<i>]`, which the `schedule` of `simulate` sets
  like any parameter.

An element set and a variadic role both bind a list. An element set applies the laws once per element, each element
read like a single element. A variadic role gives one law the whole sequence. `RandomBooleanNetworkSystem` and
`YeastCellCycleSystem` of `skcomplex.systems.biology.systems_biology` read the inputs of every node through one variadic
role.

A role declared with `RoleSpec(noise=True)` binds a `WienerProcessSpec` instead of a path, as the shared and
correlated noise of [Delays, events, drivers, and noise](delays_events_drivers_and_noise.md) shows.

## Several groups and member offsets

This section lets a member read its neighbour in the group order, such as a car that reads the car ahead on a ring
road. A member offset gives that read. Here 20 cars sit 2.0 apart on a ring of length 40, and each speeds toward a
velocity that depends on its headway, the distance to the car ahead. A push of 0.1 to the first car breaks the even
spacing.

```python
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import tanh

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import RateLaw
from skcomplex.core.components.laws.utils import mod, shift_member
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.specs import ParameterSpec, RoleSpec, StateSpec


class CarElement(Element):
    def __init__(self, label, *, x: Annotated[float, StateSpec()] = 0.0, v: Annotated[float, StateSpec()] = 1.0):
        super().__init__(label)
        self.x = x
        self.v = v

    def _get_laws(self, s):
        return [RateLaw(s.x, s.v)]


class FollowingInteraction(Interaction):
    def __init__(self, label, *, cars: Annotated[str, RoleSpec()], length: Annotated[float, ParameterSpec()] = 40.0):
        super().__init__(label)
        self.cars = cars
        self.length = length

    def _get_laws(self, s):
        headway = mod(shift_member(s.cars.x, 1) - s.cars.x, s.length)
        return [RateLaw(s.cars.v, tanh(headway - 2.0) + tanh(2.0) - s.cars.v)]


positions = np.arange(20) * 2.0 + np.eye(20)[0] * 0.1
ring = CompositeSystem(
    elements=[GroupElement('cars', element=CarElement, size=20, values={'x': positions})],
    interactions=[FollowingInteraction('following', cars='cars', length=40.0)],
)
bool(simulate(ring, until=50, dt=1.0).get_values('cars.v')[-1].std() > 0.1)
```

At time 50 the velocities of the cars differ with a standard deviation above 0.1, so the small push has grown into a
traffic jam.

`shift_member(expression, offset, boundary='periodic', fill=0.0)` reads an expression of a group's members at the
member `offset` places along the group order, the member index.

- Past the ends, `'periodic'` wraps around, `'reflecting'` mirrors the order, and `'fixed'` reads `fill`.
- Nested offsets add, and an offset read of a single element raises an error.
- The read compiles to an index table of the group, so one kernel runs over every member on the NumPy and the Numba
  [backends](../glossary.md#backend).

The ring roads of `OptimalVelocitySystem` and `IntelligentDriverSystem` of `skcomplex.systems.engineering.traffic`
read the car ahead this way, with `mod` for the headway across the point where positions wrap.

Roles of one interaction can also bind several groups of one size, and the laws read row `i` of every group together.
Groups of different sizes raise a `RoleError`, since coupling every member of one group with every member of another
is the work of a pair interaction or a reduction.

## Jacobians of reductions

This section computes the Jacobian of a system with reductions, the matrix of the derivatives of every rate with
respect to every state. Implicit solvers need it, and the library computes it exactly, by the chain rule through the
reductions. Here the 50 units of the first reduction example give their Jacobian at the initial state.

```python
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import tanh

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.laws import RateLaw
from skcomplex.core.components.laws.utils import total
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.specs import RunStateSpec, StateSpec


class UnitElement(Element):
    def __init__(self, label, *, x: Annotated[float, StateSpec()] = 0.0):
        super().__init__(label)
        self.x = x

    def _get_laws(self, s):
        return [RateLaw(s.x, -s.x + tanh(total(s.x)))]


units = CompositeSystem(elements=[GroupElement('units', element=UnitElement, size=50, values={'x': np.linspace(-1, 1, 50)})])
problem = simulate(units, until=1).problem
state = RunStateSpec(0.0, problem.initial_state, problem.initial_parameters)
jacobian = problem.jacobian.prepare(problem).run(state).values_
jacobian.shape, problem.jacobian.get_pattern().nnz
```

The Jacobian is a 50 by 50 matrix, and all of its 2500 entries can be nonzero. A reduction makes every reading member
depend on every reduced member. The Jacobian of a drift with reductions therefore has the rank-one block
`dF_i/dR dR/dx_j` of every reduction `R`.

- `problem.jacobian` of an ODE problem is a kernel. Prepared with the problem and run at a `RunStateSpec`, it sets in
  `values_` the exact Jacobian, by the chain rule through every reduction, as a SciPy sparse matrix.
- `problem.jacobian.get_pattern()` states its structure, which the implicit methods of `ScipyODESolver` receive (see
  [Discretizations and stiff solvers](fields.md#discretizations-and-stiff-solvers)).
- When the dense blocks exceed 4 million entries, `get_pattern()` returns `None` and an implicit solver estimates a dense
  Jacobian.

Mean-field groups are rarely stiff, so an explicit method such as `ScipyODESolver(method='RK45')` suits large groups
with reductions.

`stability` and the other analyses of the vector field in `skcomplex.core.analysis` expand a group of at most
64 members into one equation per member, with every reduction written as an explicit sum. The fixed points of
replicator dynamics are therefore derived symbolically. A larger group raises an error that states the limit.

## Pair interactions

This section adds forces between members that depend on their distance, such as the forces between the atoms of a
liquid. A [`PairInteraction`](../core_components/interactions.md#pairinteraction) gives them. Here 100 atoms on a
square grid of spacing 1.2 interact through the Lennard-Jones potential `4 (r^-12 - r^-6)`, cut off at the distance
2.5.

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
        self, label, *, x: Annotated[object, StateSpec(vector=True, order=2, role='position')] = 0.0, x_d1: object = 0.0
    ):
        super().__init__(label)
        self.x = x
        self.x_d1 = x_d1


def lennard_jones(s, r):
    return 4 * (r**-12 - r**-6)


side = np.arange(10) * 1.2
positions = np.stack(np.meshgrid(side, side), axis=-1).reshape(-1, 2)
liquid = CompositeSystem(
    elements=[GroupElement('atoms', element=AtomElement, size=100, values={'x': positions})],
    interactions=[PairInteraction('lj', 'atoms', 'atoms', cutoff=2.5, potential=lennard_jones)],
    space=ContinuousSpace(n_dim=2),
)
simulate(liquid, until=1, dt=0.05).get_values('atoms.x').shape
```

The positions hold 21 frames of 100 atoms in two dimensions. A pair interaction is written once for a pair of members and summed over every pair closer than its cutoff, inside one
group or between two groups. `PairInteraction` adds to the rate of a state of both members a function of the
pair, or minus the gradient of a pair potential of their distance, divided by an optional `divisor`. On a state of
order 2, such as the position of an `AtomElement`, the rate is its second derivative.

`period` gives the side of a periodic box, so pair distances use the nearest periodic image.

## Boundaries

This section keeps members inside a box, by walls that reflect them or by edges that wrap around. The gas at the top
of this page bounces off the walls of its `ContinuousSpace(boundary='reflecting')`. Boundaries follow these rules.

- A [`BoundaryLaw`](../core_components/laws.md#boundarylaw) sets the value of a state after every internal solver step, however long the frame, so a frame never
  reports a state that a boundary would change.
- Adaptive solvers find the first time in a step where a boundary acts, and restart there.
- Boundaries combine with every form, so walls and periodic boxes work for differential equations, stochastic
  systems, jump processes, and discrete maps.
- A `ContinuousSpace` with `boundary='reflecting'` mirrors the position states of its elements at the walls, and
  reverses the first derivatives of the positions of order 2. With `boundary='periodic'`, it wraps positions around
  the box.
