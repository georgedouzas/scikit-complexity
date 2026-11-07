# Groups of many elements

Use a group when your system has many identical elements, such as the molecules of a gas, the cars on a road, or the
neurons of a network. You write the laws once for one member, and they run over all members at array speed.

```python
import numpy as np

from skcomplex.core.components.elements import GroupElement, PointElement
from skcomplex.core.components.spaces import ContinuousSpace
from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import Simulation

rng = np.random.default_rng(0)
values = {'x': rng.uniform(0, 1, (1_000, 2)), 'x_d1': rng.normal(0, 1, (1_000, 2))}
gas = System(
    elements=[GroupElement('gas', element=PointElement, size=1_000, values=values)],
    space=ContinuousSpace(extent=(1.0, 1.0), boundary='reflecting'),
)
trajectory = Simulation(gas, dt=0.01).prepare().run(until=0.5).trajectory_
trajectory.values('gas.x').shape
```

A group holds N identical elements of one type as arrays, with one row per member. It compiles once, whatever its
size, and every law written for one member runs over all members with array operations.

- A group is `GroupElement(label, element=..., size=..., values={...})`, with per-member values by state name. The
  states and parameters of agents cannot take the names of the agent view, such as `location` or `alive`.
- A value is an array whose first dimension is the size of the group, or one value shared by every member.
- An interaction whose role binds to a group applies to every member. A single element bound next to the group
  receives the sum over the members, unless its law reads the group only through reductions.
- Group variables are named by the group path, such as `gas.x` with shape `(1000, 2)`.
- A group parameter takes one value for every member, or one value per member, in `update_parameters`.

The catalog system `ideal_gas` builds such a gas of non-interacting molecules in a reflecting box, with the parameters
`n_molecules`, `kT`, and `size`:

```python
from skcomplex.systems import create_system

create_system('ideal_gas', n_molecules=200, kT=2.0, size=1.0).elements[0].size
```

## Reductions over a group

Use a reduction when a member's law needs a quantity of the whole group, such as its total, its mean, or the number of
members above a threshold. Here every unit reads the total of the group in its own law:

```python
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import tanh

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.laws import RateLaw
from skcomplex.core.components.laws.utils import total
from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.specs import StateSpec


class UnitElement(Element):
    def __init__(self, label, *, x: Annotated[float, StateSpec()] = 0.0):
        super().__init__(label)
        self.x = x

    def _get_laws(self, s):
        return [RateLaw(s.x, -s.x + tanh(total(s.x)))]


units = System(elements=[GroupElement('units', element=UnitElement, size=50, values={'x': np.linspace(-1, 1, 50)})])
Simulation(units, dt=0.5).prepare().run(until=2).trajectory_.values('units.x')[-1, :3].round(3).tolist()
```

`total`, `mean`, `maximum`, `minimum`, `variance`, and `count` of `skcomplex.core.components.laws.utils` reduce an
expression over the members of a group inside any law. Inside the reduction, the symbols of a member stand for the
reduced member. Outside it, they stand for the reading member. So `RateLaw(s.x, -s.x + tanh(total(s.x)))` reads the
group total in every member's own law.

A reduction is evaluated once per evaluation, before the member laws, at a cost linear in the size of the group, on
NumPy and Numba.

Reductions also take weights and nest. Replicator dynamics, where each strategy grows by how much its payoff beats the
mean payoff, needs both:

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


game = System(
    elements=[GroupElement('population', element=StrategyElement, size=2, values={'x': [0.1, 0.9]})],
    interactions=[ReplicationInteraction('replication', population='population')],
)
Simulation(game, dt=1.0).prepare().run(until=40).trajectory_.values('population.x')[-1].round(3).tolist()
```

The arguments of a reduction are:

- `where` selects the members with a condition on the reduced member, such as `count(where=s.x > 0.5)`. A reduction
  that selects no member is 0.
- `weights` gives `total` and `mean` a constant matrix, a NumPy array or a SciPy sparse matrix, with one row per
  reading member and one column per reduced member. Member `i` then reads `sum_j W_ij a_j`, such as a payoff matrix.
  A vector with one weight per reduced member is a row that every reading member shares, such as the activities of a
  moment. A shaped parameter, such as `weights=s.W`, gives weights that a schedule, an experiment, or a training
  interaction changes, as [Shaped parameters and coupling tensors](learning_by_gradient_steps.md#shaped-parameters-and-coupling-tensors) shows.
- `over` names the group by its namespace, such as `over=s.neurons`, when the expression reads no symbol of it.

Reductions follow a few rules:

- A reduction reads one group, and an argument that reads two groups raises an error.
- Reductions nest, so the mean fitness `total(s.x * total(s.x, weights=A))` of replicator dynamics evaluates the inner
  reduction first.
- An interaction reads groups member by member outside reductions, which needs groups of one size. It reads other
  groups through reductions.
- A law that sets a single element from reductions alone runs once, not once per member.
- Discrete maps read the reductions at step `n`, jump propensities after every event, and events at the state they
  receive.

**Element sets.** A role also binds to an element set, a list of single elements of one type. The interaction's laws
apply to each element, the elements keep their own state names, and reductions over the role reduce over the set.
Here three named units inhibit each other:

```python
from typing import Annotated

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import RateLaw
from skcomplex.core.components.laws.utils import total
from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import Simulation
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


network = System(
    elements=[ActivityElement(f'unit_{index}', x=0.1 * (index + 1)) for index in range(3)],
    interactions=[InhibitionInteraction('inhibition', units=['unit_0', 'unit_1', 'unit_2'])],
)
(Simulation(network, dt=1.0).prepare().run(until=30).state_.round(3) + 0.0).tolist()
```

## Variadic roles

Use a variadic role when one law takes any number of inputs, such as a node that sums weighted signals from a chosen
list of other nodes:

```python
from typing import Annotated

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import UpdateLaw
from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import Simulation
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


summed = System(
    elements=[
        NodeElement('a', x=1.0),
        NodeElement('b', x=2.0),
        NodeElement('out'),
        GroupElement('nodes', element=NodeElement, size=5, values={'x': [0.0, 1.0, 2.0, 3.0, 4.0]}),
    ],
    interactions=[WeightedSumInteraction('sum', target='out', inputs=['a', 'b', 'nodes[3]'], weights=[1.0, 1.0, 10.0])],
)
simulation = Simulation(summed, dt=1).prepare().run(until=1)
simulation.update_parameters({'sum.weights[2]': 0.0})
simulation.trajectory_.values('out.x').tolist(), simulation.parameters_.tolist()
```

- A role declared with `RoleSpec(variadic=True)` binds a list of element paths and group member references
  `<group>[<i>]`. Laws read it as a tuple of namespaces in the given order.
- A parameter declared with `ParameterSpec(per='<role>')` holds one value per entry, and laws read it as a tuple of
  symbols. Each value is the parameter component `<path>.<name>[<i>]`, which `update_parameters` and `schedule` set
  like any parameter.

An element set and a variadic role both bind a list. An element set applies the laws once per element, each element
read like a single element. A variadic role gives one law the whole sequence. `random_boolean_network` and
`yeast_cell_cycle` read the inputs of every node through one variadic role.

A role declared with `RoleSpec(noise=True)` binds a `WienerProcessSpec` instead of a path, as the shared and
correlated noise of [Delays, events, drivers, and noise](delays_events_drivers_and_noise.md) shows.

## Several groups and member offsets

Use a member offset when a member reads its neighbour in the group order, such as a car that reads the car ahead on a
ring road. A small push to the first car grows into a traffic jam:

```python
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import tanh

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import RateLaw
from skcomplex.core.components.laws.utils import mod, shift_member
from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import Simulation
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
ring = System(
    elements=[GroupElement('cars', element=CarElement, size=20, values={'x': positions})],
    interactions=[FollowingInteraction('following', cars='cars', length=40.0)],
)
bool(Simulation(ring, dt=1.0).prepare().run(until=50).trajectory_.values('cars.v')[-1].std() > 0.1)
```

`shift_member(expression, offset, boundary='periodic', fill=0.0)` reads an expression of a group's members at the
member `offset` places along the group order, the member index.

- Past the ends, `'periodic'` wraps around, `'reflecting'` mirrors the order, and `'fixed'` reads `fill`.
- Nested offsets add, and an offset read of a single element raises an error.
- The read compiles to an index table of the group, so one kernel runs over every member on the NumPy and the Numba
  backends.

The ring roads of `optimal_velocity` and `intelligent_driver` read the car ahead this way, with `mod` for the
headway across the point where positions wrap.

Roles of one interaction can also bind several groups of one size, and the laws read row `i` of every group together.
Groups of different sizes raise a `RoleError`, since coupling every member of one group with every member of another
is the work of a pair interaction or a reduction.

## Jacobians of reductions

Implicit solvers need the Jacobian of the system, and the library computes it exactly, even through reductions:

```python
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import tanh

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.laws import RateLaw
from skcomplex.core.components.laws.utils import total
from skcomplex.core.components.system import System
from skcomplex.core.engine.compile import PassCompiler
from skcomplex.core.specs import RunStateSpec, StateSpec


class UnitElement(Element):
    def __init__(self, label, *, x: Annotated[float, StateSpec()] = 0.0):
        super().__init__(label)
        self.x = x

    def _get_laws(self, s):
        return [RateLaw(s.x, -s.x + tanh(total(s.x)))]


units = System(elements=[GroupElement('units', element=UnitElement, size=50, values={'x': np.linspace(-1, 1, 50)})])
problem = PassCompiler().prepare(units).run().problem_
state = RunStateSpec(0.0, problem.initial_state, problem.initial_parameters)
jacobian = problem.jacobian.prepare(problem).run(state).values_
jacobian.shape, problem.jacobian.pattern().nnz
```

A reduction makes every reading member depend on every reduced member. The Jacobian of a drift with reductions
therefore has the rank-one block `dF_i/dR dR/dx_j` of every reduction `R`.

- `problem.jacobian` of an ODE problem is a kernel. Prepared with the problem and run at a `RunStateSpec`, it sets in
  `values_` the exact Jacobian, by the chain rule through every reduction, as a SciPy sparse matrix.
- `problem.jacobian.pattern()` states its structure, which the implicit methods of `ScipyODESolver` receive (see
  [Discretizations and stiff solvers](fields.md#discretizations-and-stiff-solvers)).
- When the dense blocks exceed 4 million entries, `pattern()` returns `None` and an implicit solver estimates a dense
  Jacobian.

Mean-field groups are rarely stiff, so an explicit method such as `ScipyODESolver(method='RK45')` suits large groups
with reductions.

`StabilityAnalysis` and the other analyses of the vector field in `skcomplex.core.analysis` expand a group of at most
64 members into one equation per member, with every reduction written as an explicit sum. The fixed points of
replicator dynamics are therefore derived symbolically. A larger group raises an error that states the limit.

## Pair interactions

Use a pair interaction for forces between members that depend on their distance, such as the atoms of a liquid:

```python
import numpy as np

from skcomplex.core.components.elements import GroupElement, PointElement
from skcomplex.core.components.interactions import PairCouplingInteraction
from skcomplex.core.components.spaces import ContinuousSpace
from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import Simulation


def lennard_jones(s, r):
    return 4 * (r**-12 - r**-6)


side = np.arange(10) * 1.2
positions = np.stack(np.meshgrid(side, side), axis=-1).reshape(-1, 2)
liquid = System(
    elements=[GroupElement('atoms', element=PointElement, size=100, values={'x': positions})],
    interactions=[PairCouplingInteraction('lj', 'atoms', 'atoms', cutoff=2.5, potential=lennard_jones)],
    space=ContinuousSpace(n_dim=2),
)
Simulation(liquid, dt=0.05).prepare().run(until=1).trajectory_.values('atoms.x').shape
```

A pair interaction is written once for a pair of members and summed over every pair closer than its cutoff, inside one
group or between two groups. `PairCouplingInteraction` adds to the rate of a state of both members a function of the
pair, or minus the gradient of a pair potential of their distance, divided by an optional `divisor`. On a state of
order 2, such as the position of a `PointElement`, the rate is its second derivative.

`period` gives the side of a periodic box, so pair distances use the nearest periodic image.

## Boundaries

Boundaries keep members inside a box, by walls that reflect them or by edges that wrap around. The gas at the top of
this section bounces off the walls of its `ContinuousSpace(boundary='reflecting')`.

- A `BoundaryLaw` sets the value of a state after every internal solver step, however long the frame, so a frame never
  reports a state that a boundary would change.
- Adaptive solvers find the first time in a step where a boundary acts, and restart there.
- Boundaries combine with every form, so walls and periodic boxes work for differential equations, stochastic
  systems, jump processes, and discrete maps.
- A `ContinuousSpace` with `boundary='reflecting'` mirrors the position states of its elements at the walls, and
  reverses the first derivatives of the positions of order 2. With `boundary='periodic'`, it wraps positions around
  the box.
