# Groups of many elements

This page builds systems of many identical elements, such as the molecules of a gas, the cars on a road, or the
neurons of a network. A group writes the laws once for one member and runs them over all members with array
operations.

Here one [`Element`](../core_components/elements.md#element) with `size=1000` holds 1000 atoms, each with a position
`x` in the plane. The law `x.diff(t, 2) = 0` lets every atom coast, so the position has a velocity `"x'"`, given with
the starting values. The atoms start at random places with random velocities inside a unit box, a
[`ContinuousSpace`](../core_components/spaces.md#continuousspace) with reflecting walls, and a
[`System`](../core_components/systems.md#system) holds them.

```python
import numpy as np

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import RateLaw
from skcomplex.core.components.spaces import ContinuousSpace
from skcomplex.core.systems import System

rng = np.random.default_rng(0)
values = {'x': rng.uniform(0, 1, (1_000, 2)), "x'": rng.normal(0, 1, (1_000, 2))}
box = ContinuousSpace(extent=(1.0, 1.0), boundary='reflecting')
coast = Interaction('coast', roles={'target': 'gas'}, laws=lambda s: [RateLaw(s.target.x.diff(s.t, 2), 0.0)])
gas = System(elements=[Element('gas', values=values, size=1_000, space=box)], interactions=[coast]).prepare()
trajectory = gas.simulate(until=0.5, dt=0.01)
trajectory.get_values('gas.x').shape
```

The positions hold 51 frames of 1000 atoms in two dimensions. A group holds N members of one element as arrays, with
one row per member. It compiles once, whatever its size, and every law written for one member runs over all members
with array operations. A group follows these rules.

- A group is `Element(label, values={...}, size=...)`, with per-member values by name, or an element class of your own
  given `size=`. The values of agents cannot take the names of the agent view, such as `location` or `alive`.
- A value is an array whose first dimension is the size of the group, or one value shared by every member.
- An interaction whose [role](../glossary.md#role) binds to a group applies to every member. A single element bound
  next to the group receives the sum over the members, unless its law reads the group only through reductions.
- Group values are named by the group path, such as `gas.x` with shape `(1000, 2)`.
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
import numpy as np
from sage.all__sagemath_symbolics import tanh

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import RateLaw
from skcomplex.core.components.laws.utils import total
from skcomplex.core.systems import System


def relax(s):
    x = s.target.x
    return [RateLaw(x.diff(s.t), -x + tanh(total(x)))]


units = Element('units', values={'x': np.linspace(-1, 1, 50)}, size=50)
system = System(elements=[units], interactions=[Interaction('relax', roles={'target': 'units'}, laws=relax)]).prepare()
system.simulate(until=2, dt=0.5).get_values('units.x')[-1, :3].round(3).tolist()
```

The first three units end at -0.861, -0.855, and -0.85 at time 2. The initial values sum to zero up to a rounding
error of about `4e-15`, but this balance is unstable. Each unit feeds the total back to all 50, so the error grows, the
total turns strongly negative, and every unit is pulled toward -1.

`total`, `mean`, `maximum`, `minimum`, `variance`, and `count` of `skcomplex.core.components.laws.utils` reduce an
expression over the members of a group inside any law. Inside the reduction, the symbols of a member stand for the
reduced member. Outside it, they stand for the reading member. So `RateLaw(x.diff(t), -x + tanh(total(x)))` reads
the group total in every member's own law, and `system.equations_` shows it as
`{'units.x[i]': '-units.x[i] + tanh(sum(units.x))'}`.

The engine evaluates a reduction once per evaluation of the laws, before the member laws, at a cost linear in the
size of the group, on NumPy and Numba.

Reductions also take weights and nest. Replicator dynamics, where the share of each strategy grows by how much its
payoff beats the mean payoff, needs both. Here an [`Interaction`](../core_components/interactions.md#interaction)
reads a population of two strategies, with a payoff matrix in which each strategy does better against the other.

```python continuation
from skcomplex.core.types import Role

payoff = np.array([[-0.5, 2.0], [0.0, 1.0]])


class Replication(Interaction):
    def __init__(self, label, *, population: Role = 'population'):
        super().__init__(label)
        self.population = population

    def _prepare(self, s):
        super()._prepare(s)
        x = s.population.x
        self.laws_ = [RateLaw(x.diff(s.t), x * (total(x, weights=payoff) - total(x * total(x, weights=payoff))))]


population = Element('population', values={'x': [0.1, 0.9]}, size=2)
game = System(elements=[population], interactions=[Replication('replication')]).prepare()
game.simulate(until=40, dt=1.0).get_values('population.x')[-1].round(3).tolist()
```

The shares settle at 0.667 and 0.333, where the two payoffs `-0.5 x_1 + 2 x_2` and `x_2` are equal. A reduction takes
these arguments.

- `where` selects the members with a condition on the reduced member, such as `count(where=s.x > 0.5)`. A reduction
  that selects no member is 0.
- `weights` gives `total` and `mean` a constant matrix, a NumPy array or a SciPy sparse matrix, with one row per
  reading member and one column per reduced member. Member `i` then reads `sum_j W_ij a_j`, such as a payoff matrix.
  A vector with one weight per reduced member is a row that every reading member shares, such as the activities of a
  moment. A parameter of a matrix shape, such as `weights=s.W`, gives weights that a schedule, an experiment, or a
  training interaction changes, as [Shaped parameters](learning_by_gradient_steps.md#shaped-parameters) and
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

**Element sets.** A role also binds to an element set, a list of single elements. The laws of the interaction apply to
each element, the elements keep their own value names, and reductions over the role reduce over the set. Here three
named units grow logistically and inhibit each other with the strength `beta = 2`.

```python
from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import RateLaw
from skcomplex.core.components.laws.utils import total
from skcomplex.core.systems import System
from skcomplex.core.types import Role


class Inhibition(Interaction):
    def __init__(self, label, *, units: Role, beta=2.0):
        super().__init__(label)
        self.units = units
        self.beta = beta

    def _prepare(self, s):
        super()._prepare(s)
        x = s.units.x
        self.laws_ = [RateLaw(x.diff(s.t), x * (1 - x) - s.beta * x * (total(x) - x))]


network = System(
    elements=[Element(f'unit_{index}', values={'x': 0.1 * (index + 1)}) for index in range(3)],
    interactions=[Inhibition('inhibition', units=['unit_0', 'unit_1', 'unit_2'])],
).prepare()
(network.simulate(until=30, dt=1.0).states[-1].round(3) + 0.0).tolist()
```

The unit that starts highest, `unit_2`, ends at 1.0 and the other two at 0.0. Strong mutual inhibition lets one winner
take all.

## Variadic roles

This section gives one law any number of inputs, such as a node that sums weighted signals from a chosen list of
other nodes. A variadic role binds such a list. Here the node `out` sums two single nodes and the member 3 of a group,
with the weights 1, 1, and 10, through an [`UpdateLaw`](../core_components/laws.md#updatelaw). A schedule sets the
third weight to 0 from time 1.

```python
from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import UpdateLaw
from skcomplex.core.systems import System
from skcomplex.core.types import Role


class WeightedSum(Interaction):
    def __init__(self, label, *, target: Role, inputs: tuple[Role, ...], weights=1.0):
        super().__init__(label)
        self.target = target
        self.inputs = inputs
        self.weights = weights

    def _prepare(self, s):
        super()._prepare(s)
        total_input = sum(weight * item.x for weight, item in zip(s.weights, s.inputs, strict=True))
        self.laws_ = [UpdateLaw(s.target.x, total_input)]


summed = System(
    elements=[
        Element('a', values={'x': 1.0}),
        Element('b', values={'x': 2.0}),
        Element('out', values={'x': 0.0}),
        Element('nodes', values={'x': [0.0, 1.0, 2.0, 3.0, 4.0]}, size=5),
    ],
    interactions=[WeightedSum('sum', target='out', inputs=('a', 'b', 'nodes[3]'), weights=[1.0, 1.0, 10.0])],
).prepare()
trajectory = summed.simulate(until=2, dt=1, schedule={'sum.weights[2]': lambda t: 10.0 if t < 1 else 0.0})
trajectory.get_values('out.x').tolist(), trajectory.get_values('sum.weights')[-1].tolist()
```

The node `out` reads 0.0, then `1 + 2 + 10 * 3 = 33.0`, then 3.0 once the third weight is 0. The weights end at
`[1.0, 1.0, 0.0]`. The role and the parameter follow these rules.

- A role typed `tuple[Role, ...]` binds a tuple of element paths and group member references `<group>[<i>]`. Laws
  read it as a tuple of namespaces in the given order.
- A parameter given as a list, such as `weights=[1.0, 1.0, 10.0]`, holds one value per entry, and laws read it as a
  tuple of symbols. Each value is the parameter component `<path>.<name>[<i>]`, which the `schedule` of `simulate`
  sets like any parameter.

An element set and a variadic role both bind a list. An element set applies the laws once per element, each element
read like a single element. A variadic role gives one law the whole sequence. `YeastCellCycleSystem` of
`skcomplex.systems.biology.systems_biology` reads the inputs of every node through one variadic role.

An argument typed `WienerProcessSpec` binds a shared noise process instead of a path, as the shared and correlated
noise of [Delays, events, drivers, and noise](delays_events_drivers_and_noise.md) shows.

## Several groups and member offsets

This section lets a member read its neighbour in the group order, such as a car that reads the car ahead on a ring
road. A member offset gives that read. Here 20 cars sit 2.0 apart on a ring of length 40, and each speeds toward a
velocity that depends on its headway, the distance to the car ahead. A push of 0.1 to the first car breaks the even
spacing.

```python
import numpy as np
from sage.all__sagemath_symbolics import tanh

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import RateLaw
from skcomplex.core.components.laws.utils import mod, shift_member
from skcomplex.core.systems import System
from skcomplex.core.types import Role


class Following(Interaction):
    def __init__(self, label, *, cars: Role = 'cars', length=40.0):
        super().__init__(label)
        self.cars = cars
        self.length = length

    def _prepare(self, s):
        super()._prepare(s)
        x, v = s.cars.x, s.cars.v
        headway = mod(shift_member(x, 1) - x, s.length)
        self.laws_ = [RateLaw(x.diff(s.t), v), RateLaw(v.diff(s.t), tanh(headway - 2.0) + tanh(2.0) - v)]


positions = np.arange(20) * 2.0 + np.eye(20)[0] * 0.1
cars = Element('cars', values={'x': positions, 'v': 1.0}, size=20)
ring = System(elements=[cars], interactions=[Following('following', length=40.0)]).prepare()
bool(ring.simulate(until=50, dt=1.0).get_values('cars.v')[-1].std() > 0.1)
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
import numpy as np
from sage.all__sagemath_symbolics import tanh

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import RateLaw
from skcomplex.core.components.laws.utils import total
from skcomplex.core.engine.problems import RunStateSpec
from skcomplex.core.systems import System


def relax(s):
    x = s.target.x
    return [RateLaw(x.diff(s.t), -x + tanh(total(x)))]


units = Element('units', values={'x': np.linspace(-1, 1, 50)}, size=50)
system = System(elements=[units], interactions=[Interaction('relax', roles={'target': 'units'}, laws=relax)]).prepare()
problem = system.simulate(until=1).problem
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
- When the dense blocks exceed 4 million entries, `get_pattern()` returns `None` and an implicit solver estimates a
  dense Jacobian.

Mean-field groups are rarely stiff, so an explicit method such as `ScipyODESolver(method='RK45')` suits large groups
with reductions.

`stability` and the other analyses of the vector field in `skcomplex.core.analysis` expand a group of at most
64 members into one equation per member, with every reduction written as an explicit sum. The fixed points of
replicator dynamics are therefore derived symbolically. A larger group raises an error that states the limit.

## Pair interactions

This section adds forces between members that depend on their distance, such as the forces between the atoms of a
liquid. An [`Interaction`](../core_components/interactions.md#interaction) with `within=RadiusNeighborhood(r)` gives
them. Here 100 atoms on a square grid of spacing 1.2 interact through the Lennard-Jones potential `4 (r^-12 - r^-6)`,
cut off at the distance 2.5. Its force on the first atom of a pair is minus its derivative times the unit vector from
the second atom, `24 (2 r^-12 - r^-6) d / r^2` for the gap `d`, and the second atom feels the opposite force.

```python
import numpy as np
from sage.all__sagemath_symbolics import sqrt

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import RateLaw
from skcomplex.core.components.spaces import ContinuousSpace, RadiusNeighborhood
from skcomplex.core.systems import System


def lennard_jones(s):
    gap = s.first.x - s.second.x
    r = sqrt(sum(item**2 for item in gap))
    force = 24 * (2 * r**-12 - r**-6) * gap / r**2
    return [RateLaw(s.first.x.diff(s.t, 2), force), RateLaw(s.second.x.diff(s.t, 2), -force)]


side = np.arange(10) * 1.2
positions = np.stack(np.meshgrid(side, side), axis=-1).reshape(-1, 2)
plane = ContinuousSpace(n_dim=2)
atoms = Element('atoms', values={'x': positions}, size=100, space=plane)
pairs = Interaction(
    'lj', roles={'first': 'atoms', 'second': 'atoms'}, within=RadiusNeighborhood(2.5), laws=lennard_jones
)
liquid = System(elements=[atoms], interactions=[pairs], space=plane).prepare()
liquid.simulate(until=1, dt=0.05).get_values('atoms.x').shape
```

The positions hold 21 frames of 100 atoms in two dimensions. The laws are written once for a pair of members and run
for every pair closer than the radius, inside one group or between the groups of the roles `first` and `second`. They
read the two members as `s.first` and `s.second` and give the laws of both. The laws give the second derivative of the
positions, so the velocities `atoms.x'` are states too.

## Boundaries

This section keeps members inside a box, by walls that reflect them or by edges that wrap around. The gas at the top
of this page bounces off the walls of its `ContinuousSpace(boundary='reflecting')`. A wall of your own is an
[`UpdateLaw`](../core_components/laws.md#updatelaw) beside the rate laws, which corrects the state after every step.
Here 1000 walkers drift toward 0 with noise, and `UpdateLaw(x, abs(x))` reflects them at the wall at 0.

```python
from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import NoiseLaw, RateLaw, UpdateLaw
from skcomplex.core.systems import System


def reflected(s):
    x = s.target.x
    return [RateLaw(x.diff(s.t), -0.5), NoiseLaw(x, 1.0), UpdateLaw(x, abs(x))]


walkers = Element('walkers', values={'x': 0.1}, size=1000)
wall = Interaction('wall', roles={'target': 'walkers'}, laws=reflected)
trajectory = System(elements=[walkers], interactions=[wall]).prepare().simulate(until=10, dt=0.01, random_state=0)
x = trajectory.get_values('walkers.x')
bool(x.min() >= 0), round(float(x[-1].mean()), 2)
```

No walker goes below 0, and their mean position at time 10 is 0.95, close to the mean 1 of the exponential spread
that a reflected walk with this drift settles into. Boundaries follow these rules.

- An `UpdateLaw` of a state that rate or noise laws also change is a correction. It sets the value of the state after
  every internal solver step, however long the frame, so a frame never reports a state that a boundary would change.
- Adaptive solvers find the first time in a step where a boundary acts, and restart there.
- The walls and periodic boxes of a space work for differential equations, stochastic systems, jump processes, and
  discrete maps.
- A `ContinuousSpace` with `boundary='reflecting'` mirrors the position states of its elements at the walls, and
  reverses their velocities `x'`. With `boundary='periodic'`, it wraps positions around
  the box.
