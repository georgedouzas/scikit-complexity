# User guide

## Concepts

`scikit-complexity` separates the description of a model from the way it is solved.

- An **element** is an entity that carries state, such as a particle, a population, a compartment, or a market.
- An **interaction** couples elements through **roles** and describes its effect with **laws**.
- A **law** says how a state changes. It is one of four kinds.
    - `Rate` adds a term to the time derivative of a state. Rates on one state add up.
    - `Noise` drives a state with an independent Wiener process.
    - `Jump` fires a random event that changes states by fixed increments.
    - `Update` gives the next value of a state after one discrete step.
- A **space** gives vector states, such as positions, their dimension.
- A **system** holds elements, interactions, child systems, an optional space, and **observables**, named quantities
  computed from its states at every frame.

These concepts carry no field-specific meaning. A population, a particle, a firm, and a cell are all elements, so a
model from any science is a configuration of the same interface, and every ready-made model in `skcomplex.models` is a
factory that builds such a configuration.

Compiling a system infers its mathematical form from its law kinds and builds fast numeric functions.

| Law kinds | Form | Default solver |
| --- | --- | --- |
| `Rate` | ordinary differential equations | `ScipyODE` |
| `Rate` and `Noise` | stochastic differential equations | `EulerMaruyama` |
| `Jump` | jump process | `Gillespie` |
| `Update` | discrete map | `Iteration` |

## Defining elements and interactions

Elements and interactions declare their fields with `state`, `parameter`, and `role`. Each field becomes a constructor
argument, stored unmodified under its own name.

```python
from skcomplex.core import Domain, parameter, role, state
from skcomplex.model import Element, Interaction, Jump, Rate


class Species(Element):
    n: float = state(10.0, domain=Domain.NONNEGATIVE)


class Birth(Interaction):
    target: str = role()
    rate: float = parameter(0.5, domain=Domain.NONNEGATIVE)
    stochastic: bool = False

    def laws(self, s):
        if self.stochastic:
            return [Jump(s.rate * s.target.n, {s.target.n: 1})]
        return [Rate(s.target.n, s.rate * s.target.n)]
```

Inside `laws`, `s.rate` is the symbol of the parameter and `s.target.n` is the symbol of the state of the element bound
to the role `target`. Because parameters stay symbols, a running simulation can change them without compiling again.

A law can also be a Python function of time and current values, which covers rules that have no closed form.

```python
class Threshold(Interaction):
    target: str = role()

    def laws(self, s):
        name = f'{self.target}.n'
        return [Rate(s.target.n, lambda t, values: -1.0 if values[name] > 50 else 0.0)]
```

### Symbolic expressions

Laws are Sage expressions. The symbols in `s` are Sage symbols that carry the domains of their variables, vector states
are Sage vectors, and mathematical functions come from Sage. Sage has no condition-based piecewise expression or
symbolic remainder, so `skcomplex.model` adds `where`, `either`, `both`, and `mod`, which every backend lowers, LAMMPS
included.

```python
from sage.all__sagemath_symbolics import exp, sqrt

from skcomplex.model import where


class Crowding(Interaction):
    target: str = role()
    capacity: float = parameter(100.0, domain=Domain.POSITIVE)

    def laws(self, s):
        n = s.target.n
        return [Rate(n, where(n > s.capacity, -sqrt(n - s.capacity), 0.1 * n * exp(-n / s.capacity)))]
```

Models written with SymPy move to Sage as follows. A SymPy expression in a law raises a `ValidationError`, and
`expression._sage_()` converts one that is already built.

| SymPy | Sage and `skcomplex` |
| --- | --- |
| `sympy.sqrt`, `sympy.exp`, `sympy.sin`, and other functions | `sqrt`, `exp`, `sin`, and others from `sage.all__sagemath_symbolics` |
| `sympy.Symbol('x', positive=True)` | `skcomplex.core.symbol('x', Domain.POSITIVE)` |
| `sympy.Matrix([...])` and `a.dot(b)` | `skcomplex.core.vector([...])` and `a.dot_product(b)` |
| `sympy.Piecewise((a, condition), (b, True))` | `where(condition, a, b)` |
| `condition1 \| condition2` and `condition1 & condition2` | `either(condition1, condition2)` and `both(condition1, condition2)` |
| `sympy.Mod(x, m)` | `mod(x, m)` |
| `sympy.diff(f, x)` and `f.subs(x, y)` | `f.diff(x)` and `f.subs({x: y})` |
| `sympy.Eq(a, b)`, `.lhs`, and `.rhs` | `a == b`, `.lhs()`, and `.rhs()` |

### Defining without classes

`Element.define` and `Interaction.define` build the same types from declarations and a laws function, which suits
one-off models. The types behave like classes in groups, cloning, and parameter introspection, and pickle when they are
assigned to a module-level name equal to their name.

```python
from skcomplex.model import Element

SpeciesDefined = Element.define('SpeciesDefined', states={'n': state(10.0, domain=Domain.NONNEGATIVE)})
BirthDefined = Interaction.define(
    'BirthDefined',
    roles=['target'],
    parameters={'rate': parameter(0.5, domain=Domain.NONNEGATIVE)},
    laws=lambda s: [Rate(s.target.n, s.rate * s.target.n)],
)
print(BirthDefined('birth', target='deer', rate=0.2))
```

### State domains

Besides real, positive, nonnegative, and integer values, a state takes booleans, angles, or the labels of a finite
alphabet. Values are stored as floats: booleans as 0 and 1, labels as their index, and angles wrapped into one turn
after every frame. Trajectories return booleans and labels, and `Simulation(..., checked=True)` raises an error naming
any state that leaves its domain.

```python
health = Domain.categorical('susceptible', 'infected', 'recovered')
Person = Element.define('Person', states={'health': state('susceptible', domain=health)})
Rotor = Element.define('Rotor', states={'theta': state(0.0, domain=Domain.ANGLE)})
```

### Equations of any order

Laws give the highest time derivative of a state, and a state declares how high that is with `order`. Newton's second
law for a pendulum, `m l θ'' = -m g sin θ`, is written as it reads. The framework adds the angular velocity as the
companion state `theta_d1`, sets its initial value from the constructor, and reduces the equation to first order for
the solvers.

```python
from sage.all__sagemath_symbolics import sin

from skcomplex.model import Equation, derivative

Pendulum = Element.define(
    'Pendulum',
    states={'theta': state(default=0.5, domain=Domain.ANGLE, order=2)},
    parameters={'m': 1.0, 'g': 9.81, 'l': 1.0},
    laws=lambda s: [Equation(s.m * s.l * derivative(s.theta, 2) == -s.m * s.g * sin(s.theta))],
)
```

- On a state of order `k`, rate and noise laws contribute to the `k`-th derivative and add up, so separate interactions
  supply separate forces, voltages, or growth terms: `Rate(s.target.x, -s.k * s.target.x / s.target.m)` from a spring
  and `Rate(s.target.x, -s.c * derivative(s.target.x, 1) / s.target.m)` from a damper superpose.
- `derivative(x, j)` reads the companion `x_d<j>` below the state's order. An `Equation` may contain the derivative of
  the state's own order, linearly, and the compiler solves all equations of a system jointly for those derivatives,
  which couples, for example, the two angles of a double pendulum.
- Any order works the same way, such as Sprott's chaotic jerk equation `x''' = -A x'' - x' + |x| - 1` on a state of
  order 3.
- Difference equations keep past values with `lags`. `lag(x, j)` reads the value `j` steps before the current one, and
  an update law gives the next value, so `Update(s.x, s.x + lag(s.x, 1))` with one lag is the Fibonacci recurrence.

`equations_of` returns these equations in the order they were written, and `solve_symbolically` takes the initial
values of every derivative and returns a solution only when it has a closed form.

## Composing systems

Systems nest. An interaction refers to an element of a child system by a slash-separated path.

```python
from skcomplex.model import System
from skcomplex.models.ecology import Population, Predation

landscape = System(
    systems=[
        System('forest', elements=[Population('foxes', n=5)]),
        System('meadow', elements=[Population('rabbits', n=50)]),
    ],
    interactions=[Predation('hunt', predator='forest/foxes', prey='meadow/rabbits', a=0.02, b=0.01)],
)
```

States and parameters are named by their path, such as `meadow/rabbits.n` and `hunt.a`.

## Simulating

`Simulation` compiles the system, picks the solver of its form, and advances it in frames of `dt`.

```python
from skcomplex.runtime import RealTime, Simulation

trajectory = Simulation(landscape, dt=0.1).run(until=10).trajectory_
print(trajectory['meadow/rabbits.n'][-1])
```

- `clock=RealTime(speed)` paces frames so that simulated time advances `speed` times faster than wall time.
- `start(until)` runs in a background thread, and `pause`, `resume`, `stop`, and `wait` control it.
- `update_parameters({'hunt.a': 0.05})` changes parameters before the next frame.
- `random_state` makes stochastic runs reproducible.

## Observables

Observables are named quantities of a system, computed at every frame, recorded in trajectories under the system's
path, and logged to Rerun. An observable is either an expression of the system namespace, where earlier observables
appear by label, or a reduction over a group's members of a variable or an expression of one member: `count`, `sum`,
`mean`, `min`, `max`, `var`, or a function.

```python
import numpy as np
from sage.all__sagemath_symbolics import I, exp

from skcomplex.model import Observable

Oscillator = Element.define(
    'Oscillator',
    states={'theta': state(0.0, domain=Domain.ANGLE)},
    parameters={'omega': 1.0},
    laws=lambda s: [Rate(s.theta, s.omega)],
)
oscillators = System(
    elements=[Oscillator.group('phases', 100, theta=np.linspace(0, 1, 100), omega=np.linspace(0.9, 1.1, 100))],
    observables=[
        Observable('z', lambda member: exp(I * member.theta), group='phases'),
        Observable('order', lambda s: abs(s.z)),
    ],
)
order = Simulation(oscillators, dt=1.0).run(until=50).trajectory_['order']
print(round(float(order[0]), 3), round(float(order[-1]), 3))
```

A law reads a reduction observable of another partition through a role bound to the observable's path, as it reads an
aggregate, so agents and equations exchange macroscopic quantities in a co-simulation.

## The model catalog

Every ready-made model is a factory registered with its field, description, tags, references, and default run
settings. `catalog` lists them, filtered by field or tag, `describe` gives a model's parameters and the formalism
inferred by compiling it, `create` builds it by name, and `run` simulates it with its defaults.

```python
from skcomplex.models import catalog, create, describe, run

print([entry.name for entry in catalog(field='ecology')])
print(describe('lotka_volterra').formalism)
predation = create('lotka_volterra', a=0.2)
print(run('sir').trajectory_['I.n'].max().round())
```

`register_model` adds a factory of your own to the catalog. Every registered factory needs defaults for all of its
arguments, so the catalog can create, compile, simulate, visualize, and clone every model through the same interface.

## Groups of many elements

A group holds N identical elements of one type as arrays with one row per member. It compiles once, whatever its size,
and every law written for one member runs over all members with array operations.

```python
import numpy as np

from skcomplex.model import EuclideanSpace, MesaSystem
from skcomplex.models.physics import Particle, ReflectingBox

rng = np.random.default_rng(0)
gas = System(
    elements=[Particle.group('gas', 10_000, x=rng.uniform(0, 1, (10_000, 2)), v=rng.normal(0, 1, (10_000, 2)))],
    interactions=[ReflectingBox('box', target='gas', size=1.0)],
    space=EuclideanSpace(2),
)
trajectory = Simulation(gas, dt=0.01).run(until=1).trajectory_
print(trajectory['gas.x'].shape)
```

- A value is an array whose first dimension is the size of the group, or one value shared by every member.
- An interaction whose role binds to a group applies to every member. A single element bound next to the group
  receives the sum over the members.
- Group variables are named by the group path, such as `gas.x` with shape `(10000, 2)`, and a group parameter takes
  one value for every member or one value per member in `update_parameters`.
- `ideal_gas(n_molecules, kT=..., size=...)` builds a gas of non-interacting molecules in a reflecting box.

### Pair interactions

A pair interaction is written once for a pair of members and summed over every pair closer than its cutoff, inside one
group or between two groups. `PairPotential` derives the forces from a pair energy.

```python
from skcomplex.core import Domain, parameter
from skcomplex.models.physics import PairPotential


class LennardJones(PairPotential):
    epsilon: float = parameter(1.0, domain=Domain.POSITIVE)
    sigma: float = parameter(1.0, domain=Domain.POSITIVE)

    def pair_energy(self, s, r):
        return 4 * s.epsilon * ((s.sigma / r) ** 12 - (s.sigma / r) ** 6)


side = np.arange(10) * 1.2
liquid = System(
    elements=[Particle.group('atoms', 100, x=np.stack(np.meshgrid(side, side), axis=-1).reshape(-1, 2))],
    interactions=[LennardJones('lj', first='atoms', second='atoms', cutoff=2.5)],
    space=EuclideanSpace(2),
)
Simulation(liquid, dt=0.05).run(until=1)
```

`period` gives the side of a periodic box, so pair distances use the nearest periodic image.

### Boundaries

A `Boundary` law sets the value of a state after every solver step. Boundaries combine with every form, so walls and
periodic boxes work for differential equations, stochastic systems, jump processes, and discrete maps.
`ReflectingBox` mirrors positions and velocities at the walls, and `PeriodicBox` wraps positions around the box.

## Backends and speed

Compiling a system produces numeric functions with a backend. `'numpy'` evaluates every law with NumPy arrays, and
`'numba'` generates one compiled function per form, with compiled cell lists for pair interactions and compiled loops
for `RK4`, `EulerMaruyama`, `Gillespie`, and `Iteration`. The default `'auto'` picks Numba when the groups of a system
hold at least 1000 members in total, because compiling takes a second or two that small systems do not win back.

```python
from skcomplex.compile import NumbaBackend
from skcomplex.models.physics import ideal_gas
from skcomplex.solvers import RK4

simulation = Simulation(ideal_gas(50_000, random_state=0), solver=RK4(dt=0.01), backend='numba', dt=0.01)
print(simulation.run(until=0.5).problem_.backend)
```

Compiled kernels are cached on disk, under `~/.cache/skcomplex` or the `cache_dir` of `NumbaBackend(cache_dir=...)`,
so running the same system again skips compilation.

A law value can be a function called once per member with `Call(function, *arguments)`. A Numba-compiled function
runs inside the compiled kernels, and the NumPy backend calls it from Python.

```python
import numba

from skcomplex.model import Call, Rate
from skcomplex.models.ecology import Population


@numba.njit
def saturating(t, n, r):
    return r * n / (1 + n)


class Saturation(Interaction):
    target: str = role()
    r: float = parameter(1.0)

    def laws(self, s):
        return [Rate(s.target.n, Call(saturating, s.target.n, s.r))]


herds = System(
    elements=[Population.group('herds', 2000, n=1.0)],
    interactions=[Saturation('growth', target='herds')],
)
print(Simulation(herds, dt=0.1).run(until=1).trajectory_['herds.n'][-1, 0])
```

A plain Python callable of the values, as in `Rate(s.target.n, lambda t, values: ...)`, also works with the Numba
backend, but runs in Python and turns off the compiled stepper loops for that kind of law.

## External backends

An external backend runs a whole system in a third-party solver library, and the simulation keeps its frames,
trajectories, live parameters, and Rerun visualization. `LammpsBackend` runs particle systems in LAMMPS, which
integrates them with velocity Verlet.

```python
from skcomplex.backends import LammpsBackend
from skcomplex.models.physics import LangevinThermostat, LennardJones

atoms = System(
    elements=[Particle.group('atoms', 100, x=np.stack(np.meshgrid(side, side), axis=-1).reshape(-1, 2))],
    interactions=[
        LennardJones('lj', first='atoms', second='atoms', cutoff=2.5),
        LangevinThermostat('bath', target='atoms', temperature=0.5, gamma=1.0),
    ],
    space=EuclideanSpace(2),
)
trajectory = Simulation(atoms, solver=LammpsBackend(dt=0.005), dt=0.05, random_state=0).run(until=1).trajectory_
print(trajectory['atoms.x'].shape)
```

Before the first frame the backend checks the system and raises one error that lists every element, interaction, and
law it cannot run. The translation is the following.

| Framework | LAMMPS |
| --- | --- |
| `Particle` and particle groups | one atom type per element or group, with one mass per group |
| two or three dimensions | `dimension 2` with `fix enforce2d`, or `dimension 3` |
| `ReflectingBox` | `fix wall/reflect` at `0` and `size` |
| `PeriodicBox` | periodic boundaries of side `size`, shared by every group |
| `LennardJones` | `pair_style lj/cut` |
| other `PairPotential` | `pair_style lepton` with the pair energy printed as a Lepton expression |
| `ConstantForce` | `fix addforce` |
| `LinearDrag` | `fix viscous` |
| `LangevinThermostat` | `fix langevin`, seeded from the simulation's random state |
| any other interaction that only changes velocities | `fix external`, evaluated by the framework every step |

Importing `skcomplex.backends` also registers the solver name `'lammps'`.

## Agent-based models

Agents follow rules written as code. An `AgentType` declares the states every agent holds and the parameters its group
shares, and returns `Rule` laws whose functions take the agent and the model. A system with rules is agent-based and
runs on Mesa, one Mesa step per unit of time, with every agent activated in random order.

```python
from skcomplex.core import Domain, VisualSpec, parameter, state
from skcomplex.model import AgentType, GridSpace, Rule


class Forager(AgentType):
    visual = VisualSpec('agents', {'color_by': 'food'})
    food: float = state(5.0)
    hunger: float = parameter(1.0, domain=Domain.NONNEGATIVE)

    def laws(self, s):
        return [Rule(self.forage)]

    def forage(self, agent, model):
        cells = agent.neighbor_cells()
        agent.move_to(cells[int(model.rng.integers(len(cells)))])
        agent.food += model.rng.poisson(0.8) - agent.hunger
        if agent.food <= 0:
            agent.remove()
        elif agent.food > 10:
            agent.food /= 2
            agent.create(food=agent.food)


foragers = System(elements=[Forager.group('foragers', 50)], space=GridSpace(30, 30, torus=True))
trajectory = Simulation(foragers, random_state=0).run(until=20).trajectory_
print(trajectory['foragers.count'][-1], len(trajectory.agents('foragers', 20)))
```

- An agent reads and sets its states as attributes and reads its group's parameters, which change live like any
  parameter. `neighbors`, `neighbor_cells`, `agents_here`, `move_to`, `create`, and `remove` act on its place and its
  population.
- The model gives the random generator `model.rng`, the other agents with `model.agents(group)`, and empty cells.
- `GridSpace` has Moore or von Neumann neighbourhoods and an optional cell capacity, `NetworkSpace` places agents on
  the nodes of a NetworkX graph, and `ContinuousSpace` finds neighbours within a radius.
- Frames and trajectories hold every group's agents with their ids, states, and locations, through
  `trajectory.agents(group, frame)`, and the count of every group as the reporter `<group>.count`.
- `skcomplex.models.social` has Schelling segregation, Boltzmann wealth exchange, and the voter model, and
  `skcomplex.models.ecology` has the wolf and sheep model.

### Existing Mesa models

`MesaSystem` runs any Mesa model with the framework's runtime, Rerun visualization, and live parameters, without
changing its code. Numeric constructor arguments become parameters, model reporters become reporters, and agent
reporters become agent states.

```python
import mesa


class Coin(mesa.Agent):
    def __init__(self, model):
        super().__init__(model)
        self.heads = 0

    def step(self):
        self.heads += int(self.model.random.random() < self.model.bias)


class Coins(mesa.Model):
    def __init__(self, n=10, bias=0.5, rng=None):
        super().__init__(rng=rng)
        self.bias = bias
        Coin.create_agents(self, n)

    def step(self):
        self.agents.do('step')


coins = MesaSystem(
    'coins',
    model=Coins,
    parameters={'n': 10, 'bias': 0.5},
    model_reporters={'heads': lambda model: sum(agent.heads for agent in model.agents)},
    agent_reporters={'heads': 'heads'},
)
print(Simulation(coins, random_state=0).run(until=10).trajectory_['coins.heads'][-1])
```

## Co-simulation

A system whose laws do not fit one form, such as agents next to differential equations, is split into partitions at
subsystem boundaries. Every subsystem whose laws fit one form becomes a partition, and a law joins the partition that
owns its target, so an interaction of the parent system may contribute to several partitions. A subsystem whose own
elements mix law kinds raises `IncompatibleLawsError` with its path and the suggestion to split it.

```python
from skcomplex.compile import compile_system
from skcomplex.models.social import harvesting

farming = harvesting(10, effort=0.05)
print([(partition.name, partition.problem.form.value) for partition in compile_system(farming).partitions])
trajectory = Simulation(farming, random_state=0).run(until=50).trajectory_
print(round(trajectory['land/stock.n'][-1]), trajectory.agents('village/farmers', 50).values['wealth'].size)
```

- Every partition holds the states of the other partitions as inputs, which keep their values from the last
  synchronisation point while it advances.
- `Aggregate(label, group=..., reduce='count' | 'sum' | 'mean' | callable, variable=None)` turns an agent group or an
  element group into the single state `value`, which other partitions read by its dotted name, such as
  `village/size.value`.
- Agent rules read the states of other partitions through `model.inputs[name]`, such as
  `model.inputs['land/stock.n']`.
- The `CoSimulation` solver advances every partition with its own solver, chosen per partition through
  `solvers={'land': ScipyODE()}`, and exchanges inputs every `sync_dt`, which defaults to the frame duration and must
  divide it, and must be a whole number of steps for discrete and agent partitions.
- The Gauss-Seidel scheme, the default, advances partitions in system order, each reading the values already advanced
  by those before it. `scheme='jacobi'` advances all of them from the values at the start of the interval.
- Splitting adds an error of first order in `sync_dt`, so halving `sync_dt` roughly halves it at the cost of more
  exchanges. `CoSimulation(split=True)` splits a system that fits one form, which shows the error against the joint
  solution.

## Visualizing with Rerun

`RerunObserver` streams frames to a spawned viewer, a running viewer, or a `.rrd` file. Entity paths follow the system
tree, each element type chooses its visual, and parameters appear under `parameters/`.

```python
import tempfile
from pathlib import Path

from skcomplex.model import EuclideanSpace
from skcomplex.models.physics import Particle, Spring
from skcomplex.viz import RerunObserver

molecule = System(
    elements=[Particle('a', x=[0.0, 0.0, 0.0]), Particle('b', x=[1.5, 0.0, 0.0], v=[0.0, 0.5, 0.0])],
    interactions=[Spring('bond', first='a', second='b', k=4.0, length=1.0)],
    space=EuclideanSpace(3),
)
path = Path(tempfile.mkdtemp()) / 'molecule.rrd'
Simulation(molecule, dt=0.02, observers=[RerunObserver('molecule', save_path=path)]).run(until=5)
```

Particles appear as 3D points, populations as time series, agents on grids and in continuous spaces as points
colored by a state, agents on networks as graph nodes, and reporters under `reporters/`. A group is logged as one batch per state, and
`PointsVisual(position='x', color_by='v')` colors its points by speed. A group can set its own visual with
`Particle.group(..., visual=VisualSpec('points', {...}))`. A new visual subclasses `Visual` and is registered with
`register_visual`, and an element type selects it with its `visual` class variable.

## Experiments

Complex systems are studied through many runs. `Experiment` runs a model, a registered name, a factory, or a system,
over a design of factory arguments and dotted paths, scenarios, and replicates. Every design row and replicate gets
its own random stream spawned from `random_state` in a fixed order, so results are the same serially, on threads, or on
processes, and every scenario of a row and replicate shares its stream. Parameter and initial-state overrides reuse one
compiled problem, so sweeps of thousands of runs compile once.

```python
from skcomplex.experiments import Experiment, Grid, LatinHypercube, Scenario, calibrate, sensitivity

ensemble = Experiment('sir', Grid(stochastic=[True], gamma=[0.1, 0.2]), replicates=10, until=60, random_state=0)
print(ensemble.run().summary()[0]['I.n.mean'])

policies = [Scenario('baseline'), Scenario('quota', schedule=[(50.0, 'harvest.effort', 0.025)])]
stocks = Experiment('harvesting', scenarios=policies, until=80, outputs=['land/stock.n']).run()
print([round(float(row['land/stock.n'])) for row in stocks.rows])
```

- Designs are `Grid`, `RandomDesign`, `LatinHypercube`, `SobolDesign`, and `Rows`. They cross with `*` and
  concatenate with `+`, and their bounds are pairs or SciPy distributions.
- `outputs` keeps states and observables as series, after `warmup`, and functions of a trajectory as scalars. Results
  give one row per run, `series`, `summary` with means, standard deviations, and quantiles across replicates, and
  `to_frame` for pandas.
- `sensitivity` ranks parameters one at a time, by Morris screening, or by Sobol indices with SALib.
- `calibrate` fits dotted paths to observed series by least squares or a Gaussian likelihood, and reports the
  covariance of the estimates with R², RMSE, MAE, and MAPE.

```python
import numpy as np

times = np.arange(0.0, 21.0)
observed = 500 / (1 + 49 * np.exp(-0.8 * times))
fit = calibrate('logistic_growth', {'population.n': (times, observed)}, {'growth.r': (0.1, 2.0), 'growth.K': (100, 1000)})
print(round(fit.estimates['growth.r'], 2), round(fit.r2, 4))
```

## Analysing symbolically

`equations_of` returns the equations of a system as Sage relations, and `solve_symbolically` solves an ODE system
exactly where Maxima can, with `desolve` for one equation and `desolve_system` for linear systems with constant
coefficients.

```python
from skcomplex.analysis import equations_of, solve_symbolically
from skcomplex.models.physics import HarmonicPotential

oscillator = System(
    elements=[Particle('p', x=1.0)],
    interactions=[HarmonicPotential('trap', target='p', k=4.0)],
    space=EuclideanSpace(1),
)
print(equations_of(oscillator)['p.v[0]'])
print(solve_symbolically(oscillator, use_initial_state=True)['p.x[0]'])
```

