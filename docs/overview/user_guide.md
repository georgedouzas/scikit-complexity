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
- A **system** holds elements, interactions, child systems, and an optional space.

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

## Analysing symbolically

`equations_of` returns the equations of a system and `solve_symbolically` solves an ODE system exactly where SymPy can.

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

With the `sage` extra installed, a law also accepts Sage expressions, which are converted to SymPy.
