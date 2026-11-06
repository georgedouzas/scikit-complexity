# Simulate

Run a system frame by frame, choose its solver and backend, hand parts of it to external simulators, and couple
simulators together.

## Simulating

Run any system, ready-made or your own, for as long as you want, change its parameters on the way, and continue a run
where it stopped. `Simulation` compiles the system, picks the solver of its form, and advances it in frames of `dt`.

```python
from skcomplex.core.engine.runtime import Simulation
from skcomplex.systems import create_system

system = create_system('lotka_volterra')
trajectory = Simulation(system, dt=0.1).prepare().run(until=10).trajectory_
trajectory.values('prey.n')[-1].round(3)
```

`prepare` validates the system, compiles it, and chooses the solver. `run(until)` runs from time 0 to `until`, and
`trajectory_` holds the recorded frames. States and parameters are named by their dotted path, such as `prey.n` and
`predation.a`.

The default ODE solver is `ScipyODESolver` with SciPy's `LSODA`. It starts with Adams methods and switches to BDF
methods while the system is stiff, so stiff systems also run with the default solver.

### Changing a run

Change a parameter between frames with `update_parameters`, and continue a finished run with `extend`.

```python
from sklearn.base import clone

from skcomplex.core.engine.runtime import Simulation
from skcomplex.systems import create_system

simulation = Simulation(create_system('lotka_volterra'), dt=0.1).prepare().run(until=10)
simulation.update_parameters({'predation.a': 0.0})
len(simulation.extend(until=20).trajectory_.times)
clone(simulation).prepare().run(until=5).time_
```

- `update_parameters({'predation.a': 0.05})` changes parameters before the next frame. Values set between runs stay in
  force for the later runs of the same simulation, and `clone(simulation).prepare().run(until)` starts afresh.
- `extend(until)` continues a finished run from its last frame, with the same stepper, clock, and observers, and
  appends the new frames to the trajectory. `run` starts again from time 0.

### Run settings

Set how a run starts and evolves with the arguments of `Simulation`.

```python
from skcomplex.core.engine.runtime import RealTimeClock, Simulation
from skcomplex.systems import create_system

simulation = Simulation(
    create_system('lotka_volterra'),
    dt=0.1,
    schedule={'predation.a': lambda t: 0.4 + 0.001 * t},
    initial_state={'prey.n': 20},
    clock=RealTimeClock(speed=1000.0),
).prepare()
simulation.run(until=10).trajectory_.values('prey.n')[0]
simulation.start(until=20)
simulation.wait()
simulation.time_
```

- `dt` defaults to 1 for discrete maps and agents. For other forms it defaults to 0.01, raised to the smallest of 0.02,
  0.05, 0.1, and so on, in steps of 1, 2, and 5 times a power of ten, that keeps the run at most 1000 frames long. So
  `run(until=1000)` takes frames of 1.
- `until` sets the end time that `run` uses when it is given none.
- `clock=RealTimeClock(speed)` paces frames so that simulated time advances `speed` times faster than wall time.
- `start(until)` runs in a background thread, and `pause`, `resume`, `stop`, and `wait` control it.
- `random_state` makes stochastic runs reproducible. A fixed-step solver splits every frame into
  `step_count(t, until, dt)` equal steps on every backend, so the same `random_state` gives the same path on the NumPy
  and Numba backends.
- `schedule={'predation.a': lambda t: 0.4 + 0.001 * t}` sets parameters from functions of time at every frame, such as
  a slow ramp of a control parameter.
- `initial_state={'prey.n': 20}` replaces initial values of the elements for one simulation.

## Backends and speed

Run large systems fast by choosing the backend that turns their laws into numeric functions. Compiling a system
produces these functions with a backend.

```python
from skcomplex.core.engine.backends import NumbaBackend
from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.engine.solvers import RK4Solver
from skcomplex.systems import create_system

gas = create_system('ideal_gas', n_molecules=50_000, random_state=0)
simulation = Simulation(gas, solver=RK4Solver(dt=0.01), backend=NumbaBackend(), dt=0.01).prepare()
simulation.run(until=0.5).problem_.backend
```

The library has three backends:

- `NumpyBackend` evaluates every law with NumPy arrays.
- `NumbaBackend` generates one compiled function per form. It adds compiled cell lists for pair interactions and
  compiled loops for `RK4Solver`, `EulerMaruyamaSolver`, `GillespieSolver`, and `IterationSolver`.
- `JaxBackend` writes discrete maps and ordinary differential equations as pure JAX functions, which JAX traces,
  compiles, and differentiates. It is the backend of the inner system of a system that trains, and a forward backend
  that agrees with NumPy within 1e-10 in double precision.

Without a backend, `Simulation` picks Numba when the groups of a system hold at least 1000 members in total. Compiling
takes a second or two, which small systems do not win back. `problem_.backend` names the backend a run used.

Compiled kernels are cached on disk, under `~/.cache/skcomplex` or the `cache_dir` of `NumbaBackend(cache_dir=...)`.
So running the same system again skips compilation.

### Compiled functions in laws

Call a fast function of your own inside a law. A law value can be a function called once per member, with
`CallSpec(function, arguments)`.

```python
from typing import Annotated

import numba

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import RateLaw
from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.specs import CallSpec, ParameterSpec, RoleSpec, StateSpec


class PopulationElement(Element):
    def __init__(self, label, *, n: Annotated[float, StateSpec()] = 1.0):
        super().__init__(label)
        self.n = n


@numba.njit
def saturating(t, n, r):
    return r * n / (1 + n)


class SaturationInteraction(Interaction):
    def __init__(self, label, *, target: Annotated[str, RoleSpec()], r: Annotated[float, ParameterSpec()] = 1.0):
        super().__init__(label)
        self.target = target
        self.r = r

    def _get_laws(self, s):
        return [RateLaw(s.target.n, CallSpec(saturating, (s.target.n, s.r)))]


herds = System(
    elements=[GroupElement('herds', element=PopulationElement, size=2000, values={'n': 1.0})],
    interactions=[SaturationInteraction('growth', target='herds')],
)
round(float(Simulation(herds, dt=0.1).prepare().run(until=1).trajectory_.values('herds.n')[-1, 0]), 4)
```

A Numba-compiled function runs inside the compiled kernels, and the NumPy backend calls it from Python. The group above
holds 2000 members, so the run picks Numba by itself.

A plain Python callable of the values, as in `RateLaw(s.target.n, lambda t, values: ...)`, also works with the Numba
backend. It runs in Python, though, and turns off the compiled stepper loops for that kind of law.

## External backends

Hand a whole system to a third-party simulator, and keep the frames, trajectories, live parameters, and Rerun
visualization of the library. `LammpsSolver` runs particle systems in LAMMPS, which integrates them with velocity
Verlet.

```python
from typing import Annotated

import numpy as np

from skcomplex.core.components.elements import GroupElement, PointElement
from skcomplex.core.components.interactions import PairCouplingInteraction, RateInteraction
from skcomplex.core.components.laws.utils import derivative
from skcomplex.core.components.spaces import ContinuousSpace
from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.engine.solvers import LammpsSolver
from skcomplex.core.specs import ParameterSpec


class LennardJonesInteraction(PairCouplingInteraction):
    def __init__(
        self,
        label,
        first,
        second,
        cutoff=1.0,
        position='x',
        *,
        epsilon: Annotated[float, ParameterSpec()] = 1.0,
        sigma: Annotated[float, ParameterSpec()] = 1.0,
    ):
        super().__init__(label, first, second, cutoff=cutoff, position=position, potential=self._get_energy)
        self.epsilon = epsilon
        self.sigma = sigma

    @staticmethod
    def _get_energy(s, r):
        return 4 * s.epsilon * ((s.sigma / r) ** 12 - (s.sigma / r) ** 6)


class DragInteraction(RateInteraction):
    def __init__(self, label, *, target, gamma: Annotated[float, ParameterSpec()] = 1.0):
        super().__init__(label, target=target, state='x', rate=self._get_rate)
        self.gamma = gamma

    @staticmethod
    def _get_rate(s):
        return -s.gamma * s.target.x.apply_map(lambda item: derivative(item, 1))


def get_pair_coefficients(value):
    return f'{float(value("epsilon")[0])} {float(value("sigma")[0])}'


def get_drag_fix(fix):
    return f'fix {fix.name} {fix.group} viscous {float(fix.value("gamma")[0])}'


side = np.arange(10) * 1.2
grid = np.stack(np.meshgrid(side, side), axis=-1).reshape(-1, 2)
atoms = System(
    elements=[GroupElement('atoms', element=PointElement, size=100, values={'x': grid})],
    interactions=[
        LennardJonesInteraction('lj', 'atoms', 'atoms', cutoff=2.5),
        DragInteraction('air', target='atoms', gamma=1.0),
    ],
    space=ContinuousSpace(n_dim=2),
)
pairs = {LennardJonesInteraction: ('lj/cut', get_pair_coefficients)}
fixes = {DragInteraction: get_drag_fix}
solver = LammpsSolver(dt=0.005, pairs=pairs, fixes=fixes)
trajectory = Simulation(atoms, solver=solver, dt=0.05, random_state=0).prepare().run(until=1).trajectory_
trajectory.values('atoms.x').shape
```

Two maps tell the solver how to write your interactions in LAMMPS:

- `pairs` maps pair potential classes to native LAMMPS pair styles, with a function that writes their coefficients.
- `fixes` maps interaction classes to native fixes, with a function that writes the fix.

You write both maps for your own classes, as above, and the prebuilt particle systems come with theirs. A function in
`fixes` reads a seed from the random state of the simulation with `fix.seed()`, for a stochastic fix such as
`fix langevin`.

Before the first frame the backend checks the system. It raises one error that lists every element, interaction, and
law it cannot run. The library translates a system to LAMMPS as follows:

- Elements and groups with a vector position of order 2, such as `PointElement`, become one atom type per element or
  group. The velocities of the atoms are the first derivatives of the positions. Their mass is the parameter that the
  `divisor` of their interactions names, or 1 without one.
- Two or three dimensions become `dimension 2` with `fix enforce2d`, or `dimension 3`.
- `ContinuousSpace(extent, boundary='reflecting')` becomes `fix wall/reflect` at `0` and the extent.
- `ContinuousSpace(extent, boundary='periodic')` becomes periodic boundaries of the extent, shared by every group.
- A class in `pairs`, such as `LennardJonesInteraction`, becomes its native pair style, such as `pair_style lj/cut`.
- Any other `PairCouplingInteraction` with a potential becomes `pair_style lepton`, with the potential printed as a
  Lepton expression.
- A class in `fixes` becomes its native fix, such as `fix addforce`, `fix viscous`, or `fix langevin`.
- Any other interaction that only changes the derivatives of the positions becomes `fix external`, which the library
  evaluates at every step.

## Co-simulation

Run parts of a system that need different solvers side by side, such as farming agents next to the differential
equation of the land they harvest. A system whose laws do not fit one form is split into partitions at subsystem
boundaries, and each partition runs with its own solver.

```python
from skcomplex.core.engine.compile import PassCompiler
from skcomplex.core.engine.runtime import Simulation
from skcomplex.systems import create_system

farming = create_system('harvesting', farmers=10, effort=0.05)
partitions = PassCompiler().prepare(farming).run().problem_.partitions
[(partition.name, partition.problem.form.value) for partition in partitions]
trajectory = Simulation(farming, random_state=0).prepare().run(until=50).trajectory_
round(trajectory.values('land/stock.n')[-1]), trajectory.agents('village/farmers', 50).values['wealth'].size
trajectory.values('village/farmers_count.value')[-1]
```

Every subsystem whose laws fit one form becomes a partition. A law joins the partition that owns its target, so an
interaction of the parent system may contribute to several partitions. A subsystem whose own elements mix law kinds
raises `IncompatibleLawsError`, with its path and the suggestion to split it.

The partitions exchange values in these ways:

- Every partition holds the states of the other partitions as inputs. They keep their values from the last
  synchronisation point while it advances.
- A law of one partition reads a reduction of a group of another through a role bound to the group, such as
  `count(s.farmers)` or `total(s.farmers.wealth)`. The co-simulation computes every reduction of agents at each
  synchronisation point and holds it until the next, in the state `value` of an element next to the group, such as
  `village/farmers_count.value`. A reduction of an element group reads the states of the other partition, which keep
  their values from the last synchronisation point too.
- Agent rules read the states of other partitions through `system.inputs[name]`, such as
  `system.inputs['land/stock.n']`.

The `CoSimulationSolver` advances the partitions and controls the exchange:

- It advances every partition with its own solver, chosen per partition through `solvers={'land': ScipyODESolver()}`.
- It exchanges inputs every `sync_dt`. This defaults to the frame duration and must divide it, and it must be a whole
  number of steps for discrete and agent partitions.
- The Gauss-Seidel scheme, the default, advances partitions in system order, each reading the values already advanced
  by those before it. `scheme='jacobi'` advances all of them from the values at the start of the interval.
- Splitting adds an error of first order in `sync_dt`. Halving `sync_dt` roughly halves it, at the cost of more
  exchanges. `CoSimulationSolver(split=True)` splits a system that fits one form, which shows the error against the
  joint solution.
