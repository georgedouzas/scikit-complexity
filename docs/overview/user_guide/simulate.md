# Simulate a system

This tutorial runs the predator and prey model of the [build tutorial](build_a_custom_system/index.md) and reads what
the run returns. It sets the [run settings](glossary.md#run-settings), chooses a [solver](glossary.md#solver) and a [backend](glossary.md#backend) from the [tags](glossary.md#tag) of the [system](glossary.md#system), makes random runs
reproducible, records and replays their draws, and closes with large systems, external simulators, and systems whose
components need different solvers.

## The first run

The goal of this section is a [trajectory](glossary.md#trajectory) of the hares and the lynx from 1900 to 1920, compared with the pelt records.
The Hudson's Bay Company bought the pelts of hares and lynx from trappers for two centuries, and the counts rise and
fall in cycles of about ten years. The counts below are the pelts of 1900 to 1920, in thousands, as distributed with
the case study of Carpenter (2018, *Predator-Prey Population Dynamics: the Lotka-Volterra model in Stan*) in the file
`knitr/lotka-volterra/hudson-bay-lynx-hare.csv` of the Stan example models. That file cites the lecture notes of
Howard (2009) at Texas A&M University, which trace the numbers through the notes of Mahaffy (Math 636,
San Diego State University) to Purves, Sadava, Orians, and Heller, *Life: The Science of Biology*.

```python
import numpy as np

from skcomplex.systems.biology.ecology import LotkaVolterraSystem

years = np.arange(1900, 1921)
hares = np.array(
    [30.0, 47.2, 70.2, 77.4, 36.3, 20.6, 18.1, 21.4, 22.0, 25.4, 27.1]
    + [40.3, 57.0, 76.6, 52.3, 19.5, 11.2, 7.6, 14.6, 16.2, 24.7]
)
lynx = np.array(
    [4.0, 6.1, 9.8, 35.2, 59.4, 41.7, 19.0, 13.0, 8.3, 9.1, 7.4]
    + [8.0, 12.3, 19.5, 45.7, 51.1, 29.7, 15.8, 9.7, 10.1, 8.6]
)
system = LotkaVolterraSystem(prey=30.0, predators=4.0, r=0.55, a=0.028, b=0.024, d=0.80)
trajectory = system.prepare().simulate(until=20, dt=1.0)
trajectory.get_values('prey.x').round(1)[:6].tolist()
```

[prepare][skcomplex.core.systems.System.prepare] checks the system, compiles its [laws](glossary.md#law) into one
problem, and picks a solver. [simulate][skcomplex.core.systems.System.simulate] advances the problem from time 0 to
`until` in [frames](glossary.md#frame) of `dt`. Here a frame is one year. The simulated hares rise from 30.0 to 81.2
thousand in 1903 and fall to 18.6 thousand in 1905, as the records do.

```python continuation
fit = np.sqrt(np.mean((trajectory.get_values('prey.x') - hares) ** 2))
round(float(fit), 1), int(years[trajectory.get_values('prey.x').argmax()])
```

The root mean square difference to the records is 7.8 thousand pelts, and the simulated hares peak in 1903, the year
of the highest count. The system itself does not change in a run, so the same instance runs again with other settings.

## The trajectory

A [Trajectory][skcomplex.core.engine.runtime.Trajectory] holds the frames of a run. Every [state](glossary.md#state) and
parameter has a [dotted name](glossary.md#dotted-name), the [label](glossary.md#label) of its component and the name of
the value, such as `prey.x` or `predation.a`.

```python continuation
trajectory.names, trajectory.times[:3].tolist(), trajectory.states.shape, trajectory.parameters.shape
```

- `names` lists the states `prey.x` and `predators.x`, then the parameters `growth.r`, `predation.a`, `predation.b`,
  and `death.d`, the parameters of the built components.
- `times` holds the time of every frame, 0, 1, 2, and so on.
- `states` holds one row per frame and one column per state, here 21 frames of 2 states.
- `parameters` holds one row per frame and one column per parameter, 21 by 4.

`get_values(name)` returns the column of one name, a state or a parameter. A parameter is recorded at every frame,
because a [schedule](glossary.md#schedule) may change it during the run, as the next section shows.

## Run settings

The arguments of `simulate` set how a run starts and evolves. Each one left out comes from the tuned run settings of
the class of the system, its class attributes such as `until` and `dt`.

```python continuation
system.until, system.dt, len(system.prepare().simulate().times)
```

`LotkaVolterraSystem` tunes `until=50.0` and `dt=0.1`, so a call without them runs fifty years in 501 frames. The
settings are these.

- `until` sets the end time. A system without a tuned end time, such as a plain `System`, raises an error that asks
  for it.
- `dt` sets the time between frames. Without a tuned value it is 1 for discrete maps and [agents](glossary.md#agent). For other forms it
  is 0.01, raised to the smallest of 0.02, 0.05, 0.1, and so on, in steps of 1, 2, and 5 times a power of ten, that
  keeps the run at most 1000 frames long. So a plain `System` run with `until=1000` takes frames of 1.
- `initial_state` replaces initial values of the [elements](glossary.md#element) for one run, by dotted name.
- `schedule` sets parameters from functions of time at every frame, such as a slow ramp of a control parameter.
- `solver` and `backend` choose how the run steps and evaluates its laws, as the section on solvers shows.
- `random_state` makes a [stochastic](glossary.md#stochastic) run reproducible.
- `observables` adds quantities computed at every frame, `invariants` names those that must stay constant, and
  `observers` stream the run to a viewer, as the tutorial [Observe](observe.md) shows.

### Changing a run

A schedule changes a parameter while the system runs, and `initial_state` continues a run from where another stopped.
Here the first run covers 1900 to 1910, and the second starts from its last frame with predation switched off.

```python continuation
first = system.prepare().simulate(until=10, dt=1.0)
last = {name: first.get_values(name)[-1] for name in ('prey.x', 'predators.x')}
second = system.prepare().simulate(until=10, dt=1.0, initial_state=last, schedule={'predation.a': lambda t: 0.0})
second.get_values('prey.x')[0] == first.get_values('prey.x')[-1], second.get_values('prey.x').round(1)[-1]
```

The second run starts at the 23.4 thousand hares where the first stopped, and its times start again from 0. Without
predation the hares grow at the rate 0.55 a year and reach 5735.6 thousand in ten years, the exponential growth the
lynx hold back. A ramp works the same way: `schedule={'growth.r': lambda t: 0.55 + 0.01 * t}` raises the birth rate
from 0.55 to 0.75 over twenty years.

A changed [model parameter](glossary.md#model-parameter), such as `LotkaVolterraSystem(r=0.65)` or `set_params(r=0.65)`,
builds a new set of components, which compile again. A schedule changes a value of the built components, so it names
them by their path, such as `growth.r`.

### Observables and invariants

An [observable](glossary.md#observable) is a value computed from the states at every frame. The Lotka and Volterra model keeps the quantity
`V = b H - d ln H + a L - r ln L` constant along every orbit, so it serves as a check of the numerical integration.

```python continuation
from skcomplex.core.components import Observable


def get_constant(t, values):
    prey, predators = values['prey.x'], values['predators.x']
    return 0.024 * prey - 0.80 * np.log(prey) + 0.028 * predators - 0.55 * np.log(predators)


constant = Observable('constant', function=get_constant)
observed = system.prepare().simulate(until=20, dt=1.0, observables=[constant], invariants={'constant': 1e-3})
observed.get_values('constant').round(4)[[0, -1]].tolist()
```

The [Observable][skcomplex.core.components.Observable] reads the dotted names of the states and returns a number,
which the trajectory records under its label. The constant is -2.6514 at the start and at the end, and the run raises
no error, because it stays within the tolerance of 0.001 that `invariants` gives it. The section on solvers below shows
a run that leaves it.

## Solver and backend

The tags of a system choose its default solver before anything compiles. A solver advances the compiled problem in
time, and a backend turns its laws into numeric functions. This section reads which ones a run used and replaces them.

```python continuation
tags = system.get_tags()
sorted(tags.time), sorted(tags.laws), trajectory.problem.form.value, trajectory.problem.backend
```

The model runs in continuous time with rate laws, so it compiles to [ordinary differential equations](glossary.md#ordinary-differential-equation), which run on the
NumPy backend. The default solver of this form is
`ScipyODESolver` with the method `LSODA` of SciPy. It starts with Adams methods and switches to BDF methods while the
system is [stiff](glossary.md#stiff), that is, while it mixes fast and slow time scales, so stiff systems also run with the default solver.

A solver passed by hand must support the tags of the system, and a solver that lacks one raises a `TagError` that names
it, as the tutorial [Check a system](check.md#the-solver) shows. A fixed-step solver suits a check of accuracy. Forward
Euler with a step of 0.1 drifts off the orbit, and the [invariant](glossary.md#invariant) catches it.

```python continuation
from skcomplex.core.engine.solvers import EulerSolver, RK4Solver
from skcomplex.core.errors import ConservationError

rk4 = system.set_params(solver=RK4Solver()).prepare().simulate(until=20, dt=1.0).get_values('prey.x').round(1)[:6].tolist()
euler = EulerSolver(dt=0.1)
try:
    system.set_params(solver=euler).prepare().simulate(until=20, dt=1.0, observables=[constant], invariants={'constant': 1e-3})
except ConservationError as error:
    message = str(error)
rk4, message
```

`RK4Solver`, the classical Runge-Kutta method of order four, gives the same hares as the default solver. The message
says that the invariant changed from -2.65142 to -2.64349 at frame 1, time 1, beyond its tolerance. Forward Euler gains
a little on every turn of the cycle, so a smaller step or a solver of higher order fixes it.

### One model, two solvers

The solver is a setting of the system, not of its components, so the same laws can run in two ways. In the reaction
`A + B -> C` of the [running examples](index.md#the-running-examples), a `JumpLaw` fires at the rate `0.01 a b` and
turns one molecule of `A` and one of `B` into one of `C`. By default the system is a jump process, run molecule by
molecule by `GillespieSolver`. Given `ScipyODESolver()`, the same system runs the rate equations of the mean, which
chemists write for large amounts.

```python continuation
from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import JumpLaw
from skcomplex.core.engine.solvers import ScipyODESolver
from skcomplex.core.systems import System

bind = Interaction(
    'bind',
    roles={'a': 'A', 'b': 'B', 'c': 'C'},
    laws=lambda s: [JumpLaw(0.01 * s.a.n * s.b.n, {s.a.n: -1, s.b.n: -1, s.c.n: +1})],
)
molecules = [Element('A', values={'n': 100}), Element('B', values={'n': 80}), Element('C', values={'n': 0})]
network = System(elements=molecules, interactions=[bind]).prepare()
counted = network.simulate(until=5, dt=1, random_state=0).get_values('C.n').tolist()
averaged = network.set_params(solver=ScipyODESolver()).prepare().simulate(until=5, dt=1).get_values('C.n')
counted, averaged.round(1).tolist()
```

The jumps make 43, 58, 65, 68, and 69 molecules of `C` after one to five time units, and the rate equations make 42.0,
56.9, 64.3, 68.8, and 71.7. With a hundred molecules the two agree within a few, and the fewer the molecules, the more
chance decides what happens.

### Backends

The library has three backends.

- `NumpyBackend` evaluates every law with NumPy arrays.
- `NumbaBackend` generates one compiled function per form. It adds compiled cell lists for pair [interactions](glossary.md#interaction) and
  compiled loops for `RK4Solver`, `EulerMaruyamaSolver`, `GillespieSolver`, and `IterationSolver`.
- `JaxBackend` writes discrete maps and ordinary differential equations as pure JAX functions, which JAX traces,
  compiles, and differentiates. It is the backend of the inner system of a system that trains, and a forward backend
  that agrees with NumPy within 1e-10 in double precision.

Without a backend, `simulate` picks Numba when the [groups](glossary.md#group) of a system hold at least 1000 [members](glossary.md#member) in total. Compiling
takes a second or two, which small systems such as the two populations above do not win back. Compiled kernels are
cached on disk, under `~/.cache/skcomplex` or the `cache_dir` of `NumbaBackend(cache_dir=...)`, so running the same
system again skips compilation.

## Random runs

The equations follow the mean of many animals. A small population also feels chance: a lynx may die before it finds a
hare, and the last predators may vanish. This section runs the model as random events and makes the runs reproducible.

The pelt model is a set of four reactions, and
[ReactionNetworkSystem](core_components/general.md#reactionnetworksystem) of `skcomplex.systems.general.reactions`
runs them as random events with `mode='jump'`. Each reaction fires one at a time, at random times whose rate is its
[propensity](glossary.md#propensity), the rate constant times the product of the reactant amounts. The amounts become counts of animals, as if
the records described a small local population.

```python continuation
from skcomplex.systems.general.reactions import ReactionNetworkSystem

reactions = {
    'births': ({'hares': 1}, {'hares': 2}, 0.55),
    'hunting': ({'hares': 1, 'lynx': 1}, {'lynx': 1}, 0.028),
    'feeding': ({'hares': 1, 'lynx': 1}, {'hares': 1, 'lynx': 2}, 0.024),
    'deaths': ({'lynx': 1}, {}, 0.80),
}
jumps = ReactionNetworkSystem(species={'hares': 30, 'lynx': 4}, reactions=reactions, mode='jump')
runs = [jumps.prepare().simulate(until=20, dt=1.0, random_state=seed).get_values('hares.x')[:6].tolist() for seed in (0, 0, 1)]
jumps.get_tags().randomness, runs
```

The tags call the system `stochastic`, so the default solver is `GillespieSolver`, the exact [algorithm of Gillespie](glossary.md#gillespie-algorithm) for
random events in continuous time. The `random_state` fixes its random numbers. Two runs with the seed 0 give the same
hares, 30, 41, 58, 78, 44, and 9, and the seed 1 gives another path, 30, 49, 64, 48, 35, and 43. A run without a
`random_state` draws a fresh path each time.

A random run answers a question the equations cannot, such as how often the lynx die out within ten years.

```python continuation
ends = [jumps.prepare().simulate(until=10, dt=1.0, random_state=seed).get_values('lynx.x')[-1] for seed in range(20)]
sum(end == 0 for end in ends)
```

The lynx die out in 2 of 20 runs. The fraction is a rough estimate, and a real study would run thousands of seeds,
as `ensemble` of the tutorial [Run experiments](run_experiments.md) does. The runs stop after ten years for speed.
Once the lynx are gone, nothing holds the hares back, and the Gillespie solver fires one event per birth, so a longer
run of an extinct path counts millions of events.

The same `random_state` gives the same path on every backend. A fixed-step solver splits every frame into the same
number of equal steps on NumPy and Numba, and the Gillespie solver draws its events in the same order.

```python continuation
from skcomplex.core.engine.backends import NumbaBackend, NumpyBackend

paths = [
    jumps.set_params(backend=backend).prepare().simulate(until=20, dt=1.0, random_state=0).get_values('hares.x')
    for backend in (NumpyBackend(), NumbaBackend())
]
bool((paths[0] == paths[1]).all())
```

The two paths are equal.

### Recording and replaying draws

A recorded run keeps every random draw, so the same path can be scored under other parameters. This is the [likelihood](glossary.md#likelihood)
of a path, which [calibration](glossary.md#calibration) uses as its event [loss](glossary.md#loss).

`record_draws=True` stores in `trajectory.draws` every named draw by its owner and name, with the step it fell in and
the arguments of its distribution, and every firing of a [jump](glossary.md#jump) law under Gillespie with its time and law.
`replay=trajectory` runs a system with those draws forced to their recorded values. It computes the arguments of each
distribution from its own parameters and states, and its trajectory holds the total log-probability of the forced
draws as `log_probability`.

```python continuation
recorded = jumps.prepare().simulate(until=20, dt=1.0, random_state=0, record_draws=True)
scores = {}
for birth_rate in (0.45, 0.55, 0.65):
    changed = {**reactions, 'births': ({'hares': 1}, {'hares': 2}, birth_rate)}
    candidate = ReactionNetworkSystem(species={'hares': 30, 'lynx': 4}, reactions=changed, mode='jump')
    replayed = candidate.prepare().simulate(until=20, dt=1.0, random_state=0, replay=recorded)
    scores[birth_rate] = round(float(replayed.log_probability), 1)
len(recorded.draws.firing_times), bool((replayed.states == recorded.states).all()), scores
```

The recorded run fires 614 reactions in twenty years. Every replay follows the same path, and its log-probability is
1537.1 for a birth rate of 0.45, 1538.8 for 0.55, and 1535.9 for 0.65. The rate that drew the path scores highest.

- The draws of the laws of stochastic maps, such as `normal('kick', 0, s)`, and the named draws that agent rules make
  through `system.draw(name, distribution, **arguments)` are recorded. A draw from `system.rng` is not, so rules draw
  every random outcome that matters to a likelihood by name.
- A replayed Gillespie run fires the recorded laws at the recorded times and adds the log propensity of every firing
  minus the integral of the total propensity, the exact log-likelihood of the path.
- A replay takes the same random numbers as the recorded run, so replay with its random state, and the draws that are
  not named, such as the order of agents, stay the same.
- A run that records or replays uses the NumPy backend unless one is given. A system whose draws differ from those of
  the trajectory raises an error that names the draw and the step.

## Large systems

The two populations run in a fraction of a second. A system of many members needs a compiled backend, and the next
example, a gas of 50,000 molecules, shows one.

```python
from skcomplex.core.engine.backends import NumbaBackend
from skcomplex.core.engine.solvers import RK4Solver
from skcomplex.systems.physics.statistical_mechanics import IdealGasSystem

gas = IdealGasSystem(n_molecules=50_000, random_state=0, solver=RK4Solver(dt=0.01), backend=NumbaBackend())
trajectory = gas.prepare().simulate(until=0.5, dt=0.01)
trajectory.get_values('gas.x').shape
```

The positions of the molecules form an array of 51 frames, 50,000 molecules, and 2 coordinates.

### Compiled functions in laws

A law can call a fast function of its own, once per member, with `CallSpec(function, arguments)`. Here a herd of 2000
groups grows at a rate that saturates with its size.

```python
import numba

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import CallSpec, RateLaw
from skcomplex.core.systems import System
from skcomplex.core.types import Role


@numba.njit
def saturating(t, n, r):
    return r * n / (1 + n)


class Saturation(Interaction):
    def __init__(self, label, *, target: Role, r=1.0):
        super().__init__(label)
        self.target = target
        self.r = r

    def _prepare(self, s):
        super()._prepare(s)
        self.laws_ = [RateLaw(s.target.n, CallSpec(saturating, (s.target.n, s.r)))]


herds = Element('herds', values={'n': 1.0}, size=2000)
system = System(elements=[herds], interactions=[Saturation('growth', target='herds')])
round(float(system.prepare().simulate(until=1, dt=0.1).get_values('herds.n')[-1, 0]), 4)
```

Each herd grows from 1 to 1.5571 in one time unit. A function compiled with Numba runs inside the compiled kernels,
and the NumPy backend calls it from Python. The group holds 2000 members, so the run picks Numba by itself.

A plain Python function of the values, as in `RateLaw(s.target.n, lambda t, values: ...)`, also works with the Numba
backend. It runs in Python, though, and turns off the compiled stepper loops for that kind of law.

## External simulators

A whole system can run in a third-party simulator while the library keeps its frames, trajectories, schedules, and
Rerun visualization. `LammpsSolver` runs particle systems in LAMMPS, which integrates them with velocity Verlet. Here
100 atoms under the Lennard-Jones potential and a drag start on a square grid.

```python
import numpy as np
from sage.all__sagemath_symbolics import sqrt

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import RateLaw
from skcomplex.core.components.laws.utils import neighbors
from skcomplex.core.components.spaces import ContinuousSpace, RadiusNeighborhood
from skcomplex.core.engine.solvers import LammpsSolver
from skcomplex.core.systems import System
from skcomplex.core.types import Role


class LennardJones(Interaction):
    def __init__(self, label, *, target: Role, cutoff=2.5, epsilon=1.0, sigma=1.0):
        super().__init__(label)
        self.target = target
        self.cutoff = cutoff
        self.epsilon = epsilon
        self.sigma = sigma

    def _prepare(self, s):
        super()._prepare(s)
        x = s.target.x
        separation = x - s.target.other.x
        r = sqrt(separation.dot_product(separation))
        force = 24 * s.epsilon * (2 * (s.sigma / r) ** 12 - (s.sigma / r) ** 6) / r**2 * separation
        self.laws_ = [RateLaw(x.diff(s.t, 2), neighbors(force, RadiusNeighborhood(self.cutoff)))]


class Drag(Interaction):
    def __init__(self, label, *, target: Role, gamma=1.0):
        super().__init__(label)
        self.target = target
        self.gamma = gamma

    def _prepare(self, s):
        super()._prepare(s)
        x = s.target.x
        self.laws_ = [RateLaw(x.diff(s.t, 2), -s.gamma * x.diff(s.t))]


def get_pair_coefficients(value):
    return f'{float(value("epsilon")[0])} {float(value("sigma")[0])}'


def get_drag_fix(fix):
    return f'fix {fix.name} {fix.group} viscous {float(fix.value("gamma")[0])}'


side = np.arange(10) * 1.2
grid = np.stack(np.meshgrid(side, side), axis=-1).reshape(-1, 2)
pairs = {LennardJones: ('lj/cut', get_pair_coefficients)}
fixes = {Drag: get_drag_fix}
atoms = System(
    elements=[Element('atoms', values={'x': grid}, size=100)],
    interactions=[LennardJones('lj', target='atoms'), Drag('air', target='atoms', gamma=1.0)],
    space=ContinuousSpace(n_dim=2),
    solver=LammpsSolver(dt=0.005, pairs=pairs, fixes=fixes),
)
trajectory = atoms.prepare().simulate(until=1, dt=0.05, random_state=0)
trajectory.get_values('atoms.x').shape
```

The positions come back from LAMMPS as 21 frames of 100 atoms in 2 dimensions. Two maps tell the solver how to write
the interactions in LAMMPS.

- `pairs` maps pair potential classes to native LAMMPS pair styles, with a function that writes their coefficients.
- `fixes` maps interaction classes to native fixes, with a function that writes the fix.

Both maps are written for each class of one's own, as above, and the ready-made particle systems come with theirs. A
function in `fixes` reads a seed from the random state of the run with `fix.seed()`, for a stochastic fix such as
`fix langevin`.

Before the first frame the solver checks the system. It raises one error that lists every element, interaction, and
law it cannot run. The library translates a system to LAMMPS as follows.

- Elements and groups with a vector position whose laws set its second derivative, such as `atoms`, become one atom
  type per element or group. The velocities of the atoms are the first derivatives of the positions. Their mass is their value `m`, or 1
  without one.
- Two or three dimensions become `dimension 2` with `fix enforce2d`, or `dimension 3`.
- `ContinuousSpace(extent, boundary='reflecting')` becomes `fix wall/reflect` at `0` and the extent.
- `ContinuousSpace(extent, boundary='periodic')` becomes periodic boundaries of the extent, shared by every group.
- A class in `pairs`, such as `LennardJonesInteraction`, becomes its native pair style, such as `pair_style lj/cut`.
- A class in `fixes` becomes its native fix, such as `fix addforce`, `fix viscous`, or `fix langevin`.
- Any other interaction that only changes the derivatives of the positions becomes `fix external`, which the library
  evaluates at every step.

## Co-simulation

Some systems hold components that need different solvers, such as farming agents next to the differential equation of
the land they harvest. Such a system is split into [partitions](glossary.md#partition) at the boundaries of its [nested
systems](glossary.md#nested-system), and each partition runs with its own solver.

```python
from skcomplex.systems.economics.resources import HarvestingSystem

trajectory = HarvestingSystem(farmers=10, effort=0.05).prepare().simulate(until=50, random_state=0)
round(trajectory.get_values('land/stock.n')[-1]), trajectory.get_agents('village/farmers', 50).values['wealth'].size
trajectory.get_values('village/farmers_count.value')[-1]
```

The land is a differential equation and the village holds agents, so each runs as its own partition. After fifty years
the village still counts 10 farmers.

Each nested system keeps the time its own laws imply, and no time is given to any of them. A nested system whose laws
are updates steps once per unit of time, and one whose laws are rates runs in continuous time between those steps. A
heater that a thermostat switches once a minute is a discrete controller next to a continuous room.

```python
from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import RateLaw, UpdateLaw
from skcomplex.core.systems import System
from skcomplex.core.types import Role


class Heating(Interaction):
    def __init__(self, label, *, room: Role, heater: Role, k=0.1):
        super().__init__(label)
        self.room = room
        self.heater = heater
        self.k = k

    def _prepare(self, context):
        super()._prepare(context)
        room, t = context.room, context.t
        self.laws_ = [RateLaw(room.T.diff(t), -context.k * room.T + context.heater.power)]


class Thermostat(Interaction):
    def __init__(self, label, *, heater: Role, room: Role):
        super().__init__(label)
        self.heater = heater
        self.room = room

    def _prepare(self, context):
        super()._prepare(context)
        self.laws_ = [UpdateLaw(context.heater.power, (context.room.T < 5) * 1.0)]


house = System(
    systems=[
        System('control', elements=[Element('heater', values={'power': 0.0})]),
        System('house', elements=[Element('room', values={'T': 0.0})]),
    ],
    interactions=[
        Thermostat('thermostat', heater='control/heater', room='house/room'),
        Heating('heating', room='house/room', heater='control/heater'),
    ],
).prepare()
trajectory = house.simulate(until=60, dt=1)
house.form_, set(trajectory.get_values('control/heater.power').tolist())
```

The thermostat changes the heater once a minute, and the room warms or cools in continuous time in between.

Every nested system whose laws fit one form becomes a partition. A law joins the partition that owns its target, so
an interaction of the parent system may contribute to several partitions. A nested system whose own elements mix law
kinds raises `IncompatibleLawsError`, with its path and the suggestion to split it.

The partitions exchange values in these ways.

- Every partition holds the states of the other partitions as inputs. They keep their values from the last
  synchronisation point while it advances.
- A law of one partition reads a reduction of a group of another through a [role](glossary.md#role) bound to the group, such as
  `count(s.farmers)` or `total(s.farmers.wealth)`. The [co-simulation](glossary.md#co-simulation) computes every reduction of agents at each
  synchronisation point and holds it until the next, in the state `value` of an element next to the group, such as
  `village/farmers_count.value`. A reduction of an element group reads the states of the other partition, which keep
  their values from the last synchronisation point too.
- Agent rules read the states of other partitions through `system.inputs[name]`, such as
  `system.inputs['land/stock.n']`.

The `CoSimulationSolver` advances the partitions and controls the exchange.

- It advances every partition with its own solver, chosen per partition through `solvers={'land': ScipyODESolver()}`.
- It exchanges inputs every `sync_dt`. This defaults to the frame duration and must divide it, and it must be a whole
  number of steps for discrete and agent partitions.
- The Gauss-Seidel scheme, the default, advances partitions in system order, each reading the values already advanced
  by those before it. `scheme='jacobi'` advances all of them from the values at the start of the interval.
- Splitting adds an error of first order in `sync_dt`. Halving `sync_dt` roughly halves it, at the cost of more
  exchanges. `CoSimulationSolver(split=True)` splits a system that fits one form, which shows the error against the
  joint solution.
