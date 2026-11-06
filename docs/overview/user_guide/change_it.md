# Change it

Give a ready-made system your own arguments and run settings, or nest it inside a larger system.

## Changing a ready-made system

A ready-made system changes in three places. Its factory arguments change the system itself, its run settings change
how the simulation runs, and its `system` can leave the catalog and run under a simulation of your own.

### Factory arguments

Every factory argument has a default, and `describe` lists them. Pass new values by name to `create_simulation` or to
`create_system`.

```python
from skcomplex.systems import create_simulation, create_system, describe

describe('lotka_volterra').parameters
simulation = create_simulation('lotka_volterra', r=1.5, predators=5.0)
simulation.prepare().run().trajectory_.values('prey.n').max().round()
logistic = create_system('lotka_volterra', K=200.0)
logistic.interactions
```

An argument can change more than a number. With a finite capacity `K`, the factory of `lotka_volterra` replaces
exponential growth of the prey by logistic growth. With `stochastic=True`, the factory of `sir` writes the epidemic as
random jumps of whole individuals, so the compiler infers a jump process instead of differential equations.

```python
from skcomplex.systems import create_simulation, describe

describe('sir').parameters
epidemic = create_simulation('sir', stochastic=True).prepare().run()
epidemic.problem_.form.value, epidemic.trajectory_.values('I.n').max()
```

### Run settings of a ready-made system

The simulation keeps its run settings as parameters, so `set_params` changes them before you prepare it. Every
argument of `Simulation` is a run setting, such as `until`, `dt`, `solver`, `random_state`, and `initial_state`.

```python
from skcomplex.systems import create_simulation

simulation = create_simulation('lotka_volterra')
simulation.until, simulation.dt
simulation.set_params(dt=0.5, until=20.0, initial_state={'prey.n': 20.0})
trajectory = simulation.prepare().run().trajectory_
trajectory.times[-1], trajectory.values('prey.n')[0]
```

`run(until=...)` overrides the end time for one run and leaves `until` as it is.

```python continuation
simulation.run(until=5.0).trajectory_.times[-1], simulation.until
```

A new `initial_state` replaces the whole mapping that the catalog set, so repeat any tuned value you want to keep.

### Simulating the system your own way

`simulation.system` is an ordinary `System`. Give it to a `Simulation` of your own to pick every setting yourself, such
as a fixed-step solver, a schedule of a parameter, or other observables.

```python
from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.engine.solvers import RK4Solver
from skcomplex.systems import create_simulation

system = create_simulation('lotka_volterra').system
simulation = Simulation(system, solver=RK4Solver(dt=0.01), dt=0.5).prepare().run(until=30.0)
simulation.trajectory_.values('predators.n')[-1].round(3)
```

States and parameters are named by their dotted path, such as `prey.n` and `predation.a`. The labels of the elements
and interactions are in `system.elements` and `system.interactions`. The section [Simulating](simulate.md#simulating) lists every
run setting.

## Composing systems

Build a larger system out of smaller ones, such as a landscape of a forest and a meadow, and couple their elements.
A system holds nested systems in `systems`, and an interaction reaches an element of a nested system by a
slash-separated path.

```python
from typing import Annotated

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import RateLaw
from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.specs import ParameterSpec, RoleSpec, StateSpec


class PopulationElement(Element):
    def __init__(self, label, *, n: Annotated[float, StateSpec()] = 1.0):
        super().__init__(label)
        self.n = n


class PredationInteraction(Interaction):
    def __init__(
        self,
        label,
        *,
        predator: Annotated[str, RoleSpec(states=('n',))],
        prey: Annotated[str, RoleSpec(states=('n',))],
        a: Annotated[float, ParameterSpec()] = 0.4,
        b: Annotated[float, ParameterSpec()] = 0.1,
    ):
        super().__init__(label)
        self.predator = predator
        self.prey = prey
        self.a = a
        self.b = b

    def _get_laws(self, s):
        contacts = s.predator.n * s.prey.n
        return [RateLaw(s.prey.n, -s.a * contacts), RateLaw(s.predator.n, s.b * contacts)]


landscape = System(
    systems=[
        System('forest', elements=[PopulationElement('foxes', n=5)]),
        System('meadow', elements=[PopulationElement('rabbits', n=50)]),
    ],
    interactions=[PredationInteraction('hunt', predator='forest/foxes', prey='meadow/rabbits', a=0.02, b=0.01)],
)
trajectory = Simulation(landscape, dt=0.1).prepare().run(until=10).trajectory_
trajectory.values('meadow/rabbits.n')[-1].round(3)
```

The interaction `hunt` sits in the outer system and binds `forest/foxes` and `meadow/rabbits` to its roles. States and
parameters are named by their path, such as `meadow/rabbits.n` and `hunt.a`, in the trajectory, in initial values, and
in parameter changes.

A ready-made system from `create_system` nests in the same way. Give each nested system its own label with
`set_params`, since a system from the catalog has the label `system`.

```python
from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import Simulation
from skcomplex.systems import create_system

island = create_system('lotka_volterra').set_params(label='island')
town = create_system('sir').set_params(label='town')
trajectory = Simulation(System(systems=[island, town]), dt=0.1).prepare().run(until=5).trajectory_
trajectory.values('town/I.n')[-1].round(1)
```
