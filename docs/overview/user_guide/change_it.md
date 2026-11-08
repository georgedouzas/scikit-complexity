# Change it

Give a ready-made system your own parameters and run settings, or nest it inside a larger system.

## Changing a ready-made system

A ready-made system changes in three places. Its model parameters change the system itself, the arguments of
`simulate` change how it runs, and its parts can leave the catalog and run as a system of your own.

### Model parameters

Every model parameter is an argument of the class with a default, and `get_params` lists them. Pass new values by
name to the class, or change them later with `set_params`, as for any scikit-learn estimator.

```python
from skcomplex.core.engine.runtime import simulate
from skcomplex.systems.biology.ecology import LotkaVolterraSystem

LotkaVolterraSystem().get_params()
simulate(LotkaVolterraSystem(r=1.5, predators=5.0)).values('prey.n').max().round()
logistic = LotkaVolterraSystem().set_params(K=200.0)
logistic.prepare().system_.interactions
```

`prepare` builds the parts of a system from its current parameters, and keeps them as `system_`, the composite
`System` of its elements and interactions.

A parameter can change more than a number. With a finite capacity `K`, `LotkaVolterraSystem` replaces exponential
growth of the prey by logistic growth. With `stochastic=True`, `SIRSystem` writes the epidemic as random jumps of whole
individuals, so its tags change from rates to jumps, and `simulate` picks a jump process solver instead of an ODE
solver.

```python
from skcomplex.core.engine.runtime import simulate
from skcomplex.systems.biology.epidemiology import SIRSystem

epidemic = SIRSystem(stochastic=True)
tags = epidemic.get_tags()
tags.laws, tags.randomness, simulate(epidemic, random_state=0).values('I.n').max()
```

### Run settings of a ready-made system

A system class keeps tuned run settings, which `simulate` uses for every argument a call leaves out. Pass an argument
to override one for a run, such as `until`, `dt`, `solver`, `initial_state`, or `random_state`.

```python
from skcomplex.core.engine.runtime import simulate
from skcomplex.systems.biology.ecology import LotkaVolterraSystem

settings = LotkaVolterraSystem().get_run_settings()
settings['until'], settings['dt']
trajectory = simulate(LotkaVolterraSystem(), until=20.0, dt=0.5, initial_state={'prey.n': 20.0})
trajectory.times[-1], trajectory.values('prey.n')[0]
```

A new `initial_state` replaces the whole mapping that the class tuned, so repeat any tuned value you want to keep.

### Simulating the system your own way

Pick every setting yourself, such as a fixed-step solver, a schedule of a parameter, or other observables.

```python
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.solvers import RK4Solver
from skcomplex.systems.biology.ecology import LotkaVolterraSystem

trajectory = simulate(LotkaVolterraSystem(), until=30.0, dt=0.5, solver=RK4Solver(dt=0.01))
trajectory.values('predators.n')[-1].round(3)
```

States and parameters are named by their dotted path, such as `prey.n` and `predation.a`. The labels of the elements
and interactions are those of the parts in `system_`, after `prepare`. The section [Simulating](simulate.md#simulating)
lists every run setting.

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
from skcomplex.core.engine.runtime import simulate
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
trajectory = simulate(landscape, until=10, dt=0.1)
trajectory.values('meadow/rabbits.n')[-1].round(3)
```

The interaction `hunt` sits in the outer system and binds `forest/foxes` and `meadow/rabbits` to its roles. States and
parameters are named by their path, such as `meadow/rabbits.n` and `hunt.a`, in the trajectory, in initial values, and
in parameter changes.

A ready-made system nests in the same way, through its parts. Give each nested system its own label, since a system
class has the label `system` by default, and nest the composite `system_` of its parts.

```python
from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import simulate
from skcomplex.systems.biology.ecology import LotkaVolterraSystem
from skcomplex.systems.biology.epidemiology import SIRSystem

island = LotkaVolterraSystem(label='island').prepare().system_
town = SIRSystem(label='town').prepare().system_
trajectory = simulate(System(systems=[island, town]), until=5, dt=0.1)
trajectory.values('town/I.n')[-1].round(1)
```

To change what a ready-made system builds, rather than its values, subclass it, as the page
[System classes](system_classes.md) shows.
