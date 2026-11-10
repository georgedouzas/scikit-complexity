# Change it

This page gives a ready-made system your own parameters and run settings, and nests it inside a larger system. The
running example is `LotkaVolterraSystem`, the predator and prey model of Lotka (1925) and Volterra (1926).

## Changing a ready-made system

A ready-made system changes in three places. Its model parameters change the system itself, the arguments of
`simulate` change how it runs, and its parts can leave the class and run as a system of your own.

### Model parameters

This section changes the parameters of a model. Every model parameter is an argument of the class with a default, and
`get_params` lists them.

```python
from skcomplex.core.engine.runtime import simulate
from skcomplex.systems.biology.ecology import LotkaVolterraSystem

LotkaVolterraSystem().get_params()
```

The parameters are the initial populations `prey=40.0` and `predators=9.0`, the prey growth rate `r=1.1`, the
predation rates `a=0.4` and `b=0.1`, the predator death rate `d=0.4`, the capacity `K=None`, and the label. You pass new
values by name to the class, as for any scikit-learn estimator.

```python continuation
simulate(LotkaVolterraSystem(r=1.5, predators=5.0)).get_values('prey.x').max().round()
```

With a faster growth and fewer predators at the start, the prey peak at 41. `set_params` changes the parameters of an
existing system, and `prepare` builds its parts from them.

```python continuation
logistic = LotkaVolterraSystem().set_params(K=200.0)
logistic.prepare().system_.interactions
```

The parts are a `LogisticGrowthInteraction` with `K=200.0`, a `PredationInteraction`, and a `MortalityInteraction`.
`prepare` keeps them as `system_`, the [`CompositeSystem`](core_components/systems.md#compositesystem) of the elements
and interactions of the system.

A parameter can change more than a number. With a finite capacity `K`, `LotkaVolterraSystem` replaces the exponential
growth of the prey by logistic growth, as above. With `stochastic=True`, `SIRSystem` writes the epidemic as random
jumps of whole individuals. Its tags then change from rates to jumps, and `simulate` picks a solver of jump processes
instead of one of differential equations.

```python
from skcomplex.core.engine.runtime import simulate
from skcomplex.systems.biology.epidemiology import SIRSystem

epidemic = SIRSystem(stochastic=True)
tags = epidemic.get_tags()
tags.laws, tags.randomness, simulate(epidemic, random_state=0).get_values('I.x').max()
```

The laws are of the kind `'jump'`, the system is `'stochastic'`, and the run with the seed 0 peaks at 252 infected
people, against 304 for the deterministic model.

### Run settings of a ready-made system

This section overrides the tuned run settings for one run. A system class keeps tuned run settings, which `simulate`
uses for every argument a call leaves out. An argument of the call overrides one of them, such as `until`, `dt`,
`solver`, `initial_state`, or `random_state`.

```python
from skcomplex.core.engine.runtime import simulate
from skcomplex.systems.biology.ecology import LotkaVolterraSystem

settings = LotkaVolterraSystem().get_run_settings()
settings['until'], settings['dt']
```

The tuned run ends at time 50 with a frame every 0.1. The call below ends it at 20, records a frame every 0.5, and
starts from 20 prey.

```python continuation
trajectory = simulate(LotkaVolterraSystem(), until=20.0, dt=0.5, initial_state={'prey.x': 20.0})
trajectory.times[-1], trajectory.get_values('prey.x')[0]
```

The run ends at 20.0 and starts from 20.0 prey. A new `initial_state` replaces the whole mapping that a class tuned, so
repeat any tuned value you want to keep.

### Simulating the system your own way

This section picks every setting of a run yourself, such as a fixed-step [solver](glossary.md#solver).
`RK4Solver` integrates ordinary differential equations by the classical Runge-Kutta method with a fixed step.

```python
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.solvers import RK4Solver
from skcomplex.systems.biology.ecology import LotkaVolterraSystem

trajectory = simulate(LotkaVolterraSystem(), until=30.0, dt=0.5, solver=RK4Solver(dt=0.01))
trajectory.get_values('predators.x')[-1].round(3)
```

The predators end at 0.48 at time 30. States and parameters are named by their dotted path, such as `prey.x` and
`predation.a`. The labels of the elements and interactions are those of the parts in `system_` after `prepare`. The
section [Run settings](simulate.md#run-settings) of the simulation tutorial lists every run setting.

## Composing systems

This section builds a larger system out of smaller ones, a landscape of a forest and a meadow, and couples their
elements. A system holds nested systems in `systems`, and an [interaction](glossary.md#interaction) reaches an element
of a nested system by a path separated by slashes. The [element](glossary.md#element) and the interaction below are
written from the base classes [`Element`](core_components/elements.md#element) and
[`Interaction`](core_components/interactions.md#interaction), as the page
[Build a custom system](build_a_custom_system/index.md) explains.

```python
from typing import Annotated

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import RateLaw
from skcomplex.core.components.systems import CompositeSystem
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


landscape = CompositeSystem(
    systems=[
        CompositeSystem('forest', elements=[PopulationElement('foxes', n=5)]),
        CompositeSystem('meadow', elements=[PopulationElement('rabbits', n=50)]),
    ],
    interactions=[PredationInteraction('hunt', predator='forest/foxes', prey='meadow/rabbits', a=0.02, b=0.01)],
)
trajectory = simulate(landscape, until=10, dt=0.1)
trajectory.get_values('meadow/rabbits.n')[-1].round(3)
```

The rabbits fall from 50 to 0.735 in ten time units. The interaction `hunt` sits in the outer system and binds
`forest/foxes` and `meadow/rabbits` to its roles, the names under which it reads and changes them. States and
parameters are named by their path, such as `meadow/rabbits.n` and `hunt.a`, in the trajectory, in initial values, and
in parameter changes.

A ready-made system nests in the same way, through its parts. Give each nested system its own label, since a system
class has the label `system` by default, and nest the composite `system_` of its parts.

```python
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.systems.biology.ecology import LotkaVolterraSystem
from skcomplex.systems.biology.epidemiology import SIRSystem

island = LotkaVolterraSystem(label='island').prepare().system_
town = SIRSystem(label='town').prepare().system_
trajectory = simulate(CompositeSystem(systems=[island, town]), until=5, dt=0.1)
trajectory.get_values('town/I.x')[-1].round(1)
```

The epidemic of the town reaches 26.4 infected people at time 5, while the island runs beside it. To change what a
ready-made system builds, rather than its values, subclass it, as the page [System classes](system_classes.md) shows.
