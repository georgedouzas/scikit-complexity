# System classes

Change what a ready-made system builds by subclassing it, and write a system of your own as a class that takes the
parameters of its model.

## Four kinds of system

Every system derives from `BaseSystem`, and comes in one of four kinds, from the most general to the most specific:

- **`BaseSystem`** is the base of every system. It holds the main public methods, derives the tags, and builds the parts of a system
  from its model parameters.
- **`System`** is the generic composite of elements, interactions, and nested systems, on a space and in a time. It
  plays the role of `Pipeline` in scikit-learn: it holds parts and adds no model of its own.
- **General systems of the core**, such as `MassActionSystem`, name a mathematical structure that several fields share,
  such as species in compartments that react at mass-action rates.
- **Field realizations**, such as `SIRSystem`, give a general system or `BaseSystem` the parameters and the words of one
  field. Every ready-made system of the catalog is one.

The kinds mirror the linear models of scikit-learn, where `Ridge` and `Lasso` realize one general structure with
parameters of their own.

```python
from skcomplex.core.components.mass_action import MassActionSystem
from skcomplex.core.components.system import BaseSystem, System
from skcomplex.systems.biology.epidemiology import SIRSystem

issubclass(SIRSystem, MassActionSystem), issubclass(MassActionSystem, BaseSystem), issubclass(System, BaseSystem)
```

## How a system class builds its parts

Write a system as a class when you want parameters that read like those of the model, and a system that clones, sweeps,
and calibrates like any estimator. A subclass of `BaseSystem` does two things:

- its `__init__` takes the parameters of the model and stores each under its own name, and nothing else, as in
  scikit-learn, so `get_params`, `set_params`, and `clone` work on them,
- its hook `_build_system` returns the `System` of its parts, written in terms of these parameters.

```python
from typing import Annotated

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import RateInteraction
from skcomplex.core.components.system import BaseSystem, System
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.specs import StateSpec


class TankElement(Element):
    def __init__(self, label, *, level: Annotated[float, StateSpec()] = 1.0):
        super().__init__(label)
        self.level = level


class DrainingTankSystem(BaseSystem):
    until = 10.0
    dt = 0.1

    def __init__(self, level=1.0, *, k=0.5, label='system'):
        super().__init__(label)
        self.level = level
        self.k = k

    def _build_system(self):
        drain = RateInteraction('drain', target='tank', state='level', rate=lambda s: -self.k * s.target.level)
        return System(self.label, elements=[TankElement('tank', level=self.level)], interactions=[drain])


tank = DrainingTankSystem(k=0.2)
tank.get_params(), sorted(tank.get_tags().laws)
round(float(simulate(tank).values('tank.level')[-1]), 3)
```

`prepare` builds the parts from the current parameters and keeps them as `system_`. It builds them again only when a
parameter changed, so `set_params(k=0.4)` changes the parts of the next `prepare` and of the next `simulate`. The tags
come from the parts and are never declared, so they cannot drift from what the system does.

The class attributes `until` and `dt` above are tuned run settings. `simulate` reads them, with `solver`,
`observables`, `initial_state`, `invariants`, `schedule`, `compiler`, and `backend`, for every argument a call leaves
out. A system whose settings depend on its parameters overrides `get_run_settings` instead.

## General systems of the core

Write a compartment model of any field without writing an element or an interaction. `MassActionSystem` takes the
initial amount of every species and a table of reactions. Every reaction maps its label to its reactants and products,
as coefficients by species name, and to its rate constant.

```python
from skcomplex.core.components.mass_action import MassActionSystem
from skcomplex.core.engine.runtime import simulate

isomerization = MassActionSystem(
    species={'A': 10.0, 'B': 0.0},
    reactions={'forth': ({'A': 1}, {'B': 1}, 0.3), 'back': ({'B': 1}, {'A': 1}, 0.1)},
)
round(float(simulate(isomerization, until=50.0, dt=1.0).values('A.n')[-1]), 3)
isomerization.set_params(mode='jump').get_tags().randomness
```

Every species is a `CompartmentElement` named after it, whose state `n` holds its amount, and every reaction is a
`ReactionInteraction`. The `mode` writes the reactions as rate equations, `'rate'`, as random events of single
molecules, `'jump'`, or as chemical Langevin equations, `'langevin'`. The amount of `A` relaxes to
`0.1 / (0.3 + 0.1)` of the total.

## Field realizations

A field realization takes the parameters of its model under its own names, and gives the general system its parts
through hooks. `SIRSystem` subclasses `MassActionSystem`. It takes `susceptible`, `infected`, `beta`, and `gamma`,
returns its compartments from `_get_species`, and its infection and recovery from `_get_interactions`. Epidemics of
susceptible, infected, and recovered individuals and enzyme kinetics are both mass-action systems, so they share the
general system and differ in their realizations.

The hooks of `MassActionSystem` are these:

- `_get_species` returns the initial amount of every species by name,
- `_get_reactions` returns the reaction table,
- `_get_mode` returns the mode,
- `_get_interactions` returns the reactions as interactions, for a realization whose reactions have roles of their own
  names, such as the `susceptible` and `infected` roles of an infection.

## Changing a ready-made system by subclassing it

Add a process to a ready-made system, and keep everything else it does. A subclass of `SIRSystem` that vaccinates
susceptible individuals at the rate `nu` adds one parameter and one reaction:

```python
from skcomplex.core.components.interactions import ReactionInteraction
from skcomplex.core.engine.runtime import simulate
from skcomplex.systems.biology.epidemiology import SIRSystem


class VaccinatedSIRSystem(SIRSystem):
    until = 100.0

    def __init__(
        self,
        susceptible=990.0,
        infected=10.0,
        recovered=0.0,
        *,
        beta=0.0003,
        gamma=0.1,
        nu=0.02,
        stochastic=False,
        label='system',
    ):
        super().__init__(susceptible, infected, recovered, beta=beta, gamma=gamma, stochastic=stochastic, label=label)
        self.nu = nu

    def _get_interactions(self):
        vaccination = ReactionInteraction(
            'vaccination',
            reactants={('S', 'n'): 1},
            products={('R', 'n'): 1},
            k=self.nu,
            mode='jump' if self.stochastic else 'rate',
        )
        return [*super()._get_interactions(), vaccination]


vaccinated = VaccinatedSIRSystem()
[interaction.label for interaction in vaccinated.prepare().system_.interactions]
simulate(SIRSystem()).values('I.n').max().round(), simulate(vaccinated).values('I.n').max().round()
```

The subclass repeats the arguments of its parent in its `__init__`, since `get_params` reads the signature of the
class, as in scikit-learn. It overrides one hook and calls the parent for the rest, so the infection, the recovery,
the stochastic mode, and the tuned time step keep working. Vaccination lowers the peak of the epidemic.

The subclass is an estimator like its parent, so every function of the library takes it:

```python continuation
stochastic = vaccinated.set_params(stochastic=True)
stochastic.get_tags().randomness, simulate(stochastic, random_state=0).values('I.n').max()
```

The [contributing guide](../../development/contributing.md#adding-a-system-to-the-catalog) shows how a system class of
your own joins the catalog, with its field, topics, references, tests, and gallery example.
