# System classes

Change what a ready-made system builds by subclassing it, and write a system of your own as a class that takes the
parameters of its model.

## Four kinds of system

Every system derives from `BaseSystem`, and comes in one of four kinds, from the most general to the most specific:

- **`BaseSystem`** is the base of every system. It holds the main public methods, derives the tags, and builds the parts of a system
  from its model parameters.
- **`CompositeSystem`** is the composite of elements, interactions, and nested systems, on a space and in a time. It
  plays the role of `Pipeline` in scikit-learn: it holds parts and adds no model of its own.
- **General realizations**, such as `ReactionNetworkSystem` of `skcomplex.systems.general.reactions`, name a structure
  that several fields share, such as a reaction network, whose reactions fire at a rate constant times the product of
  the reactant amounts. The general systems of `skcomplex.systems.general.dynamics` name the structures of states and
  their evolution: `DynamicalSystem` for rates or maps of declared states, `NetworkDynamicalSystem` for states on the
  nodes of a graph, `ReactionDiffusionSystem` for fields on a lattice, and `StockFlowSystem` for stocks that change
  through named flows. They are recipes written with the core, in
  [General realizations](core_components/general.md#reaction-networks) and
  [Dynamical systems](core_components/general.md#dynamical-systems).
- **Field realizations**, such as `SIRSystem`, give a general system or `BaseSystem` the parameters and the words of one
  field. Every ready-made system of the catalog is one.

The kinds mirror the linear models of scikit-learn, where `Ridge` and `Lasso` realize one general structure with
parameters of their own.

```python
from skcomplex.core.components.systems import BaseSystem, CompositeSystem
from skcomplex.systems.general.dynamics import DynamicalSystem
from skcomplex.systems.general.reactions import ReactionNetworkSystem
from skcomplex.systems.biology.epidemiology import SIRSystem
from skcomplex.systems.mathematics.dynamical_systems import HenonMapSystem

(
    issubclass(SIRSystem, ReactionNetworkSystem),
    issubclass(HenonMapSystem, DynamicalSystem),
    issubclass(ReactionNetworkSystem, BaseSystem),
    issubclass(CompositeSystem, BaseSystem),
)
```

## How a system class builds its parts

Write a system as a class when you want parameters that read like those of the model, and a system that clones, sweeps,
and calibrates like any estimator. A subclass of `BaseSystem` does two things:

- its `__init__` takes the parameters of the model and stores each under its own name, and nothing else, as in
  scikit-learn, so `get_params`, `set_params`, and `clone` work on them,
- its hook `_build_system` returns the `CompositeSystem` of its parts, written in terms of these parameters.

```python
from typing import Annotated

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import RateInteraction
from skcomplex.core.components.systems import BaseSystem, CompositeSystem
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
        return CompositeSystem(self.label, elements=[TankElement('tank', level=self.level)], interactions=[drain])


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

## Field realizations

A field realization takes the parameters of its model under its own names, and gives a general system or `BaseSystem`
its parts through hooks. `SIRSystem` subclasses `ReactionNetworkSystem`, returns its compartments from `_get_species`,
and its infection and recovery from `_get_interactions`. Epidemics and enzyme kinetics are both reaction networks with
a product propensity, so they share the general realization and differ in their own. `HenonMapSystem` subclasses
`DynamicalSystem` in discrete time, and returns its two states from `_get_states`, its parameters from
`_get_parameters`, and its map from `_get_equations`, so its module defines no element class.

## Changing a ready-made system by subclassing it

Add a process to a ready-made system, and keep everything else it does. A subclass of `SIRSystem` that vaccinates
susceptible individuals at the rate `nu` adds one parameter and one reaction:

```python
from skcomplex.systems.general.reactions import ReactionInteraction
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
            reactants={('S', 'x'): 1},
            products={('R', 'x'): 1},
            k=self.nu,
            mode='jump' if self.stochastic else 'rate',
        )
        return [*super()._get_interactions(), vaccination]


vaccinated = VaccinatedSIRSystem()
[interaction.label for interaction in vaccinated.prepare().system_.interactions]
simulate(SIRSystem()).values('I.x').max().round(), simulate(vaccinated).values('I.x').max().round()
```

The subclass repeats the arguments of its parent in its `__init__`, since `get_params` reads the signature of the
class, as in scikit-learn. It overrides one hook and calls the parent for the rest, so the infection, the recovery,
the stochastic mode, and the tuned time step keep working. Vaccination lowers the peak of the epidemic.

The subclass is an estimator like its parent, so every function of the library takes it:

```python continuation
stochastic = vaccinated.set_params(stochastic=True)
stochastic.get_tags().randomness, simulate(stochastic, random_state=0).values('I.x').max()
```

The [contributing guide](../../development/contributing.md#adding-a-system-to-the-catalog) shows how a system class of
your own joins the catalog, with its field, topics, references, tests, and gallery example.
