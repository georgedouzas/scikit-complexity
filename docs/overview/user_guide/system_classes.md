# System classes

This page changes what a ready-made system builds by subclassing it, and writes a system of your own as a class that
takes the parameters of its model. Such a class clones, sweeps, and calibrates like any estimator of scikit-learn.

## Four kinds of system

This section places every system class in one of four kinds, from the most general to the most specific. Every system
derives from [`BaseSystem`](core_components/systems.md#basesystem).

- **`BaseSystem`** is the base of every system. It holds the main public methods, derives the tags, and builds the
  parts of a system from its model parameters.
- **[`CompositeSystem`](core_components/systems.md#compositesystem)** is the composite of elements, interactions, and
  nested systems, on a space and in a time. It plays the role of `Pipeline` in scikit-learn: it holds parts and adds no
  model of its own.
- **General realizations** name a structure that several fields share.
  [`ReactionNetworkSystem`](core_components/general.md#reactionnetworksystem) of `skcomplex.systems.general.reactions`
  is a reaction network, whose reactions fire at a rate constant times the product of the reactant amounts. The general
  systems of `skcomplex.systems.general.dynamics` name structures of states and their evolution.
  [`DynamicalSystem`](core_components/general.md#dynamicalsystem) holds rates or maps of declared states,
  [`NetworkDynamicalSystem`](core_components/general.md#networkdynamicalsystem) holds states on the nodes of a graph,
  [`ReactionDiffusionSystem`](core_components/general.md#reactiondiffusionsystem) holds fields on a lattice, and
  [`StockFlowSystem`](core_components/general.md#stockflowsystem) holds stocks that change through named flows. They
  are recipes written with the core.
- **Field realizations**, such as `SIRSystem`, give a general system or `BaseSystem` the parameters and the words of
  one field. Every ready-made system of the catalog is one.

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

All four checks return `True`. The epidemic is a reaction network, the map of Hénon is a dynamical system, and both
general systems and the composite derive from `BaseSystem`.

## How a system class builds its parts

This section writes a system as a class whose parameters read like those of its model. A subclass of `BaseSystem` does
two things.

- Its `__init__` takes the parameters of the model and stores each under its own name, and nothing else, as in
  scikit-learn. So `get_params`, `set_params`, and `clone` work on them.
- Its hook `_build_system` returns the `CompositeSystem` of its parts, written in terms of these parameters.

The example is a tank that drains at a rate proportional to its level, written with the base class
[`Element`](core_components/elements.md#element) and the general
[`RateInteraction`](core_components/interactions.md#rateinteraction).

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
```

`get_params` returns `k=0.2`, the label, and `level=1.0`, and the tags hold one kind of [law](glossary.md#law),
`'rate'`. The class runs like any system.

```python continuation
round(float(simulate(tank).get_values('tank.level')[-1]), 3)
```

The level falls to 0.135 at time 10, which is `exp(-0.2 * 10)`. `prepare` builds the parts from the current parameters
and keeps them as `system_`. It builds them again only when a parameter changed, so `set_params(k=0.4)` changes the
parts of the next `prepare` and of the next `simulate`. The tags come from the parts and are never declared, so they
cannot drift from what the system does.

The class attributes `until` and `dt` above are tuned run settings. `simulate` reads them, with `solver`,
`observables`, `initial_state`, `invariants`, `schedule`, `compiler`, and `backend`, for every argument a call leaves
out. A system whose settings depend on its parameters overrides `get_run_settings` instead.

## Field realizations

This section shows how a field realization fills a general system. It takes the parameters of its model under its own
names and gives the general system its parts through hooks. `SIRSystem` subclasses `ReactionNetworkSystem`. It returns
its compartments from `_get_species`, and its infection and recovery from `_get_interactions`. Epidemics and enzyme
kinetics are both reaction networks with a product propensity, so they share the general realization and differ only
in their field realizations. `HenonMapSystem` subclasses `DynamicalSystem` in discrete time. It returns its two states
from `_get_states`, its parameters from `_get_parameters`, and its map from `_get_equations`, so its module defines no
element class.

## Changing a ready-made system by subclassing it

This section adds a process to a ready-made system and keeps everything else it does. A subclass of `SIRSystem` that
vaccinates susceptible people at the rate `nu` adds one parameter and one reaction.
[`ReactionInteraction`](core_components/general.md#reactioninteraction) moves people from `S` to `R` at that rate.

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
```

The parts hold the interactions `infection`, `recovery`, and `vaccination`. The subclass repeats the arguments of its
parent in its `__init__`, since `get_params` reads the signature of the class, as in scikit-learn. It overrides one
hook and calls the parent for the rest, so the infection, the recovery, the stochastic mode, and the tuned time step
keep working.

```python continuation
simulate(SIRSystem()).get_values('I.x').max().round(), simulate(vaccinated).get_values('I.x').max().round()
```

Vaccination lowers the peak of the epidemic from 304 to 132 infected people. The subclass is an estimator like its
parent, so every function of the library takes it.

```python continuation
stochastic = vaccinated.set_params(stochastic=True)
stochastic.get_tags().randomness, simulate(stochastic, random_state=0).get_values('I.x').max()
```

With `stochastic=True` the system is `'stochastic'`, and the run with the seed 0 peaks at 110 infected people. The
[contributing guide](../../development/contributing.md#adding-a-system-to-the-catalog) shows how a system class of
your own joins the catalog, with its field, topics, references, tests, and gallery example.
