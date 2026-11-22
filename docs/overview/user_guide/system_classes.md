# System classes

This page changes what a ready-made system builds by subclassing it, and writes a system of your own as a class that
takes the parameters of its model. Such a class clones, sweeps, and calibrates like any estimator of scikit-learn.

## Four kinds of system

This section places every system class in one of four kinds, from the most general to the most specific. Every system
derives from [`BaseSystem`](core_components/systems.md#basesystem).

- **`BaseSystem`** is the base of every system. It holds the main public methods, derives the tags, and builds the
  components of a system from its model parameters.
- **[`System`](core_components/systems.md#system)** is the composite of elements, interactions, and nested systems,
  on a space. It plays the role of `Pipeline` in scikit-learn: it holds components and
  adds no model of its own.
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
from skcomplex.core.systems import BaseSystem, System
from skcomplex.systems.general.reactions import ReactionNetworkSystem
from skcomplex.systems.biology.epidemiology import SIRSystem

(
    issubclass(SIRSystem, ReactionNetworkSystem),
    issubclass(ReactionNetworkSystem, BaseSystem),
    issubclass(System, BaseSystem),
)
```

All three checks return `True`. The epidemic is a reaction network, and both general systems and the composite derive
from `BaseSystem`.

## How a system class builds its components

This section writes a system as a class whose parameters read like those of its model. A subclass of `BaseSystem` does
two things.

- Its `__init__` takes the parameters of the model and stores each under its own name, and nothing else, as in
  scikit-learn. So `get_params`, `set_params`, and `clone` work on them.
- Its hook `_prepare` sets `system_`, the `System` of its components, written in terms of these parameters.

The example is a tank that drains at a rate proportional to its level, written with an
[`Element`](core_components/elements.md#element) that holds the level and a one-off
[`Interaction`](core_components/interactions.md#interaction) that drains it.

```python
from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import RateLaw
from skcomplex.core.systems import BaseSystem, System


class DrainingTankSystem(BaseSystem):
    until = 10.0
    dt = 0.1

    def __init__(self, level=1.0, *, k=0.5, label='system'):
        super().__init__(label)
        self.level = level
        self.k = k

    def _prepare(self, context=None):
        super()._prepare(context)
        k = self.k
        tank = Element('tank', values={'level': self.level})

        def outflow(s):
            return [RateLaw(s.target.level, -k * s.target.level)]

        drain = Interaction('drain', roles={'target': 'tank'}, laws=outflow)
        self.system_ = System(self.label, elements=[tank], interactions=[drain])


tank = DrainingTankSystem(k=0.2).prepare()
tank.get_params(), tank.states_, tank.form_
```

`get_params` returns `k=0.2`, the label, and `level=1.0`. The level is the one state, and the system is an ordinary
differential equation. The class runs like any system.

```python continuation
round(float(tank.simulate().get_values('tank.level')[-1]), 3)
```

The level falls to 0.135 at time 10, which is `exp(-0.2 * 10)`. `prepare` builds the components from the current
parameters, keeps them as `system_`, and compiles them once. After `set_params(k=0.4)` the system must be prepared
again before it simulates, and the next `prepare` builds the components with the new rate.

The class attributes `until` and `dt` above are tuned run settings, which `simulate` reads when a call leaves them
out. The solver and the backend are the settings `solver=` and `backend=` of a system, as for `System`.

## Field realizations

This section shows how a field realization fills a general system. It takes the parameters of its model under its own
names and gives the general system its components through hooks. `SIRSystem` subclasses `ReactionNetworkSystem`. It
returns its compartments from `_get_species`, and its infection and recovery from `_get_interactions`. Epidemics and
enzyme kinetics are both reaction networks with a product propensity, so they share the general realization and differ
only in their field realizations.

## Changing a ready-made system by subclassing it

This section adds a process to a ready-made system and keeps everything else it does. A subclass of `SIRSystem` that
vaccinates susceptible people at the rate `nu` adds one parameter and one reaction.
[`ReactionInteraction`](core_components/general.md#reactioninteraction) moves people from `S` to `R` at that rate.

```python
from skcomplex.systems.general.reactions import ReactionInteraction
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

The components hold the interactions `infection`, `recovery`, and `vaccination`. The subclass repeats the arguments of
its parent in its `__init__`, since `get_params` reads the signature of the class, as in scikit-learn. It overrides one
hook and calls the parent for the rest, so the infection, the recovery, the stochastic mode, and the tuned time step
keep working.

```python continuation
SIRSystem().prepare().simulate().get_values('I.x').max().round(), vaccinated.prepare().simulate().get_values('I.x').max().round()
```

Vaccination lowers the peak of the epidemic from 304 to 132 infected people. The subclass is an estimator like its
parent, so every function of the library takes it.

```python continuation
stochastic = vaccinated.set_params(stochastic=True)
stochastic.get_tags().randomness, stochastic.prepare().simulate(random_state=0).get_values('I.x').max()
```

With `stochastic=True` the system is `'stochastic'`, and the run with the seed 0 peaks at 110 infected people. The
[contributing guide](../../development/contributing.md#adding-a-system-to-the-catalog) shows how a system class of
your own joins the catalog, with its field, topics, references, tests, and gallery example.
