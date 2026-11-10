# Build a custom system

This tutorial builds one model three ways: from the parts of the core, as a subclass of a shared recipe, and as a
system class with model parameters. The model is the predator and prey cycle of the snowshoe hare and the Canada lynx,
and every later tutorial of the guide starts from it.

## The data

The Hudson's Bay Company bought the pelts of hares and lynx from trappers for two centuries, and the counts rise and
fall in cycles of about ten years. Hewitt (1921, *The Conservation of the Wild Life of Canada*) analysed these records.
The counts below are the pelts of 1900 to 1920, in thousands, as distributed with the case study of Carpenter (2018,
*Predator-Prey Population Dynamics: the Lotka-Volterra model in Stan*) in the file
`knitr/lotka-volterra/hudson-bay-lynx-hare.csv` of the Stan example models. That file cites the lecture notes of
Howard (2009) at Texas A&M University, which trace the numbers through the notes of Mahaffy (Math 636,
San Diego State University) to Purves, Sadava, Orians, and Heller, *Life: The Science of Biology*.

```python
import numpy as np

years = np.arange(1900, 1921)
hares = np.array(
    [30.0, 47.2, 70.2, 77.4, 36.3, 20.6, 18.1, 21.4, 22.0, 25.4, 27.1]
    + [40.3, 57.0, 76.6, 52.3, 19.5, 11.2, 7.6, 14.6, 16.2, 24.7]
)
lynx = np.array(
    [4.0, 6.1, 9.8, 35.2, 59.4, 41.7, 19.0, 13.0, 8.3, 9.1, 7.4]
    + [8.0, 12.3, 19.5, 45.7, 51.1, 29.7, 15.8, 9.7, 10.1, 8.6]
)
int(years[hares.argmax()]), int(years[lynx.argmax()])
```

The hares peak in 1903 and the lynx a year later, in 1904. The predators follow their prey.

## The model

Lotka and Volterra wrote the cycle as two equations. Hares `H` are born at the rate `alpha H` and are hunted at the
rate `beta H L`. Lynx `L` gain at the rate `delta H L` from the hares they eat and die at the rate `gamma L`:

```text
dH/dt = alpha H - beta H L
dL/dt = delta H L - gamma L
```

Carpenter estimated `alpha = 0.55`, `beta = 0.028`, `gamma = 0.80`, and `delta = 0.024` per year, the means of the
posterior of a Bayesian fit. The sections below use these rates and start from the records of 1900, 30.0 thousand
hares and 4.0 thousand lynx.

## From the parts of the core

Each population is a `DeclaredElement` with one state `x`, and each term of the equations is a `RateInteraction`. An
interaction that reads the other population names it in `sources`, and its rate reads it as `s.sources[0]`.
[How the parts fit](../core_components/index.md) explains each part in turn.

```python continuation
from skcomplex.core.components.elements import DeclaredElement
from skcomplex.core.components.interactions import RateInteraction
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.specs import DeclaredVariableSpec, StateSpec

hare_population = DeclaredElement('hares', variables={'x': DeclaredVariableSpec(30.0, StateSpec())})
lynx_population = DeclaredElement('lynx', variables={'x': DeclaredVariableSpec(4.0, StateSpec())})
births = RateInteraction('births', target='hares', state='x', rate=lambda s: 0.55 * s.target.x)
hunting = RateInteraction(
    'hunting', target='hares', state='x', sources=['lynx'], rate=lambda s: -0.028 * s.target.x * s.sources[0].x
)
feeding = RateInteraction(
    'feeding', target='lynx', state='x', sources=['hares'], rate=lambda s: 0.024 * s.target.x * s.sources[0].x
)
deaths = RateInteraction('deaths', target='lynx', state='x', rate=lambda s: -0.80 * s.target.x)
pelts = CompositeSystem(
    elements=[hare_population, lynx_population], interactions=[births, hunting, feeding, deaths]
)
trajectory = simulate(pelts, until=20, dt=1.0)
trajectory.get_values('hares.x').round(1)[:6].tolist()
```

The simulated hares rise from 30.0 to 81.2 thousand in 1903, the year of the highest count of pelts, and fall to 18.6
thousand in 1905. The root mean square difference to the records, 7.8 thousand pelts, measures the fit.

```python continuation
fit = np.sqrt(np.mean((trajectory.get_values('hares.x') - hares) ** 2))
round(float(fit), 1)
```

## As a subclass of a shared recipe

The model is a set of reactions between two species. `ReactionNetworkSystem` of `skcomplex.systems.general.reactions`
builds one element per species and one reaction per row of a table, with a propensity of mass action, the rate
constant times the product of the reactant amounts. A subclass takes the parameters of the model under its own names
and returns the species and the reactions. Each row holds the reactants, the products, and the rate constant.

```python continuation
from skcomplex.systems.general.reactions import ReactionNetworkSystem


class HareLynxSystem(ReactionNetworkSystem):
    until = 20.0
    dt = 1.0

    def __init__(
        self, hares=30.0, lynx=4.0, *, alpha=0.55, beta=0.028, gamma=0.80, delta=0.024, mode='rate', label='system'
    ):
        super().__init__(mode=mode, label=label)
        self.hares = hares
        self.lynx = lynx
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta

    def _get_species(self):
        return {'hares': self.hares, 'lynx': self.lynx}

    def _get_reactions(self):
        return {
            'births': ({'hares': 1}, {'hares': 2}, self.alpha),
            'hunting': ({'hares': 1, 'lynx': 1}, {'lynx': 1}, self.beta),
            'feeding': ({'hares': 1, 'lynx': 1}, {'hares': 1, 'lynx': 2}, self.delta),
            'deaths': ({'lynx': 1}, {}, self.gamma),
        }


simulate(HareLynxSystem()).get_values('hares.x').round(1)[:6].tolist()
```

The values match the composite system above. The class variables `until` and `dt` are tuned run settings, which
`simulate` reads when a call leaves them out. The recipe also runs the same reactions as random events, which suits a
small population.

```python continuation
jumps = HareLynxSystem(hares=30, lynx=4, mode='jump')
sorted(jumps.get_tags().laws), jumps.get_tags().randomness
```

## As a system class with model parameters

A `BaseSystem` subclass writes its parts in `_build_system` from its model parameters, so `set_params`, `clone`, and
the experiments of the library change them by name.

```python continuation
from skcomplex.core.components.systems import BaseSystem


class PeltSystem(BaseSystem):
    until = 20.0
    dt = 1.0

    def __init__(self, hares=30.0, lynx=4.0, *, alpha=0.55, beta=0.028, gamma=0.80, delta=0.024, label='system'):
        super().__init__(label)
        self.hares = hares
        self.lynx = lynx
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta

    def _build_system(self):
        prey = DeclaredElement('hares', variables={'x': DeclaredVariableSpec(self.hares, StateSpec())})
        predators = DeclaredElement('lynx', variables={'x': DeclaredVariableSpec(self.lynx, StateSpec())})
        alpha, beta, gamma, delta = self.alpha, self.beta, self.gamma, self.delta
        interactions = [
            RateInteraction('births', target='hares', state='x', rate=lambda s: alpha * s.target.x),
            RateInteraction(
                'hunting',
                target='hares',
                state='x',
                sources=['lynx'],
                rate=lambda s: -beta * s.target.x * s.sources[0].x,
            ),
            RateInteraction(
                'feeding',
                target='lynx',
                state='x',
                sources=['hares'],
                rate=lambda s: delta * s.target.x * s.sources[0].x,
            ),
            RateInteraction('deaths', target='lynx', state='x', rate=lambda s: -gamma * s.target.x),
        ]
        return CompositeSystem(elements=[prey, predators], interactions=interactions)


system = PeltSystem()
simulate(system).get_values('lynx.x').round(1)[:6].tolist()
```

A faster birth rate of the hares raises their highest count over the twenty years from 81.2 to 85.8 thousand.

```python continuation
faster = simulate(system.set_params(alpha=0.65)).get_values('hares.x')
round(float(faster.max()), 1)
```

## The catalog class

`LotkaVolterraSystem` of `skcomplex.systems.biology.ecology` is the same model, ready made, with the rates `r`, `a`,
`b`, and `d`. The later tutorials start from it in one line.

```python continuation
from skcomplex.systems.biology.ecology import LotkaVolterraSystem

catalog = LotkaVolterraSystem(prey=30.0, predators=4.0, r=0.55, a=0.028, b=0.024, d=0.80)
simulate(catalog, until=20, dt=1.0).get_values('prey.x').round(1)[:6].tolist()
```

## Recipes by kind of system

The pages below build other kinds of system from the same parts.

- [Choosing compatible components](choosing_compatible_components.md)
- [Defining elements](defining_elements.md)
- [Kinds of dynamics](kinds_of_dynamics.md)
- [General interactions](general_interactions.md)
- [Connected components](connected_components.md)
- [Delays, events, drivers, and noise](delays_events_drivers_and_noise.md)
- [Stochastic maps](stochastic_maps.md)
- [Anomalous processes](anomalous_processes.md)
- [Geometry of chains](geometry_of_chains.md)
- [Groups of many elements](groups_of_many_elements.md)
- [Lattice systems](lattice_systems.md)
- [Networks](networks.md)
- [Fields](fields.md)
- [Phase-space fields](phase_space_fields.md)
- [Kinetic systems of active particles](kinetic_systems_of_active_particles.md)
- [Agent-based systems](agent_based_systems.md)
- [Cell complexes and link fields](cell_complexes_and_link_fields.md)
- [Rewriting systems](rewriting_systems.md)
- [Quantum systems](quantum_systems.md)
- [Stocks and flows](stocks_and_flows.md)
- [Linear programs](linear_programs.md)
- [Static systems](static_systems.md)
- [Constraints and search](constraints_and_search.md)
- [Control](control.md)
- [Learning by gradient steps](learning_by_gradient_steps.md)
