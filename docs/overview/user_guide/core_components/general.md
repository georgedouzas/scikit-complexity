# General realizations

`skcomplex.systems.general` holds the recipes that systems of several fields share, such as a nonnegative amount, a
reaction, an agent that moves, or a market that matches buyers with sellers. A recipe is a component written with the
public components of the core alone, so each one is also an example of how to build a component of your own. The core
never imports these recipes, and every field imports them freely. [How the components fit](index.md) shows the
components of the core that the recipes are written with, and the section [Where a component
goes](#where-a-component-goes) at the end of this page gives the rule that decides whether a new component is a recipe.

The package has one subpackage per concept, and every recipe is imported from its subpackage.

| Subpackage | Concept | Recipes |
| --- | --- | --- |
| `elements` | amounts and bodies | `NonnegativeElement`, `PointElement` |
| `flows` | chaotic flows | `LorenzElement` |
| `reactions` | reactions at a product propensity | `ReactionInteraction`, `ReactionNetworkSystem` |
| `kinetics` | encounters of distributions | `EncounterInteraction` |
| `dynamics` | systems given as equations | `DynamicalSystem`, `NetworkDynamicalSystem`, `ReactionDiffusionSystem`, `StockFlowSystem` |
| `agents` | agent rules | `MovementInteraction`, `ChoiceInteraction`, `ImitationInteraction`, `EvolutionInteraction`, `BirthDeathInteraction` |
| `matching` | markets that pair two groups | `MatchingInteraction`, its rule specs, and `MatchSpec` |

## NonnegativeElement

A `NonnegativeElement` holds one number `x` that stays at zero or above, such as a count of people or the moles of a
chemical species. It is an [`Element`](elements.md#element) whose one state carries the domain `[0, +oo)`, and it
returns no laws of its own. Here 100 infected people recover at a rate of 0.1 a day through a
[`RateInteraction`](interactions.md#rateinteraction).

```python
from skcomplex.core.components.interactions import RateInteraction
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.systems.general.elements import NonnegativeElement

infected = NonnegativeElement('infected', x=100.0)
recovery = RateInteraction('recovery', target='infected', state='x', rate=lambda s: -0.1 * s.target.x)
epidemic = CompositeSystem(elements=[infected], interactions=[recovery])
simulate(epidemic, until=10, dt=5.0).get_values('infected.x').round(1).tolist()
```

The infected fall from 100 to 60.7 after five days and to 36.8 after ten.

The same element holds the moles of a chemical species in a reactor. A species that decays at the rate 0.5 loses about
two thirds of its amount in two seconds.

```python continuation
species = NonnegativeElement('A', x=2.0)
decay = RateInteraction('decay', target='A', state='x', rate=lambda s: -0.5 * s.target.x)
simulate(CompositeSystem(elements=[species], interactions=[decay]), until=2, dt=1.0).get_values('A.x').round(3).tolist()
```

The amount falls from 2.0 to 1.213 and then to 0.736, which is `2 exp(-1)`.

**How the engine runs it.** The element adds no laws. The compiler reads its state as it reads any annotated state, and
the laws of the interactions that target it form the equations a solver integrates.

**How it combines.** It is the species of [`ReactionInteraction`](#reactioninteraction) and
[`ReactionNetworkSystem`](#reactionnetworksystem), and a [`GroupElement`](elements.md#groupelement) of it on the sites
of a lattice is the distribution that [`EncounterInteraction`](#encounterinteraction) reads.

**Limits and errors.** It holds one number. A system that tracks several numbers per member declares its own
`Element`. The check of the domain runs when the system compiles, so a negative initial value raises a
`ValidationError` from `simulate`.

```python continuation
from skcomplex.core.errors import ValidationError

try:
    simulate(CompositeSystem(elements=[NonnegativeElement('A', x=-1.0)]), until=1, dt=1.0)
except ValidationError as error:
    message = str(error)
message
```

The message names the value `A.x = -1.0` and the domain `[0, +oo)` that does not hold it. The fix is an initial
value of zero or above.

**A field realization.** Biology names it a compartment of an epidemic and declares the same state again, so the
docstring of the subclass can say that `x` counts people.

```python continuation
from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from skcomplex.core.specs import StateSpec


class CompartmentElement(NonnegativeElement):
    def __init__(self, label, visual=None, *, x: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0):
        super().__init__(label, visual=visual, x=x)


CompartmentElement('susceptible', x=990.0).get_params()
```

See its [API entry][skcomplex.systems.general.elements].

## PointElement

A `PointElement` is a body that moves in a continuous space by its acceleration. Its state `x` is a vector of order 2,
the position, and the companion state `x_d1` holds the velocity. Here a ball leaves the origin at the velocity
`(1, 2)` and falls under a gravity of 1, which a `RateInteraction` gives as the second derivative of `x`.

```python
from skcomplex.core.components.interactions import RateInteraction
from skcomplex.core.components.spaces import ContinuousSpace
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.systems.general.elements import PointElement

ball = PointElement('ball', x=[0.0, 0.0], x_d1=[1.0, 2.0])
gravity = RateInteraction('gravity', target='ball', state='x', rate=lambda s: [0.0, -1.0])
flight = CompositeSystem(elements=[ball], interactions=[gravity], space=ContinuousSpace(n_dim=2))
simulate(flight, until=2.0, dt=1.0).get_values('ball.x').round(3).tolist()
```

The ball reaches `(1, 1.5)` after one second and `(2, 2)` after two, the parabola `(t, 2t - t^2 / 2)`.

A star that drifts through a cluster at a constant velocity is the same element with no force on it.

```python continuation
star = PointElement('star', x_d1=[1.0, 0.0])
drift = CompositeSystem(elements=[star], space=ContinuousSpace(n_dim=2))
simulate(drift, until=2.0, dt=1.0).get_values('star.x')[-1].tolist()
```

The star stands at `(2, 0)` after two units of time.

**How the engine runs it.** The laws on `x` give its second derivative, and the compiler adds the first-order
equations of the pair `x` and `x_d1`. A backend for molecular dynamics, such as LAMMPS, reads the velocity of a point
from the declared state `x_d1`, and the boundary of the space acts on `x`.

**How it combines.** It lives in a [`ContinuousSpace`](spaces.md#continuousspace). A
[`PairInteraction`](interactions.md#pairinteraction) adds forces between points, such as those of a gas or of a flock.

**Limits and errors.** Its state is a vector, so the system needs a space with a dimension, and `simulate` raises a
`SpaceError` without one.

```python continuation
from skcomplex.core.errors import SpaceError

try:
    simulate(CompositeSystem(elements=[PointElement('p', x_d1=[1.0, 0.0])]), until=2.0, dt=1.0)
except SpaceError as error:
    message = str(error)
message
```

The message says that `p.x` is a vector, so the system that holds it must have a space. The fix is
`space=ContinuousSpace(n_dim=2)`.

**A field realization.** Physics names the point an atom and fixes its start at rest.

```python continuation
class AtomElement(PointElement):
    def __init__(self, label, x=0.0, visual=None):
        super().__init__(label, x=x, x_d1=0.0, visual=visual)


AtomElement('argon', x=[1.0, 2.0]).get_params()
```

See its [API entry][skcomplex.systems.general.elements].

## LorenzElement

A `LorenzElement` follows the Lorenz flow `x' = sigma (y - x)`, `y' = x (rho - z) - y`, `z' = x y - beta z`, which
Lorenz derived from convection in the atmosphere. It is an `Element` that returns three
[`RateLaw`](laws.md#ratelaw) laws. For `sigma = 10`, `rho = 28`, and `beta = 8/3` its runs never settle and never
repeat.

```python
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.systems.general.flows import LorenzElement

trajectory = simulate(CompositeSystem(elements=[LorenzElement('lorenz')]), until=1.0, dt=0.5)
trajectory.get_values('lorenz.x').round(3).tolist()
```

The state `x` moves from 1.0 to 1.198 and swings to -9.379 after one unit of time.

Below the critical value of `rho` the same flow describes a water wheel that settles into a steady rotation. With
`rho = 10` the state `z` settles at `rho - 1`.

```python continuation
wheel = simulate(CompositeSystem(elements=[LorenzElement('wheel', rho=10.0)]), until=20.0, dt=10.0)
wheel.get_values('wheel.z').round(3).tolist()
```

The state `z` passes 9.023 at time 10 and rests at 9.0 at time 20.

**How the engine runs it.** The three rates form an ordinary differential equation that `ScipyODESolver` integrates.

**How it combines.** It runs alone as a model, or it drives a response system through an interaction that reads its
states, as in studies of chaotic synchronization.

**Limits and errors.** Its parameters are `sigma`, `rho`, and `beta`, and `set_params` raises a `ValueError` for any
other name.

```python continuation
try:
    LorenzElement('lorenz').set_params(r=1.0)
except ValueError as error:
    message = str(error)
message
```

The message lists the valid parameters `beta`, `label`, `rho`, `sigma`, `visual`, `x`, `y`, and `z`. The fix is
`rho=...`.

**A field realization.** `LorenzSystem` of mathematics wraps the element in a system class that takes the parameters
of the model, and the chaotic synchronization of physics drives a response system with it.

```python continuation
from skcomplex.systems.mathematics.dynamical_systems import LorenzSystem

LorenzSystem(rho=20.0).prepare().system_.elements[0].rho
```

The system builds one `LorenzElement` with `rho = 20.0`.

See its [API entry][skcomplex.systems.general.flows].

## ReactionInteraction

A `ReactionInteraction` moves units from its reactants to its products, at a propensity that is the rate constant `k`
times the product of the reactant amounts, each raised to its coefficient. It is a
[`JumpInteraction`](interactions.md#jumpinteraction) whose changes are the stoichiometric vector, the number of units
each reaction takes from or gives to every species. A species is a pair of the path of an element and a state name.
Here an epidemic runs as two reactions, infection `S + I -> 2 I` and recovery `I -> R`.

```python
from skcomplex.core.components.interactions import JumpInteraction
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.systems.general.elements import NonnegativeElement
from skcomplex.systems.general.reactions import ReactionInteraction

infection = ReactionInteraction('infection', reactants={('S', 'x'): 1, ('I', 'x'): 1}, products={('I', 'x'): 2}, k=5e-4)
recovery = ReactionInteraction('recovery', reactants={('I', 'x'): 1}, products={('R', 'x'): 1}, k=0.1)
people = [NonnegativeElement('S', x=990), NonnegativeElement('I', x=10), NonnegativeElement('R')]
sir = CompositeSystem(elements=people, interactions=[infection, recovery])
trajectory = simulate(sir, until=50, dt=10.0)
infected = trajectory.get_values('I.x').round(1).tolist()
infected, float(trajectory.states[-1].sum()), issubclass(ReactionInteraction, JumpInteraction)
```

The infected rise from 10 to 400.3 at day 20 and fall to 26.4 at day 50, while the population stays at 1000. The last
value confirms that the class is a `JumpInteraction`.

In chemistry the same class is the law of mass action. Two species bind into a complex, `A + B -> C`.

```python continuation
binding = ReactionInteraction('binding', reactants={('A', 'x'): 1, ('B', 'x'): 1}, products={('C', 'x'): 1}, k=0.01)
species = [NonnegativeElement('A', x=100.0), NonnegativeElement('B', x=50.0), NonnegativeElement('C')]
reactor = CompositeSystem(elements=species, interactions=[binding])
simulate(reactor, until=10, dt=5.0).get_values('C.x').round(2).tolist()
```

The complex reaches 47.86 after five seconds and 49.83 after ten, close to the 50 units of `B`, the reactant that
runs out.

**How the engine runs it.** The argument `mode` chooses the laws. With `'rate'` the reaction is a
[`RateLaw`](laws.md#ratelaw) per species, the rate equation. With `'jump'` it is a [`JumpLaw`](laws.md#jumplaw) that
fires at the propensity and moves the units of the coefficients, which the algorithm of Gillespie runs. With
`'langevin'` it adds to the rate a [`NoiseLaw`](laws.md#noiselaw) whose amplitude is the square root of the
propensity. The tags follow the mode.

```python continuation
jumps = binding.set_params(mode='jump')
sorted(jumps.get_tags().laws), jumps.get_tags().randomness
```

The laws are `['jump']` and the randomness is `'stochastic'`.

**How it combines.** Its species are usually `NonnegativeElement` components. With `target`, a species may be the name
of a state of the target alone, and a target that is a group or a field reacts in every member. `modifiers` enter the
propensity without changing, as a catalyst does, and `propensity` replaces the product form, such as a saturating rate
of enzyme kinetics.

**Limits and errors.** The propensity is a product of amounts unless `propensity` says otherwise, and a reaction needs
at least one reactant or product. `prepare` raises a `ValidationError` for an empty reaction.

```python continuation
from skcomplex.core.errors import ValidationError

try:
    ReactionInteraction('nothing', k=1.0).prepare()
except ValidationError as error:
    message = str(error)
message
```

The message reads "Reaction 'nothing' has neither reactants nor products. Give at least one of them." The fix is a
reactant, such as `reactants={('A', 'x'): 1}` for a decay.

**A field realization.** Epidemiology binds the species under roles of its own names. The subclass declares the roles
with `RoleSpec` and returns the stoichiometry from the hooks `_get_reactants` and `_get_products`.

```python continuation
from typing import Annotated

from skcomplex.core.specs import RoleSpec


class InfectionInteraction(ReactionInteraction):
    def __init__(
        self, label, *, susceptible: Annotated[str, RoleSpec()], infected: Annotated[str, RoleSpec()], k=5e-4
    ):
        super().__init__(label, k=k)
        self.susceptible = susceptible
        self.infected = infected

    def _get_reactants(self):
        return {(self.susceptible, 'x'): 1, (self.infected, 'x'): 1}

    def _get_products(self):
        return {(self.infected, 'x'): 2}


spread = InfectionInteraction('infection', susceptible='S', infected='I')
compartments = [NonnegativeElement('S', x=990), NonnegativeElement('I', x=10)]
outbreak = CompositeSystem(elements=compartments, interactions=[spread])
simulate(outbreak, until=10, dt=10.0).get_values('I.x').round(1).tolist()
```

Without recovery the infected reach 599.9 after ten days.

See its [API entry][skcomplex.systems.general.reactions].

## ReactionNetworkSystem

A `ReactionNetworkSystem` writes a whole network of reactions as a table, with no element or interaction to write. It
takes the initial amount of every species and maps the label of every reaction to its reactants, its products, and its
rate constant. Here `A` and `B` convert into each other.

```python
from skcomplex.core.engine.runtime import simulate
from skcomplex.systems.general.reactions import ReactionNetworkSystem

isomerization = ReactionNetworkSystem(
    species={'A': 10.0, 'B': 0.0},
    reactions={'forth': ({'A': 1}, {'B': 1}, 0.3), 'back': ({'B': 1}, {'A': 1}, 0.1)},
)
round(float(simulate(isomerization, until=50.0, dt=1.0).get_values('A.x')[-1]), 3)
```

The amount of `A` relaxes to 2.5, which is `0.1 / (0.3 + 0.1)` of the total.

An epidemic of susceptible, infected, and recovered people is the same table with other names.

```python
from skcomplex.core.engine.runtime import simulate
from skcomplex.systems.general.reactions import ReactionNetworkSystem

sir = ReactionNetworkSystem(
    species={'S': 990.0, 'I': 10.0, 'R': 0.0},
    reactions={'infection': ({'S': 1, 'I': 1}, {'I': 2}, 5e-4), 'recovery': ({'I': 1}, {'R': 1}, 0.1)},
)
simulate(sir, until=50, dt=10.0).get_values('I.x').round(1).tolist()
```

The infected follow the curve of the epidemic of the previous section, 10, 294.4, 400.3, 175.0, 68.6, and 26.4 every
ten days.

**How the engine runs it.** `prepare` builds a [`CompositeSystem`](systems.md#compositesystem) with one
`NonnegativeElement` per species, named after it with its amount in `x`, and one `ReactionInteraction` per reaction.
The `mode` writes the reactions as rate equations, `'rate'`, as random events of single units, `'jump'`, or as
chemical Langevin equations, `'langevin'`.

```python continuation
sir.set_params(mode='jump').get_tags().randomness
```

The jump mode makes the system `'stochastic'`.

**How it combines.** It is a [`BaseSystem`](systems.md#basesystem), so `simulate`, `calibrate`, and the analyses take
it as they take any system. A field system subclasses it and gives its components through hooks:

- `_get_species` returns the initial amount of every species by name.
- `_get_reactions` returns the reaction table.
- `_get_mode` returns the mode.
- `_get_interactions` returns the reactions as interactions, for a realization whose reactions bind roles of their
  own names, such as the `susceptible` and `infected` roles of an infection.

**Limits and errors.** Every propensity is a product of amounts. A reaction with another propensity goes through
`_get_interactions`. An unknown mode raises a `ValidationError` from `prepare`.

```python continuation
from skcomplex.core.errors import ValidationError

try:
    sir.set_params(mode='fast').prepare()
except ValidationError as error:
    message = str(error)
message
```

The message reads "Reaction 'infection' has the mode 'fast'. Use one of rate, jump, langevin."

**A field realization.** A field system takes the parameters of its model under its own names. `SIRSystem` of
biology takes `beta` and `gamma` this way. Epidemics and enzyme kinetics are both reaction networks with a product
propensity, which epidemiology calls homogeneous mixing and chemistry the law of mass action, so they share the
general system and differ in their realizations.

```python continuation
class OutbreakSystem(ReactionNetworkSystem):
    def __init__(self, *, beta=5e-4, gamma=0.1, label='system'):
        super().__init__(label=label)
        self.beta = beta
        self.gamma = gamma

    def _get_species(self):
        return {'S': 990.0, 'I': 10.0, 'R': 0.0}

    def _get_reactions(self):
        return {'infection': ({'S': 1, 'I': 1}, {'I': 2}, self.beta), 'recovery': ({'I': 1}, {'R': 1}, self.gamma)}


simulate(OutbreakSystem(gamma=0.2), until=50, dt=25.0).get_values('I.x').round(1).tolist()
```

With a faster recovery of 0.2 a day the infected number 132.1 at day 25 and 4.1 at day 50, so the outbreak stays
smaller.

See its [API entry][skcomplex.systems.general.reactions].

## EncounterInteraction

An `EncounterInteraction` models the meetings of members of two distributions in a kinetic system. A distribution is
a `GroupElement` of `NonnegativeElement` with one member per class of a trait, such as the levels of an opinion. A
candidate at class `h` meets the field at class `k` at the rate `rate[h, k]`, and a conservative encounter moves it to
class `i` with the probability `kernel[h, k, i]`. Here candidates at three opinion levels meet a field that pulls every
one of them to the first level.

```python
import numpy as np

from skcomplex.core.components.elements import GroupElement
from skcomplex.core.components.spaces import LatticeSpace
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.systems.general.elements import NonnegativeElement
from skcomplex.systems.general.kinetics import EncounterInteraction

kernel = np.zeros((3, 1, 3))
kernel[:, :, 0] = 1.0
gather = EncounterInteraction(
    'gather', candidate='candidates', field='field', rate=lambda s: 1.0, kernel=lambda s: kernel
)
system = CompositeSystem(
    elements=[
        GroupElement('candidates', element=NonnegativeElement, size=3, on='sites', values={'x': [0.2, 0.3, 0.5]}),
        GroupElement('field', element=NonnegativeElement, size=1, values={'x': 1.0}),
    ],
    interactions=[gather],
    space=LatticeSpace(3, boundary='fixed'),
)
np.round(simulate(system, until=4.0, dt=0.5).get_values('candidates.x')[-1], 2).tolist()
```

After four units of time 0.99 of the candidates hold the first level.

In immunology a destructive encounter removes the tumour cells that immune cells meet. A tumour density of 1 meets an
immune density of 0.4 at the rate 0.5, so it decays at the rate 0.2.

```python continuation
killing = EncounterInteraction('killing', candidate='tumour', field='immune', rate=lambda s: 0.5, outcome='destructive')
cells = CompositeSystem(
    elements=[
        GroupElement('tumour', element=NonnegativeElement, size=1, on='sites', values={'x': 1.0}),
        GroupElement('immune', element=NonnegativeElement, size=1, values={'x': 0.4}),
    ],
    interactions=[killing],
    space=LatticeSpace(1, boundary='fixed'),
)
simulate(cells, until=10.0, dt=5.0).get_values('tumour.x').round(3).tolist()
```

The tumour falls to 0.368 at time 5 and to 0.135 at time 10, which is `exp(-2)`.

**How the engine runs it.** The interaction lowers to a [`BilinearLaw`](laws.md#bilinearlaw), a rate that is a sum
of products of the two densities, which the compiler evaluates as arrays over the classes. A proliferative outcome adds
a candidate at class `i`, and with a `cross_section` the rate is the collision integral of kinetic theory, evaluated
by a fast spectral method.

**How it combines.** The candidates sit on the sites of a [`LatticeSpace`](spaces.md#latticespace) of classes, or on a
[`PhaseSpace`](spaces.md#phasespace), where encounters pair classes at the same position. Without a field the
encounter is a linear redistribution between classes.

**Limits and errors.** The bilinear law needs a lattice of classes. A conservative encounter moves candidates by a
kernel, so `prepare` raises a `ValidationError` without one.

```python continuation
from skcomplex.core.errors import ValidationError

try:
    EncounterInteraction('exchange', candidate='candidates', rate=lambda s: 1.0).prepare()
except ValidationError as error:
    message = str(error)
message
```

The message reads "The conservative encounter 'exchange' moves candidates by a kernel. Give `kernel`."

See its [API entry][skcomplex.systems.general.kinetics].

## DynamicalSystem

A `DynamicalSystem` is a set of states and the equations that evolve them, rates in continuous time or next values in
discrete time. It takes its data by name and writes one [`DeclaredElement`](elements.md#declaredelement) with one
`RateInteraction` or [`MapInteraction`](interactions.md#mapinteraction) per state. The populations of Wilson and Cowan
in neuroscience are rates.

```python
from sage.all__sagemath_symbolics import exp

from skcomplex.core.engine.runtime import simulate
from skcomplex.systems.general.dynamics import DynamicalSystem

populations = DynamicalSystem(
    states={'e': 0.1, 'i': 0.05},
    equations={
        'e': lambda s: -s.target.e + 1 / (1 + exp(-(2 * s.target.e - 3 * s.target.i))),
        'i': lambda s: -s.target.i + 1 / (1 + exp(-(2 * s.target.e - s.target.i))),
    },
    element='populations',
)
round(float(simulate(populations, until=20.0, dt=1.0).get_values('populations.e')[-1]), 3)
```

The excitatory activity `e` is 0.272 at time 20.

The multiplier and accelerator model of economics is a map. In discrete time every equation reads the states at the
start of the step, so `previous` takes the income of the last year.

```python continuation
economy = DynamicalSystem(
    states={'y': 1.0, 'previous': 1.0},
    equations={
        'y': lambda s: (
            s.target.g + s.target.c * (1 + s.target.v) * s.target.y - s.target.c * s.target.v * s.target.previous
        ),
        'previous': lambda s: s.target.y,
    },
    parameters={'g': 1.0, 'c': 0.5, 'v': 0.5},
    time='discrete',
    element='economy',
)
simulate(economy, until=3).get_values('economy.y').round(3).tolist()
```

The income runs through 1.0, 1.5, 1.875, and 2.031, a damped oscillation around the level `g / (1 - c)`, which is 2.

**How the engine runs it.** In continuous time the rates form an ordinary differential equation, or a stochastic one
with `noises`, whose amplitudes are read in the Itô or Stratonovich convention. In discrete time the maps set the
next values once per step. An equation is a function of a namespace whose `target` holds the states and parameters
and whose `t` holds the time.

**How it combines.** It is a `BaseSystem`, so every workflow takes it. A subclass takes the parameters of its model
under its own names and returns its data from the hooks `_get_states`, `_get_equations`, `_get_parameters`,
`_get_noises`, `_get_time`, and `_get_element_label`. The hook `_get_variables` gives the specs of the states and the
parameters, such as a domain or the order of a Caputo derivative.

**Limits and errors.** All states live in one element, so a model with many members uses
[`NetworkDynamicalSystem`](#networkdynamicalsystem) or [`ReactionDiffusionSystem`](#reactiondiffusionsystem). A noise
needs continuous time, and `prepare` raises a `ValidationError` otherwise.

```python continuation
from skcomplex.core.errors import ValidationError

noisy = DynamicalSystem(
    states={'y': 1.0}, equations={'y': lambda s: s.target.y}, noises={'y': lambda s: 0.1}, time='discrete'
)
try:
    noisy.prepare()
except ValidationError as error:
    message = str(error)
message
```

The message reads "DynamicalSystem has noise amplitudes in discrete time. Noise amplitudes need continuous time." The
fix is `time='continuous'`, or no noise.

**A field realization.** Ecology writes the logistic growth of a population with its own parameters.

```python continuation
class LogisticSystem(DynamicalSystem):
    def __init__(self, *, r=1.0, K=100.0, label='system'):
        super().__init__(label=label)
        self.r = r
        self.K = K

    def _get_states(self):
        return {'n': 10.0}

    def _get_equations(self):
        return {'n': lambda s: s.target.r * s.target.n * (1 - s.target.n / s.target.K)}

    def _get_parameters(self):
        return {'r': self.r, 'K': self.K}


round(float(simulate(LogisticSystem(), until=10.0, dt=10.0).get_values('state.n')[-1]), 2)
```

The population approaches its capacity of 100 and reaches 99.96 at time 10.

See its [API entry][skcomplex.systems.general.dynamics].

## NetworkDynamicalSystem

A `NetworkDynamicalSystem` puts the states on every node of a graph and adds to the rate of every coupled state the
sum over the neighbours `sum_j A_ij g(x_i, x_j)`. Cells of a tissue exchange ions through gap junctions this way.

```python
import networkx as nx

from skcomplex.core.engine.runtime import simulate
from skcomplex.systems.general.dynamics import NetworkDynamicalSystem

cells = NetworkDynamicalSystem(
    nx.cycle_graph(4),
    states={'v': [1.0, 0.0, 0.0, 0.0]},
    equations={'v': lambda s: -s.target.v / s.target.tau},
    coupling={'v': lambda s: s.member.g * (s.other.v - s.member.v)},
    parameters={'tau': 10.0, 'g': 0.5},
    element='cells',
)
simulate(cells, until=5.0, dt=1.0).get_values('cells.v')[-1].round(3).tolist()
```

After five units of time the voltages are 0.154, 0.152, 0.15, and 0.152. The charge of the first cell has spread over
the ring and decays.

Heat that flows between three rooms in a row is diffusion on a path graph, with no local equation.

```python continuation
rooms = NetworkDynamicalSystem(
    nx.path_graph(3),
    states={'u': [3.0, 0.0, 0.0]},
    equations={'u': lambda s: 0 * s.target.u},
    coupling={'u': lambda s: s.other.u - s.member.u},
    element='rooms',
)
simulate(rooms, until=10.0, dt=10.0).get_values('rooms.u')[-1].round(3).tolist()
```

The rooms reach the same temperature, 1.0 each, and the total stays 3.

**How the engine runs it.** The states are those of a `DeclaredElement` that is the member of a
[`FieldElement`](elements.md#fieldelement) on the [`NetworkSpace`](spaces.md#networkspace) of the graph. The equations
are `RateInteraction` components, and the coupling is a [`CouplingInteraction`](interactions.md#couplinginteraction)
that reads node `i` as `s.member` and its neighbour as `s.other`. The system runs in continuous time.

**How it combines.** A subclass returns the graph from `_get_graph` and the couplings from `_get_coupling`, besides
the hooks of `DynamicalSystem`. A local equation may also read the graph Laplacian, as network diffusion does.

**Limits and errors.** Every node holds the same states. A system without a graph raises a `ValidationError` from
`prepare`.

```python continuation
from skcomplex.core.errors import ValidationError

try:
    NetworkDynamicalSystem(states={'v': 1.0}, equations={'v': lambda s: -s.target.v}).prepare()
except ValidationError as error:
    message = str(error)
message
```

The message reads "NetworkDynamicalSystem has no graph. Give a NetworkX graph whose nodes hold the states."

See its [API entry][skcomplex.systems.general.dynamics].

## ReactionDiffusionSystem

A `ReactionDiffusionSystem` puts the states as fields on the sites of a lattice, with local kinetics and the diffusion
`D laplacian(u)` of every field that has a diffusivity `D`. The front of Fisher and Kolmogorov in ecology invades an
empty line.

```python
import numpy as np

from skcomplex.core.engine.runtime import simulate
from skcomplex.systems.general.dynamics import ReactionDiffusionSystem

initial = np.zeros(50)
initial[:5] = 1.0
front = ReactionDiffusionSystem(
    states={'u': initial},
    equations={'u': lambda s: s.target.r * s.target.u * (1 - s.target.u)},
    diffusivities={'u': 1.0},
    shape=50,
    boundary='reflecting',
    parameters={'r': 1.0},
)
density = simulate(front, until=10.0, dt=5.0).get_values('medium.u')[-1]
int((density > 0.5).sum())
```

After ten units of time 20 sites hold more than half the capacity.

A pollutant that spreads along a channel and decays at the rate 0.1 is the same system with linear kinetics.

```python continuation
spill = np.zeros(20)
spill[10] = 1.0
channel = ReactionDiffusionSystem(
    states={'u': spill},
    equations={'u': lambda s: -s.target.k * s.target.u},
    diffusivities={'u': 0.5},
    shape=20,
    boundary='reflecting',
    parameters={'k': 0.1},
)
round(float(simulate(channel, until=5.0, dt=5.0).get_values('medium.u')[-1].sum()), 3)
```

The total falls to 0.607, which is `exp(-0.5)`, since diffusion keeps the total and only the decay removes it.

**How the engine runs it.** The states are those of a `DeclaredElement` that is the member of a `FieldElement` on a
`LatticeSpace`. The diffusion of every field is a separate `RateInteraction`, so a split solver can treat it apart from
the kinetics, and the compiler turns the Laplacian into a stencil of the lattice.

**How it combines.** A subclass returns the diffusivities from `_get_diffusivities`, the lattice from `_get_space`, and
the boundary of every field from `_get_boundaries`, besides the hooks of `DynamicalSystem`. A chemistry system whose
kinetics are reactions returns them from `_get_equation_interactions`, as the Gray-Scott system of pattern formation
does.

**Limits and errors.** Every diffusivity must name a declared state, and `prepare` raises a `ValidationError`
otherwise.

```python continuation
from skcomplex.core.errors import ValidationError

broken = ReactionDiffusionSystem(
    states={'u': 0.0}, equations={'u': lambda s: 0 * s.target.u}, diffusivities={'w': 1.0}, shape=5
)
try:
    broken.prepare()
except ValidationError as error:
    message = str(error)
message
```

The message says that the system has a diffusivity for the state `'w'`, which it does not declare, and lists the
declared states `['u']`.

See its [API entry][skcomplex.systems.general.dynamics].

## StockFlowSystem

A `StockFlowSystem` is a set of stocks that change only through named flows, each from a stock or a source to a stock
or a sink. The trajectory reports every flow. The arms race of Richardson is two stocks of arms that grow with the arms
of the other nation.

```python
from skcomplex.core.engine.runtime import simulate
from skcomplex.systems.general.dynamics import StockFlowSystem

race = StockFlowSystem(
    stocks={'a': 1.0, 'b': 1.0},
    flows={
        'arming_a': (None, 'a', lambda s: s.source.k * s.source.b),
        'arming_b': (None, 'b', lambda s: s.source.k * s.source.a),
    },
    parameters={'k': 0.1},
    element='nations',
)
trajectory = simulate(race, until=10.0, dt=5.0)
trajectory.get_values('nations.a').round(3).tolist(), trajectory.get_values('nations.arming_a').round(3).tolist()
```

The arms of each nation grow as `exp(0.1 t)`, to 2.718 at time 10, and the flow `arming_a` grows with them.

The bathtub of engineering fills from a tap at a rate of 1 and drains at half its content.

```python
from skcomplex.core.engine.runtime import simulate
from skcomplex.systems.general.dynamics import StockFlowSystem

tub = StockFlowSystem(
    stocks={'water': 0.0},
    flows={'tap': (None, 'water', 1.0), 'drain': ('water', None, lambda s: 0.5 * s.source.water)},
    element='tub',
)
simulate(tub, until=10.0, dt=5.0).get_values('tub.water').round(3).tolist()
```

The water rises to 1.836 at time 5 and to 1.987 at time 10, towards the level 2 where the drain matches the tap.

**How the engine runs it.** The stocks and parameters are the variables of one `DeclaredElement`. Every flow is a
[`TransferInteraction`](interactions.md#transferinteraction) from the element to itself, so a flow between two stocks
keeps their total, and the trajectory reports it as `<element>.<flow>`. `None` stands for the environment, and a
labelled [`Source`](laws.md#source) or [`Sink`](laws.md#sink) of `skcomplex.core.components.laws` in its place names
it.

**How it combines.** A subclass returns its data from the hooks `_get_stocks`, `_get_flows`, `_get_parameters`,
`_get_element_label`, and `_get_variables`. The translator of system dynamics files reads models into this system.

**Limits and errors.** Every flow must leave or enter a declared stock, and `prepare` raises a `ValidationError`
otherwise.

```python continuation
from skcomplex.core.errors import ValidationError

try:
    StockFlowSystem(stocks={'water': 0.0}, flows={'tap': (None, 'bath', 1.0)}).prepare()
except ValidationError as error:
    message = str(error)
message
```

The message says that the flow `'tap'` names the stock `'bath'`, which the system does not declare, and lists the
declared stocks `['water']`.

See its [API entry][skcomplex.systems.general.dynamics].

## MovementInteraction

A `MovementInteraction` moves the agents of a group by steps at every step of discrete time. It returns a
[`RuleLaw`](laws.md#rulelaw), an agent rule that the agent solver runs on every agent. Here two sheep start in the
middle of a pasture of 5 by 5 cells and each takes one step to a neighbouring cell.

```python
from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.spaces import LatticeSpace
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.systems.general.agents import MovementInteraction

sheep = GroupElement('sheep', element=Element, size=2, location=[(2, 2), (2, 2)])
walk = MovementInteraction('walk', target='sheep', step='neighbor')
pasture = CompositeSystem(elements=[sheep], interactions=[walk], space=LatticeSpace((5, 5), metric='manhattan'))
trajectory = simulate(pasture, until=1, random_state=0)
[int(abs(cell - 2).sum()) for cell in trajectory.get_agents('sheep', 1).locations]
```

Each sheep stands one cell away from the middle.

Divers swim in a box of three dimensions by Gaussian steps, the random walk of physics.

```python
from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.spaces import ContinuousSpace
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.systems.general.agents import MovementInteraction

divers = GroupElement('divers', element=Element, size=4)
swim = MovementInteraction('swim', target='divers', step='gaussian', scale=0.1)
space = ContinuousSpace(extent=(10.0, 5.0, 2.0), boundary='periodic')
system = CompositeSystem(elements=[divers], interactions=[swim], space=space)
trajectory = simulate(system, until=5, random_state=0)
trajectory.get_agents('divers', 5).locations.shape
```

The trajectory holds one location of three coordinates for each of the four divers.

**How the engine runs it.** `MesaSolver` runs the rule once per step for every agent for which `condition` holds, in
the order of `schedule`. The `step` chooses where an agent goes:

- `'neighbor'`: a random cell or node next to its own, on a lattice or a network.
- `'gaussian'`: its position plus a normal vector of deviation `scale`, in a continuous space.
- `'levy'`: a random direction and a Pareto length, in a continuous space.
- `'best'`: the next cell with the highest `score(agent, cell, system)`.
- a function `step(agent, system)` that returns the destination, or `None` to stay.

An agent takes a step with the probability `probability` and then with the probability `accept(agent, destination,
system)`. An exclusive movement never enters a cell held by another agent, and every step charges its `cost`.

**How it combines.** It moves the members of a `GroupElement` on a `LatticeSpace`, a `NetworkSpace`, or a
`ContinuousSpace` of two or three sides, and it changes where agents stand, while a
[`TopologyInteraction`](interactions.md#topologyinteraction) changes the space itself. In three dimensions, placement,
steps, radius queries, the minimum image of a periodic box, and the locations of a trajectory use every side, and
`system.dimensions` returns every side. The solver rejects a continuous space with any other number of sides, or
without an extent, before the first frame, and the compile report holds the issue `location-dimension` for an initial
location whose length differs from the dimension of its space. `AgentsVisual` draws agents in three dimensions as
`Points3D`.

**Limits and errors.** It runs in discrete time with the agent solver. The step `'best'` maximizes a score, so
`prepare` raises a `ValidationError` without one.

```python continuation
from skcomplex.core.errors import ValidationError

try:
    MovementInteraction('forage', target='divers', step='best').prepare()
except ValidationError as error:
    message = str(error)
message
```

The message reads "The step `best` of 'forage' maximizes a score. Pass `score(agent, cell, system)`." The fix is a
score, such as the food in a cell.

**A field realization.** `ZoneCouplingInteraction` of biology and the boids of `Boids3dSystem` move agents by forces
of their own. A short realization fixes the step of a foraging animal.

```python continuation
class LevyForagingInteraction(MovementInteraction):
    def __init__(self, label, *, target, exponent=1.5):
        super().__init__(label, target=target, step='levy', exponent=exponent)


LevyForagingInteraction('forage', target='divers').step
```

See its [API entry][skcomplex.systems.general.agents].

### Agents on region layers

Agents can move between named regions, such as census tracts. A space of two sides takes layers of polygons, and a
step that returns the polygon of a region moves an agent into it.

```python
from shapely import Point, box

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.spaces import ContinuousSpace
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.systems.general.agents import MovementInteraction

census = [box(0.0, 0.0, 4.0, 4.0), box(5.0, 0.0, 9.0, 4.0)]
places = [box(1.0, 6.0, 3.0, 8.0)]
people = GroupElement('people', element=Element, size=3, location=[census[0], census[0], census[1]])
outing = MovementInteraction('outing', target='people', step=lambda agent, system: system.get_region('places', 0))
space = ContinuousSpace(extent=(10.0, 10.0), regions={'census': census, 'places': places})
system = CompositeSystem(elements=[people], interactions=[outing], space=space)
trajectory = simulate(system, until=1, random_state=0)
all(places[0].contains(Point(point)) for point in trajectory.get_agents('people', 1).locations)
```

Every person stands inside the place after one step.

`ContinuousSpace(regions=...)` maps the name of a layer to a sequence of Shapely polygons in the coordinates of the
space. `prepare` keeps one `STRtree` per layer in `regions_`, and it raises `SpaceError`, naming the layer and the
region, for a region that is not a valid polygon.

- A polygon is a location in `GroupElement(location=...)`, `SystemView.create`, and `AgentView.move_to`. The solver
  draws a uniform point inside it from the generator of the run. It chooses a triangle of the constrained Delaunay
  triangulation of the polygon with a probability proportional to its area.
- `AgentView.get_regions_here(layer)` returns the indices of the regions of a layer that contain the agent.
- `SystemView.get_region(layer, index)` returns the polygon of a region, and `SystemView.get_regions(layer)` the
  number of regions of a layer.
- `AgentsVisual` draws every layer once as static `LineStrips2D` under the agents.

**Real maps.** The polygons of a space are in the coordinates of the space, and the library reads no coordinate
reference system. Census shapefiles usually come in longitude and latitude, where every distance, such as a radius, is
in degrees. Reproject them to a metric system first. With GeoPandas,
`geopandas.read_file(path).to_crs(epsg=32604).geometry` gives the polygons of a shapefile in metres for a Hawaiian
island. Then shift them so that no coordinate is negative, since a box runs from 0 to its sides. The next example does
the same with a plain local projection of one census polygon.

```python
import math

import numpy as np
import shapely
from shapely import Polygon
from shapely.affinity import translate

EARTH_RADIUS = 6371000.0
tract = Polygon([(-159.6, 21.9), (-159.4, 21.9), (-159.4, 22.1), (-159.6, 22.1)])


def project(points):
    scale = math.cos(math.radians(22.0))
    return EARTH_RADIUS * np.radians(points) * np.array([scale, 1.0])


projected = shapely.transform(tract, project)
low_x, low_y, _, _ = projected.bounds
tract_in_metres = translate(projected, -low_x, -low_y)
[round(value, -3) for value in tract_in_metres.bounds]
```

The tract spans about 21000 metres from west to east and 22000 metres from south to north.

`KauaiPopulationSystem` of `skcomplex.systems.social.organizations` places one agent per resident of the census
regions of the synthetic island of its package, and `Boids3dSystem` of `skcomplex.systems.biology.collective_behaviour`
flies in a periodic cube.

```python
from skcomplex.systems.biology.collective_behaviour import Boids3dSystem
from skcomplex.systems.social.organizations import KauaiPopulationSystem

system = KauaiPopulationSystem(populations=(10, 5, 0)).prepare().system_
system.elements[0].size, list(system.space.regions), Boids3dSystem(n_boids=10).prepare().system_.space.extent
```

The island holds 15 agents, its space has the layers `boundary`, `census`, `places`, and `facilities`, and the
boids fly in a cube of side 20.

## ChoiceInteraction

A `ChoiceInteraction` lets every member of a group pick an option by its utility. By the logit rule of discrete
choice, option `k` has the probability `exp(beta u_k) / sum_j exp(beta u_j)`, where `beta` is the intensity of
choice. Here 500 members prefer to agree with the mean of the group, and a small majority for the option 1 grows into
a consensus.

```python
from typing import Annotated

import numpy as np

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.laws.utils import mean
from skcomplex.core.components.spaces import DiscreteTime
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.specs import StateSpec
from skcomplex.systems.general.agents import ChoiceInteraction


class ChooserElement(Element):
    def __init__(self, label, *, option: Annotated[float, StateSpec()] = 1.0):
        super().__init__(label)
        self.option = option


utility = lambda s, option: 2.0 * option * mean(s.target.option)
choice = ChoiceInteraction('choice', target='members', state='option', options=(-1, 1), utility=utility)
initial = np.where(np.random.default_rng(0).random(500) < 0.6, 1.0, -1.0)
members = GroupElement('members', element=ChooserElement, size=500, values={'option': initial})
system = CompositeSystem(elements=[members], interactions=[choice], time=DiscreteTime())
values = simulate(system, until=10, random_state=0).get_values('members.option')
round(float(initial.mean()), 2), round(float(values[-1].mean()), 2)
```

The mean option rises from 0.13 to 0.96 in ten steps, close to the root of `m = tanh(2 m)`, the consensus that the
mean field predicts.

In transport economics commuters choose between the bus, option 0, and the car, option 1, whose utility is one unit
higher. The logit rule gives the car the share `e / (1 + e)`, about 0.73.

```python continuation
class CommuterElement(Element):
    def __init__(self, label, *, mode: Annotated[float, StateSpec()] = 0.0):
        super().__init__(label)
        self.mode = mode


modes = (0, 1)
commute = ChoiceInteraction('commute', target='commuters', state='mode', options=modes, utility=lambda s, k: k)
commuters = GroupElement('commuters', element=CommuterElement, size=1000)
city = CompositeSystem(elements=[commuters], interactions=[commute], time=DiscreteTime())
float(simulate(city, until=1, random_state=0).get_values('commuters.mode')[-1].mean())
```

In one draw 0.75 of the commuters take the car, close to the share of the logit rule.

**How the engine runs it.** In the symbolic form `utility(namespace, option)` returns an expression, and the
interaction returns an [`UpdateLaw`](laws.md#updatelaw) that draws a new option for every member at once. In the
agent form `utility(agent, system, option)` returns a number, and a `RuleLaw` draws for the members one after another.
The rule `'metropolis'` or `'fermi'` accepts a proposed option, and the rule `'best'` takes an option of largest
utility.

**How it combines.** It writes a discrete state of the members of a `GroupElement` in
[`DiscreteTime`](spaces.md#discretetime), and the utility may read any state of the group, such as its mean, or of
neighbours. Each member revises with the probability `activity`.

**Limits and errors.** A choice needs at least two options, and `prepare` raises a `ValidationError` otherwise.

```python continuation
from skcomplex.core.errors import ValidationError

try:
    single = ChoiceInteraction('commute', target='commuters', state='mode', options=(1,), utility=lambda s, k: k)
    single.prepare()
except ValidationError as error:
    message = str(error)
message
```

The message reads "'commute' chooses among [1]. Give at least 2 options."

**A field realization.** Social science names the choice above conformity. The catalog writes the standardization of
a technology the same way, as `StandardizationInteraction` of `skcomplex.systems.social.opinion_dynamics`.

```python continuation
class ConformityInteraction(ChoiceInteraction):
    def __init__(self, label, *, target, strength=2.0):
        super().__init__(
            label,
            target=target,
            state='option',
            options=(-1, 1),
            utility=lambda s, option: strength * option * mean(s.target.option),
        )
        self.strength = strength


ConformityInteraction('conformity', target='members').options
```

See its [API entry][skcomplex.systems.general.agents].

## ImitationInteraction

An `ImitationInteraction` lets members copy a discrete state from their neighbours. It is a
[`NeighborInteraction`](interactions.md#neighborinteraction) whose `rule` says whom a member copies: a random
neighbour, the local majority, the neighbour with the best payoff, or a neighbour chosen by the Fermi rule. Here
voters on a ring of twelve follow the majority of their two neighbours.

```python
from typing import Annotated

import numpy as np

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.spaces import DiscreteTime, LatticeSpace
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.specs import StateSpec
from skcomplex.systems.general.agents import ImitationInteraction


class VoterElement(Element):
    def __init__(self, label, *, state: Annotated[float, StateSpec()] = 0.0):
        super().__init__(label)
        self.state = state


initial = np.array([1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0], dtype=float)
majority = ImitationInteraction('majority', target='voters', rule='majority', form='probability')
voters = GroupElement('voters', element=VoterElement, space=LatticeSpace(12), values={'state': initial}, on='sites')
system = CompositeSystem(elements=[voters], interactions=[majority], time=DiscreteTime())
simulate(system, until=1, random_state=0).get_values('voters.state')[-1].tolist()
```

After one step the lone 0 at the fourth site has joined its neighbours, and the votes read
`[1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]`.

Farmers adopt the crops of their neighbours on a network. On a star, the hub grows a new crop and four farms around it
do not. With the rule `'random'` every farm copies a random neighbour at once.

```python continuation
import networkx as nx

from skcomplex.core.components.spaces import NetworkSpace

crops = {'state': [1, 0, 0, 0, 0]}
farms = GroupElement('farms', element=VoterElement, space=NetworkSpace(nx.star_graph(4)), values=crops, on='sites')
adoption = ImitationInteraction('adoption', target='farms', rule='random', form='probability')
village = CompositeSystem(elements=[farms], interactions=[adoption], time=DiscreteTime())
simulate(village, until=1, random_state=0).get_values('farms.state')[-1].tolist()
```

The four farms take the crop of the hub, and the hub takes the old crop of a farm, so the states read
`[0, 1, 1, 1, 1]`.

**How the engine runs it.** The `form` chooses the laws. With `'rate'` imitation is a [`JumpLaw`](laws.md#jumplaw) at
the rate `rate` in continuous time. With `'probability'` it is an `UpdateLaw` that every member applies with the
probability `probability` per step, all at once. With `'agent'` it is a `RuleLaw` that runs the members in the order
of `schedule`.

**How it combines.** Its neighbours come from the space of the group, a `LatticeSpace` or a `NetworkSpace`, or from
the `layer` of a network with several layers. The rules `'best'` and `'fermi'` read the payoff state, which another
interaction, such as a game, writes.

**Limits and errors.** It copies one discrete value, while a
[`CouplingInteraction`](interactions.md#couplinginteraction) averages numeric states. The rule `'better'` runs in the
agent form only, and `prepare` raises a `ValidationError` in another form.

```python continuation
from skcomplex.core.errors import ValidationError

try:
    ImitationInteraction('adoption', target='farms', rule='better').prepare()
except ValidationError as error:
    message = str(error)
message
```

The message reads "The rule of imitation 'better' runs in the agent form only. Use form `agent`."

**A field realization.** The voter model of social science is an imitation of a random neighbour at rate 1.

```python continuation
class VoterModelInteraction(ImitationInteraction):
    def __init__(self, label, *, target):
        super().__init__(label, target=target, rule='random', form='rate')


VoterModelInteraction('voting', target='voters').rule
```

See its [API entry][skcomplex.systems.general.agents].

## EvolutionInteraction

An `EvolutionInteraction` breeds the strategies of a group by selection, crossover, and mutation, as a genetic
algorithm does. Every `period` steps it selects parents by their fitness and replaces strategies by offspring. Here the
fitness of a bit string is its number of ones, which a rule of the element writes in a phase before the breeding.

```python
from typing import Annotated

import numpy as np

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.laws import RuleLaw
from skcomplex.core.components.spaces import DiscreteTime
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.specs import PhaseSpec, PhasesScheduleSpec, StateSpec
from skcomplex.systems.general.agents import EvolutionInteraction


class BitStringElement(Element):
    def __init__(
        self,
        label,
        *,
        strategy: Annotated[object, StateSpec()] = 0.0,
        fitness: Annotated[float, StateSpec()] = 0.0,
    ):
        super().__init__(label)
        self.strategy = strategy
        self.fitness = fitness

    def _get_laws(self, s):
        return [RuleLaw(self.score, phase='scoring')]

    def score(self, agent, system):
        agent.fitness = float(np.sum(agent.strategy))


initial = np.random.default_rng(0).integers(0, 2, size=(40, 20)).astype(float)
agents = GroupElement('agents', element=BitStringElement, size=40, values={'strategy': initial})
evolution = EvolutionInteraction('evolution', target='agents', phase='evolution')
schedule = PhasesScheduleSpec([PhaseSpec('scoring'), PhaseSpec('evolution')])
system = CompositeSystem(elements=[agents], interactions=[evolution], time=DiscreteTime(schedule=schedule))
trajectory = simulate(system, until=30, random_state=0)
final = np.array(trajectory.get_agents('agents', 30).values['strategy'].tolist())
float(initial.sum(axis=1).mean()), float(final.sum(axis=1).mean())
```

The mean number of ones rises from 10.675 to 19.675 of 20 in thirty generations.

In economics firms learn a price by imitation of the profitable ones and by small trials. A real strategy, with
`alphabet=None`, mutates by a normal step of deviation `scale`, and the function `evaluate` gives the profit of every
price, highest at 3.

```python continuation
class FirmElement(Element):
    def __init__(self, label, *, strategy: Annotated[object, StateSpec()] = 0.0):
        super().__init__(label)
        self.strategy = strategy


prices = np.random.default_rng(0).uniform(0, 6, size=(30, 1))
firms = GroupElement('firms', element=FirmElement, size=30, values={'strategy': prices})
learning = EvolutionInteraction(
    'learning',
    target='firms',
    fitness=None,
    alphabet=None,
    scale=0.2,
    mutation=0.5,
    evaluate=lambda strategies, system: -((strategies[:, 0] - 3.0) ** 2),
)
market = CompositeSystem(elements=[firms], interactions=[learning], time=DiscreteTime())
learned = np.array(simulate(market, until=20, random_state=0).get_agents('firms', 20).values['strategy'].tolist())
round(float(prices.std()), 2), round(float(learned.mean()), 2), round(float(learned.std()), 2)
```

The prices start spread with a deviation of 1.9 and gather near 3.01 with a deviation of 0.14 after twenty steps.

**How the engine runs it.** It returns one `RuleLaw` that the agent solver runs once per `period`. The `selection`
picks parents by tournament, roulette, proportion, or rank. The `replacement` replaces the whole generation, or one
member at a time by the process of Moran, or by death and birth, truncation, or the best mutant.

**How it combines.** The strategies and the fitness are states of the members of a `GroupElement`. A rule of the
element, a game, or the function `evaluate` writes the fitness, and a [`PhasesScheduleSpec`](spaces.md#discretetime)
of `DiscreteTime` orders the scoring before the breeding.

**Limits and errors.** It replaces strategies across the whole group, while `ImitationInteraction` copies neighbours.
The counts must be valid, and `prepare` raises a `ValidationError` for a negative elite.

```python continuation
from skcomplex.core.errors import ValidationError

try:
    EvolutionInteraction('learning', target='firms', elite=-1).prepare()
except ValidationError as error:
    message = str(error)
message
```

The message reads "'learning' has elite -1 and tournament 2. Give an elite of at least 0 and a tournament of at least
1."

**A field realization.** Biology writes the process of Moran, one birth and one death at a time.

```python continuation
class MoranInteraction(EvolutionInteraction):
    def __init__(self, label, *, target, mutation=0.01):
        super().__init__(label, target=target, replacement='moran', mutation=mutation)


MoranInteraction('moran', target='agents').replacement
```

See its [API entry][skcomplex.systems.general.agents].

## BirthDeathInteraction

A `BirthDeathInteraction` creates and removes the agents of a group. At every step an agent dies with the probability
`death`, and an agent that survives gives birth with the probability `birth` or at a Poisson `rate`. Here a cell of
size 8 divides twice, and each division splits the size between the parent and its offspring.

```python
from typing import Annotated

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.spaces import LatticeSpace
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.specs import StateSpec
from skcomplex.systems.general.agents import BirthDeathInteraction


class CellElement(Element):
    def __init__(self, label, *, size: Annotated[float, StateSpec()] = 0.0):
        super().__init__(label)
        self.size = size


cells = GroupElement('cells', element=CellElement, size=1, values={'size': 8.0}, variable=True)
division = BirthDeathInteraction('division', target='cells', birth=1.0, split=('size',))
system = CompositeSystem(elements=[cells], interactions=[division], space=LatticeSpace((3, 3)))
simulate(system, until=2, random_state=0).get_agents('cells', 2).values['size'].tolist()
```

After two steps four cells of size 2.0 share the size 8 of the first.

Rabbits in a warren breed with the probability 0.3 and die with the probability 0.1 each step. The interaction reports
the births and deaths of every step.

```python continuation
rabbits = GroupElement('rabbits', element=CellElement, size=10, variable=True)
breeding = BirthDeathInteraction('breeding', target='rabbits', birth=0.3, death=0.1)
field = CompositeSystem(elements=[rabbits], interactions=[breeding], space=LatticeSpace((10, 10)))
warren = simulate(field, until=5, random_state=0)
warren.get_values('breeding.births')[1:].tolist(), warren.get_values('breeding.deaths')[1:].tolist()
```

Over five steps 15 rabbits are born and 6 die.

**How the engine runs it.** It returns one `RuleLaw`, and the agent solver runs it for every agent in the order of
`schedule`. Offspring stand where `placement` says, copy the states of `inherit`, take the values of `values`, share
the states of `split` with the parent, and may then mutate.

**How it combines.** Its group must be a `GroupElement` with `variable=True`, since members come and go. Further
options reset or replace the parent, renew a dying agent in place, cull a member of a kin group, and add arrivals
without a parent. A `JumpInteraction` changes numeric states of a fixed set of members, while this interaction adds
and removes the members themselves.

**Limits and errors.** Births come by a probability or by a rate, not both, and `prepare` raises a
`ValidationError` when both are given.

```python continuation
from skcomplex.core.errors import ValidationError

try:
    BirthDeathInteraction('breeding', target='rabbits', birth=0.3, rate=1.0).prepare()
except ValidationError as error:
    message = str(error)
message
```

The message reads "'breeding' gives births by a probability and by a rate. Give one of them."

**A field realization.** Biology writes cell division as a birth that halves the size.

```python continuation
class DivisionInteraction(BirthDeathInteraction):
    def __init__(self, label, *, target, probability=1.0):
        super().__init__(label, target=target, birth=probability, split=('size',))


DivisionInteraction('division', target='cells').split
```

See its [API entry][skcomplex.systems.general.agents].

## MatchingInteraction

A `MatchingInteraction` pairs requesters with providers by a rule and settles every match. Requesters state the
highest value they accept in their `reserve` state and the quantity they want in `quantity`, and providers the lowest
value they accept and the quantity they offer. Every match lowers the quantities of both sides and moves the states of
`transfers`. Here four buyers meet four sellers in a market for one good, and the function `trade` runs one round
under a given rule.

```python
from typing import Annotated

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.spaces import DiscreteTime
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.specs import StateSpec
from skcomplex.systems.general.matching import BatchMatchingSpec, MatchingInteraction


class TraderElement(Element):
    def __init__(
        self,
        label,
        *,
        reserve: Annotated[float, StateSpec()] = 0.0,
        quantity: Annotated[float, StateSpec()] = 1.0,
        money: Annotated[float, StateSpec()] = 0.0,
        goods: Annotated[float, StateSpec()] = 0.0,
        partner: Annotated[float, StateSpec()] = -1.0,
    ):
        super().__init__(label)
        self.reserve = reserve
        self.quantity = quantity
        self.money = money
        self.goods = goods
        self.partner = partner


def trade(rule, on_match=None):
    bids = {'reserve': [10.0, 8.0, 6.0, 4.0], 'money': 20.0}
    asks = {'reserve': [3.0, 5.0, 7.0, 9.0], 'goods': 1.0}
    buyers = GroupElement('buyers', element=TraderElement, size=4, values=bids)
    sellers = GroupElement('sellers', element=TraderElement, size=4, values=asks)
    transfers = (('money', -1), ('goods', 1))
    market = MatchingInteraction(
        'market', requesters='buyers', providers='sellers', rule=rule, transfers=transfers, on_match=on_match
    )
    system = CompositeSystem(elements=[buyers, sellers], interactions=[market], time=DiscreteTime())
    trajectory = simulate(system, until=1, random_state=0)
    value, volume = trajectory.get_values('market.value')[-1], trajectory.get_values('market.volume')[-1]
    return round(float(value), 2), float(volume)


trade(BatchMatchingSpec())
```

A batch auction pairs the buyer at 10 with the seller at 3 and the buyer at 8 with the seller at 5, and both units
clear at 6.5, the midpoint of the last paired limits.

In a labour market the requesters are firms that offer a wage up to their reserve, and the providers are workers who
accept a wage above theirs. The same interaction with `transfers=(('money', -1),)` moves the wage from the firm to the
worker for every match, and the section [A market in two phases](#a-market-in-two-phases) below runs a market over
several rounds.

**How the engine runs it.** It returns a [`MatchLaw`](laws.md#matchlaw), the primitive of the core that pairs two
groups, and a `RuleLaw` that settles the matches. The matching runs once per round of its phase, runs the algorithm of
its rule on the offers, and reports `<label>.value`, the clearing value weighted by quantity, `<label>.volume`, and
`<label>.unmatched`. With `('money', -1)` the value times the quantity moves from the requester to the provider, and
with `('goods', 1)` the quantity moves back, so the totals stay the same. `on_match(match, system)` adds any other
effect.

**How it combines.** The rule is one of the specs below, or a function `rule(requesters, providers, rng, capacity)`
that returns pairs, which match at the midpoint of both limits. A name given as a pair, such as
`reserve=('bid', 'ask')`, names the state of the requesters and of the providers. A
[`TransferInteraction`](interactions.md#transferinteraction) moves amounts between given pairs, while this interaction
forms the pairs first.

**Limits and errors.** A transfer moves value with the sign -1 or quantity with the sign 1, and `prepare` raises a
`ValidationError` for another sign.

```python continuation
from skcomplex.core.errors import ValidationError

try:
    MatchingInteraction('market', requesters='buyers', providers='sellers', transfers=(('money', 2),)).prepare()
except ValidationError as error:
    message = str(error)
message
```

The message reads "The transfer of 'money' has the sign 2. Use one of (-1, 1)."

See its [API entry][skcomplex.systems.general.matching].

## RandomSearchSpec

A `RandomSearchSpec` is the default rule of a matching, the decentralised exchange of Hamill and Gilbert (2016,
chapter 5). Every requester, in random order, meets up to `tries` providers that still offer something and matches
when the limits cross, at the midpoint of the two limits or at the limit of one side. Job seekers who meet a few firms
and buyers in a bazaar search this way. It continues the market of [`MatchingInteraction`](#matchinginteraction).

```python continuation
from skcomplex.systems.general.matching import RandomSearchSpec

trade(RandomSearchSpec()), trade(RandomSearchSpec(tries=4))
```

With one meeting per buyer two units trade at a mean value of 8.0. With four meetings three units trade at 6.83, since
more buyers find a seller whose limit they accept. The value rule is `'midpoint'`, `'requester'`, or `'provider'`, and
the matching raises a `ValidationError` for any other when it checks its rule. See its
[API entry][skcomplex.systems.general.matching].

## PartnerMatchingSpec

A `PartnerMatchingSpec` lets requesters pick partners among candidates and then trade at a value. It is a
`PartnerSpec` of the core with the value of a pair, which only a matching reads. Its settings `candidates`, `choice`,
`fallback`, `capacity`, and `score` are those of `PartnerSpec`, such as the nearest shops of a buyer or the firms
within reach of a worker.

```python continuation
from skcomplex.systems.general.matching import PartnerMatchingSpec

trade(PartnerMatchingSpec())
```

Each buyer picks a seller whose limit it accepts, and four units trade at a mean value of 6.5. See its
[API entry][skcomplex.systems.general.matching].

## PartnerSwitchSpec

A `PartnerSwitchSpec` keeps a partner across steps and switches with friction, as in Caiani et al. (2016, chapter 2).
Every requester samples `sample` providers, and when the lowest limit among them is below that of its partner it
switches with the probability `1 - exp(intensity (p_new - p_old) / p_new)`. Households that keep their shops and firms
that keep their banks follow this rule. The state `partner` holds the id of the partner, which is why the traders
above declare it.

```python continuation
from skcomplex.systems.general.matching import PartnerSwitchSpec

trade(PartnerSwitchSpec())
```

Three units trade at a mean value of 5.0, the limits of the chosen sellers. See its
[API entry][skcomplex.systems.general.matching].

## QueueSpec

A `QueueSpec` serves requesters one by one, first come first served. Requesters queue in random order, by decreasing
reserve, or in order of creation, and each matches at the reserve of the cheapest provider whose reserve it accepts.
Customers at a shop and households waiting for housing queue this way.

```python continuation
from skcomplex.systems.general.matching import QueueSpec

trade(QueueSpec(order='reserve'))
```

The buyers at 10 and 8 buy from the sellers at 3 and 5, so two units trade at a mean value of 4.0, and the buyers at
6 and 4 get nothing. An unknown order raises a `ValidationError` when the matching checks its rule.

```python continuation
from skcomplex.core.errors import ValidationError

try:
    trade(QueueSpec(order='alphabetical'))
except ValidationError as error:
    message = str(error)
message
```

The message reads "Unknown order 'alphabetical'. Use one of ('random', 'reserve', 'sequential')." See its
[API entry][skcomplex.systems.general.matching].

## IterativeMatchingSpec

An `IterativeMatchingSpec` moves a value until supply meets demand, the tâtonnement of Walras, as in Hamill and
Gilbert (2016, chapter 5). From the last clearing value, or the median of the limits, the value moves by
`p <- p (1 + adjustment (D - S) / (D + S))` until the relative excess of requests falls below `tolerance`. The members
then match at that value, and the long side is rationed in random order. Prices in general equilibrium and wages in a
labour market adjust this way.

```python continuation
from skcomplex.systems.general.matching import IterativeMatchingSpec

trade(IterativeMatchingSpec())
```

The value settles where two units are requested and offered, and both trade at 6.5. See its
[API entry][skcomplex.systems.general.matching].

## ContinuousMatchingSpec

A `ContinuousMatchingSpec` matches offers as they arrive, as a limit order book does, following Caiani et al. (2016,
chapter 3). An arriving offer matches the best resting offers of the other side at their limits as long as the limits
cross, and its remainder rests until the end of the matching.

```python continuation
from skcomplex.systems.general.matching import ContinuousMatchingSpec

trade(ContinuousMatchingSpec())
```

Three units trade at a mean value of 7.0, each at the limit of the offer that was resting. The order of arrival is
`'random'` or `'sequential'`. See its [API entry][skcomplex.systems.general.matching].

## StableMatchingSpec

A `StableMatchingSpec` pairs by mutual preferences, the stable matching of Gale and Shapley that matches medical
residents with hospitals and students with schools. Requesters rank providers by `requester_preference`, by default
the lower reserve, and providers rank requesters by `provider_preference`, by default the higher reserve. The solver
of the `matching` library pairs every requester with at most one provider, so no requester and provider prefer each
other to their partners.

```python continuation
from skcomplex.systems.general.matching import StableMatchingSpec

trade(StableMatchingSpec())
```

Two units trade at a mean value of 4.0, the limits of the providers, since `value='provider'` is the default. See its
[API entry][skcomplex.systems.general.matching].

## MatchSpec

A `MatchSpec` records one match that a rule made: the requester, the provider, the quantity, and the value per unit.
The function `on_match` of a `MatchingInteraction` receives one for every match, such as the loan a borrower takes at a
rate.

```python continuation
matches = []
trade(BatchMatchingSpec(), on_match=lambda match, system: matches.append((match.quantity, match.value)))
matches
```

The batch auction makes two matches of one unit, each at 6.5. See its
[API entry][skcomplex.systems.general.matching].

## BatchMatchingSpec

A `BatchMatchingSpec` collects every offer and clears them at one value, as the call auction that opens a stock
exchange does. Requesters are sorted by decreasing and providers by increasing limit, and they pair while the limit of
the requester covers that of the provider. The value is the midpoint of the last paired limits.
[`MatchingInteraction`](#matchinginteraction) runs it above, where two units clear at 6.5, and the
[market in two phases](#a-market-in-two-phases) below runs it over several rounds. It takes no settings.

```python continuation
BatchMatchingSpec()
```

See its [API entry][skcomplex.systems.general.matching].

### A market in two phases

Traders quote prices in one phase and trade in a batch auction in the next, while a clerk sets a fee from the volume
traded. The phases and the clerk are components of the core, which the section
[Phases, institutions, and learning](../build_a_custom_system/agent_based_systems.md#phases-institutions-and-learning)
explains.

```python
from typing import Annotated, ClassVar

import numpy as np

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import RuleLaw
from skcomplex.core.components.spaces import DiscreteTime
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.specs import ParameterSpec, PhaseSpec, PhasesScheduleSpec, StateSpec
from skcomplex.systems.general.matching import BatchMatchingSpec, MatchingInteraction


class QuotingElement(Element):
    def __init__(
        self,
        label,
        *,
        value: Annotated[float, StateSpec()] = 0.0,
        bid: Annotated[float, StateSpec()] = 0.0,
        ask: Annotated[float, StateSpec()] = 0.0,
        demand: Annotated[float, StateSpec()] = 0.0,
        supply: Annotated[float, StateSpec()] = 0.0,
        money: Annotated[float, StateSpec()] = 0.0,
        goods: Annotated[float, StateSpec()] = 0.0,
    ):
        super().__init__(label)
        self.value = value
        self.bid = bid
        self.ask = ask
        self.demand = demand
        self.supply = supply
        self.money = money
        self.goods = goods

    def _get_laws(self, s):
        return [RuleLaw(self.quote, phase='quote')]

    def quote(self, agent, system):
        agent.bid = system.rng.uniform(0, agent.value) if agent.demand else 0.0
        agent.ask = system.rng.uniform(agent.value, 2.0) if agent.supply else 2.0


class ClerkInteraction(Interaction):
    law_classes: ClassVar[tuple[type, ...]] = (RuleLaw,)
    reports: ClassVar[tuple[str, ...]] = ('traded',)

    def __init__(self, label, *, fee: Annotated[float, ParameterSpec()] = 0.0):
        super().__init__(label)
        self.fee = fee

    def _get_laws(self, s):
        return [RuleLaw(self.act, phase='trade')]

    def act(self, system):
        traded = sum(agent.goods for agent in system.get_agents('buyers'))
        system.report('traded', traded)
        system.set_parameter('fee', 0.01 * traded)


rng = np.random.default_rng(0)
buyers = {'value': rng.uniform(0.5, 2, 20), 'demand': 1.0, 'money': 2.0}
sellers = {'value': rng.uniform(0, 1.5, 20), 'supply': 1.0}
auction = CompositeSystem(
    elements=[
        GroupElement('buyers', element=QuotingElement, size=20, values=buyers),
        GroupElement('sellers', element=QuotingElement, size=20, values=sellers),
    ],
    interactions=[
        MatchingInteraction(
            'auction',
            requesters='buyers',
            providers='sellers',
            rule=BatchMatchingSpec(),
            reserve=('bid', 'ask'),
            quantity=('demand', 'supply'),
            cap='money',
            transfers=(('money', -1), ('goods', 1)),
            phase='trade',
        ),
        ClerkInteraction('clerk'),
    ],
    time=DiscreteTime(schedule=PhasesScheduleSpec([PhaseSpec('quote'), PhaseSpec('trade')])),
)
trajectory = simulate(auction, until=5, random_state=0)
prices = [round(float(price), 3) for price in trajectory.get_values('auction.value')[1:]]
prices, trajectory.get_values('auction.volume')[1:].tolist(), round(float(trajectory.get_values('clerk.fee')[-1]), 3)
```

The clearing prices of the rounds are 1.018, 1.231, 1.074, none in the fourth round, and 1.155, with the volumes 4,
4, 1, 0, and 2. No bid of the fourth round covers an ask, so nothing trades. The clerk counts 11 goods bought and
ends with a fee of 0.11. The state `cap` names the money of a buyer, the most value it can give.

## Where a component goes

One ordered test, the placement rule of the package layout, decides where any new component goes. The source tests check
it mechanically. It reads, word for word:

> Every new component SHALL be placed by one ordered test. A structural base SHALL go to the core. A component the
> engine handles by type, or an operator generated from one function, SHALL go to the core under a mathematical name.
> Any other component is a recipe, placed by its users: `skcomplex.systems.general.<concept>` for several fields,
> `systems/<field>/common/<family>` for one field, its system's module for one system. The core SHALL never import
> `skcomplex.systems`.

A field with its own term names its recipe, and `general` uses neutral names. One example for each step follows.

**A structural base.** A base where laws run once per triangle of a cell complex goes to the core, as
[`HyperedgeInteraction`](interactions.md#hyperedgeinteraction) does, which runs its laws once per hyperedge, and
[`CompositeSystem`](systems.md#compositesystem), which holds the components of a system. Components of every field build
on them.

**A component the engine handles by type.** A law that a kernel lowers goes to the core under a mathematical name.
[`GroupingInteraction`](interactions.md#groupinginteraction) stays in the core, since a compile pass lowers it by type
to kernels over random groups, and economics and social science both form groups. `BilinearLaw` is in the core
because the compiler names it, while `EncounterInteraction`, which only builds bilinear laws, is a recipe.

**A recipe of several fields.** An agent rule that economics and social science both use goes to a concept subpackage
of `skcomplex.systems.general`. `ChoiceInteraction` returns only laws the core already runs, and the oligopoly markets
of economics and the standardization of social science use it, so it lives in `skcomplex.systems.general.agents`.

**A recipe of one field.** The toppling of a sandpile serves several systems of physics only, so it lives in the
family `avalanches` of the physics `common` package, under the physics name `TopplingInteraction`.

**A recipe of one system.** The satisfiability of logical constraints serves one system, so
`SatisfiabilityInteraction` lives in the module of `PerceptronSatisfiabilitySystem`.

```python
from skcomplex.core.components.interactions import GroupingInteraction, HyperedgeInteraction
from skcomplex.core.components.laws import BilinearLaw
from skcomplex.systems.general.agents import ChoiceInteraction

components = [HyperedgeInteraction, GroupingInteraction, BilinearLaw, ChoiceInteraction]
[component.__module__.rsplit('.', 1)[0] for component in components]
```

The first three components live in `skcomplex.core.components` and the choice in `skcomplex.systems.general.agents`. The
toppling lives in the physics `common` package and the satisfiability in the module of its system, so the guide names
them without importing them, since only system classes are imported from a field.

**A recipe that reveals a missing primitive.** A recipe that needs an operation the engine cannot lower reveals a
missing primitive, which enters the core by the second step, and the recipe keeps no engine code. The satisfiability
once had a check of its own in the compiler, engine support inside a recipe. The check that a role accepts only
Boolean states became a primitive of the core, `RoleSpec(domains=...)`, so the recipe declares its role with it and
the compiler keeps no code for the class. Any interaction declares such a role the same way. Here a rule on bits
accepts only Boolean states, and a group of real states fails the check of the role before any run.

```python continuation
from typing import Annotated

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.errors import RoleElementError
from skcomplex.core.specs import RoleSpec, StateSpec
from skcomplex.core.types import ElementKind


class BitRuleInteraction(Interaction):
    def __init__(
        self,
        label,
        *,
        target: Annotated[str, RoleSpec(elements=(ElementKind.GROUP,), states=('state',), domains=('boolean',))],
        state,
    ):
        super().__init__(label)
        self.target = target
        self.state = state


class BitElement(Element):
    def __init__(self, label, *, on: Annotated[float, StateSpec()] = 0.0):
        super().__init__(label)
        self.on = on


rule = BitRuleInteraction('sat', target='bits', state='on')
try:
    CompositeSystem(elements=[GroupElement('bits', element=BitElement, size=2)], interactions=[rule]).prepare()
except RoleElementError as error:
    message = str(error)
message
```

The message says that the role `target` of `sat` accepts the domain `boolean`, but the element `bits` has the state
`on` of the domain `real`. The fix is a state of the domain `{0, 1}`, declared with
`StateSpec(domain=FiniteEnumeratedSet([0, 1]))`.

Finally, `sparse_identification` of the core returns a `CompositeSystem` of core components, never a class of
`skcomplex.systems`, so the core runs without the catalog.

### Recipes of one field

A recipe that only one field uses lives in that field, under the name the field gives it. Its system classes are the
examples to read, and the gallery example of each system runs it.

- `TopplingInteraction` drives the members of a lattice and relaxes every member at or above a threshold, as in
  `SandpileSystem` of `skcomplex.systems.physics.criticality`. The section
  [Cascades](../build_a_custom_system/lattice_systems.md#cascades) writes a cascade with the core `CascadeLaw`.
- `PolicyRuleInteraction` sets parameters once per round of its phase from what happened in the round, as in
  `BrockHommesSystem` of `skcomplex.systems.economics.markets`. A plain `Interaction` that returns
  `RuleLaw(self.act, phase=...)` does the same, as the clerk of the market above shows.
- `SpacetimeSpace` and `GeodesicInteraction` move a particle along the geodesics of a metric, as in
  `SchwarzschildPrecessionSystem` of `skcomplex.systems.physics.celestial_mechanics`. They need the extra
  `relativity`, which installs EinsteinPy.
- `ZoneCouplingInteraction` couples agents by zones of repulsion, alignment, and attraction, as in `BoidsSystem` of
  `skcomplex.systems.biology.collective_behaviour`.
- `CellAdhesionInteraction` samples the regions of a cellular Potts model, as in `CellularPottsSystem` of
  `skcomplex.systems.biology.physiology`. `SequenceSamplingInteraction` samples the rows of sequences, as in
  `RNANeutralNetworksSystem` of `skcomplex.systems.biology.evolution`. `WilsonSamplingInteraction` samples the matrix
  links of a lattice gauge field, as in `LatticeYangMillsSystem` of `skcomplex.systems.physics.waves`.
- `OptimizationInteraction` moves a population of candidates through pymoo, as in `EvolutionaryOptimisersSystem` of
  `skcomplex.systems.computer_science.evolutionary_computation`.
- `SatisfiabilityInteraction` reveals logical constraints step by step to a PySAT solver, as in
  `PerceptronSatisfiabilitySystem` of `skcomplex.systems.computer_science.neural_networks`.
