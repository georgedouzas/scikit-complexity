# General interactions

This page finds the general interaction behind a process and configures it for a model. An interaction reads the
states of the elements it binds and returns the laws that change them. Most processes already exist as a general
interaction: you pick the class that does the mathematical operation you need, bind it to your elements, and pass the
functions that make it your process. Here a stock of 8 units decays at the rate 0.5.

```python
from typing import Annotated

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import RateInteraction
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.specs import StateSpec


class AmountElement(Element):
    def __init__(self, label, *, x: Annotated[float, StateSpec()] = 0.0):
        super().__init__(label)
        self.x = x


decay = RateInteraction('decay', target='stock', state='x', rate=lambda s: -0.5 * s.target.x)
system = CompositeSystem(elements=[AmountElement('stock', x=8.0)], interactions=[decay])
round(float(simulate(system, until=2.0, dt=0.01).get_values('stock.x')[-1]), 3)
```

The stock holds 2.943 after two time units, which is `8 exp(-1)`.

`skcomplex.core.components.interactions` holds one general class per mathematical operation, such as a rate, a
gradient, a map, a coupling, a jump, a transition, a transfer, or a grouping. Find the operation you need in the list
below, then follow the link to its subsection.

- Change a state continuously by a rate, towards a goal, or with noise: [`RateInteraction`](../core_components/interactions.md#rateinteraction), in
  [Rates and relaxation](#rates-and-relaxation).
- Move states down a potential, or write the mechanics of a Lagrangian or a Hamiltonian:
  [`GradientInteraction`](../core_components/interactions.md#gradientinteraction),
  [`EulerLagrangeInteraction`](../core_components/interactions.md#eulerlagrangeinteraction), and
  [`HamiltonianInteraction`](../core_components/interactions.md#hamiltonianinteraction), in [Gradients and mechanics](#gradients-and-mechanics).
- Evolve a metric by the Einstein equations, or move a particle along its geodesics: the physics classes in
  [Curved spacetimes](#curved-spacetimes).
- Conserve, carry, or spread a quantity over a field:
  [`ConservationInteraction`](../core_components/interactions.md#conservationinteraction) and `RateInteraction`, in
  [Conservation laws](#conservation-laws).
- Update a state in steps, search by trial and error, or run a cellular automaton:
  [`MapInteraction`](../core_components/interactions.md#mapinteraction) and [`LocalRuleInteraction`](../core_components/interactions.md#localruleinteraction), in [Discrete time](#discrete-time).
- Couple members over the edges of a lattice or a network, over a group, or over close pairs:
  [`CouplingInteraction`](../core_components/interactions.md#couplinginteraction) and [`PairInteraction`](../core_components/interactions.md#pairinteraction), in [Coupling over a structure](#coupling-over-a-structure).
- Move particles through a viscous fluid by the forces, torques, and active slip of all of them: the physics
  `StokesianMobilityInteraction`, in [Coupling over a structure](#coupling-over-a-structure).
- Fire random events: [`JumpInteraction`](../core_components/interactions.md#jumpinteraction), in [Jumps](#jumps).
- Move a member between discrete values, such as susceptible and infected:
  [`TransitionInteraction`](../core_components/interactions.md#transitioninteraction), in
  [Transitions between values](#transitions-between-values).
- Reset states when a condition crosses zero, or at chosen times:
  [`CrossingInteraction`](../core_components/interactions.md#crossinginteraction), in
  [Crossings and impulses](#crossings-and-impulses).
- Sample configurations by Monte Carlo: [`SamplingInteraction`](../core_components/interactions.md#samplinginteraction), in [Monte Carlo sampling](#monte-carlo-sampling).
- Set a state from the time or the random stream of the step: [`ArrayUpdateLaw`](../core_components/laws.md#arrayupdatelaw), in
  [Array updates that read the step](#array-updates-that-read-the-step).
- Run an operation as symbolic laws or as agent rules: the `form` and `schedule` arguments, in
  [Agent and symbolic forms](#agent-and-symbolic-forms).
- Learn from payoffs: `MapInteraction`, in [Learning](#learning).
- Transfer amounts, form groups, or rewire a network: [`TransferInteraction`](../core_components/interactions.md#transferinteraction),
  [`GroupingInteraction`](../core_components/interactions.md#groupinginteraction), and [`TopologyInteraction`](../core_components/interactions.md#topologyinteraction),
  in [Agents, groups, and exchange](#agents-groups-and-exchange).
- Move agents, create and remove them, let them choose, imitate, and evolve, match partners, fire reactions, or give
  the encounters of a kinetic system: the recipes that several fields share, in
  [General realizations](../core_components/general.md).
- Write laws with Hill functions, delays, pulses, or group reductions: `skcomplex.core.components.laws.utils`, in
  [Law utilities](#law-utilities).
- Evolve a tensor state of a field: `StateSpec(tensor=...)`, in [Tensor fields](#tensor-fields).
- Do something no general class covers: the binding bases, in [When no general class fits](#when-no-general-class-fits).

## How the general classes are configured

This section shows the arguments that turn a general class into a process of a field. A class names what it does,
never what a state means. No core class or argument names a force, a position, a price,
an infection, or a harvest. You configure a class by the roles it binds, the names of the states it reads and writes,
and the functions you pass.

- `target` is the [role](../glossary.md#role) whose states the laws change, the slot of the interaction that an
  element fills. `sources` are roles the laws only read.
- `state`, `states`, `read`, and `write` name the states, so one class serves any meaning. A force is a rate on a
  state of order 2, and a mass is its `divisor`.
- Functions are callables of the namespace `s` that return symbolic expressions, such as
  `rate=lambda s: -s.k * s.target.x`. Where an argument allows it, a number stands for a constant function.
- `divisor` divides what a rate, a gradient, or a pair coupling adds to the highest derivative, such as a mass, a
  friction, or a capacitance. It is a named argument, not part of the function, so a backend such as LAMMPS can read
  it.
- A variant is a string checked against the constants of its module, such as `mode='increment'`. A variant that
  carries an algorithm of its own is a frozen spec instead, such as the totalistic rule of a cellular automaton.

## What enters the core

This section tells where a part of your own belongs. A class enters the core when it is a structural base, or when the
engine handles it by type, such as a variational principle, a conservation law, or a Markov jump process, and when it
applies in principle to other domains than the one that first needs it. Every other part is a recipe written with core
parts, placed by the systems that use it. The [placement rule](../core_components/general.md#where-a-part-goes) states
the test.

The case of one field is a thin subclass of a general class in `skcomplex.systems`, even when it only renames
arguments. It lives with the one system that uses it, or with the code its field shares when several systems use it.
A recipe that several fields share lives in `skcomplex.systems.general`.
These subclasses double as examples of how to build a part of your own, and the page of each system shows them. The
interactions of `LotkaVolterraSystem`, for example, are rate interactions except predation. `prepare` builds the parts
of a system class in `system_`:

```python
from skcomplex.core.components.interactions import RateInteraction
from skcomplex.systems.biology.ecology import LotkaVolterraSystem

system = LotkaVolterraSystem().prepare().system_
{type(part).__name__: isinstance(part, RateInteraction) for part in system.interactions}
```

`ExponentialGrowthInteraction` and `MortalityInteraction` subclass `RateInteraction`. `PredationInteraction` does not,
because it changes both populations at once.

## Elements of an interaction

This section picks the element that an interaction binds. The general interactions act on any element that declares
the states they read, so a small subclass of [`Element`](../core_components/elements.md#element), or a
[`DeclaredElement`](../core_components/elements.md#declaredelement) that takes its states as data, is all an interaction needs.

```python
from skcomplex.core.components.elements import DeclaredElement
from skcomplex.core.components.interactions import RateInteraction
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.specs import DeclaredVariableSpec, StateSpec

stock = DeclaredElement('stock', variables={'x': DeclaredVariableSpec(5.0, StateSpec())})
decay = RateInteraction('decay', target='stock', state='x', rate=lambda s: -s.target.x)
round(float(simulate(CompositeSystem(elements=[stock], interactions=[decay]), until=1.0).get_values('stock.x')[-1]), 3)
```

The stock falls from 5 to 1.839, which is `5 exp(-1)`.

- `Element` declares states and parameters as annotated arguments of its `__init__`, and `DeclaredElement` takes
  them as data.
- [`GroupElement`](../core_components/elements.md#groupelement) holds many identical members as arrays, and on `'sites'` it places one
  member at every site of a space, as [`FieldElement`](../core_components/elements.md#fieldelement) does.
- A state of order 2 with a vector value, `StateSpec(vector=True, order=2)`, is the position of a point in a
  continuous space. Its first derivative is the companion state `x_d1`, so a backend such as LAMMPS reads the
  velocity of a point from its declared states. The general realization
  [`PointElement`](../core_components/general.md#pointelement) is such an element.

## Continuous time

This section writes differential equations, with or without noise, from rates, potentials, and fluxes.

### Rates and relaxation

Here you add a rate to a state, as the decay at the top of this page does, or pull a state towards a goal with a
`RateInteraction`.

`RateInteraction(target, state, rate, sources=(), noise=None, divisor=None, hurst=None)` adds `rate(s)`, and
`noise(s) dW`, to the highest derivative of a state of any order. A Hurst index `hurst` makes the noise fractional.

```python
from typing import Annotated

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import RateInteraction
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.specs import StateSpec


class AmountElement(Element):
    def __init__(self, label, *, x: Annotated[float, StateSpec()] = 0.0):
        super().__init__(label)
        self.x = x


smoothing = RateInteraction('smoothing', target='belief', state='x', rate=lambda s: (10.0 - s.target.x) / 0.5)
system = CompositeSystem(elements=[AmountElement('belief', x=0.0)], interactions=[smoothing])
round(float(simulate(system, until=5.0, dt=0.01).get_values('belief.x')[-1]), 3)
```

The belief reaches the goal 10.0 within five time units, ten times its relaxation time of 0.5. The rate
`(goal - x) / time` relaxes a state towards a goal. With noise, it is an Ornstein-Uhlenbeck process, a random walk
pulled back towards a mean. A field
that has its own term for this form names it by a subclass, such as the physics `RelaxationTimeInteraction` or the
neuroscience `LeakyIntegrationInteraction`.

### Gradients and mechanics

Here a state moves down a potential through a `GradientInteraction`. On a state of order 1 it is a gradient flow.

```python
from typing import Annotated

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import GradientInteraction
from skcomplex.core.components.spaces import ContinuousSpace
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.specs import StateSpec


class AmountElement(Element):
    def __init__(self, label, *, x: Annotated[float, StateSpec()] = 0.0):
        super().__init__(label)
        self.x = x


class BallElement(Element):
    def __init__(self, label, *, x: Annotated[object, StateSpec(vector=True, order=2)] = 0.0, x_d1: object = 0.0):
        super().__init__(label)
        self.x = x
        self.x_d1 = x_d1


well = GradientInteraction('well', target='walker', states='x', potential=lambda s: s.target.x**2 / 2)
flow = CompositeSystem(elements=[AmountElement('walker', x=1.0)], interactions=[well])
round(float(simulate(flow, until=1.0, dt=0.01).get_values('walker.x')[-1]), 4)
```

The walker slides from 1.0 to 0.3679 in the well `x^2 / 2`, the decay `exp(-t)` of the flow `x' = -x`.

On a state of order 2, such as the position of a point in a
[`ContinuousSpace`](../core_components/spaces.md#continuousspace), the same class gives Newton's equations.

```python continuation
spring = GradientInteraction('spring', target='point', states='x', potential=lambda s: s.target.x * s.target.x / 2)
ball = BallElement('point', x=[1.0])
newton = CompositeSystem(elements=[ball], interactions=[spring], space=ContinuousSpace(n_dim=1))
round(float(simulate(newton, until=3.1416, dt=0.01).get_values('point.x')[-1][0]), 3)
```

The point starts at 1.0 and reaches -1.0 after half a period `pi` of the harmonic spring.

`GradientInteraction(target, states, potential, sources=(), divisor=None, variational=False, conserved=False)` also
gives the variational or conserved dynamics of a field. The divisor is a name, a function of the member, or a mapping
of one divisor per state.

When you know the Lagrangian of a mechanical system, the kinetic minus the potential energy, an
`EulerLagrangeInteraction` writes its equations of motion. Here a pendulum swings from the angle 0.1.

```python
from typing import Annotated

from sage.all__sagemath_symbolics import cos

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import EulerLagrangeInteraction
from skcomplex.core.components.laws.utils import derivative
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.specs import StateSpec


class AngleElement(Element):
    def __init__(self, label, *, q: Annotated[float, StateSpec(order=2)] = 0.1, q_d1: float = 0.0):
        super().__init__(label)
        self.q = q
        self.q_d1 = q_d1


lagrangian = lambda s: derivative(s.target.q, 1) ** 2 / 2 + cos(s.target.q)
swing = EulerLagrangeInteraction('swing', target='bob', states='q', lagrangian=lagrangian)
system = CompositeSystem(elements=[AngleElement('bob')], interactions=[swing])
round(float(simulate(system, until=3.2, dt=0.01).get_values('bob.q')[-1]), 2)
```

The angle is -0.1 at the time 3.2, close to half the small-swing period `2 pi`.

`EulerLagrangeInteraction(target, states, lagrangian, dissipation=None, forcing=None)` writes the equations of motion
of a Lagrangian, with a Rayleigh dissipation and a generalized forcing. The equations of one coordinate from several
interactions add, so a coupling is an interaction whose Lagrangian holds only the coupling energy.

When you know the Hamiltonian, the total energy in coordinates and momenta, a `HamiltonianInteraction` writes
Hamilton's equations. Here a planet circles a sun.

```python
from typing import Annotated

from sage.all__sagemath_symbolics import sqrt

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import HamiltonianInteraction
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.solvers import Yoshida4Solver
from skcomplex.core.specs import StateSpec


class PlanetElement(Element):
    def __init__(
        self,
        label,
        *,
        x: Annotated[float, StateSpec()] = 1.0,
        y: Annotated[float, StateSpec()] = 0.0,
        px: Annotated[float, StateSpec()] = 0.0,
        py: Annotated[float, StateSpec()] = 1.0,
    ):
        super().__init__(label)
        self.x = x
        self.y = y
        self.px = px
        self.py = py


def kepler(s):
    return (s.target.px**2 + s.target.py**2) / 2 - 1 / sqrt(s.target.x**2 + s.target.y**2)


coordinates, momenta = ('x', 'y'), ('px', 'py')
gravity = HamiltonianInteraction('gravity', target='sun', coordinates=coordinates, momenta=momenta, hamiltonian=kepler)
system = CompositeSystem(elements=[PlanetElement('sun')], interactions=[gravity])
trajectory = simulate(system, until=6.2832, solver=Yoshida4Solver(dt=0.01), dt=1.0)
round(float(trajectory.get_values('sun.x')[-1]), 6)
```

The planet returns to `x = 1.0` after one period `2 pi` of its circular orbit.

`HamiltonianInteraction(target, coordinates='q', momenta='p', hamiltonian=None)` writes Hamilton's equations
`q' = dH/dp` and `p' = -dH/dq` as rate laws of canonical coordinates and momenta. These are states of order 1, and
they may be angles. The rates of one state add, so the Hamiltonians of separate interactions sum.

The [solver](../glossary.md#solver) `Yoshida4Solver` integrates a separable Hamiltonian at fourth order by three velocity Verlet steps. It first splits the
states into two blocks whose rates read only the other block. It keeps the energy bounded over long runs.

`invariants` computes Poisson brackets over the canonical pairs. It decides whether expressions given as
`ObservableSpec`s are conserved, and checks a `SymmetrySpec` against the Hamiltonian or the Lagrangian, with its
Noether charge, the quantity that a continuous symmetry conserves.

```python continuation
from skcomplex.core.analysis import invariants
from skcomplex.core.specs import ObservableSpec, SymmetrySpec

momentum = ObservableSpec('angular_momentum', lambda s: s.sun.x * s.sun.py - s.sun.y * s.sun.px)
rotation = SymmetrySpec('rotation', lambda s: {'sun.x': -s.sun.y, 'sun.y': s.sun.x})
invariants(system, expressions=[momentum]).conserved, invariants(system, kind='symmetry', symmetry=rotation).charge
```

The angular momentum is conserved, and the charge of the rotation is `py_sun*x_sun - px_sun*y_sun`, the same angular
momentum.

### Curved spacetimes

This section points to the recipes that evolve a metric or move a point along its geodesics. General relativity is a
recipe of physics, written with the rate laws of the core. `SpacetimeSpace` derives the
Christoffel symbols of a metric through EinsteinPy, and `GeodesicInteraction` moves a point along the geodesics of the
metric, as in `SchwarzschildPrecessionSystem` of `skcomplex.systems.physics.celestial_mechanics`, whose page shows the
precession of an orbit around a static mass. Both need the extra `relativity`.

The physics `EinsteinFieldEquationsInteraction` of `skcomplex.systems.physics.cosmology` evolves the metric when the
metric itself is the unknown, reduced by a symmetry to functions of time, such as the scale factor `a(t)` of a flat
universe of dust in `FLRWCosmologySystem`.

`EinsteinFieldEquationsInteraction(target, states, space, components, source=None, coupling=8π,
cosmological_constant=0.0, time=0, axes=None)` names every function of the metric after a state or parameter of the
target. It takes one component `(mu, nu)` of the mixed equations `G^mu_nu + Λ δ^mu_nu = κ T^mu_nu` per state, and solves
them for the second time derivatives. The source returns the mixed energy-momentum tensor, for dust of density `ρ /
a³`, so the scale factor grows as `(3 t / 2 + 1)^(2/3)`. The other components, such as the Friedmann equation `G^t_t`,
are constraints the initial values satisfy. A function of a spatial coordinate too, such as the radius `R(t, r)` of the
shells of a dust ball, needs that coordinate on a lattice axis of a field, `axes={1: 0}`, and its derivatives become
`partial` of the field. An equation that mixes time and space derivatives, or keeps a coordinate, raises an error.

### Conservation laws

Here a quantity of a field is only moved around, never created or lost except by a source. The example runs
Burgers' flux, a drift, and diffusion on a periodic finite-volume field of a
[`LatticeSpace`](../core_components/spaces.md#latticespace). A finite-volume field stores the mean of a cell, so the total changes only
through the boundary.

```python
from typing import Annotated

import numpy as np

from skcomplex.core.components.elements import Element, FieldElement
from skcomplex.core.components.interactions import ConservationInteraction, RateInteraction
from skcomplex.core.components.laws.utils import laplacian
from skcomplex.core.components.spaces import LatticeSpace
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.specs import StateSpec


class CellElement(Element):
    def __init__(self, label, *, u: Annotated[float, StateSpec()] = 0.0):
        super().__init__(label)
        self.u = u


space = LatticeSpace((50,), spacing=0.1, boundary='periodic')
values = {'u': 1.0 + np.sin(2 * np.pi * np.arange(50) / 50)}
interactions = {
    'burgers': ConservationInteraction('burgers', target='wave', state='u', flux=lambda s: s.target.u**2 / 2),
    'drift': ConservationInteraction('drift', target='wave', state='u', flux=lambda s: 1.0 * s.target.u),
    'heat': RateInteraction('heat', target='wave', state='u', rate=lambda s: 0.1 * laplacian(s.target.u)),
}
totals = {}
for label, interaction in interactions.items():
    field = FieldElement('wave', element=CellElement, space=space, values=values, discretization='finite-volume')
    system = CompositeSystem(elements=[field], interactions=[interaction])
    total = simulate(system, until=0.5, dt=0.01).get_values('wave.u')[-1].sum()
    totals[label] = round(float(total), 6)
totals
```

The total stays 50.0 under each of the three interactions.

- `ConservationInteraction(target, state, flux, source=None, limiter='minmod', riemann='rusanov')` conserves a state of
  a finite-volume field under a flux and a source. The flux laws of one field form one system. With `riemann=None`, it
  lowers to the divergence of the flux on spectral and finite-difference fields.
- A drift `a` carries a state as the flux `a u` of a `ConservationInteraction`.
- Diffusion is the rate `D laplacian(u)` of a `RateInteraction`, the divergence of the flux `-D gradient(u)`, which keeps
  the total of a periodic field. Physics names these forms by the subclasses `AdvectionInteraction` and
  `DiffusivityInteraction`.

## Discrete time

This section updates states in steps, as maps, searches, cellular automata, and cascades. A system steps when it
declares [`DiscreteTime`](../core_components/spaces.md#discretetime). A `MapInteraction` sets the next value of a state from the current
one, here by the logistic map `x(t+1) = 3 x (1 - x)`.

```python
from typing import Annotated

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import MapInteraction
from skcomplex.core.components.spaces import DiscreteTime
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.specs import StateSpec


class ValueElement(Element):
    def __init__(self, label, *, x: Annotated[float, StateSpec()] = 0.5):
        super().__init__(label)
        self.x = x


logistic = MapInteraction('map', target='value', state='x', function=lambda s: 3 * s.target.x * (1 - s.target.x))
system = CompositeSystem(elements=[ValueElement('value')], interactions=[logistic], time=DiscreteTime())
simulate(system, until=2).get_values('value.x').tolist()
```

The values are `[0.5, 0.75, 0.5625]`.

`MapInteraction(target, state, function, sources=(), mode='replace', schedule='synchronous', form='symbolic')` sets
`x(t+1) = f(s)`, or `x + f(s)` with `mode='increment'`. The schedule decides the order of the members:

- `'synchronous'` updates all members at once.
- `'sequential'` updates them one after the other.
- `'random-sequential'` updates them in random order.
- `'member'` updates them in the activation of every member, in the agent form.

A state that searches for a target by trial and error, such as a price that rises while demand exceeds supply, is a
`MapInteraction` that steps it by the sign of a signal and keeps it within bounds:

```python
from typing import Annotated

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import MapInteraction
from skcomplex.core.components.laws.utils import clip, sign
from skcomplex.core.components.spaces import DiscreteTime
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.specs import StateSpec


class QuoteElement(Element):
    def __init__(self, label, *, x: Annotated[float, StateSpec()] = 1.0):
        super().__init__(label)
        self.x = x


step = lambda s: clip(s.target.x * (1 + 0.5 * sign(2 - s.target.x)), None, 2.0)
search = MapInteraction('search', target='quote', state='x', function=step)
system = CompositeSystem(elements=[QuoteElement('quote')], interactions=[search], time=DiscreteTime())
simulate(system, until=3).get_values('quote.x').tolist()
```

The quote rises from 1.0 to 1.5 and to 2.0, where the signal `2 - x` vanishes, and it stays there. The next value is `x (1 + step sign(signal))` kept within `[lower, upper]`. Economics names this search
`PriceAdjustmentInteraction`, with an additive stepping and an agent form.

A cellular automaton updates each cell from the cells around it by a fixed rule. A `LocalRuleInteraction` runs one,
and here its [`BallNeighborhood`](../core_components/spaces.md#ballneighborhood) takes the two nearest cells. The helper `run` below places
a row of five cells and runs one interaction on it.

```python
from typing import Annotated

import numpy as np

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.interactions import LocalRuleInteraction
from skcomplex.core.components.spaces import BallNeighborhood, DiscreteTime, LatticeSpace
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.specs import StateSpec, TotalisticSpec


class CellElement(Element):
    def __init__(self, label, *, c: Annotated[float, StateSpec()] = 0.0):
        super().__init__(label)
        self.c = c


def run(interaction, values, until):
    cells = GroupElement('cells', element=CellElement, space=LatticeSpace(5), values={'c': values}, on='sites')
    system = CompositeSystem(elements=[cells], interactions=[interaction], time=DiscreteTime())
    return simulate(system, until=until, dt=until, random_state=0)


parity = LocalRuleInteraction(
    'parity',
    target='cells',
    state='c',
    rule=TotalisticSpec({1: 1}),
    neighborhood=BallNeighborhood(metric='manhattan', n_dim=1),
)
run(parity, np.eye(5)[2], 1).states[-1].tolist()
```

The rule turns on a cell whose neighbourhood holds exactly one live cell. The single live cell in the middle gives
`[0.0, 1.0, 0.0, 1.0, 0.0]` after one step.

`LocalRuleInteraction(target, state, rule, neighborhood)` runs a cellular automaton from a rule table, a totalistic
rule, or a function of the ordered neighbourhood.

A cascade, which relaxes every member at or above a threshold until none is left, is a
[`CascadeLaw`](../core_components/laws.md#cascadelaw) of the core, in
[Cascades](lattice_systems.md#cascades). Physics names the cascade of a sandpile `TopplingInteraction`, as in
`SandpileSystem` of `skcomplex.systems.physics.criticality`.

## Coupling over a structure

This section lets members influence each other: neighbours on a lattice or a network, every member of a group, or
pairs of points closer than a cutoff. Kuramoto oscillators, for example, couple every member of a group through the
mean of a sine of their phase differences, in a `CouplingInteraction`.

```python
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import sin

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.interactions import CouplingInteraction
from skcomplex.core.components.laws import RateLaw
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.specs import ParameterSpec, StateSpec


class OscillatorElement(Element):
    def __init__(
        self,
        label,
        *,
        theta: Annotated[float, StateSpec()] = 0.0,
        omega: Annotated[float, ParameterSpec()] = 0.0,
    ):
        super().__init__(label)
        self.theta = theta
        self.omega = omega

    def _get_laws(self, s):
        return [RateLaw(s.theta, s.omega)]


rng = np.random.default_rng(0)
values = {'theta': rng.uniform(-np.pi, np.pi, 50), 'omega': rng.normal(0.0, 0.5, 50)}
oscillators = GroupElement('oscillators', element=OscillatorElement, size=50, values=values)
kuramoto = CouplingInteraction(
    'kuramoto',
    target='oscillators',
    read='theta',
    write='theta',
    function=lambda s: 2.0 * sin(s.other.theta - s.member.theta),
    reduce='mean',
    mode='group',
)
system = CompositeSystem(elements=[oscillators], interactions=[kuramoto])
phases = simulate(system, until=10, dt=1.0).get_values('oscillators.theta')[-1]
round(float(np.abs(np.exp(1j * phases).mean())), 2)
```

The order parameter, the length of the mean of `exp(i theta)`, reaches 0.97 at the time 10. The phases have locked.

`CouplingInteraction(target, source=None, read, write, function, reduce='sum', mode='edges', ...)` gives member `i` a
reduction over `j` of `function(s)`. Inside the function, `s.member` reads `i`, `s.other` reads `j`, and `s.edge`
reads the edge between them.

- It works over the edges of a lattice or a network, over the whole group or a separate `source` group
  (`mode='group'`), or over the next member of an ordered group.
- The reductions are sum, mean, max, min, normalized mean, and circular mean.
- The weights are an edge state, a matrix, or a function.
- It adds to a rate (`form='rate'`), or sets the next value with `combine(s, reduction)` (`form='map'`).

A `PairInteraction` gives forces between pairs of points, here a Morse potential between two atoms.

```python
import numpy as np
from sage.all__sagemath_symbolics import exp

from typing import Annotated

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.interactions import PairInteraction
from skcomplex.core.components.spaces import ContinuousSpace
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.specs import StateSpec


class AtomElement(Element):
    def __init__(self, label, *, x: Annotated[object, StateSpec(vector=True, order=2)] = 0.0, x_d1: object = 0.0):
        super().__init__(label)
        self.x = x
        self.x_d1 = x_d1


morse = PairInteraction('morse', 'pair', 'pair', cutoff=3.0, potential=lambda s, r: 2 * (1 - exp(1 - r)) ** 2)
pair = GroupElement('pair', element=AtomElement, size=2, values={'x': [[0.0, 0.0], [1.5, 0.0]]})
system = CompositeSystem(elements=[pair], interactions=[morse], space=ContinuousSpace(n_dim=2))
positions = simulate(system, until=5, dt=0.1).get_values('pair.x')
distances = np.linalg.norm(positions[:, 1] - positions[:, 0], axis=1)
round(float(distances.min()), 2), round(float(distances.max()), 2)
```

The distance between the atoms oscillates between 0.67 and 1.5 around the minimum of the potential at 1.

`PairInteraction(first, second, cutoff, read, write, function=None, potential=None, divisor=None,
period=None)` adds to both members of every pair within the cutoff either `function(s)`, or minus the gradient of
`potential(s, r)` at their distance `r`. With several states in `write`, the function returns one value per state.

The physics `StokesianMobilityInteraction` moves spheres in a viscous fluid at low Reynolds number, where the force on
one sphere moves all the others through the flow it creates, as in `SquirmersSystem`. Two spheres that sediment side by
side fall faster than one alone.

`StokesianMobilityInteraction(target, position='x', orientation=None, tensor='rotne-prager-yamakawa', form='rate',
force, torque, velocity, stresslet, dipole, potential, cutoff, root='cholesky', radius, viscosity, temperature, dt)`
moves the spheres of a group by the grand mobility `M`, the linear map from the forces and torques on all spheres to
their velocities and angular velocities. A lone sphere moves at `F / (6 pi eta a)`, and a second sphere adds the
fraction `3 a / (4 r) + a^3 / (2 r^3)` to the speed of the first, about a quarter at the distance `r = 3`.

- The tensor is `'oseen'`, the flow of a point force, `'rotne-prager-yamakawa'`, which adds the finite size of the
  spheres and stays positive definite when they overlap, or `'free-draining'`, which couples no pair. The rotlet couples
  torques to velocities and forces to angular velocities, and an `orientation` state turns at `Omega x p`.
- The force, the torque, an intrinsic `velocity`, a `stresslet`, and a potential `dipole` are functions of the namespace
  of a member. The stresslet and the dipole are force-free sources whose flows move and turn the other spheres, so
  together with the intrinsic velocity they carry the active slip of a microswimmer. A pair `potential(s, r)` of the
  distance adds its forces within `cutoff`.
- In `form='map'`, in discrete time, every step is the Ermak-McCammon step `x + M F dt + sqrt(2 kT dt) B xi` of Brownian
  dynamics, where `B B^T = M`. `root='cholesky'` factors `M`, and `root='lanczos'` applies a Krylov approximation of
  its square root with the matrix-free product, for larger groups. The time of the namespace then counts steps.
- Positions in the plane are the plane `z = 0` of a three-dimensional fluid. The fluid is unbounded, every evaluation
  costs `O(N^2)`, and Cholesky costs `O(N^3)`.

The library has `SquirmersSystem` of `skcomplex.systems.physics.active_matter`, spherical microswimmers whose slip
modes make them pushers or pullers, and `HydrodynamicallyCoupledTrapsSystem` of `skcomplex.systems.physics.soft_matter`,
two Brownian spheres in blinking optical tweezers whose free steps correlate through the mobility.

## Events and discrete states

This section changes states by discrete, often random, events.

### Jumps

A `JumpInteraction` fires random events that change states by fixed amounts, here arrivals to a queue at the rate 2.

```python
from typing import Annotated

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import JumpInteraction
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.specs import StateSpec


class AmountElement(Element):
    def __init__(self, label, *, x: Annotated[float, StateSpec()] = 0.0):
        super().__init__(label)
        self.x = x


arrivals = JumpInteraction('arrivals', target='queue', changes={'x': 1}, rate=lambda s: 2.0)
queue = CompositeSystem(elements=[AmountElement('queue')], interactions=[arrivals])
counts = simulate(queue, until=5, dt=1.0, random_state=0).get_values('queue.x')
all(count == round(count) for count in counts)
```

Every recorded length of the queue is a whole number.

`JumpInteraction(target, changes, rate=None, probability=None, waiting=None, reset=None)` fires stochastic events that
change states by signed amounts. They fire at a rate, with a probability per step, or after drawn waiting times.

A reaction between species at a product propensity is a recipe on these jumps and on rates, the general realization
[`ReactionInteraction`](../core_components/general.md#reactioninteraction).

### Transitions between values

A `TransitionInteraction` moves a member between discrete values, such as states of health or of ownership. Here
each of 1000 holders leaves the value 1 with the probability 0.2 per step.

```python
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import NN

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.interactions import TransitionInteraction
from skcomplex.core.components.laws.utils import neighbors, where
from skcomplex.core.components.spaces import DiscreteTime, LatticeSpace
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.specs import StateSpec


class HolderElement(Element):
    def __init__(self, label, *, state: Annotated[int, StateSpec(domain=NN)] = 1):
        super().__init__(label)
        self.state = state


move = TransitionInteraction(
    'move', target='holders', state='state', source=1, destination=2, probability=lambda s: 0.2
)
holders = GroupElement('holders', element=HolderElement, size=1000)
system = CompositeSystem(elements=[holders], interactions=[move], time=DiscreteTime())
values = simulate(system, until=3, random_state=0).get_values('holders.state')
round(float((values[-1] == 1).mean()), 1)
```

After three steps half the holders keep the value 1, close to `0.8^3 = 0.512`.

`TransitionInteraction(target, state, source, destination, rate=None, probability=None, condition=None, matrix=None,
...)` moves a discrete state from a value, a set of values, or the values a function accepts, to another value. It
fires at a rate, with a probability per step, when a condition holds, or by a transition or an emission matrix. With
`after`, it fires after a residence time.

A rate can read the neighbours of a member, so spreading along edges with recovery takes two transitions.

```python continuation
contact = TransitionInteraction(
    'contact',
    target='sites',
    state='state',
    source=0,
    destination=1,
    rate=lambda s: 4.0 * neighbors(where(s.target.other.state == 1, 1, 0), reduce='mean'),
)
recovery = TransitionInteraction('recovery', target='sites', state='state', source=1, destination=0, rate=lambda s: 1.0)
sites = GroupElement('sites', element=HolderElement, space=LatticeSpace((20, 20)), on='sites')
system = CompositeSystem(elements=[sites], interactions=[contact, recovery])
values = simulate(system, until=10, dt=1.0, random_state=0).get_values('sites.state')
values[-1].mean() > 0.3
```

More than three in ten sites are infected at the time 10. The ready-made systems of
`skcomplex.systems.biology.epidemiology` build on such a subclass.

### Crossings and impulses

A `CrossingInteraction` acts when a condition is met, or at chosen times with a guard of the time. The example wraps
a clock back to zero each time it reaches 1, and kicks it forward once at the time 1.5.

```python
from typing import Annotated

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import CrossingInteraction
from skcomplex.core.components.laws import RateLaw
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.specs import StateSpec


class ClockElement(Element):
    def __init__(self, label, *, x: Annotated[float, StateSpec()] = 0.0):
        super().__init__(label)
        self.x = x

    def _get_laws(self, s):
        return [RateLaw(s.x, 1.0)]


wrap = CrossingInteraction(
    'wrap', target='clock', guard=lambda s: s.target.x - 1, reset={'x': lambda s: 0}, direction=1
)
kick = CrossingInteraction(
    'kick', target='clock', guard=lambda s: s.t - 1.5, reset={'x': lambda s: s.target.x + 0.25}, direction=1
)
system = CompositeSystem(elements=[ClockElement('clock')], interactions=[wrap, kick])
events = simulate(system, until=2.2, dt=0.5).events
[(event.label, round(event.time, 3)) for event in events]
```

The events are a wrap at 1.0, the kick at 1.5, and a second wrap at 1.75, because the kick moved the clock from 0.5
to 0.75.

- `CrossingInteraction(target, guard, reset=None, direction=0, terminal=False)` fires an event when a guard crosses
  zero. The event resets states, or ends the run.
- A guard `s.t - t_k` resets states at the time `t_k`, and the guard `periodic_guard(s.t, period, start, count)` of
  `skcomplex.core.components.laws.utils` at periodic times. Engineering names these resets `PulseInteraction`, and
  physics names periodic ones `KickInteraction`.

### Monte Carlo sampling

A `SamplingInteraction` samples the configurations of a lattice, here the spins of an Ising magnet at low
temperature, which align.

```python
from typing import Annotated

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.interactions import SamplingInteraction
from skcomplex.core.components.laws.utils import neighbors
from skcomplex.core.components.spaces import DiscreteTime, LatticeSpace
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.specs import StateSpec


class SpinElement(Element):
    def __init__(self, label, *, s: Annotated[float, StateSpec()] = 1.0):
        super().__init__(label)
        self.s = s


objective = lambda s: -s.target.s * neighbors(s.target.other.s)
ising = SamplingInteraction('ising', target='spins', state='s', objective=objective, intensity=1.0)
spins = GroupElement('spins', element=SpinElement, space=LatticeSpace((8, 8)), on='sites')
system = CompositeSystem(elements=[spins], interactions=[ising], time=DiscreteTime())
abs(simulate(system, until=20, random_state=0).states[-1].mean()) > 0.9
```

The mean spin stays above 0.9 in absolute value after 20 sweeps.

`SamplingInteraction(target, state, objective, move='flip', intensity=1.0)` runs Monte Carlo sweeps. A move is accepted
from the change of an objective with probability `exp(-intensity * change)`. With `move='cluster'`, it makes Wolff
cluster flips instead. To sample an energy at a temperature, pass the energy as the objective and one over the
temperature as the intensity.

A subclass that scores more than local terms overrides the hooks `_check_objective` and `_check_move`, and returns its
own Monte Carlo sweep as an `ArrayUpdateLaw`. Biology samples whole rows of sequences this way with
`SequenceSamplingInteraction`, as in `RNANeutralNetworksSystem` of `skcomplex.systems.biology.evolution`, and physics
samples matrix links with `WilsonSamplingInteraction`, as in `LatticeYangMillsSystem` of
`skcomplex.systems.physics.waves`. An infinite intensity is the zero temperature, where the Metropolis rule accepts a
move only when the objective does not grow.

## Array updates that read the step

This section sets a state from outside, by the time of the step or by a random draw, with an `ArrayUpdateLaw`. Here
it replays an ensemble of recorded configurations.

```python
import numpy as np
from sage.all__sagemath_symbolics import var

from skcomplex.core.components.laws import ArrayUpdateLaw

ensemble = np.zeros((38, 20, 3))
replay = ArrayUpdateLaw(var('x'), lambda values, *, time: ensemble[min(int(time), 37)])
replay.prepare().targets_
```

The law targets the one state `x`.

`ArrayUpdateLaw` passes the time of the step to a function that names `time`, and the random generator to a function
that names `random`. A function can then set a lattice state to one member of an ensemble per step, as
`FuzzyOilDropSystem` of `skcomplex.systems.biology.systems_biology` does, and hold the last member after the end.

## Agent and symbolic forms

This section runs one operation either as symbolic laws or as agent rules. A class whose operation agent systems also
use takes `form='symbolic'` or `form='agent'`. Choose the agent form when your members are agents that need to read the
system's history, storage, or their own neighbours and cell.

```python
from types import SimpleNamespace

from skcomplex.core.components.interactions import MapInteraction

symbolic = MapInteraction('map', target='walkers', state='x', function=lambda s: s.target.x + 1)
agent = MapInteraction('map', target='walkers', state='x', function=lambda agent, system: agent.x + 1, form='agent')
type(agent.prepare().run(SimpleNamespace()).laws_[0]).__name__, symbolic.form
```

The agent form returns a `RuleLaw`, and the default form is `'symbolic'`.

- The symbolic form lowers to symbolic laws, which every solver compiles.
- The agent form lowers to an agent [`RuleLaw`](../core_components/laws.md#rulelaw) in its `phase`. Its functions take the agent and the system view,
  `f(agent, system)`, so they read the system's history and storage and the agent's neighbours and cell.
- The form is an argument, not a guess, because the engine probes laws with empty namespaces and cannot tell an agent
  group from a symbolic one.

**The member schedule.** Use `schedule='member'` when every agent should take all of its actions in one go, such as
stepping and then recording the step it reached:

```python
from typing import Annotated

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.interactions import MapInteraction
from skcomplex.core.components.spaces import LatticeSpace
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.specs import StateSpec


class WalkerElement(Element):
    def __init__(self, label, *, x: Annotated[float, StateSpec()] = 0.0, steps: Annotated[float, StateSpec()] = 0.0):
        super().__init__(label)
        self.x = x
        self.steps = steps


member = {'schedule': 'member', 'form': 'agent'}
walk = MapInteraction('walk', target='walkers', state='x', function=lambda a, m: a.x + 1, **member)
count = MapInteraction('count', target='walkers', state='steps', function=lambda a, m: a.x, **member)
walkers = GroupElement('walkers', element=WalkerElement, size=3)
system = CompositeSystem(elements=[walkers], interactions=[walk, count], space=LatticeSpace((5, 5)))
simulate(system, until=4, random_state=0).get_agents('walkers', 4).values['steps'].tolist()
```

Each of the three walkers records 4.0 steps after four steps.

An agent rule may name the role whose members run it, `RuleLaw(function, members='target')`. The Mesa solver runs it
in the activation of every member of that role, after the rules of the member's element and the member rules of the
interactions listed before it. `MapInteraction`, `TransitionInteraction`, and `TransferInteraction` take
`schedule='member'` for it, and so do the movement and the birth and death of the general realizations. The updates,
transfers, transitions, and life cycle of one agent then run together as one step of that agent, in the order of the
system's interactions, so the count above reads the step its walker took in the same activation.

## Learning

This section lets members learn from payoffs. When members keep a score per option, a `MapInteraction` per score
moves it towards the payoff its option realized, here 1.0 for the first option and 2.0 for the second.

```python
from typing import Annotated

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.interactions import MapInteraction
from skcomplex.core.components.spaces import DiscreteTime
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.specs import StateSpec


class LearnerElement(Element):
    def __init__(
        self,
        label,
        *,
        first: Annotated[float, StateSpec()] = 0.0,
        second: Annotated[float, StateSpec()] = 0.0,
    ):
        super().__init__(label)
        self.first = first
        self.second = second


first = lambda s: 0.5 * (1.0 - s.target.first)
second = lambda s: 0.5 * (2.0 - s.target.second)
learning = [
    MapInteraction('first', target='learners', state='first', function=first, mode='increment'),
    MapInteraction('second', target='learners', state='second', function=second, mode='increment'),
]
learners = GroupElement('learners', element=LearnerElement, size=2)
system = CompositeSystem(elements=[learners], interactions=learning, time=DiscreteTime())
trajectory = simulate(system, until=10, random_state=0)
first, second = trajectory.get_values('learners.first')[-1][0], trajectory.get_values('learners.second')[-1][0]
round(float(first), 2), round(float(second), 2)
```

After ten steps the scores are 1.0 and 2.0, the payoffs of the two options.

Each score moves by `rate (payoff - score)`. Economics names this learning `PayoffLearningInteraction`, with the
average and cumulative rules, a chosen option, and an agent form.

## Agents, groups, and exchange

This section passes amounts between agents, gathers them into groups, and changes the network they live on. The
movement, the birth and death, and the matching of agents are general realizations, such as
[`MovementInteraction`](../core_components/general.md#movementinteraction).

### Transfers

A `TransferInteraction` moves an amount from one holder to another, so that the total is kept. Here an upper tank
drains into a lower one.

```python
from typing import Annotated

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import TransferInteraction
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.specs import StateSpec


class TankElement(Element):
    def __init__(self, label, *, level: Annotated[float, StateSpec()] = 0.0):
        super().__init__(label)
        self.level = level


flow = TransferInteraction('flow', source='upper', target='lower', state='level', amount=lambda s: s.source.level / 2)
system = CompositeSystem(elements=[TankElement('upper', level=4.0), TankElement('lower')], interactions=[flow])
trajectory = simulate(system, until=1.0)
round(float(trajectory.get_values('upper.level')[-1] + trajectory.get_values('lower.level')[-1]), 6)
```

The two levels still sum to 4.0.

`TransferInteraction(source, target, state, into=None, amount, ...)` moves amounts of states between members, the
agents of one cell, a [`Source()`](../core_components/laws.md#source), and a [`Sink()`](../core_components/laws.md#sink). In the symbolic form these are transfer laws, which the trajectory
records as named flows. In the agent form they move between paired agents.

Transfers within one group can follow the group order, as in the exclusion process of particles hopping along a lane.
Every site holds at most one particle, and the number of particles stays the same.

```python
from typing import Annotated

import numpy as np

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.interactions import TransferInteraction
from skcomplex.core.components.laws.utils import bernoulli
from skcomplex.core.components.spaces import DiscreteTime
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.specs import StateSpec


class SiteElement(Element):
    def __init__(self, label, *, occupied: Annotated[float, StateSpec()] = 0.0):
        super().__init__(label)
        self.occupied = occupied


hops = TransferInteraction(
    'hops',
    source='lane',
    target='lane',
    state='occupied',
    amount=lambda s: s.source.occupied * (1 - s.target.occupied) * bernoulli('hop', 0.5),
    offset=1,
)
lane = GroupElement('lane', element=SiteElement, size=20, values={'occupied': np.tile([1.0, 0.0], 10)})
system = CompositeSystem(elements=[lane], interactions=[hops], time=DiscreteTime())
trajectory = simulate(system, until=10, dt=1, random_state=0)
set(trajectory.get_values('lane.occupied').sum(axis=1))
```

Every recorded step holds 10.0 particles.

- With `offset=k`, a symbolic transfer within one group moves amounts along the edges of the group order, from every
  member to the member `k` places along it, which `namespace.target` reads.
- The `boundary` is `'periodic'` for a ring, or `'fixed'` for a line, whose last member moves nothing.
- With a `rate`, a `reverse_rate` adds jumps that move the amounts back.
- A draw in an amount is one draw per edge, read by both of its ends. The parallel update of the exclusion process
  therefore keeps one particle per site.
- A leg end `'<role>.<state>'` names a state of any role. One transfer, such as the circuit of money of a stock-flow
  consistent economy, can then move amounts between several holders, and a subclass declares the parameters its
  amounts read.

### Grouping

A `GroupingInteraction` lets agents meet in small groups and act together, here the majority vote of Galam's
majority rule.

```python
from typing import Annotated

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.interactions import GroupingInteraction
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.specs import StateSpec


class VoterElement(Element):
    def __init__(self, label, *, s: Annotated[float, StateSpec()] = 0.0):
        super().__init__(label)
        self.s = s


def adopt_majority(members, system):
    majority = float(2 * sum(member.s for member in members) > len(members))
    for member in members:
        member.s = majority


voters = GroupElement('voters', element=VoterElement, size=3, values={'s': [1.0, 1.0, 0.0]})
galam = GroupingInteraction('galam', target='voters', groups='partition', size=3, act=adopt_majority)
trajectory = simulate(CompositeSystem(elements=[voters], interactions=[galam]), until=1)
trajectory.get_agents('voters', 1).values['s'].tolist()
```

The three voters held `[1.0, 1.0, 0.0]` and all hold 1.0 after one vote.

`GroupingInteraction(target, groups='pairs', act, size=2)` draws pairs, a random partition, neighbourhoods, or groups
by a state every step. It then calls `act(members, system)` on every group.

### Network topology

A `TopologyInteraction` changes the network itself. Here every node of a ring of five in a
[`NetworkSpace`](../core_components/spaces.md#networkspace) removes an edge in one step.

```python
import networkx as nx

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.interactions import TopologyInteraction
from skcomplex.core.components.spaces import NetworkSpace
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate

nodes = GroupElement('nodes', element=Element, on='sites', variable=True, capacity=20)
links = GroupElement('links', element=Element, on='edges', variable=True, capacity=20)
expiry = TopologyInteraction('expiry', target='nodes', action='remove-edge')
system = CompositeSystem(elements=[nodes, links], interactions=[expiry], space=NetworkSpace(nx.cycle_graph(5)))
trajectory = simulate(system, until=1, random_state=0)
len(trajectory.get_network(0).sources), len(trajectory.get_network(1).sources)
```

The ring has 5 edges at the start and none after the step.

`TopologyInteraction(target=None, action='add-edge', rate=None, probability=None, toward=None, ...)` adds or removes
nodes and edges of a network, and rewires edges, preferentially towards a weight. Without a target, it adds one node
per step from outside the network.

### When no general class fits

This section names the classes to subclass for an operation the general classes do not cover. These binding bases
fix only which members an interaction binds.

- [`Interaction`](../core_components/interactions.md#interaction) for laws of its own over any roles.
- [`NeighborInteraction`](../core_components/interactions.md#neighborinteraction) over the edges of a lattice or a network.
- `PairInteraction` over the pairs within a cutoff.
- [`HyperedgeInteraction`](../core_components/interactions.md#hyperedgeinteraction) over the members of hyperedges.

The sections [Groups of many elements](groups_of_many_elements.md), [Networks](networks.md), and
[Pair interactions](groups_of_many_elements.md#pair-interactions) show them.

## Law utilities

This section writes laws with the common functions of `skcomplex.core.components.laws.utils`, which every
[backend](../glossary.md#backend) lowers.

```python
from sage.all__sagemath_symbolics import var

from skcomplex.core.components.laws.utils import circular_mean, clip, hill, logistic, pulse_train, sign

x, t, theta = var('x t theta')
pointwise = float(hill(x, 2, 3).subs(x=2)), float(logistic(x, 1, 2).subs(x=1))
clipped = [clip(x, 0, 1).subs(x=value) for value in (-2, 0.5, 3)]
signs = [sign(x).subs(x=value) for value in (-2, 0, 3)]
pulses = [float(pulse_train(t, 10, 1, count=2).subs({t: time})) for time in (10.5, 20.5)]
pointwise, clipped, signs, pulses, circular_mean(theta)
```

The Hill function and the logistic function are both 0.5 at their centre. Clipping to `[0, 1]` gives `[0, 0.5, 1]`,
the signs are `[-1, 0, 1]`, and a train of two pulses is on at 10.5 and off at 20.5, after its count. The circular mean
is the symbolic `arctan2(total(sin(theta)), total(cos(theta)))`.

- `where`, `either`, `both`, and `mod` select, combine conditions, and take remainders.
- `hill(x, half, n)`, `logistic(x, center, width)`, `clip(x, low, high)`, and `sign(x)` are pointwise functions.
- `minimum` and `maximum` of several expressions compare them. Of one expression, they reduce it over a group.
- `total`, `mean`, `count`, and `circular_mean` reduce over a group. With `by=`, they reduce over the members that
  share the value of a state, such as the members of one household or one cell.
- `distributed_delay(value, time, order, conserved=True)` is the material delay of stock-and-flow systems, and the
  smoothing with `conserved=False`.
- `pulse_train(t, period, width, height, start, count=None)` is a train of rectangular pulses, finite with a count.

## Tensor fields

This section evolves a field that carries a matrix at every site, such as the orientation of a liquid crystal or a
stress. The example stirs a symmetric traceless tensor by the flow it drives, and diffusion and decay shrink it.

```python
from typing import Annotated

import numpy as np

from skcomplex.core.components.elements import Element, FieldElement
from skcomplex.core.components.laws import RateLaw
from skcomplex.core.components.laws.utils import (
    advection,
    inverse_helmholtz,
    laplacian,
    leray_projection,
    tensor_divergence,
)
from skcomplex.core.components.spaces import LatticeSpace
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.specs import StateSpec


class StirredElement(Element):
    def __init__(self, label, Q: Annotated[object, StateSpec(tensor='traceless')] = 0.0):
        super().__init__(label)
        self.Q = Q

    def _get_laws(self, s):
        force = tensor_divergence(s.Q)
        velocity = leray_projection(force.apply_map(lambda item: inverse_helmholtz(-item)))
        return [RateLaw(s.Q, -s.Q.apply_map(lambda item: advection(item, velocity)) + s.Q.apply_map(laplacian) - s.Q)]


size = 16
spacing = 2 * np.pi / size
x, y = np.meshgrid(np.arange(size) * spacing, np.arange(size) * spacing, indexing='ij')
start = np.stack([0.1 * np.cos(x).ravel(), 0.1 * np.sin(y).ravel()], axis=1)
film = FieldElement(
    'film',
    element=StirredElement,
    space=LatticeSpace((size, size), spacing=spacing),
    values={'Q': start},
    discretization='spectral',
)
trajectory = simulate(CompositeSystem(elements=[film]), until=0.2, dt=0.01)
bool(np.max(np.abs(trajectory.get_values('film.Q')[-1])) < np.max(np.abs(start)))
```

The largest component at the time 0.2 is smaller than at the start, so the block returns `True`.

`StateSpec(tensor=symmetry)` declares a rank-2 tensor state of a field on a lattice of 2 or 3 dimensions. The symmetry
is `'full'`, `'symmetric'`, `'antisymmetric'`, or `'traceless'`, which means symmetric and traceless.

- The element stores only the independent components, such as `Q_xx` and `Q_xy` for a symmetric traceless state in 2
  dimensions. Initial values give one value per independent component.
- Laws read `s.Q` as a Sage matrix whose dependent entries are written in terms of the independent ones. A `RateLaw`
  on it takes a matrix.

The operations on vectors and tensors are:

- `vector_gradient(v)` is the matrix `partial(v_i, j)`, and `tensor_divergence(T)` the vector of the sums of
  `partial(T_ij, j)` over `j`.
- `symmetric_part(M)` and `antisymmetric_part(M)` split a matrix, so the strain rate of a velocity is
  `symmetric_part(vector_gradient(u))`.
- `T.apply_map(laplacian)` is the Laplacian of a matrix, entry by entry, as `v.apply_map(laplacian)` is that of a
  vector.
- The transport of a tensor or a vector by a flow advects it, rotates it with the vorticity, and aligns it with the
  strain rate. It is an expression of `advection`, `vector_gradient`, `symmetric_part`, and `antisymmetric_part`, and a
  law subtracts it from its rate.
- `leray_projection(v)` is the divergence-free part of a vector, `v - gradient(inverse_helmholtz(divergence(v)))`. It
  is exact on spectral fields of periodic lattices. A velocity odd about the middle of an axis, set by the `symmetry`
  of the field, gives walls there.

The `boundary` of a field takes a matrix of the symmetry of the state as the fixed value `('fixed', matrix)`, which
gives every edge of the lattice that matrix. `simulate` reports a rate law that breaks the declared symmetry with
the code `tensor-symmetry`, and a tensor operation given the wrong shape with the code `law-domain`.
