# General interactions

Most processes you want to describe already exist as a general interaction. You pick the class that does the
mathematical operation you need, bind it to your elements, and pass the functions that make it your process.

```python
from skcomplex.core.components.elements import NonnegativeElement
from skcomplex.core.components.interactions import RateInteraction
from skcomplex.core.components.systems import System
from skcomplex.core.engine.runtime import simulate

decay = RateInteraction('decay', target='stock', state='x', rate=lambda s: -0.5 * s.target.x)
system = System(elements=[NonnegativeElement('stock', x=8.0)], interactions=[decay])
round(float(simulate(system, until=2.0, dt=0.01).values('stock.x')[-1]), 3)
```

`skcomplex.core.components.interactions` holds one general class per mathematical operation: a rate, a gradient, a
map, a coupling, a jump, a transition, a choice, a movement, a transfer, a grouping. Find the operation you need in the
list, then follow the link to its subsection.

- Change a state continuously by a rate, towards a goal, or with noise: `RateInteraction`, in
  [Rates and relaxation](#rates-and-relaxation).
- Move states down a potential, or write the mechanics of a Lagrangian or a Hamiltonian: `GradientInteraction`,
  `EulerLagrangeInteraction`, and `HamiltonianInteraction`, in [Gradients and mechanics](#gradients-and-mechanics).
- Move a particle along the geodesics of a curved metric: `PseudoRiemannianSpace` and `GeodesicInteraction`, in
  [Curved spacetimes](#curved-spacetimes). Physics evolves a metric ansatz by the Einstein equations with
  `EinsteinFieldEquationsInteraction`.
- Conserve, carry, or spread a quantity over a field: `ConservationInteraction` and `RateInteraction`, in
  [Conservation laws](#conservation-laws).
- Update a state in steps, search by trial and error, or run a cellular automaton or a cascade: `MapInteraction`,
  `LocalRuleInteraction`, and `CascadeInteraction`, in [Discrete time](#discrete-time).
- Couple members over the edges of a lattice or a network, over a group, or over close pairs: `CouplingInteraction`
  and `PairCouplingInteraction`, in [Coupling over a structure](#coupling-over-a-structure).
- Move particles through a viscous fluid by the forces, torques, and active slip of all of them: the physics
  `StokesianMobilityInteraction`, in [Coupling over a structure](#coupling-over-a-structure).
- Fire random events or chemical reactions: `JumpInteraction` and `ReactionInteraction`, in
  [Jumps and reactions](#jumps-and-reactions).
- Move a member between discrete values, such as susceptible and infected: `TransitionInteraction`, in
  [Transitions between values](#transitions-between-values).
- Give the encounters of a kinetic system: `EncounterInteraction`, in
  [Encounters of a kinetic system](#encounters-of-a-kinetic-system).
- Reset states when a condition crosses zero, or at chosen times: `CrossingInteraction`, in
  [Crossings and impulses](#crossings-and-impulses).
- Sample configurations by Monte Carlo: `SamplingInteraction`, in [Monte Carlo sampling](#monte-carlo-sampling) and
  [Row scores and zero temperature](#row-scores-and-zero-temperature).
- Set a state from the time or the random stream of the step: `ArrayUpdateLaw`, in
  [Array updates that read the step](#array-updates-that-read-the-step).
- Run an operation as symbolic laws or as agent rules: the `form` and `schedule` arguments, in
  [Agent and symbolic forms](#agent-and-symbolic-forms).
- Choose, imitate, learn, or evolve: `ChoiceInteraction`, `ImitationInteraction`, `MapInteraction`, and
  `EvolutionInteraction`, in [Choice and learning](#choice-and-learning).
- Move agents, transfer amounts, create and remove agents, form groups, match partners, or rewire a network:
  `MovementInteraction`, `TransferInteraction`, `BirthDeathInteraction`, `GroupingInteraction`,
  `MatchingInteraction`, `FeedbackInteraction`, and `TopologyInteraction`, in
  [Agents, groups, and exchange](#agents-groups-and-exchange).
- Place agents in three dimensions or in regions such as census tracts: `ContinuousSpace` and `MovementInteraction`,
  in [Agents in three dimensions and on region layers](#agents-in-three-dimensions-and-on-region-layers).
- Write laws with Hill functions, delays, pulses, or group reductions: `skcomplex.core.components.laws.utils`, in
  [Law utilities](#law-utilities).
- Evolve a tensor state of a field: `StateSpec(tensor=...)`, in [Tensor fields](#tensor-fields).
- Do something no general class covers: the binding bases, in [When no general class fits](#when-no-general-class-fits).

## How the general classes are configured

A class names what it does, never what a state means. No core class or argument names a force, a position, a price,
an infection, or a harvest. You configure a class by the roles it binds, the names of the states it reads and writes,
and the functions you pass.

- `target` is the role whose states the laws change. `sources` are roles the laws only read.
- `state`, `states`, `read`, and `write` name the states, so one class serves any meaning. A force is a rate on a
  state of order 2, and a mass is its `divisor`.
- Functions are callables of the namespace `s` that return symbolic expressions, such as
  `rate=lambda s: -s.k * s.target.x`. Where an argument allows it, a number stands for a constant function.
- `divisor` divides what a rate, a gradient, or a pair coupling adds to the highest derivative, such as a mass, a
  friction, or a capacitance. It is a named argument, not part of the function, so a backend such as LAMMPS can read
  it.
- A variant is a string checked against the constants of its module, such as `mode='increment'`. A variant that
  carries an algorithm of its own is a frozen spec instead, such as the rule of a matching.

## What enters the core

A class enters the core when it is a fundamental paradigm, such as a variational principle, a conservation law, a
Markov jump process, a reaction network, or imitation dynamics, and when it applies in principle to other domains than
the one that first needs it. How many fields use it today does not decide it.

The case of one field is a thin subclass of a general class in `skcomplex.systems`, even when it only renames
arguments. It lives with the one system that uses it, or with the code its field shares when several systems use it.
These subclasses double as examples of how to build a part of your own, and the page of each system shows them. The
interactions of `LotkaVolterraSystem`, for example, are all rate interactions. `prepare` builds the parts of a system
class in `system_`:

```python
from skcomplex.core.components.interactions import RateInteraction
from skcomplex.systems.biology.ecology import LotkaVolterraSystem

system = LotkaVolterraSystem().prepare().system_
{type(part).__name__: isinstance(part, RateInteraction) for part in system.interactions}
```

## Elements of an interaction

The general interactions act on a few general elements, so you rarely need an element class of your own.

```python
from skcomplex.core.components.elements import NonnegativeElement, PointElement
from skcomplex.core.components.spaces import ContinuousSpace
from skcomplex.core.components.systems import System
from skcomplex.core.engine.runtime import simulate

system = System(
    elements=[NonnegativeElement('stock', x=5.0), PointElement('point', x_d1=[1.0, 0.0])],
    space=ContinuousSpace(n_dim=2),
)
trajectory = simulate(system, until=2.0, dt=1.0)
trajectory.values('stock.x')[-1], trajectory.values('point.x')[-1].tolist()
```

- `Element` declares states and parameters. `GroupElement` holds many identical members as arrays, and on `'sites'` it
  places one member at every site of a space, as `FieldElement` does.
- `NonnegativeElement` holds one nonnegative state `x`, such as a stock, a species, a queue, or the density of one
  activity class of a kinetic system. A field names its own quantity by a thin subclass, such as the biology
  `CompartmentElement`.
- `PointElement` is a member with a vector state `x` of order 2 in a continuous space. Its first derivative is `x_d1`,
  so a backend such as LAMMPS reads the velocity of a point from its declared states.

## Continuous time

These classes write differential equations, with or without noise.

### Rates and relaxation

Use `RateInteraction` to add any rate to a state, as the decay at the top of this section does, or to pull a state
towards a goal.

`RateInteraction(target, state, rate, sources=(), noise=None, divisor=None, hurst=None)` adds `rate(s)`, and
`noise(s) dW`, to the highest derivative of a state of any order. A Hurst index `hurst` makes the noise fractional.

```python
from skcomplex.core.components.elements import NonnegativeElement
from skcomplex.core.components.interactions import RateInteraction
from skcomplex.core.components.systems import System
from skcomplex.core.engine.runtime import simulate

smoothing = RateInteraction('smoothing', target='belief', state='x', rate=lambda s: (10.0 - s.target.x) / 0.5)
system = System(elements=[NonnegativeElement('belief', x=0.0)], interactions=[smoothing])
round(float(simulate(system, until=5.0, dt=0.01).values('belief.x')[-1]), 3)
```

The rate `(goal - x) / time` relaxes a state towards a goal. With noise, it is an Ornstein-Uhlenbeck process. A field
that has its own term for this form names it by a subclass, such as the physics `RelaxationTimeInteraction` or the
neuroscience `LeakyIntegrationInteraction`.

### Gradients and mechanics

Use `GradientInteraction` when a state moves down a potential. On a state of order 1 it is a gradient flow:

```python
from skcomplex.core.components.elements import NonnegativeElement, PointElement
from skcomplex.core.components.interactions import GradientInteraction
from skcomplex.core.components.spaces import ContinuousSpace
from skcomplex.core.components.systems import System
from skcomplex.core.engine.runtime import simulate

well = GradientInteraction('well', target='walker', states='x', potential=lambda s: s.target.x**2 / 2)
flow = System(elements=[NonnegativeElement('walker', x=1.0)], interactions=[well])
round(float(simulate(flow, until=1.0, dt=0.01).values('walker.x')[-1]), 4)
```

On a state of order 2, such as the position of a point, the same class gives Newton's equations:

```python continuation
spring = GradientInteraction('spring', target='point', states='x', potential=lambda s: s.target.x * s.target.x / 2)
newton = System(elements=[PointElement('point', x=[1.0])], interactions=[spring], space=ContinuousSpace(n_dim=1))
round(float(simulate(newton, until=3.1416, dt=0.01).values('point.x')[-1][0]), 3)
```

`GradientInteraction(target, states, potential, sources=(), divisor=None, variational=False, conserved=False)` also
gives the variational or conserved dynamics of a field. The divisor is a name, a function of the member, or a mapping
of one divisor per state.

Use `EulerLagrangeInteraction` when you know the Lagrangian of a mechanical system, such as a pendulum:

```python
from typing import Annotated

from sage.all__sagemath_symbolics import cos

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import EulerLagrangeInteraction
from skcomplex.core.components.laws.utils import derivative
from skcomplex.core.components.systems import System
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.specs import StateSpec


class AngleElement(Element):
    def __init__(self, label, *, q: Annotated[float, StateSpec(order=2)] = 0.1, q_d1: float = 0.0):
        super().__init__(label)
        self.q = q
        self.q_d1 = q_d1


lagrangian = lambda s: derivative(s.target.q, 1) ** 2 / 2 + cos(s.target.q)
swing = EulerLagrangeInteraction('swing', target='bob', states='q', lagrangian=lagrangian)
system = System(elements=[AngleElement('bob')], interactions=[swing])
round(float(simulate(system, until=3.2, dt=0.01).values('bob.q')[-1]), 2)
```

`EulerLagrangeInteraction(target, states, lagrangian, dissipation=None, forcing=None)` writes the equations of motion
of a Lagrangian, with a Rayleigh dissipation and a generalized forcing. The equations of one coordinate from several
interactions add, so a coupling is an interaction whose Lagrangian holds only the coupling energy.

Use `HamiltonianInteraction` when you know the Hamiltonian, such as a planet around a sun:

```python
from typing import Annotated

from sage.all__sagemath_symbolics import sqrt

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import HamiltonianInteraction
from skcomplex.core.components.systems import System
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
system = System(elements=[PlanetElement('sun')], interactions=[gravity])
trajectory = simulate(system, until=6.2832, solver=Yoshida4Solver(dt=0.01), dt=1.0)
round(float(trajectory.values('sun.x')[-1]), 6)
```

`HamiltonianInteraction(target, coordinates='q', momenta='p', hamiltonian=None)` writes Hamilton's equations
`q' = dH/dp` and `p' = -dH/dq` as rate laws of canonical coordinates and momenta. These are states of order 1, and
they may be angles. The rates of one state add, so the Hamiltonians of separate interactions sum.

`Yoshida4Solver` integrates a separable Hamiltonian at fourth order by three velocity Verlet steps. It first splits the
states into two blocks whose rates read only the other block. It keeps the energy bounded over long runs.

`invariants` computes Poisson brackets over the canonical pairs. It decides whether expressions given as
`ObservableSpec`s are conserved, and checks a `SymmetrySpec` against the Hamiltonian or the Lagrangian, with its
Noether charge:

```python continuation
from skcomplex.core.analysis import invariants
from skcomplex.core.specs import ObservableSpec, SymmetrySpec

momentum = ObservableSpec('angular_momentum', lambda s: s.sun.x * s.sun.py - s.sun.y * s.sun.px)
rotation = SymmetrySpec('rotation', lambda s: {'sun.x': -s.sun.y, 'sun.y': s.sun.x})
invariants(system, expressions=[momentum]).conserved, invariants(system, kind='symmetry', symmetry=rotation).charge
```

### Curved spacetimes

Use `PseudoRiemannianSpace` and `GeodesicInteraction` when a particle falls freely in a metric you know, such as a
planet around a static mass in the Schwarzschild metric:

```python
from sage.all__sagemath_symbolics import SR, diagonal_matrix, pi, sin

from skcomplex.core.components.elements import PointElement
from skcomplex.core.components.interactions import GeodesicInteraction
from skcomplex.core.components.spaces import PseudoRiemannianSpace
from skcomplex.core.components.systems import System
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.solvers import ScipyODESolver

t, r, theta, phi = SR.var('t r theta phi')
lapse = 1 - 2 / r
metric = diagonal_matrix(SR, [-lapse, 1 / lapse, r**2, r**2 * sin(theta) ** 2])
spacetime = PseudoRiemannianSpace(metric, (t, r, theta, phi), curvature=False)
dilation = 1 / (1 - 3 / 20) ** 0.5
planet = PointElement('planet', x=[0.0, 20.0, pi / 2, 0.0], x_d1=[dilation, 0.0, 0.0, dilation / 20**1.5])
fall = GeodesicInteraction('fall', target='planet', space=spacetime)
system = System(elements=[planet], interactions=[fall], space=spacetime)
solver = ScipyODESolver(method='DOP853', rtol=1e-10, atol=1e-10)
trajectory = simulate(system, until=500.0, dt=10.0, solver=solver)
round(float(trajectory.values('planet.x')[-1][1]), 6)
```

`PseudoRiemannianSpace(metric, coordinates, curvature=True)` takes a symmetric Sage matrix and its coordinates, and
`prepare` derives through EinsteinPy the Christoffel symbols `christoffel_`, and with `curvature` the tensors
`riemann_`, `ricci_`, `ricci_scalar_`, and `einstein_`. It derives them again only when the metric, the coordinates,
or `curvature` change. The space is an unbounded continuous space with one dimension per coordinate, so a point on it
has one component per coordinate. `GeodesicInteraction(target, state='x', space)` gives that vector state of order 2
the geodesic equation `x''^mu = -Γ^mu_(alpha beta) x'^alpha x'^beta`, with the system time as the affine parameter.
The initial four-velocity decides the kind of geodesic: a norm `g_(mu nu) x'^mu x'^nu` of -1 is a massive particle on
its proper time, and 0 is a light ray. The circular orbit above, of radius `20 M`, keeps its radius.

Use the physics `EinsteinFieldEquationsInteraction` when the metric itself is the unknown, reduced by a symmetry to functions of time, such as
the scale factor `a(t)` of a flat universe of dust:

```python
import math
from typing import Annotated

from sage.all__sagemath_symbolics import SR, diagonal_matrix, function, sin

from skcomplex.core.components.elements import Element
from skcomplex.core.components.spaces import PseudoRiemannianSpace
from skcomplex.core.components.systems import System
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.specs import ParameterSpec, StateSpec
from skcomplex.systems.physics.cosmology import EinsteinFieldEquationsInteraction


class UniverseElement(Element):
    def __init__(
        self,
        label,
        *,
        a: Annotated[float, StateSpec(order=2)] = 1.0,
        a_d1: float = 1.0,
        rho: Annotated[float, ParameterSpec()] = 3 / (8 * math.pi),
    ):
        super().__init__(label)
        self.a = a
        self.a_d1 = a_d1
        self.rho = rho


t, r, theta, phi = SR.var('t r theta phi')
a = function('a')(t)
metric = diagonal_matrix(SR, [-1, a**2, a**2 * r**2, a**2 * r**2 * sin(theta) ** 2])
ansatz = PseudoRiemannianSpace(metric, (t, r, theta, phi))


def dust(s):
    return diagonal_matrix(SR, [-s.target.rho / s.target.a**3, 0, 0, 0])


expansion = EinsteinFieldEquationsInteraction(
    'expansion', target='universe', states='a', space=ansatz, components=[(1, 1)], source=dust
)
system = System(elements=[UniverseElement('universe')], interactions=[expansion])
trajectory = simulate(system, until=2.0, dt=0.1)
round(float(trajectory.values('universe.a')[-1]), 4)
```

`EinsteinFieldEquationsInteraction(target, states, space, components, source=None, coupling=8π,
cosmological_constant=0.0, time=0, axes=None)` names every function of the metric after a state or parameter of the target. It takes one component
`(mu, nu)` of the mixed equations `G^mu_nu + Λ δ^mu_nu = κ T^mu_nu` per state, and solves them for the second time
derivatives. The source returns the mixed energy-momentum tensor, here dust of density `ρ / a³`, so the scale factor
grows as `(3 t / 2 + 1)^(2/3)`. The other components, such as the Friedmann equation `G^t_t`, are constraints the
initial values satisfy. A function of a spatial coordinate too, such as the radius `R(t, r)` of the shells of a dust
ball, needs that coordinate on a lattice axis of a field, `axes={1: 0}`, and its derivatives become `partial` of the
field. An equation that mixes time and space derivatives, or keeps a coordinate, raises an error.

### Conservation laws

Use a conservation law for a quantity of a field that is only moved around, never created or lost, except by a source. The
example runs Burgers' flux, a drift, and diffusion on a periodic finite-volume field, and the total stays the same in
each case.

```python
from typing import Annotated

import numpy as np

from skcomplex.core.components.elements import Element, FieldElement
from skcomplex.core.components.interactions import ConservationInteraction, RateInteraction
from skcomplex.core.components.laws.utils import laplacian
from skcomplex.core.components.spaces import LatticeSpace
from skcomplex.core.components.systems import System
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
    system = System(elements=[field], interactions=[interaction])
    total = simulate(system, until=0.5, dt=0.01).values('wave.u')[-1].sum()
    totals[label] = round(float(total), 6)
totals
```

- `ConservationInteraction(target, state, flux, source=None, limiter='minmod', riemann='rusanov')` conserves a state of
  a finite-volume field under a flux and a source. The flux laws of one field form one system. With `riemann=None`, it
  lowers to the divergence of the flux on spectral and finite-difference fields.
- A drift `a` carries a state as the flux `a u` of a `ConservationInteraction`.
- Diffusion is the rate `D laplacian(u)` of a `RateInteraction`, the divergence of the flux `-D gradient(u)`, which keeps
  the total of a periodic field. Physics names these forms by the subclasses `AdvectionInteraction` and
  `DiffusivityInteraction`.

## Discrete time

These classes update states in steps, as maps, searches, cellular automata, and cascades. Use `MapInteraction` to set
the next value of a state from the current one, such as the logistic map:

```python
from typing import Annotated

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import MapInteraction
from skcomplex.core.components.spaces import DiscreteTime
from skcomplex.core.components.systems import System
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.specs import StateSpec


class ValueElement(Element):
    def __init__(self, label, *, x: Annotated[float, StateSpec()] = 0.5):
        super().__init__(label)
        self.x = x


logistic = MapInteraction('map', target='value', state='x', function=lambda s: 3 * s.target.x * (1 - s.target.x))
system = System(elements=[ValueElement('value')], interactions=[logistic], time=DiscreteTime())
simulate(system, until=2).values('value.x').tolist()
```

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
from skcomplex.core.components.systems import System
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.specs import StateSpec


class QuoteElement(Element):
    def __init__(self, label, *, x: Annotated[float, StateSpec()] = 1.0):
        super().__init__(label)
        self.x = x


step = lambda s: clip(s.target.x * (1 + 0.5 * sign(2 - s.target.x)), None, 2.0)
search = MapInteraction('search', target='quote', state='x', function=step)
system = System(elements=[QuoteElement('quote')], interactions=[search], time=DiscreteTime())
simulate(system, until=3).values('quote.x').tolist()
```

The next value is `x (1 + step sign(signal))` kept within `[lower, upper]`. Economics names this search
`PriceAdjustmentInteraction`, with an additive stepping and an agent form.

Use `LocalRuleInteraction` for a cellular automaton and `CascadeInteraction` for a cascade. The helper `run` below
places a row of five cells and runs one interaction on it:

```python
from typing import Annotated

import numpy as np

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.interactions import CascadeInteraction, LocalRuleInteraction
from skcomplex.core.components.spaces import BallNeighborhood, DiscreteTime, LatticeSpace
from skcomplex.core.components.systems import System
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.specs import StateSpec, TotalisticSpec


class CellElement(Element):
    def __init__(self, label, *, c: Annotated[float, StateSpec()] = 0.0):
        super().__init__(label)
        self.c = c


def run(interaction, values, until):
    cells = GroupElement('cells', element=CellElement, space=LatticeSpace(5), values={'c': values}, on='sites')
    system = System(elements=[cells], interactions=[interaction], time=DiscreteTime())
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

`LocalRuleInteraction(target, state, rule, neighborhood)` runs a cellular automaton from a rule table, a totalistic
rule, or a function of the ordered neighbourhood.

```python continuation
around = BallNeighborhood(metric='manhattan', n_dim=1, center='first')
toppling = CascadeInteraction('toppling', target='cells', state='c', threshold=2, share=1, neighborhood=around)
trajectory = run(toppling, [0.0, 1.0, 3.0, 1.0, 0.0], 1)
trajectory.states[-1].tolist(), [event.size for event in trajectory.events]
```

`CascadeInteraction(target, state, threshold, share, mode, drive=None, neighborhood=None)` first drives the members.
It then relaxes every member at or above a threshold by a reset, a redistribution, or a random share, and records each
cascade as an event.

## Coupling over a structure

Use these classes when members influence each other: neighbours on a lattice or a network, every member of a group, or
pairs of points closer than a cutoff. Kuramoto oscillators, for example, couple every member of a group through the
mean of a sine:

```python
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import sin

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.interactions import CouplingInteraction
from skcomplex.core.components.laws import RateLaw
from skcomplex.core.components.systems import System
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
system = System(elements=[oscillators], interactions=[kuramoto])
phases = simulate(system, until=10, dt=1.0).values('oscillators.theta')[-1]
round(float(np.abs(np.exp(1j * phases).mean())), 2)
```

`CouplingInteraction(target, source=None, read, write, function, reduce='sum', mode='edges', ...)` gives member `i` a
reduction over `j` of `function(s)`. Inside the function, `s.member` reads `i`, `s.other` reads `j`, and `s.edge`
reads the edge between them.

- It works over the edges of a lattice or a network, over the whole group or a separate `source` group
  (`mode='group'`), or over the next member of an ordered group.
- The reductions are sum, mean, max, min, normalized mean, and circular mean.
- The weights are an edge state, a matrix, or a function.
- It adds to a rate (`form='rate'`), or sets the next value with `combine(s, reduction)` (`form='map'`).

Use `PairCouplingInteraction` for forces between pairs of points, such as a Morse potential between two atoms:

```python
import numpy as np
from sage.all__sagemath_symbolics import exp

from skcomplex.core.components.elements import GroupElement, PointElement
from skcomplex.core.components.interactions import PairCouplingInteraction
from skcomplex.core.components.spaces import ContinuousSpace
from skcomplex.core.components.systems import System
from skcomplex.core.engine.runtime import simulate

morse = PairCouplingInteraction('morse', 'pair', 'pair', cutoff=3.0, potential=lambda s, r: 2 * (1 - exp(1 - r)) ** 2)
pair = GroupElement('pair', element=PointElement, size=2, values={'x': [[0.0, 0.0], [1.5, 0.0]]})
system = System(elements=[pair], interactions=[morse], space=ContinuousSpace(n_dim=2))
positions = simulate(system, until=5, dt=0.1).values('pair.x')
distances = np.linalg.norm(positions[:, 1] - positions[:, 0], axis=1)
round(float(distances.min()), 2), round(float(distances.max()), 2)
```

`PairCouplingInteraction(first, second, cutoff, read, write, function=None, potential=None, divisor=None,
period=None)` adds to both members of every pair within the cutoff either `function(s)`, or minus the gradient of
`potential(s, r)` at their distance `r`. With several states in `write`, the function returns one value per state.

Use the physics `StokesianMobilityInteraction` for spheres in a viscous fluid at low Reynolds number, where the force on one sphere moves all
the others through the flow it creates. Two spheres that sediment side by side fall faster than one alone:

```python
from typing import Annotated

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import vector

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.spaces import ContinuousSpace
from skcomplex.core.components.systems import System
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.specs import StateSpec
from skcomplex.systems.physics.common.particles import StokesianMobilityInteraction


class SphereElement(Element):
    def __init__(self, label, *, x: Annotated[npt.ArrayLike, StateSpec(vector=True, role='position')] = 0.0):
        super().__init__(label)
        self.x = x


def fall(positions):
    gravity = StokesianMobilityInteraction('stokes', target='spheres', force=lambda s: vector([0.0, 0.0, -1.0]))
    spheres = GroupElement('spheres', element=SphereElement, size=len(positions), values={'x': positions})
    system = System(elements=[spheres], interactions=[gravity], space=ContinuousSpace(n_dim=3))
    heights = simulate(system, until=10.0, dt=1.0).values('spheres.x')[..., 2]
    return float(heights[0, 0] - heights[-1, 0]) / 10 * 6 * np.pi


round(fall([[0.0, 0.0, 0.0]]), 3), round(fall([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]]), 3)
```

`StokesianMobilityInteraction(target, position='x', orientation=None, tensor='rotne-prager-yamakawa', form='rate',
force, torque, velocity, stresslet, dipole, potential, cutoff, root='cholesky', radius, viscosity, temperature, dt)`
moves the spheres of a group by the grand mobility `M`, the linear map from the forces and torques on all spheres to their
velocities and angular velocities. A lone sphere moves at `F / (6 pi eta a)`, and the second sphere above adds the
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

These classes change states by discrete, often random, events.

### Jumps and reactions

Use `JumpInteraction` for random events that change states by fixed amounts, such as arrivals to a queue:

```python
from skcomplex.core.components.elements import NonnegativeElement
from skcomplex.core.components.interactions import JumpInteraction, ReactionInteraction
from skcomplex.core.components.systems import System
from skcomplex.core.engine.runtime import simulate

arrivals = JumpInteraction('arrivals', target='queue', changes={'x': 1}, rate=lambda s: 2.0)
queue = System(elements=[NonnegativeElement('queue')], interactions=[arrivals])
counts = simulate(queue, until=5, dt=1.0, random_state=0).values('queue.x')
all(count == round(count) for count in counts)
```

`JumpInteraction(target, changes, rate=None, probability=None, waiting=None, reset=None)` fires stochastic events that
change states by signed amounts. They fire at a rate, with a probability per step, or after drawn waiting times.

Use `ReactionInteraction` for mass action between species, such as an epidemic written as two reactions. The total
population stays the same:

```python continuation
infection = ReactionInteraction('infection', reactants={('S', 'x'): 1, ('I', 'x'): 1}, products={('I', 'x'): 2}, k=5e-4)
recovery = ReactionInteraction('recovery', reactants={('I', 'x'): 1}, products={('R', 'x'): 1}, k=0.1)
sir = System(
    elements=[NonnegativeElement('S', x=990), NonnegativeElement('I', x=10), NonnegativeElement('R')],
    interactions=[infection, recovery],
)
round(float(simulate(sir, until=50, dt=1.0).states[-1].sum()), 6)
```

`ReactionInteraction(target=None, reactants, products, k, propensity=None, modifiers=None, mode='rate')` runs as
rates, as jumps (`mode='jump'`), or with Langevin noise. A species is a pair of a path and a state, or a state name of
the `target` alone. A reaction with a group or field target therefore reacts in every member.

### Transitions between values

Use `TransitionInteraction` when a member moves between discrete values, such as states of health or of ownership:

```python
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import NN

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.interactions import TransitionInteraction
from skcomplex.core.components.laws.utils import neighbors, where
from skcomplex.core.components.spaces import DiscreteTime, LatticeSpace
from skcomplex.core.components.systems import System
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
system = System(elements=[holders], interactions=[move], time=DiscreteTime())
values = simulate(system, until=3, random_state=0).values('holders.state')
round(float((values[-1] == 1).mean()), 1)
```

`TransitionInteraction(target, state, source, destination, rate=None, probability=None, condition=None, matrix=None,
...)` moves a discrete state from a value, a set of values, or the values a function accepts, to another value. It
fires at a rate, with a probability per step, when a condition holds, or by a transition or an emission matrix. With
`after`, it fires after a residence time.

A rate can read the neighbours of a member, so spreading along edges with recovery takes two transitions:

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
system = System(elements=[sites], interactions=[contact, recovery])
values = simulate(system, until=10, dt=1.0, random_state=0).values('sites.state')
values[-1].mean() > 0.3
```

The ready-made systems of `skcomplex.systems.biology.epidemiology` build on such a subclass.

### Encounters of a kinetic system

Use `EncounterInteraction` for the bilinear encounters of a kinetic system, whose elements are densities on activity
classes:

```python
import numpy as np

from skcomplex.core.components.elements import GroupElement, NonnegativeElement
from skcomplex.core.components.interactions import EncounterInteraction
from skcomplex.core.components.spaces import LatticeSpace
from skcomplex.core.components.systems import System
from skcomplex.core.engine.runtime import simulate

kernel = np.zeros((3, 1, 3))
kernel[:, :, 0] = 1.0
gather = EncounterInteraction(
    'gather', candidate='candidates', field='field', rate=lambda s: 1.0, kernel=lambda s: kernel
)
system = System(
    elements=[
        GroupElement('candidates', element=NonnegativeElement, size=3, on='sites', values={'x': [0.2, 0.3, 0.5]}),
        GroupElement('field', element=NonnegativeElement, size=1, values={'x': 1.0}),
    ],
    interactions=[gather],
    space=LatticeSpace(3, boundary='fixed'),
)
densities = simulate(system, until=4.0, dt=0.5).values('candidates.x')
np.round(densities[-1], 2).tolist()
```

`EncounterInteraction(candidate, field, rate, kernel, outcome)` gives encounters that are conservative, destructive,
or proliferative.

### Crossings and impulses

Use `CrossingInteraction` to act when a condition is met, or at chosen times with a guard of the time. The example
wraps a clock back to zero each time it reaches 1, and kicks it forward once at time 1.5:

```python
from typing import Annotated

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import CrossingInteraction
from skcomplex.core.components.laws import RateLaw
from skcomplex.core.components.systems import System
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
system = System(elements=[ClockElement('clock')], interactions=[wrap, kick])
events = simulate(system, until=2.2, dt=0.5).events
[(event.label, round(event.time, 3)) for event in events]
```

- `CrossingInteraction(target, guard, reset=None, direction=0, terminal=False)` fires an event when a guard crosses
  zero. The event resets states, or ends the run.
- A guard `s.t - t_k` resets states at the time `t_k`, and the guard `periodic_guard(s.t, period, start, count)` of
  `skcomplex.core.components.laws.utils` at periodic times. Engineering names these resets `PulseInteraction`, and
  physics names periodic ones `KickInteraction`.

### Monte Carlo sampling

Use `SamplingInteraction` to sample the configurations of a lattice, such as the spins of an Ising magnet at low
temperature, which align:

```python
from typing import Annotated

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.interactions import SamplingInteraction
from skcomplex.core.components.laws.utils import neighbors
from skcomplex.core.components.spaces import DiscreteTime, LatticeSpace
from skcomplex.core.components.systems import System
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.specs import StateSpec


class SpinElement(Element):
    def __init__(self, label, *, s: Annotated[float, StateSpec()] = 1.0):
        super().__init__(label)
        self.s = s


objective = lambda s: -s.target.s * neighbors(s.target.other.s)
ising = SamplingInteraction('ising', target='spins', state='s', objective=objective, intensity=1.0)
spins = GroupElement('spins', element=SpinElement, space=LatticeSpace((8, 8)), on='sites')
system = System(elements=[spins], interactions=[ising], time=DiscreteTime())
abs(simulate(system, until=20, random_state=0).states[-1].mean()) > 0.9
```

`SamplingInteraction(target, state, objective, move='flip', intensity=1.0)` runs Monte Carlo sweeps. A move is accepted
from the change of an objective with probability `exp(-intensity * change)`. With `move='cluster'`, it makes Wolff
cluster flips instead. To sample an energy at a temperature, pass the energy as the objective and one over the
temperature as the intensity.

## Row scores and zero temperature

Use a row score when the objective is a function of a whole row of a lattice, rather than a sum of local terms. The
example flips one bit of each row at a time at zero temperature, so the score of every row never grows:

```python
from typing import Annotated

import numpy as np
from sage.sets.finite_enumerated_set import FiniteEnumeratedSet

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.interactions import SamplingInteraction
from skcomplex.core.components.spaces import DiscreteTime, LatticeSpace
from skcomplex.core.components.systems import System
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.specs import RowScoreSpec, StateSpec


class BitElement(Element):
    def __init__(self, label, s: Annotated[int, StateSpec(domain=FiniteEnumeratedSet([0, 1]))] = 1):
        super().__init__(label)
        self.s = s


def flip_one(row, random):
    site = random.integers(row.size)
    row[site] = 1 - row[site]
    return row


walk = SamplingInteraction(
    'walk', target='bits', state='s', objective=RowScoreSpec(np.sum), move=flip_one, intensity=float('inf')
)
bits = GroupElement('bits', element=BitElement, space=LatticeSpace((3, 20)), on='sites')
system = System(elements=[bits], interactions=[walk], time=DiscreteTime())
scores = simulate(system, until=50, random_state=0).values('walk.score')
bool(np.all(np.diff(scores, axis=0) <= 0))
```

`SamplingInteraction` takes a `RowScoreSpec(function, proposals=1)` as its objective. The function reads the values of
one row of a state on a `LatticeSpace`, its last axis, and the parameters that the function names.

- Every unit of time proposes `proposals` moves per row through the move. Each is accepted from the change of the
  score of the row, and the rows take separate draws.
- A named move changes one random site of a row. A function move receives the row.
- A trajectory records the score of every row as `<label>.score`.

The intensity may be infinite, which is the zero temperature of the thin subclass in physics. The Metropolis rule then
accepts a move only when the objective does not grow, for local objectives and row scores alike.

## Array updates that read the step

Use an `ArrayUpdateLaw` when a state is set from outside, by the time of the step or by a random draw, such as
replaying an ensemble of recorded configurations:

```python
import numpy as np
from sage.all__sagemath_symbolics import var

from skcomplex.core.components.laws import ArrayUpdateLaw

ensemble = np.zeros((38, 20, 3))
replay = ArrayUpdateLaw(var('x'), lambda values, *, time: ensemble[min(int(time), 37)])
replay.prepare().targets_
```

`ArrayUpdateLaw` passes the time of the step to a function that names `time`, and the random generator to a function
that names `random`. A function can then set a lattice state to one member of an ensemble per step, as
`FuzzyOilDropSystem` of `skcomplex.systems.biology.systems_biology` does, and hold the last member after the end.

## Agent and symbolic forms

A class whose operation agent systems also use takes `form='symbolic'` or `form='agent'`. Choose the agent form when
your members are agents that need to read the system's history, storage, or their own neighbours and cell.

```python
from types import SimpleNamespace

from skcomplex.core.components.interactions import MapInteraction

symbolic = MapInteraction('map', target='walkers', state='x', function=lambda s: s.target.x + 1)
agent = MapInteraction('map', target='walkers', state='x', function=lambda agent, system: agent.x + 1, form='agent')
type(agent.prepare().run(SimpleNamespace()).laws_[0]).__name__, symbolic.form
```

- The symbolic form lowers to symbolic laws, which every solver compiles.
- The agent form lowers to an agent `RuleLaw` in its `phase`. Its functions take the agent and the system view,
  `f(agent, system)`, so they read the system's history and storage and the agent's neighbours and cell.
- The form is an argument, not a guess, because the engine probes laws with empty namespaces and cannot tell an agent
  group from a symbolic one.

**The member schedule.** Use `schedule='member'` when every agent should take all of its actions in one go, such as
moving and then counting its steps:

```python
from typing import Annotated

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.interactions import MapInteraction, MovementInteraction
from skcomplex.core.components.spaces import LatticeSpace
from skcomplex.core.components.systems import System
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.specs import StateSpec


class WalkerElement(Element):
    def __init__(self, label, *, steps: Annotated[float, StateSpec()] = 0.0):
        super().__init__(label)
        self.steps = steps


walk = MovementInteraction('walk', target='walkers', step='neighbor', schedule='member')
count = MapInteraction(
    'count', target='walkers', state='steps', function=lambda a, m: a.steps + 1, schedule='member', form='agent'
)
walkers = GroupElement('walkers', element=WalkerElement, size=3)
system = System(elements=[walkers], interactions=[walk, count], space=LatticeSpace((5, 5)))
simulate(system, until=4, random_state=0).agents('walkers', 4).values['steps'].tolist()
```

An agent rule may name the role whose members run it, `RuleLaw(function, members='target')`. The Mesa solver runs it
in the activation of every member of that role, after the rules of the member's element and the member rules of the
interactions listed before it. `MapInteraction`, `TransitionInteraction`, `MovementInteraction`,
`TransferInteraction`, and `BirthDeathInteraction` take `schedule='member'` for it. The movement, transfers,
transitions, and life cycle of one agent then run together as one step of that agent, in the order of the system's
interactions.

## Choice and learning

These classes let members choose between options, copy each other, learn from payoffs, and evolve strategies. Use
`ChoiceInteraction` when every member picks an option by its utility, such as members that prefer to agree with the
mean:

```python
from typing import Annotated

import numpy as np

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.interactions import ChoiceInteraction
from skcomplex.core.components.laws.utils import mean
from skcomplex.core.components.spaces import DiscreteTime
from skcomplex.core.components.systems import System
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.specs import StateSpec


class ChooserElement(Element):
    def __init__(self, label, *, option: Annotated[float, StateSpec()] = 1.0):
        super().__init__(label)
        self.option = option


utility = lambda s, option: 2.0 * option * mean(s.target.option)
choice = ChoiceInteraction('choice', target='members', state='option', options=(-1, 1), utility=utility)
members = GroupElement('members', element=ChooserElement, size=500)
system = System(elements=[members], interactions=[choice], time=DiscreteTime())
values = simulate(system, until=10, random_state=0).values('members.option')
round(float(np.mean(values[-1])), 1)
```

`ChoiceInteraction(target, state, options, utility, rule='logit', intensity=1.0)` draws an option for every member by
the logit, Metropolis, Fermi, or best rule from `utility(s, option)`, or `utility(agent, system, option)` in the agent
form.

Use `ImitationInteraction` when members copy their neighbours, such as voters that follow the local majority:

```python
from typing import Annotated

import numpy as np

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.interactions import ImitationInteraction
from skcomplex.core.components.spaces import DiscreteTime, LatticeSpace
from skcomplex.core.components.systems import System
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.specs import StateSpec


class VoterElement(Element):
    def __init__(self, label, *, state: Annotated[float, StateSpec()] = 0.0):
        super().__init__(label)
        self.state = state


initial = np.array([1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0], dtype=float)
majority = ImitationInteraction('majority', target='voters', rule='majority', form='probability')
voters = GroupElement('voters', element=VoterElement, space=LatticeSpace(12), values={'state': initial}, on='sites')
system = System(elements=[voters], interactions=[majority], time=DiscreteTime())
simulate(system, until=1, random_state=0).values('voters.state')[-1].tolist()
```

`ImitationInteraction(target, state, payoff, options, rate, rule='random', form='rate')` copies the state of a
neighbour at random, by majority, from the best, from a random better neighbour, or by the Fermi rule. It runs as
jumps, as updates with a probability per step (`form='probability'`), or as an agent rule.

When members keep a score per option and learn from payoffs, a `MapInteraction` per score moves it towards the payoff
its option realized:

```python
from typing import Annotated

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.interactions import MapInteraction
from skcomplex.core.components.spaces import DiscreteTime
from skcomplex.core.components.systems import System
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
system = System(elements=[learners], interactions=learning, time=DiscreteTime())
trajectory = simulate(system, until=10, random_state=0)
first, second = trajectory.values('learners.first')[-1][0], trajectory.values('learners.second')[-1][0]
round(float(first), 2), round(float(second), 2)
```

Each score moves by `rate (payoff - score)`. Economics names this learning `PayoffLearningInteraction`, with the
average and cumulative rules, a chosen option, and an agent form.

Use `EvolutionInteraction` for a genetic algorithm or evolutionary dynamics over strategies. In the example, the
fitness of a bit string is its number of ones, and the mean rises over the generations:

```python
from typing import Annotated

import numpy as np

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.interactions import EvolutionInteraction
from skcomplex.core.components.laws import RuleLaw
from skcomplex.core.components.spaces import DiscreteTime
from skcomplex.core.components.systems import System
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.specs import PhaseSpec, PhasesScheduleSpec, StateSpec


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
system = System(elements=[agents], interactions=[evolution], time=DiscreteTime(schedule=schedule))
trajectory = simulate(system, until=30, random_state=0)
final = np.array(trajectory.agents('agents', 30).values['strategy'].tolist())
final.sum(axis=1).mean() > initial.sum(axis=1).mean() + 4
```

`EvolutionInteraction(target, strategy, fitness, selection='tournament', replacement='generational', crossover,
mutation, ...)` replaces strategies by selection, crossover, and mutation every `period` steps.

- The selection is by tournament, roulette, proportional, or rank.
- The replacement replaces the whole generation, or replaces by Moran, death-birth, truncation, or the best mutant.

## Agents, groups, and exchange

These classes move agents, pass amounts between them, create and remove them, gather them into groups, pair them, and
change the network they live on.

### Movement

Use `MovementInteraction` to move agents. Here two walkers each take one step to a neighbouring cell:

```python
from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.interactions import MovementInteraction
from skcomplex.core.components.spaces import LatticeSpace
from skcomplex.core.components.systems import System
from skcomplex.core.engine.runtime import simulate

walkers = GroupElement('walkers', element=Element, size=2, location=[(2, 2), (2, 2)])
walk = MovementInteraction('walk', target='walkers', step='neighbor')
system = System(elements=[walkers], interactions=[walk], space=LatticeSpace((5, 5), metric='manhattan'))
trajectory = simulate(system, until=1, random_state=0)
[int(abs(cell - 2).sum()) for cell in trajectory.agents('walkers', 1).locations]
```

`MovementInteraction(target, step='neighbor', ...)` moves agents by a neighbour, Gaussian, Lévy, best-cell, or user
step, with a condition, an acceptance, exclusion, and a cost.

### Transfers

Use `TransferInteraction` to move an amount from one holder to another, so that the total is kept:

```python
from typing import Annotated

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import TransferInteraction
from skcomplex.core.components.systems import System
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.specs import StateSpec


class TankElement(Element):
    def __init__(self, label, *, level: Annotated[float, StateSpec()] = 0.0):
        super().__init__(label)
        self.level = level


flow = TransferInteraction('flow', source='upper', target='lower', state='level', amount=lambda s: s.source.level / 2)
system = System(elements=[TankElement('upper', level=4.0), TankElement('lower')], interactions=[flow])
trajectory = simulate(system, until=1.0)
round(float(trajectory.values('upper.level')[-1] + trajectory.values('lower.level')[-1]), 6)
```

`TransferInteraction(source, target, state, into=None, amount, ...)` moves amounts of states between members, the
agents of one cell, a `Source()`, and a `Sink()`. In the symbolic form these are transfer laws, which the trajectory
records as named flows. In the agent form they move between paired agents.

Transfers within one group can follow the group order, as in the exclusion process of particles hopping along a lane.
Every site holds at most one particle, and the number of particles stays the same:

```python
from typing import Annotated

import numpy as np

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.interactions import TransferInteraction
from skcomplex.core.components.laws.utils import bernoulli
from skcomplex.core.components.spaces import DiscreteTime
from skcomplex.core.components.systems import System
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
system = System(elements=[lane], interactions=[hops], time=DiscreteTime())
trajectory = simulate(system, until=10, dt=1, random_state=0)
set(trajectory.values('lane.occupied').sum(axis=1))
```

- With `offset=k`, a symbolic transfer within one group moves amounts along the edges of the group order, from every
  member to the member `k` places along it, which `namespace.target` reads.
- The `boundary` is `'periodic'` for a ring, or `'fixed'` for a line, whose last member moves nothing.
- With a `rate`, a `reverse_rate` adds jumps that move the amounts back.
- A draw in an amount is one draw per edge, read by both of its ends. The parallel update of the exclusion process
  therefore keeps one particle per site.
- A leg end `'<role>.<state>'` names a state of any role. One transfer, such as the circuit of money of a stock-flow
  consistent economy, can then move amounts between several holders, and a subclass declares the parameters its
  amounts read.

### Birth and death

Use `BirthDeathInteraction` to create and remove agents. Here a cell of size 8 divides twice, and each division splits
the size between parent and offspring:

```python
from typing import Annotated

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.interactions import BirthDeathInteraction
from skcomplex.core.components.spaces import LatticeSpace
from skcomplex.core.components.systems import System
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.specs import StateSpec


class CellElement(Element):
    def __init__(self, label, *, size: Annotated[float, StateSpec()] = 0.0):
        super().__init__(label)
        self.size = size


cells = GroupElement('cells', element=CellElement, size=1, values={'size': 8.0}, variable=True)
division = BirthDeathInteraction('division', target='cells', birth=1.0, split=('size',))
system = System(elements=[cells], interactions=[division], space=LatticeSpace((3, 3)))
simulate(system, until=2, random_state=0).agents('cells', 2).values['size'].tolist()
```

`BirthDeathInteraction(target, birth, death, rate=None, offspring=1, placement='here', inherit=(), split=(), ...)`
creates offspring by a condition, a probability, or a rate. Offspring values are inherited, given, split, or mutated.
It also removes or renews agents.

### Grouping and matching

Use `GroupingInteraction` when agents meet in small groups and act together, such as the majority vote of Galam's
majority rule:

```python
from typing import Annotated

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.interactions import GroupingInteraction
from skcomplex.core.components.systems import System
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
trajectory = simulate(System(elements=[voters], interactions=[galam]), until=1)
trajectory.agents('voters', 1).values['s'].tolist()
```

`GroupingInteraction(target, groups='pairs', act, size=2)` draws pairs, a random partition, neighbourhoods, or groups
by a state every step. It then calls `act(members, system)` on every group.

Use `MatchingInteraction` to pair requesters with providers, such as buyers and sellers in a market, and
`FeedbackInteraction` to set parameters from what happened in a round:

```python
from skcomplex.core.components.interactions import FeedbackInteraction, MatchingInteraction
from skcomplex.core.specs import QueueSpec, StableMatchingSpec

queue = MatchingInteraction('queue', requesters='buyers', providers='sellers', rule=QueueSpec())
stable = MatchingInteraction('stable', requesters='buyers', providers='sellers', rule=StableMatchingSpec())
controller = FeedbackInteraction('controller', rule=lambda system: None, phase='policy')
queue.reports, type(stable.rule).__name__, controller.phase
```

`MatchingInteraction(requesters, providers, rule, limit, quantity, budget, transfers, on_match)` pairs requesters
with providers by the algorithm its rule names, then moves the states of `transfers`. The rules are frozen specs of
`skcomplex.core.specs` that hold only the settings of their algorithm:

- `RandomSearchSpec(tries, clearing)`, the default.
- `PartnerSpec(candidates, choice, fallback, capacity, score, clearing)`.
- `PartnerSwitchSpec(sample, intensity, partner)`.
- `QueueSpec(order)`.
- `IterativeMatchingSpec(adjustment, iterations, tolerance)`.
- `ContinuousMatchingSpec(arrival)`.
- `BatchMatchingSpec()`.
- `StableMatchingSpec(requester_preference, provider_preference, clearing)`.

The matching runs each algorithm as its own method, chosen by the type of its rule, so a new algorithm is a new spec
and a method. A `FeedbackInteraction(rule)` reads observables once per round of its phase and sets parameters. The
section [Phases, markets, institutions, and learning](agent_based_systems.md#phases-markets-institutions-and-learning) runs both.

### Network topology

Use `TopologyInteraction` when the network itself changes. Here every node of a ring of five loses an edge in one
step:

```python
import networkx as nx

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.interactions import TopologyInteraction
from skcomplex.core.components.spaces import NetworkSpace
from skcomplex.core.components.systems import System
from skcomplex.core.engine.runtime import simulate

nodes = GroupElement('nodes', element=Element, on='sites', variable=True, capacity=20)
links = GroupElement('links', element=Element, on='edges', variable=True, capacity=20)
expiry = TopologyInteraction('expiry', target='nodes', action='remove-edge')
system = System(elements=[nodes, links], interactions=[expiry], space=NetworkSpace(nx.cycle_graph(5)))
trajectory = simulate(system, until=1, random_state=0)
len(trajectory.network(0).sources), len(trajectory.network(1).sources)
```

`TopologyInteraction(target=None, action='add-edge', rate=None, probability=None, toward=None, ...)` adds or removes
nodes and edges of a network, and rewires edges, preferentially towards a weight. Without a target, it adds one node
per step from outside the network.

### When no general class fits

The binding bases stay the classes to subclass for an operation the general classes do not cover:

- `Interaction` for laws of its own over any roles.
- `NeighborInteraction` over the edges of a lattice or a network.
- `PairInteraction` over the pairs within a cutoff.
- `HyperedgeInteraction` over the members of hyperedges.

The sections [Groups of many elements](groups_of_many_elements.md), [Networks](networks.md), and
[Pair interactions](groups_of_many_elements.md#pair-interactions) show them.

## Agents in three dimensions and on region layers

Agents can live in a box of three dimensions, or move between named regions such as census tracts.

```python
from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.interactions import MovementInteraction
from skcomplex.core.components.spaces import ContinuousSpace
from skcomplex.core.components.systems import System
from skcomplex.core.engine.runtime import simulate

divers = GroupElement('divers', element=Element, size=4)
swim = MovementInteraction('swim', target='divers', step='gaussian', scale=0.1)
space = ContinuousSpace(extent=(10.0, 5.0, 2.0), boundary='periodic')
system = System(elements=[divers], interactions=[swim], space=space)
trajectory = simulate(system, until=5, random_state=0)
trajectory.agents('divers', 5).locations.shape
```

`MesaSolver` runs agents in a `ContinuousSpace` of two or three sides. Placement, `MovementInteraction` steps, radius
queries, the minimum image of a periodic box, and the locations of a trajectory use every side, and
`system.dimensions` returns every side.

- The solver rejects a continuous space with any other number of sides, or without an extent, before the first frame.
- The compile report holds the issue `location-dimension` for an initial location whose length differs from the
  dimension of its space.
- `AgentsVisual` draws agents in three dimensions as `Points3D`.

To place agents in regions, give the space layers of polygons:

```python
from shapely import Point, box

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.interactions import MovementInteraction
from skcomplex.core.components.spaces import ContinuousSpace
from skcomplex.core.components.systems import System
from skcomplex.core.engine.runtime import simulate

census = [box(0.0, 0.0, 4.0, 4.0), box(5.0, 0.0, 9.0, 4.0)]
places = [box(1.0, 6.0, 3.0, 8.0)]
people = GroupElement('people', element=Element, size=3, location=[census[0], census[0], census[1]])
outing = MovementInteraction('outing', target='people', step=lambda agent, system: system.region('places', 0))
space = ContinuousSpace(extent=(10.0, 10.0), regions={'census': census, 'places': places})
system = System(elements=[people], interactions=[outing], space=space)
trajectory = simulate(system, until=1, random_state=0)
all(places[0].contains(Point(point)) for point in trajectory.agents('people', 1).locations)
```

`ContinuousSpace(regions=...)` holds layers of regions in a space of two sides. It is a mapping from the name of a
layer to a sequence of Shapely polygons in the coordinates of the space. `prepare` keeps one `STRtree` per layer in
`regions_`. For a region that is not a valid polygon, it raises `SpaceError`, which names the layer and the region.

- A polygon is a location in `GroupElement(location=...)`, `SystemView.create`, and `AgentView.move_to`. The solver
  draws a uniform point inside it from the generator of the run. It chooses a triangle of the constrained
  Delaunay triangulation of the polygon with a probability proportional to its area.
- `AgentView.regions_here(layer)` returns the indices of the regions of a layer that contain the agent.
- `SystemView.region(layer, index)` returns the polygon of a region, and `SystemView.regions(layer)` the number of
  regions of a layer.
- A `MovementInteraction` step that returns the polygon of a region moves an agent into it.
- `AgentsVisual` draws every layer once as static `LineStrips2D` under the agents.

**Real maps.** The polygons of a space are in the coordinates of the space, and the library handles no coordinate
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

`KauaiPopulationSystem` of `skcomplex.systems.social.organizations` places one agent per resident of the census
regions of the synthetic island of its package, and `Boids3dSystem` of `skcomplex.systems.biology.collective_behaviour`
flies in a periodic cube.

```python
from skcomplex.systems.biology.collective_behaviour import Boids3dSystem
from skcomplex.systems.social.organizations import KauaiPopulationSystem

system = KauaiPopulationSystem(populations=(10, 5, 0)).prepare().system_
system.elements[0].size, list(system.space.regions), Boids3dSystem(n_boids=10).prepare().system_.space.extent
```

## Law utilities

`skcomplex.core.components.laws.utils` gives the functions laws are commonly written with, and every backend lowers
them.

```python
from sage.all__sagemath_symbolics import var

from skcomplex.core.components.laws.utils import circular_mean, clip, hill, logistic, pulse_train, sign

x, t, theta = var('x t theta')
float(hill(x, 2, 3).subs(x=2)), float(logistic(x, 1, 2).subs(x=1))
[clip(x, 0, 1).subs(x=value) for value in (-2, 0.5, 3)], [sign(x).subs(x=value) for value in (-2, 0, 3)]
[float(pulse_train(t, 10, 1, count=2).subs({t: time})) for time in (10.5, 20.5)], circular_mean(theta)
```

- `where`, `either`, `both`, and `mod` select, combine conditions, and take remainders.
- `hill(x, half, n)`, `logistic(x, center, width)`, `clip(x, low, high)`, and `sign(x)` are pointwise functions.
- `minimum` and `maximum` of several expressions compare them. Of one expression, they reduce it over a group.
- `total`, `mean`, `count`, and `circular_mean` reduce over a group. With `by=`, they reduce over the members that
  share the value of a state, such as the members of one household or one cell.
- `distributed_delay(value, time, order, conserved=True)` is the material delay of stock-and-flow systems, and the
  smoothing with `conserved=False`.
- `pulse_train(t, period, width, height, start, count=None)` is a train of rectangular pulses, finite with a count.

## Tensor fields

Use a tensor state when a field carries a matrix at every site, such as the orientation of a liquid crystal or a
stress. The example stirs a symmetric traceless tensor by the flow it drives, and diffusion and decay shrink it:

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
from skcomplex.core.components.systems import System
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
trajectory = simulate(System(elements=[film]), until=0.2, dt=0.01)
bool(np.max(np.abs(trajectory.values('film.Q')[-1])) < np.max(np.abs(start)))
```

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
