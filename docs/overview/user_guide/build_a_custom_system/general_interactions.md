# General interactions

This page finds the general interaction behind a process and configures it for a model. An interaction reads the
values of the elements it binds and sets the laws that change them. Many processes already exist as a general
interaction: you pick the class that does what you need, bind it to your elements, and pass the functions that make it
your process. Here a stock of 8 units decays at the rate 0.5, through an
[`EquationInteraction`](../core_components/general.md#equationinteraction) that gives the value `x` its rate.

```python
from skcomplex.core.components.elements import Element
from skcomplex.core.systems import System
from skcomplex.systems.general.dynamics import EquationInteraction

decay = EquationInteraction('decay', target='stock', state='x', equation=lambda s: -0.5 * s.target.x)
system = System(elements=[Element('stock', values={'x': 8.0})], interactions=[decay]).prepare()
round(float(system.simulate(until=2.0, dt=0.01).get_values('stock.x')[-1]), 3)
```

The stock holds 2.943 after two time units, which is `8 exp(-1)`.

The general interactions are the classes of `skcomplex.core.components.interactions` and the recipes that several
fields share in `skcomplex.systems.general`, each for one mathematical operation, such as a rate, a gradient, a flow, a
coupling, a jump, a transition, or a grouping. Where no recipe fits, an
[`Interaction`](../core_components/interactions.md#interaction) sets one of the seven kinds of law. Find the operation
you need in the list below, then follow the link to its subsection.

- Change a state continuously by a rate, towards a goal, or with noise:
  [`EquationInteraction`](../core_components/general.md#equationinteraction), in [Rates and relaxation](#rates-and-relaxation).
- Move states down a potential, or write the mechanics of a Lagrangian or a Hamiltonian:
  [`GradientInteraction`](../core_components/general.md#gradientinteraction),
  [`EulerLagrangeInteraction`][skcomplex.systems.general.mechanics], and
  [`HamiltonianInteraction`][skcomplex.systems.physics.common.mechanics], in [Gradients and mechanics](#gradients-and-mechanics).
- Evolve a metric by the Einstein equations, or move a particle along its geodesics: the physics classes in
  [Curved spacetimes](#curved-spacetimes).
- Conserve, carry, or spread a quantity over a field: the rate `-divergence(flux)` on a finite-volume field, in
  [Conservation laws](#conservation-laws).
- Update a state in steps, search by trial and error, or run a cellular automaton: `EquationInteraction` with
  `time='discrete'` and an [`UpdateLaw`](../core_components/laws.md#updatelaw) with a neighbourhood, in
  [Discrete time](#discrete-time).
- Couple members over the whole group or over close pairs: the reductions `total` and `count`, and the binding
  `within=RadiusNeighborhood(r)`, in [Coupling over a structure](#coupling-over-a-structure).
- Move particles through a viscous fluid by the forces, torques, and active slip of all of them: the physics
  `StokesianMobilityInteraction`, in [Coupling over a structure](#coupling-over-a-structure).
- Fire random events: a [`JumpLaw`](../core_components/laws.md#jumplaw), in [Jumps](#jumps).
- Move a member between discrete values, such as susceptible and infected: an update law with a random draw or a jump
  law, in [Transitions between values](#transitions-between-values).
- Reset states when a condition crosses zero, or at chosen times: an [`EventLaw`](../core_components/laws.md#eventlaw),
  in [Crossings and impulses](#crossings-and-impulses).
- Sample configurations by Monte Carlo: an [`ObjectiveLaw`](../core_components/laws.md#objectivelaw) moved by a
  [`MetropolisSolver`](../core_components/interactions.md#samplinginteraction), in [Monte Carlo sampling](#monte-carlo-sampling).
- Set a state from the time or the random stream of the step: an update law whose value is a function, in
  [Updates that read the step](#updates-that-read-the-step).
- Write an operation as symbolic laws or as agent rules: in [Agent and symbolic forms](#agent-and-symbolic-forms).
- Learn from payoffs: `EquationInteraction`, in [Learning](#learning).
- Move amounts between stocks, form groups, or rewire a network: rate laws whose targets map stocks to signs and
  [`FlowInteraction`](../core_components/general.md#flowinteraction), the binding `along=RandomPartition(k)`, and
  update laws of a `TopologyRequestSpec`, in [Agents, groups, and exchange](#agents-groups-and-exchange).
- Move agents, create and remove them, let them choose, imitate, and evolve, match partners, fire reactions, or give
  the encounters of a kinetic system: the recipes that several fields share, in
  [General realizations](../core_components/general.md).
- Write laws with Hill functions, delays, pulses, or group reductions: `skcomplex.core.components.laws.utils`, in
  [Law utilities](#law-utilities).
- Evolve a tensor value of a field: `Tensor` in `Annotated`, in [Tensor fields](#tensor-fields).
- Do something no general class covers: the bindings of `Interaction`, in
  [When no general class fits](#when-no-general-class-fits).

## How the general classes are configured

This section shows the arguments that turn a general class into a process of a field. A class names what it does,
never what a state means. No core class or argument names a force, a position, a price,
an infection, or a harvest. You configure a class by the roles it binds, the names of the states it reads and writes,
and the functions you pass.

- `target` is the [role](../glossary.md#role) whose states the laws change, the slot of the interaction that an
  element fills. `sources` are roles the laws only read.
- `state`, `states`, `read`, and `write` name the states, so one class serves any meaning. A force is a rate of the
  second derivative of a position.
- Functions are callables of the namespace `s` that return symbolic expressions, such as
  `rate=lambda s: -s.k * s.target.x`. Where an argument allows it, a number stands for a constant function.
- `divisor` of `GradientInteraction` divides what the gradient adds to the highest derivative, such as a mass or a
  friction.
- A variant is a string checked against the constants of its module, such as `mode='increment'`. A variant that
  carries an algorithm of its own is a frozen spec instead, such as the totalistic rule of a cellular automaton.

## What enters the core

This section tells where a component of your own belongs. A class enters the core when it is a structural base, or when
the engine handles it by type, such as a variational principle, a conservation law, or a Markov jump process, and when
it applies in principle to other domains than the one that first needs it. Every other component is a recipe written
with core components, placed by the systems that use it. The [placement
rule](../core_components/general.md#where-a-component-goes) states the test.

The case of one field is a thin subclass of a general class in `skcomplex.systems`, even when it only renames arguments.
It lives with the one system that uses it, or with the code its field shares when several systems use it. A recipe that
several fields share lives in `skcomplex.systems.general`. These subclasses double as examples of how to build a
component of your own, and the page of each system shows them. The interactions of `LotkaVolterraSystem`, for example,
live with the population models of biology. `prepare` builds the components of a system class in `system_`:

```python
from skcomplex.systems.biology.ecology import LotkaVolterraSystem

system = LotkaVolterraSystem().prepare().system_
{type(component).__name__: type(component).__module__.rpartition('.')[0] for component in system.interactions}
```

`ExponentialGrowthInteraction`, `PredationInteraction`, and `MortalityInteraction` come from
`skcomplex.systems.biology.common.populations`. Each is a short subclass of `Interaction` that sets rate laws. Growth
and mortality change one population, and predation changes both at once.

## Elements of an interaction

This section picks the element that an interaction binds. A general interaction acts on any element that has the
values it reads, so an [`Element`](../core_components/elements.md#element) with its values by name is all an
interaction needs.

```python
from skcomplex.core.components.elements import Element
from skcomplex.core.systems import System
from skcomplex.systems.general.dynamics import EquationInteraction

stock = Element('stock', values={'x': 5.0})
decay = EquationInteraction('decay', target='stock', state='x', equation=lambda s: -s.target.x)
system = System(elements=[stock], interactions=[decay]).prepare()
round(float(system.simulate(until=1.0, dt=1.0).get_values('stock.x')[-1]), 3)
```

The stock falls from 5 to 1.839, which is `5 exp(-1)`.

- A subclass of `Element` lists its values as the arguments of its `__init__`, with their allowed values in
  `Annotated`, as [Defining elements](defining_elements.md) shows.
- `Element(label, values=..., size=n)` holds `n` members, and `space=` places one member at every site of a space, as
  [Groups of many elements](groups_of_many_elements.md) and [Fields](fields.md) show.
- A value whose laws give its second derivative, such as `x.diff(t, 2)`, is the position of a point, and its velocity
  `x'` is a state too. The general realization [`PointElement`](../core_components/general.md#pointelement) is such an
  element.

## Continuous time

This section writes differential equations, with or without noise, from rates, potentials, and fluxes.

### Rates and relaxation

Here you add a rate to a state, as the decay at the top of this page does, or pull a state towards a goal, with an
`EquationInteraction`.

`EquationInteraction(target, state, equation, noise=None, convention='ito', time='continuous')` adds the rate
`equation(s)`, and the noise `noise(s) dW`, to a state, or to its derivative of order `k` when the state is named with
`k` primes, such as `"x''"`.

```python
from skcomplex.core.components.elements import Element
from skcomplex.core.systems import System
from skcomplex.systems.general.dynamics import EquationInteraction

smoothing = EquationInteraction('smoothing', target='belief', state='x', equation=lambda s: (10.0 - s.target.x) / 0.5)
system = System(elements=[Element('belief', values={'x': 0.0})], interactions=[smoothing]).prepare()
round(float(system.simulate(until=5.0, dt=0.01).get_values('belief.x')[-1]), 3)
```

The belief reaches the goal 10.0 within five time units, ten times its relaxation time of 0.5. The rate
`(goal - x) / time` relaxes a state towards a goal. With noise, it is an Ornstein-Uhlenbeck process, a random walk
pulled back towards a mean. A field that has its own term for this form names it by a subclass, such as the physics
`RelaxationTimeInteraction` or the neuroscience `LeakyIntegrationInteraction`.

### Gradients and mechanics

Here a state moves down a potential through a `GradientInteraction`. On a state of order 1 it is a gradient flow.

```python
from skcomplex.core.components.elements import Element
from skcomplex.core.systems import System
from skcomplex.systems.general.gradients import GradientInteraction

well = GradientInteraction('well', target='walker', states='x', potential=lambda s: s.target.x**2 / 2)
flow = System(elements=[Element('walker', values={'x': 1.0})], interactions=[well]).prepare()
round(float(flow.simulate(until=1.0, dt=0.01).get_values('walker.x')[-1]), 4)
```

The walker slides from 1.0 to 0.3679 in the well `x^2 / 2`, the decay `exp(-t)` of the flow `x' = -x`.

With `order=2` the same class gives Newton's equations, the rate going to the second derivative of the position,
whose first derivative the trajectory reports as `"point.x'"`.

```python continuation
spring = GradientInteraction(
    'spring', target='point', states='x', potential=lambda s: s.target.x * s.target.x / 2, order=2
)
newton = System(elements=[Element('point', values={'x': [1.0]})], interactions=[spring]).prepare()
round(float(newton.simulate(until=3.1416, dt=0.01).get_values('point.x')[-1][0]), 3)
```

The point starts at 1.0 and reaches -1.0 after half a period `pi` of the harmonic spring.

`GradientInteraction(target, states, potential, sources=(), divisor=None, variational=False, conserved=False,
order=1)` also gives the variational or conserved dynamics of a field. The divisor is a name, a function of the member, or a mapping
of one divisor per state.

An interaction states Newton's or Hamilton's equations directly as rate laws written in Sage. Here a pendulum swings
from the angle 0.1 under the law `q'' = -sin q`.

```python
from sage.all__sagemath_symbolics import sin

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import RateLaw
from skcomplex.core.systems import System
from skcomplex.core.types import Role


class Swing(Interaction):
    def __init__(self, label, *, target: Role = 'bob'):
        super().__init__(label)
        self.target = target

    def _prepare(self, s):
        super()._prepare(s)
        q = s.target.q
        self.laws_ = [RateLaw(q.diff(s.t, 2), -sin(q))]


pendulum = System(elements=[Element('bob', values={'q': 0.1})], interactions=[Swing('swing')]).prepare()
round(float(pendulum.simulate(until=3.2, dt=0.01).get_values('bob.q')[-1]), 2)
```

The angle is -0.1 at the time 3.2, close to half the small-swing period `2 pi`.

Hamilton's equations `q' = dH/dp` and `p' = -dH/dq` are two rate laws per canonical pair. Here a planet circles a sun
under the Hamiltonian `H = (px^2 + py^2) / 2 - 1 / r`.

```python
from sage.all__sagemath_symbolics import sqrt

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import RateLaw
from skcomplex.core.engine.solvers import Yoshida4Solver
from skcomplex.core.systems import System
from skcomplex.core.types import Role


class Gravity(Interaction):
    def __init__(self, label, *, target: Role = 'planet'):
        super().__init__(label)
        self.target = target

    def _prepare(self, s):
        super()._prepare(s)
        x, y, px, py, t = s.target.x, s.target.y, s.target.px, s.target.py, s.t
        r3 = sqrt(x**2 + y**2) ** 3
        self.laws_ = [
            RateLaw(x.diff(t), px),
            RateLaw(y.diff(t), py),
            RateLaw(px.diff(t), -x / r3),
            RateLaw(py.diff(t), -y / r3),
        ]


planet = Element('planet', values={'x': 1.0, 'y': 0.0, 'px': 0.0, 'py': 1.0})
system = System(elements=[planet], interactions=[Gravity('gravity')], solver=Yoshida4Solver(dt=0.01)).prepare()
round(float(system.simulate(until=6.2832, dt=1.0).get_values('planet.x')[-1]), 6)
```

The planet returns to `x = 1.0` after one period `2 pi` of its circular orbit.

The [solver](../glossary.md#solver) `Yoshida4Solver` integrates a separable Hamiltonian at fourth order by three
velocity Verlet steps. It first splits the states into two blocks whose rates read only the other block. It keeps the
energy bounded over long runs.

The shared mechanics `skcomplex.systems.general.mechanics` writes the equations of motion from a Lagrangian with its
`EulerLagrangeInteraction`, whose Lagrangian writes the velocity of a coordinate `q` as `q.diff(t)`. Here the pendulum
above swings again from the Lagrangian `q'^2 / 2 + cos q`.

```python continuation
from sage.all__sagemath_symbolics import cos

from skcomplex.systems.general.mechanics import EulerLagrangeInteraction

lagrangian = lambda s: s.target.q.diff(s.t) ** 2 / 2 + cos(s.target.q)
swing = EulerLagrangeInteraction('swing', target='bob', states='q', lagrangian=lagrangian)
pendulum = System(elements=[Element('bob', values={'q': 0.1})], interactions=[swing]).prepare()
round(float(pendulum.simulate(until=3.2, dt=0.01).get_values('bob.q')[-1]), 2)
```

The angle is -0.1 at the time 3.2, as with the rate law written by hand.

`invariants` decides whether expressions given as `Observable`s are conserved by the equations of any system.

```python continuation
from skcomplex.core.analysis import invariants
from skcomplex.core.components import Observable

momentum = Observable('angular_momentum', lambda s: s.planet.x * s.planet.py - s.planet.y * s.planet.px)
invariants(system, expressions=[momentum]).conserved
```

The angular momentum is conserved. The mechanics family of physics adds `poisson_brackets`, the brackets over the canonical
pairs, and `noether`, which checks a symmetry against the Hamiltonian or the Lagrangian and gives its Noether charge,
the quantity that a continuous symmetry conserves.

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
[`LatticeSpace`](../core_components/spaces.md#latticespace). A finite-volume field stores the mean of a cell, so the total
changes only through the boundary.

```python
import numpy as np

from skcomplex.core.components.elements import Element
from skcomplex.core.components.laws.utils import divergence, laplacian
from skcomplex.core.components.spaces import LatticeSpace
from skcomplex.core.systems import System
from skcomplex.systems.general.dynamics import EquationInteraction

space = LatticeSpace((50,), spacing=0.1, boundary='periodic', discretization='finite-volume')
values = {'u': 1.0 + np.sin(2 * np.pi * np.arange(50) / 50)}
rates = {
    'burgers': lambda s: -divergence([s.target.u**2 / 2]),
    'drift': lambda s: -divergence([1.0 * s.target.u]),
    'heat': lambda s: 0.1 * laplacian(s.target.u),
}
totals = {}
for label, rate in rates.items():
    wave = Element('wave', values=values, space=space)
    interaction = EquationInteraction(label, target='wave', state='u', equation=rate)
    system = System(elements=[wave], interactions=[interaction]).prepare()
    total = system.simulate(until=0.5, dt=0.01).get_values('wave.u')[-1].sum()
    totals[label] = round(float(total), 6)
totals
```

The total stays 50.0 under each of the three rates.

- A finite-volume field reads the rate `-divergence(flux)` of a state as a conservation law and computes it by the
  differences of the flux between cells, under the limiter and numerical flux of the `FiniteVolumeSpec` of its
  lattice, as [Fields](fields.md) shows. Spectral and finite-difference fields discretize the divergence by their own
  operators.
- A drift `a` carries a state as the flux `a u`.
- Diffusion is the rate `D laplacian(u)`, the divergence of the flux `-D gradient(u)`, which keeps the total of a
  periodic field. Physics names these forms by the interactions `AdvectionInteraction` and `DiffusivityInteraction`.

## Discrete time

This section updates states in steps, as maps, searches, cellular automata, and cascades. An `EquationInteraction`
with `time='discrete'` sets the next value of a state from the current one, and a system of next values steps once per
unit of time. Here it is the logistic map `x(t+1) = 3 x (1 - x)`.

```python
from skcomplex.core.components.elements import Element
from skcomplex.core.systems import System
from skcomplex.systems.general.dynamics import EquationInteraction

logistic = EquationInteraction(
    'map', target='value', state='x', equation=lambda s: 3 * s.target.x * (1 - s.target.x), time='discrete'
)
system = System(elements=[Element('value', values={'x': 0.5})], interactions=[logistic]).prepare()
system.simulate(until=2).get_values('value.x').tolist()
```

The values are `[0.5, 0.75, 0.5625]`. Every next value reads the values at the start of the step, so all members and
states update at once.

A state that searches for a target by trial and error, such as a price that rises while demand exceeds supply, steps
by the sign of a signal and keeps within bounds:

```python
from skcomplex.core.components.elements import Element
from skcomplex.core.components.laws.utils import clip, sign
from skcomplex.core.systems import System
from skcomplex.systems.general.dynamics import EquationInteraction

step = lambda s: clip(s.target.x * (1 + 0.5 * sign(2 - s.target.x)), None, 2.0)
search = EquationInteraction('search', target='quote', state='x', equation=step, time='discrete')
system = System(elements=[Element('quote', values={'x': 1.0})], interactions=[search]).prepare()
system.simulate(until=3).get_values('quote.x').tolist()
```

The quote rises from 1.0 to 1.5 and to 2.0, where the signal `2 - x` vanishes, and it stays there. The next value is
`x (1 + step sign(signal))` kept within `[lower, upper]`. Economics names this search `PriceAdjustmentInteraction`,
with an additive stepping and a schedule that walks the agents.

A cellular automaton updates each cell from the cells around it by a fixed rule. An
[`UpdateLaw`](../core_components/laws.md#updatelaw) with a `neighborhood` runs one, and here its
[`RadiusNeighborhood`](../core_components/spaces.md#radiusneighborhood) takes the two nearest cells. The cells are an
element on a row of five sites.

```python
import numpy as np

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import TotalisticSpec, UpdateLaw
from skcomplex.core.components.spaces import LatticeSpace, RadiusNeighborhood
from skcomplex.core.systems import System

rule, nearest = TotalisticSpec({1: 1}), RadiusNeighborhood(1, metric='manhattan')
parity = Interaction(
    'parity', roles={'target': 'cells'}, laws=lambda s: [UpdateLaw(s.target.c, rule, neighborhood=nearest)]
)
cells = Element('cells', values={'c': np.eye(5)[2]}, space=LatticeSpace(5))
system = System(elements=[cells], interactions=[parity]).prepare()
system.simulate(until=1, dt=1, random_state=0).states[-1].tolist()
```

The rule turns on a cell whose neighbourhood holds exactly one live cell. The single live cell in the middle gives
`[0.0, 1.0, 0.0, 1.0, 0.0]` after one step.

`UpdateLaw(target, rule, neighborhood=...)` runs a cellular automaton from a rule table, a totalistic rule, or a
function of the ordered neighbourhood, as [Lattice systems](lattice_systems.md) shows.

A cascade, which relaxes every member at or above a threshold until none is left, is an `UpdateLaw` whose value is
the operator [`relax`](../core_components/laws.md#relax) of the core, in
[Cascades](lattice_systems.md#cascades). Physics names the cascade of a sandpile `TopplingInteraction`, as in
`SandpileSystem` of `skcomplex.systems.physics.criticality`.

## Coupling over a structure

This section lets members influence each other: neighbours on a lattice or a network, every member of a group, or
pairs of points closer than a cutoff. Kuramoto oscillators, for example, couple every member of a group through the
mean of a sine of their phase differences. The mean over the other members is a reduction over the group: since
`sin(b - a) = sin b cos a - cos b sin a`, the sum over the group needs only the totals of `sin theta` and `cos theta`,
which `total` gives, and `count()` gives the number of members.

```python
import numpy as np
from sage.all__sagemath_symbolics import cos, sin

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import RateLaw
from skcomplex.core.components.laws.utils import count, total
from skcomplex.core.systems import System


def kuramoto(s):
    theta = s.target.theta
    pull = (total(sin(theta)) * cos(theta) - total(cos(theta)) * sin(theta)) / (count() - 1)
    return [RateLaw(theta, s.target.omega + 2.0 * pull)]


rng = np.random.default_rng(0)
values = {'theta': rng.uniform(-np.pi, np.pi, 50), 'omega': rng.normal(0.0, 0.5, 50)}
oscillators = Element('oscillators', values=values, size=50)
coupling = Interaction('kuramoto', roles={'target': 'oscillators'}, laws=kuramoto)
system = System(elements=[oscillators], interactions=[coupling]).prepare()
phases = system.simulate(until=10, dt=1.0).get_values('oscillators.theta')[-1]
round(float(np.abs(np.exp(1j * phases).mean())), 2)
```

The order parameter, the length of the mean of `exp(i theta)`, reaches 0.97 at the time 10. The phases have locked.

The reductions of `skcomplex.core.components.laws.utils` give the other couplings.

- `total`, `mean`, and `count` reduce over the whole group, and with `by=` over the members that share a value.
- `neighbors(expression, reduce=...)` reduces over the neighbours of a lattice or a network, where `s.target.other`
  reads the neighbour, as [Networks](networks.md#reductions-over-neighbourhoods) shows.
- `hyperedges(expression, size=k)` reduces over the hyperedges of a member.

An [`Interaction`](../core_components/interactions.md#interaction) with `within=RadiusNeighborhood(r)` runs its laws
once for every pair of members closer than `r`. The laws read the two members as `s.first` and `s.second`, and give the
laws of both. Here a Morse potential `2 (1 - exp(1 - r))^2` holds two atoms together. Its force on the first atom is
minus its derivative times the unit vector from the second atom, and the second atom feels the opposite force.

```python
import numpy as np
from sage.all__sagemath_symbolics import exp, sqrt

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import RateLaw
from skcomplex.core.components.spaces import ContinuousSpace, RadiusNeighborhood
from skcomplex.core.systems import System


def morse(s):
    gap = s.first.x - s.second.x
    r = sqrt(sum(item**2 for item in gap))
    force = -4 * (1 - exp(1 - r)) * exp(1 - r) * gap / r
    return [RateLaw(s.first.x.diff(s.t, 2), force), RateLaw(s.second.x.diff(s.t, 2), -force)]


plane = ContinuousSpace(n_dim=2)
pair = Element('pair', values={'x': np.array([[0.0, 0.0], [1.5, 0.0]])}, size=2, space=plane)
bond = Interaction('morse', roles={'first': 'pair', 'second': 'pair'}, within=RadiusNeighborhood(3.0), laws=morse)
system = System(elements=[pair], interactions=[bond], space=plane).prepare()
positions = system.simulate(until=5, dt=0.1).get_values('pair.x')
distances = np.linalg.norm(positions[:, 1] - positions[:, 0], axis=1)
round(float(distances.min()), 2), round(float(distances.max()), 2)
```

The distance between the atoms oscillates between 0.67 and 1.5 around the minimum of the potential at 1. The laws
give the second derivative of the positions, so their velocities `pair.x'` are states too.

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

A [`JumpLaw`](../core_components/laws.md#jumplaw) fires random events that change states by fixed amounts, here
arrivals to a queue at the rate 2.

```python
from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import JumpLaw
from skcomplex.core.systems import System

arrivals = Interaction('arrivals', roles={'target': 'queue'}, laws=lambda s: [JumpLaw(2.0, {s.target.x: 1})])
queue = System(elements=[Element('queue', values={'x': 0.0})], interactions=[arrivals]).prepare()
counts = queue.simulate(until=5, dt=1.0, random_state=0).get_values('queue.x')
all(count == round(count) for count in counts)
```

Every recorded length of the queue is a whole number.

`JumpLaw(rate, changes)` fires events at a rate that change states by signed amounts, and `JumpLaw(waiting=...,
reset=...)` fires them after drawn waiting times. A system of jump laws is a jump process, which Gillespie's method
runs.

A reaction between species at a product propensity is a recipe on these jumps and on rates, the general realization
[`ReactionInteraction`](../core_components/general.md#reactioninteraction).

### Transitions between values

A member moves between discrete values, such as states of health or of ownership, by an update law with a random draw
in discrete time, or by a jump law in continuous time. Here each of 1000 holders leaves the value 1 for the value 2
with the probability 0.2 per step. `bernoulli('leave', 0.2)` is a draw of 1 with probability 0.2 for every member at
every step.

```python
from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import JumpLaw, UpdateLaw
from skcomplex.core.components.laws.utils import bernoulli, neighbors, where
from skcomplex.core.components.spaces import LatticeSpace
from skcomplex.core.systems import System


def leave(s):
    state = s.target.state
    return [UpdateLaw(state, where(state == 1, 1 + bernoulli('leave', 0.2), state))]


holders = Element('holders', values={'state': 1}, size=1000)
system = System(elements=[holders], interactions=[Interaction('move', roles={'target': 'holders'}, laws=leave)])
values = system.prepare().simulate(until=3, random_state=0).get_values('holders.state')
round(float((values[-1] == 1).mean()), 1)
```

After three steps half the holders keep the value 1, close to `0.8^3 = 0.512`.

A rate can read the neighbours of a member, so spreading along edges with recovery takes two jump laws. Each jump law
fires for every site at its rate, which is 0 for a site its change does not apply to.

```python continuation
def spread(s):
    state = s.target.state
    infected = neighbors(where(s.target.other.state == 1, 1, 0), reduce='mean')
    return [JumpLaw(where(state == 0, 4.0 * infected, 0), {state: 1}), JumpLaw(where(state == 1, 1.0, 0), {state: -1})]


sites = Element('sites', values={'state': 1}, space=LatticeSpace((20, 20)))
system = System(elements=[sites], interactions=[Interaction('spread', roles={'target': 'sites'}, laws=spread)])
values = system.prepare().simulate(until=10, dt=1.0, random_state=0).get_values('sites.state')
values[-1].mean() > 0.3
```

More than three in ten sites are infected at the time 10. The ready-made systems of
`skcomplex.systems.biology.epidemiology` build on such laws.

### Crossings and impulses

An [`EventLaw`](../core_components/laws.md#eventlaw) acts when a guard crosses zero, or at chosen times with a guard of
the time. The example wraps a clock back to zero each time it reaches 1, and kicks it forward once at the time 1.5.

```python
from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import EventLaw, RateLaw
from skcomplex.core.systems import System


def tick(s):
    x = s.target.x
    return [
        RateLaw(x, 1.0),
        EventLaw(x - 1, {x: 0}, direction=1, label='wrap'),
        EventLaw(s.t - 1.5, {x: x + 0.25}, direction=1, label='kick'),
    ]


ticking = Interaction('ticking', roles={'target': 'clock'}, laws=tick)
system = System(elements=[Element('clock', values={'x': 0.0})], interactions=[ticking]).prepare()
events = system.simulate(until=2.2, dt=0.5).events
[(event.label, round(event.time, 3)) for event in events]
```

The events are a wrap at 1.0, the kick at 1.5, and a second wrap at 1.75, because the kick moved the clock from 0.5
to 0.75.

- `EventLaw(guard, reset=None, direction=0, terminal=False)` fires an event when a guard crosses zero. The event resets
  states, or ends the run.
- A guard `s.t - t_k` resets states at the time `t_k`, and the guard `periodic_guard(s.t, period, start, count)` of
  `skcomplex.core.components.laws.utils` at periodic times. Engineering names these resets `PulseInteraction`, and
  physics names periodic ones `KickInteraction`.

### Monte Carlo sampling

An interaction gives the objective of every member as an `ObjectiveLaw`, and a `MetropolisSolver` samples the
configurations of a lattice by its moves, here the spins of an Ising magnet at low temperature, which align.

```python
import numpy as np

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import ObjectiveLaw
from skcomplex.core.components.laws.utils import neighbors
from skcomplex.core.components.spaces import LatticeSpace
from skcomplex.core.engine.solvers import MetropolisSolver
from skcomplex.core.systems import System


def ising(s):
    spin = s.target.s
    return [ObjectiveLaw(-spin * neighbors(s.target.other.s) / s.target.T, spin)]


spins = Element('spins', values={'s': np.ones(64), 'T': 1.0}, space=LatticeSpace((8, 8)))
magnet = Interaction('ising', roles={'target': 'spins'}, laws=ising)
system = System(elements=[spins], interactions=[magnet], solver=MetropolisSolver(moves={'spins.s': 'flip'}))
abs(system.prepare().simulate(until=20, random_state=0).states[-1].mean()) > 0.9
```

The mean spin stays above 0.9 in absolute value after 20 sweeps.

`MetropolisSolver(moves={'spins.s': 'flip'})` runs Monte Carlo sweeps of the states it names. A move is accepted from
the change of the objective with probability `exp(-change)`, so an energy over a temperature samples the Boltzmann
weight at that temperature, which stays a value of the system. With the move `'cluster'`, it makes Wolff cluster flips
instead.

An interaction that scores more than local terms returns its own Monte Carlo sweep as an `ArrayUpdateLaw`. Biology
samples whole rows of sequences this way with `SequenceSamplingInteraction`, as in `RNANeutralNetworksSystem` of
`skcomplex.systems.biology.evolution`, and physics samples matrix links with `WilsonSamplingInteraction`, as in
`LatticeYangMillsSystem` of `skcomplex.systems.physics.waves`. A temperature near zero accepts a move only when the
objective does not grow.

## Updates that read the step

This section sets a state from outside, by the time of the step, with an
[`UpdateLaw`](../core_components/laws.md#updatelaw) whose value is a Python function `f(t, values)`. Here it replays an
ensemble of recorded configurations.

```python
import numpy as np
from sage.all__sagemath_symbolics import var

from skcomplex.core.components.laws import UpdateLaw

ensemble = np.zeros((38, 20, 3))
replay = UpdateLaw(var('x'), lambda t, values: ensemble[min(int(t), 37)])
replay.prepare().targets_
```

The law targets the one state `x`.

The function receives the time of the step and the current values by name. It can then set a lattice state to one
member of an ensemble per step, as `FuzzyOilDropSystem` of `skcomplex.systems.biology.systems_biology` does, and hold
the last member after the end.

## Agent and symbolic forms

This section writes one operation either as symbolic laws or as an agent rule. A symbolic law, such as the next
value of an `EquationInteraction` with `time='discrete'`, updates every member at once and runs on every solver. An
agent rule, `UpdateLaw(s.target, rule)`, is a Python function that reads and changes one agent, so it can also read
the history of the system and the agent's neighbours and cell. Choose it when your members are agents, as
[Agent-based systems](agent_based_systems.md) shows. Here three walkers step by one, first as a symbolic map and then
as a rule that also records the step each walker reached.

```python
from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import UpdateLaw
from skcomplex.core.systems import System
from skcomplex.core.types import Role
from skcomplex.systems.general.dynamics import EquationInteraction


class Step(Interaction):
    def __init__(self, label, *, target: Role = 'walkers'):
        super().__init__(label)
        self.target = target

    def _prepare(self, s):
        super()._prepare(s)
        self.laws_ = [UpdateLaw(s.target, self.step)]

    def step(self, agent, system):
        agent.x += 1
        agent.steps = agent.x


symbolic = EquationInteraction('step', target='walkers', state='x', equation=lambda s: s.target.x + 1, time='discrete')
walkers = Element('walkers', values={'x': 0.0, 'steps': 0.0}, size=3)
mapped = System(elements=[walkers], interactions=[symbolic]).prepare()
ruled = System(elements=[walkers], interactions=[Step('step')]).prepare()
mapped.simulate(until=4).get_values('walkers.x')[-1].tolist(), mapped.form_, ruled.form_
```

The map takes each walker to 4.0 after four steps, as a `'discrete map'`. The rule makes an `'agent-based system'`,
which `MesaSolver` runs and whose trajectory holds the values of its agents.

```python continuation
ruled.simulate(until=4, random_state=0).get_agents('walkers', 4).values['steps'].tolist()
```

Each of the three walkers records 4.0 steps after four steps.

- The symbolic form lowers to symbolic laws, which every solver compiles.
- A rule takes all the actions of its agent in one go, such as stepping and then recording the step it reached, so
  `steps` reads the step its walker took in the same activation.
- Since a rule may change any value, every value of a group with rules is a state, recorded in the trajectory.

## Learning

This section lets members learn from payoffs. When members keep a score per option, an `EquationInteraction` per
score moves it towards the payoff its option realized, here 1.0 for the first option and 2.0 for the second.

```python
from skcomplex.core.components.elements import Element
from skcomplex.core.systems import System
from skcomplex.systems.general.dynamics import EquationInteraction

first = lambda s: s.target.first + 0.5 * (1.0 - s.target.first)
second = lambda s: s.target.second + 0.5 * (2.0 - s.target.second)
learning = [
    EquationInteraction('first', target='learners', state='first', equation=first, time='discrete'),
    EquationInteraction('second', target='learners', state='second', equation=second, time='discrete'),
]
learners = Element('learners', values={'first': 0.0, 'second': 0.0}, size=2)
trajectory = System(elements=[learners], interactions=learning).prepare().simulate(until=10)
first, second = trajectory.get_values('learners.first')[-1][0], trajectory.get_values('learners.second')[-1][0]
round(float(first), 2), round(float(second), 2)
```

After ten steps the scores are 1.0 and 2.0, the payoffs of the two options.

Each score moves by `rate (payoff - score)`. Economics names this learning `PayoffLearningInteraction`, with the
average and cumulative rules, a chosen option, and a schedule that walks the agents.

## Agents, groups, and exchange

This section passes amounts between agents, gathers them into groups, and changes the network they live on. The
movement, the birth and death, and the matching of agents are general realizations, such as
[`MovementInteraction`](../core_components/general.md#movementinteraction).

### Transfers

A flow moves an amount from one stock to another, so that the total is kept. It is one rate law whose target maps
each stock to the sign of its change, `-1` for the stock it leaves and `+1` for the stock it enters, with a label that
names the flow. Here an upper tank drains into a lower one through a pipe.

```python
from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import RateLaw
from skcomplex.core.systems import System
from skcomplex.core.types import Role


class Pipe(Interaction):
    def __init__(self, label, *, upper: Role = 'upper', lower: Role = 'lower', k=0.5):
        super().__init__(label)
        self.upper = upper
        self.lower = lower
        self.k = k

    def _prepare(self, s):
        super()._prepare(s)
        self.laws_ = [RateLaw({s.upper.level: -1, s.lower.level: +1}, s.k * s.upper.level, label='flow')]


tanks = [Element('upper', values={'level': 4.0}), Element('lower', values={'level': 0.0})]
trajectory = System(elements=tanks, interactions=[Pipe('pipe')]).prepare().simulate(until=1.0, dt=1.0)
(trajectory.get_values('upper.level') + trajectory.get_values('lower.level')).tolist(), trajectory.get_values('pipe.flow').round(3).tolist()
```

The two levels still sum to 4.0, and the flow falls from 2.0 to 1.213 as the upper tank empties.

- The trajectory records the flow under its label, `pipe.flow`. A flow whose stocks belong to one element is recorded
  under that element instead, such as `population.births`.
- A flow from outside the system names only the stock it enters, such as `RateLaw({s.tank.level: +1}, f, label='tap')`,
  and a flow out of it only the stock it leaves.
- The [loops analysis](stocks_and_flows.md#feedback-loops) and its flow matrix read the signs of the stocks, so they
  draw every flow from the stock it leaves to the stock it enters.
- [`FlowInteraction`](../core_components/general.md#flowinteraction) writes such a flow between two stocks of one
  element from a function of its amount, and [`StockFlowSystem`](../core_components/general.md#stockflowsystem) builds a
  whole system of stocks and flows.

Amounts can also move within one group in discrete time, as in the exclusion process of particles hopping along a lane.
Every site holds at most one particle, and a particle hops to the next site with probability one half when that site is
empty. `shift_member(x, k)` reads the value of the member `k` places along the group, so each site loses the particle
that hops away and gains the one that hops in from behind.

```python
import numpy as np

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import UpdateLaw
from skcomplex.core.components.laws.utils import bernoulli, shift_member
from skcomplex.core.systems import System
from skcomplex.core.types import Role


class Hops(Interaction):
    def __init__(self, label, *, target: Role = 'lane', p=0.5):
        super().__init__(label)
        self.target = target
        self.p = p

    def _prepare(self, s):
        super()._prepare(s)
        n = s.target.occupied
        hop = n * (1 - shift_member(n, 1)) * bernoulli('hop', self.p)
        self.laws_ = [UpdateLaw(n, n - hop + shift_member(hop, -1))]


lane = Element('lane', values={'occupied': np.tile([1.0, 0.0], 10)}, size=20)
trajectory = System(elements=[lane], interactions=[Hops('hops')]).prepare().simulate(until=10, dt=1, random_state=0)
set(trajectory.get_values('lane.occupied').sum(axis=1).tolist())
```

Every recorded step holds 10.0 particles.

- A draw in an expression is one draw per member, and `shift_member(hop, -1)` reads the draw of the member behind, so
  the particle that leaves one site is the one that enters the next.
- `shift_member(x, k, boundary='fixed')` reads a line instead of a ring, whose ends read zero.

### Grouping

An [`Interaction`](../core_components/interactions.md#interaction) with `along=RandomPartition(k)` lets the members of
the group `among` meet in random groups of `k` at every step and act together, here the majority vote of Galam's
majority rule. The laws read the member they change as `s.member` and its group as `s.members`.

```python
from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction, RandomPartition
from skcomplex.core.components.laws import UpdateLaw
from skcomplex.core.components.laws.utils import mean, where
from skcomplex.core.systems import System


def majority(s):
    return [UpdateLaw(s.member.s, where(mean(s.members.s, over=s.members) > 1 / 2, 1, 0))]


voters = Element('voters', values={'s': [1.0, 1.0, 0.0]}, size=3)
galam = Interaction('galam', among='voters', along=RandomPartition(3), laws=majority)
trajectory = System(elements=[voters], interactions=[galam]).prepare().simulate(until=1, random_state=0)
trajectory.get_values('voters.s')[-1].tolist()
```

The three voters held `[1.0, 1.0, 0.0]` and all hold 1.0 after one vote.

`RandomPartition(2)` draws pairs, and `along='neighborhoods'` groups every member with the members around it in the
space, as [Networks](networks.md#random-groups-and-neighbourhoods) shows.

### Network topology

An [`UpdateLaw`](../core_components/laws.md#updatelaw) whose target is a `TopologyRequestSpec` changes the network
itself, with the probability its value gives. Here every edge of a ring of five in a
[`NetworkSpace`](../core_components/spaces.md#networkspace) is removed in one step, through an interaction that runs
along the edges.

```python
import networkx as nx

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import TopologyRequestSpec, UpdateLaw
from skcomplex.core.components.spaces import NetworkSpace
from skcomplex.core.systems import System

space = NetworkSpace(nx.cycle_graph(5))
nodes, links = Element('nodes', space=space), Element('links', space=space, on='edges')
removal = TopologyRequestSpec('remove-edge')
expiry = Interaction('expiry', among='nodes', along=True, laws=lambda s: [UpdateLaw(removal, 1.0)])
system = System(elements=[nodes, links], interactions=[expiry], space=space).prepare()
trajectory = system.simulate(until=1, random_state=0)
len(trajectory.get_network(0).sources), len(trajectory.get_network(1).sources)
```

The ring has 5 edges at the start and none after the step.

`TopologyRequestSpec(kind, edges=1, toward=None, ...)` adds or removes nodes and edges of a network, and rewires edges,
preferentially towards a weight. An interaction without roles fires it once per step, such as a node that arrives from
outside the network, as [Networks](networks.md#topology-requests) shows.

### When no general class fits

This section names the bindings of [`Interaction`](../core_components/interactions.md#interaction) for an operation
the general classes do not cover. They fix only which members the laws of an interaction run for.

- Named roles, such as `Interaction('pipe', roles={'upper': 'upper', 'lower': 'lower'}, laws=...)`, for laws over the
  elements the roles bind.
- `among` and `along=True` or a layer name, over the edges of a lattice or a network.
- `within=RadiusNeighborhood(r)`, over the pairs of members closer than `r`.
- `along='hyperedges'` or `along=('hyperedges', k)`, over the members of hyperedges.

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

from skcomplex.core.components.elements import Element, Tensor
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import RateLaw
from skcomplex.core.components.laws.utils import (
    advection,
    inverse_helmholtz,
    laplacian,
    leray_projection,
    tensor_divergence,
)
from skcomplex.core.components.spaces import LatticeSpace
from skcomplex.core.systems import System


class Film(Element):
    def __init__(self, label, *, Q: Annotated[object, Tensor('traceless')] = 0.0, space=None):
        super().__init__(label, space=space)
        self.Q = Q


def stir(s):
    Q = s.target.Q
    force = tensor_divergence(Q)
    velocity = leray_projection(force.apply_map(lambda item: inverse_helmholtz(-item)))
    return [RateLaw(Q, -Q.apply_map(lambda item: advection(item, velocity)) + Q.apply_map(laplacian) - Q)]


size = 16
spacing = 2 * np.pi / size
x, y = np.meshgrid(np.arange(size) * spacing, np.arange(size) * spacing, indexing='ij')
start = np.stack([0.1 * np.cos(x).ravel(), 0.1 * np.sin(y).ravel()], axis=1)
lattice = LatticeSpace((size, size), spacing=spacing, discretization='spectral')
film = Film('film', Q=start, space=lattice)
system = System(elements=[film], interactions=[Interaction('stir', roles={'target': 'film'}, laws=stir)]).prepare()
trajectory = system.simulate(until=0.2, dt=0.01)
bool(np.max(np.abs(trajectory.get_values('film.Q')[-1])) < np.max(np.abs(start)))
```

The largest component at the time 0.2 is smaller than at the start, so the block returns `True`.

`Tensor(symmetry)` in `Annotated` says that a value holds a rank-2 tensor of a field on a lattice of 2 or 3
dimensions. The symmetry is `'full'`, `'symmetric'`, `'antisymmetric'`, or `'traceless'`, which means symmetric and
traceless.


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
