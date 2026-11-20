# Laws

A law is one equation or rule by which states change, such as a rate, the next value at a step, a random jump, or an
algebraic equation. Interactions set their laws when a system prepares them, and the compiler lowers the laws of a
system to one problem that a solver runs. The kind of law decides the form of that problem: rate laws give ordinary
differential equations, update laws give a map, and jump laws give a Markov chain in continuous time. [How the
components fit](index.md) shows the families together, and the [laws page](../../../components/laws.md)
compares every law class and holds its fact sheet.

## Updates of a whole array

An `UpdateLaw` with `arrays=True`, `UpdateLaw(state, f, arrays=True)`, takes a NumPy function of the whole array. It
hands the array of the state to the function at every step and stores the array it returns. Without `arrays`, a
function value is `f(t, values)` and gives the next value of one state. Here a sandpile of 5 by 5 sites receives one grain at a random site per step.

```python
from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import UpdateLaw
from skcomplex.core.components.spaces import LatticeSpace
from skcomplex.core.systems import System


def drop(values, *, random):
    following = values.copy()
    following[tuple(random.integers(values.shape))] += 1
    return following


pile = Element('pile', values={'height': 0.0}, space=LatticeSpace((5, 5)))
rain = Interaction('rain', roles={'target': 'pile'}, laws=lambda s: [UpdateLaw(s.target.height, drop, arrays=True)])
system = System(elements=[pile], interactions=[rain]).prepare()
heights = system.simulate(until=10, random_state=0).get_lattice('pile.height')
int(heights[-1].sum()), int(heights[-1].max())
```

After ten steps the pile holds 10 grains, and the highest site holds 3 of them.

The same law mutates a genome. Each step one random site of a sequence of 30 bases takes one of the four bases at
random.

```python continuation
def mutate(values, *, random):
    following = values.copy()
    following[random.integers(values.size)] = random.integers(4)
    return following


genome = Element('genome', values={'base': 0.0}, space=LatticeSpace(30))
mutation = Interaction(
    'mutation',
    roles={'target': 'genome'},
    laws=lambda s: [UpdateLaw(s.target.base, mutate, arrays=True)],
)
system = System(elements=[genome], interactions=[mutation]).prepare()
bases = system.simulate(until=20, random_state=1).get_values('genome.base')
int((bases[-1] != 0).sum())
```

After 20 mutations, 9 sites differ from the first base. Some mutations hit a site twice or draw the base it already
holds.

**How the engine runs it.** The law needs discrete time and a lattice or a network. The lattice solver calls the
function once per step with the array of the state, shaped like the lattice or with one row per edge, and the array it
returns becomes the state. The keyword `random` is the generator of the run, so a fixed `random_state` repeats the run.

**How it combines.** It is the drive of a relaxation, `relax(..., drive=...)`, which adds grains before the lattice
relaxes, and it runs in the same step as the local rules of other states. A function that names `time` reads the time
of the step, and one that names `space` reads the space of the element.

**Limits and errors.** The law changes one whole state at once, so it cannot read the ordered neighbourhood of a site
the way a local rule does. The function takes the array as its only positional argument and everything else by
keyword. A function `f(t, values)` of two positional arguments gives instead the next value of one state.

**A field realization.** A field names the drive in an interaction class of its own, which binds the pile through a
role.

```python continuation
from types import SimpleNamespace

from skcomplex.core.types import Role


class GrainDropInteraction(Interaction):
    def __init__(self, label, *, pile: Role):
        super().__init__(label)
        self.pile = pile

    def _prepare(self, context=None):
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            self.laws_ = [UpdateLaw(context.pile.height, drop, arrays=True)]


table = Element('table', values={'height': 0.0}, space=LatticeSpace((3, 3)))
dropping = System(elements=[table], interactions=[GrainDropInteraction('rain', pile='table')]).prepare()
int(dropping.simulate(until=4, random_state=0).get_lattice('table.height')[-1].sum())
```

Four steps drop 4 grains on the table.

See the [API entry][skcomplex.core.components.laws.UpdateLaw] of `UpdateLaw`.

## relax

The function `relax` of `skcomplex.core.components.laws.utils` is the operator of a relaxation. An `UpdateLaw` whose
value is `relax(state, rules, stable, drive=...)` drives a lattice once per step and then relaxes it until it is
stable, so one step holds a whole avalanche. In the sandpile of Bak, Tang, and Wiesenfeld, a site with 4 grains or more
topples and passes one grain to each of its four neighbours, and grains that fall off the edge leave the pile.

```python
import numpy as np

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import UpdateLaw
from skcomplex.core.components.laws.utils import relax
from skcomplex.core.components.spaces import LatticeNeighborhood, LatticeSpace, RadiusNeighborhood
from skcomplex.core.systems import System

cross = RadiusNeighborhood(1, metric='manhattan', center=True)


def drop(values, *, random):
    following = values.copy()
    following[tuple(random.integers(values.shape))] += 1
    return following


def topple(z, *around):
    return z - 4 * (z >= 4) + sum(neighbour >= 4 for neighbour in around)


def sandpile(s):
    height = s.target.height
    drive = UpdateLaw(height, drop, arrays=True)
    rule = UpdateLaw(height, topple, neighborhood=cross)
    relaxed = relax(height, rule, lambda z: bool((z < 4).all()), drive=drive)
    return [UpdateLaw(height, relaxed, label='avalanche')]


def run(label, space, laws, until, dt, values):
    """Return the trajectory of one element on a space under the laws of one interaction."""
    element = Element(label, values=values, space=space)
    interaction = Interaction('cascade', roles={'target': label}, laws=laws)
    return System(elements=[element], interactions=[interaction]).prepare().simulate(until=until, dt=dt, random_state=0)


trajectory = run('pile', LatticeSpace((6, 6), boundary='fixed'), sandpile, 300, 100, {'height': 0.0})
len(trajectory.events), max(event.size for event in trajectory.events)
```

In 300 steps the pile records 91 avalanches, and the largest updates 114 sites.

The same operator runs the earthquake model of Olami, Feder, and Christensen. The drive loads every site of a fault
until the most stressed one reaches 1. A site at 1 or above slips to 0 and passes a fifth of its stress to each
neighbour.

```python continuation
def load(values, *, random):
    return values + (1.0 - values.max())


def slip(z, *around):
    return np.where(z >= 1.0, 0.0, z) + 0.2 * sum(np.where(neighbour >= 1.0, neighbour, 0.0) for neighbour in around)


def fault(s):
    stress = s.target.stress
    drive = UpdateLaw(stress, load, arrays=True)
    rule = UpdateLaw(stress, slip, neighborhood=cross)
    relaxed = relax(stress, rule, lambda z: bool((z < 1.0).all()), drive=drive)
    return [UpdateLaw(stress, relaxed, label='earthquake')]


stress = np.random.default_rng(0).random((6, 6))
quakes = run('fault', LatticeSpace((6, 6), boundary='fixed'), fault, 200, 100, {'stress': stress}).events
len(quakes), max(event.size for event in quakes)
```

Every step triggers one earthquake, so 200 steps record 200 events, and the largest updates 66 sites.

In a forest, fire spreads in one step from a burning tree to the trees beside it. A tree is 1, a burning tree 2, and
ash 0. Lightning strikes a random site, and the fire burns until no tree is alight.

```python continuation
def strike(values, *, random):
    following = values.copy()
    site = tuple(random.integers(values.shape))
    following[site] = np.where(following[site] == 1.0, 2.0, following[site])
    return following


def burn(z, *around):
    alight = sum(neighbour == 2.0 for neighbour in around) > 0
    return np.where(z == 2.0, 0.0, np.where((z == 1.0) & alight, 2.0, z))


def wildfire(s):
    tree = s.target.tree
    rule = UpdateLaw(tree, burn, neighborhood=cross)
    drive = UpdateLaw(tree, strike, arrays=True)
    relaxed = relax(tree, rule, lambda z: bool((z < 2.0).all()), drive=drive)
    return [UpdateLaw(tree, relaxed, label='fire')]


fires = run('forest', LatticeSpace((8, 8), boundary='fixed'), wildfire, 1, 1, {'tree': 1.0}).events
fires[0].area
```

The first strike burns the whole forest, so the fire reaches all 64 sites.

In neuronal avalanches, a neuron whose potential reaches 1 fires, falls to 0, and passes a half of a unit to each
neighbour. The drive raises the potential of one neuron of a line, and `activity` counts the neurons that fire.

```python continuation
line = LatticeNeighborhood(((-1,), (0,), (1,)))


def charge(values, *, random):
    return values + 0.3 * (np.arange(values.size) == random.integers(values.size))


def fire(left, value, right):
    return value * (value < 1.0) + 0.5 * (left >= 1.0) + 0.5 * (right >= 1.0)


def neurons(s):
    potential = s.target.potential
    relaxed = relax(
        potential,
        UpdateLaw(potential, fire, neighborhood=line),
        lambda v: bool((v < 1.0).all()),
        drive=UpdateLaw(potential, charge, arrays=True),
        activity=lambda v: v >= 1.0,
    )
    return [UpdateLaw(potential, relaxed, label='avalanche')]


bursts = run('neurons', LatticeSpace(20, boundary='fixed'), neurons, 500, 500, {'potential': 0.0}).events
len(bursts) > 0, all(event.size >= event.area for event in bursts)
```

The line fires in bursts, and every burst fires at least as many times as it has distinct neurons.

**How the engine runs it.** The relaxation needs discrete time and a lattice or a network. The lattice solver applies
the drive laws once, then applies the rules again and again until `stable` holds. Each cascade that changes the lattice
becomes an event in `trajectory.events`, named by the label of the update law, with its size, the number of site
updates, its duration, the number of iterations, and its area, the number of distinct sites updated.

**How it combines.** Its drive laws are updates of a whole array or local rules, `UpdateLaw`s with a neighbourhood, and
its rules are local rules, or updates of a whole array for a relaxation that sends parcels to random neighbours. The
argument `activity` counts the sites that topple, in place of the sites that change.

**Limits and errors.** The relaxation must end. A condition `stable` that never holds stops the step after
`max_iterations` iterations with a `SolverError`.

```python continuation
from skcomplex.core.errors import SolverError


def stuck(s):
    height = s.target.height
    drive = UpdateLaw(height, drop, arrays=True)
    rule = UpdateLaw(height, topple, neighborhood=cross)
    relaxed = relax(height, rule, lambda z: bool((z < 0).all()), drive=drive, max_iterations=50)
    return [UpdateLaw(height, relaxed, label='stuck')]


try:
    run('stuck', LatticeSpace((3, 3), boundary='fixed'), stuck, 1, 1, {'height': 0.0})
except SolverError as error:
    message = str(error)
message
```

The message reads "The cascade stuck is not stable after 50 iterations. Check its stability condition or raise
max_iterations." The condition `z < 0` never holds for heights, and the fix is `z < 4`.

**A field realization.** The interactions above are realizations of the sandpile, of seismology, of forest fires, and
of neuronal avalanches. A field also wraps the drive and the relaxation in one interaction, with the threshold and the
share as its arguments, as `TopplingInteraction` does for the physics packages.

## fractional

The function `fractional` of `skcomplex.core.components.laws.utils` is the Caputo derivative of a fractional order.
A `RateLaw` whose target is `fractional(x, alpha)` states `D^alpha x = f`, with the order `alpha` between 0 and 1. The
derivative weighs the whole past of the state, so a state it moves remembers its history. Charge in a dielectric,
stress in a viscoelastic material, and a drug held in tissue relax this way, slower than any exponential.

Here a state relaxes as `D^0.5 x = -k x` from 1.0. The exact solution is the Mittag-Leffler function
`E_0.5(-k t^0.5)`, which equals `erfcx(k t^0.5)` for this order.

```python
from types import SimpleNamespace

import numpy as np
from scipy.special import erfcx

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import RateLaw
from skcomplex.core.components.laws.utils import fractional
from skcomplex.core.types import Role
from skcomplex.core.systems import System


class AnomalousRelaxation(Interaction):
    def __init__(self, label, *, target: Role = 'charge', k=0.8, alpha=0.5):
        super().__init__(label)
        self.target = target
        self.k = k
        self.alpha = alpha

    def _prepare(self, context=None):
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            x = context.target.x
            self.laws_ = [RateLaw(fractional(x, self.alpha), -context.k * x)]


system = System(
    elements=[Element('charge', values={'x': 1.0})],
    interactions=[AnomalousRelaxation('relax')],
).prepare()
trajectory = system.simulate(5.0, dt=0.5)
error = np.abs(trajectory.get_values('charge.x') - erfcx(0.8 * np.sqrt(trajectory.times))).max()
system.form_, type(system.solver_).__name__, bool(error < 1e-4)
```

The system prepares fractional differential equations, picks the predictor-corrector `FractionalABMSolver`, and
follows the exact solution to within `1e-4`.

**How the engine runs it.** Sage has no fractional derivative, so `fractional` carries the order and `prepare` reads
it. The state gets the order of its derivative, and a system with such a state is solved by a Caputo solver on a fixed
grid from the start of the run: `FractionalABMSolver` by default, or `L1Solver` and `GrunwaldLetnikovSolver` given as
the system's `solver`. The order 1 is the ordinary first derivative, so a sweep over the order reaches the ordinary
equation, solved by the default solver of ordinary equations.

**How it combines.** Several rate laws on one fractional derivative add up, as for any state, and other states of the
same system keep their ordinary derivatives. The order is a number of the interaction, such as `self.alpha`, so
`set_params(relax__alpha=0.7)` followed by `prepare()` changes it.

**Limits and errors.** An order that is no number in `(0, 1]`, or a derivative of anything but a state, raises a
`ValidationError` when the law is written. Every rate law of a state takes the same order, so a state with a fractional
and an ordinary rate law raises an error naming both orders, and so does a `NoiseLaw` that targets a fractional
derivative, since no solver integrates noise of a fractional order.

```python continuation
from sage.all__sagemath_symbolics import var

from skcomplex.core.errors import ValidationError

try:
    fractional(var('x'), 1.5)
except ValidationError as raised:
    message = str(raised)
message
```

The message reads "The order of a fractional derivative is a number in (0, 1], but got 1.5. Give an order such as
`fractional(x, 0.5)`."

## EventLaw

A `EventLaw` resets states, or ends the run, when an expression of the states, its guard, crosses zero. A ball
dropped from a height of 1 metre bounces off the floor and keeps 80 percent of its speed.

```python
from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import EventLaw, RateLaw
from skcomplex.core.systems import System


def bounce(s):
    h, e = s.target.h, s.target.e
    speed = h.diff(s.t)
    return [RateLaw(h.diff(s.t, 2), -9.81), EventLaw(h, {speed: -e * speed}, direction=-1)]


ball = Element('ball', values={'h': 1.0, 'e': 0.8})
system = System(elements=[ball], interactions=[Interaction('bounce', roles={'target': 'ball'}, laws=bounce)])
trajectory = system.prepare().simulate(until=2, dt=0.01)
[round(event.time, 3) for event in trajectory.events]
```

The ball hits the floor at 0.452, 1.174, and 1.752 seconds. The first time is `sqrt(2 / 9.81)`, and each flight after
a bounce lasts 0.8 times the one before.

The same law fires a neuron. The potential of a leaky integrate-and-fire neuron rises toward the input current, and
when it reaches the threshold 1 it resets to 0.

```python continuation
def fire(s):
    v = s.target.v
    return [RateLaw(v, (s.target.current - v) / s.target.tau), EventLaw(v - 1, {v: 0}, direction=1)]


neuron = Element('neuron', values={'v': 0.0, 'current': 1.5, 'tau': 10.0})
system = System(elements=[neuron], interactions=[Interaction('fire', roles={'target': 'neuron'}, laws=fire)])
spikes = system.prepare().simulate(until=50, dt=0.1).events
[round(event.time, 2) for event in spikes]
```

The neuron fires at the times 10.99, 21.97, 32.96, and 43.94, once every `10 ln 3`.

**How the engine runs it.** The law needs continuous time. The solver watches the sign of the guard, locates the
crossing on its interpolant within its tolerance, applies every reset at once, and restarts the integration from
there. The trajectory keeps each crossing with its time and the states before and after it, and a `terminal` crossing
ends the run.

**How it combines.** It leaves the equations as they are, so it combines with `RateLaw`s and `NoiseLaw`s. An
interaction returns it with its other laws, so a threshold can read the states of several elements.

**Limits and errors.** A guard that crosses zero again right after its reset raises an error that names the chattering
event. The direction is `1` for upward crossings, `-1` for downward ones, or `0` for both, and `prepare` raises a
`ValidationError` for any other value.

```python continuation
from sage.all__sagemath_symbolics import var

from skcomplex.core.errors import ValidationError

try:
    EventLaw(var('h'), direction=2).prepare()
except ValidationError as error:
    message = str(error)
message
```

The message reads "The direction of a crossing is -1, 0, or 1, but got 2." The fix is `direction=-1` for a fall.

**A field realization.** The neuron above is a realization of neuroscience, and a thermostat of engineering switches a
heater with the same law when the temperature crosses a set point.

See its [fact sheet](../../../components/laws.md#crossinglaw)
and its [API entry][skcomplex.core.components.laws.EventLaw].

## ConstraintLaw

An `ConstraintLaw` states a relation that holds the highest time derivatives of its states linearly, as Newton's second
law holds an acceleration. The pendulum `m l θ'' = -m g sin θ` is one such relation, written as it reads.

```python
from sage.all__sagemath_symbolics import sin

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import ConstraintLaw
from skcomplex.core.systems import System


def swing(s):
    theta, m, l = s.target.theta, s.target.m, s.target.l
    return [ConstraintLaw(m * l * theta.diff(s.t, 2) == -m * 9.81 * sin(theta))]


bob = Element('bob', values={'theta': 0.1, 'm': 1.0, 'l': 1.0})
system = System(elements=[bob], interactions=[Interaction('swing', roles={'target': 'bob'}, laws=swing)])
system.prepare().simulate(until=2, dt=1.0).get_values('bob.theta').round(3).tolist()
```

The angle swings from 0.1 to -0.1 after one second and back to 0.1 after two, close to the small period
`2 pi sqrt(l / g)` of 2.006 seconds.

The same law gives the charge `q` of a series circuit of an inductor, a resistor, and a capacitor,
`L q'' + R q' + q / C = 0`.

```python continuation
def discharge(s):
    q, L, R, C = s.target.q, s.target.L, s.target.R, s.target.C
    return [ConstraintLaw(L * q.diff(s.t, 2) + R * q.diff(s.t) + q / C == 0)]


rlc = Element('rlc', values={'q': 1.0, 'L': 1.0, 'R': 0.5, 'C': 0.25})
system = System(elements=[rlc], interactions=[Interaction('discharge', roles={'target': 'rlc'}, laws=discharge)])
system.prepare().simulate(until=3, dt=1.0).get_values('rlc.q').round(3).tolist()
```

The charge oscillates and decays: 1, -0.223, -0.467, and 0.428.

**How the engine runs it.** The law needs continuous or static time. The compiler collects every equation of the
system, solves them together for the highest derivatives, adds the rate laws on the same states, and reduces the
result to first order. `ScipyODESolver` then integrates it. Equations that share a `target` add into one before the
solve.

**How it combines.** Equations of several components couple through their derivatives, as the two angles of a double
pendulum do through a mass matrix. `EulerLagrangeInteraction` and the balances of circuits return equation laws, and a
`RateLaw` on the same state adds a force.

**Limits and errors.** The relation must be linear in the highest derivatives. It is solved symbolically, so neither
side can be a function.

```python continuation
from skcomplex.core.errors import ValidationError

try:
    ConstraintLaw(lambda t, values: 0.0, 0.0).prepare()
except ValidationError as error:
    message = str(error)
message
```

The message says that equations are solved symbolically, so a side cannot be a function, and asks for a `RateLaw` of
the highest derivative with the function as its value.

**A field realization.** The circuit above is a realization of electrical engineering, and the pendulum one of
mechanics. A robot arm writes one equation per joint in the same way.

See its [fact sheet](../../../components/laws.md#equationlaw)
and its [API entry][skcomplex.core.components.laws.ConstraintLaw].

## ObjectiveLaw

An `ObjectiveLaw` gives a scalar function of states, `U(x)`, that a static system minimises over its target states. A
mass hung on a spring of stiffness `k` comes to rest where its potential energy `k x² / 2 - m g x` is least, at
`x = m g / k`.

```python
from types import SimpleNamespace

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import ConstraintLaw, ObjectiveLaw
from skcomplex.core.systems import System
from skcomplex.core.types import Role


class HangingInteraction(Interaction):
    def __init__(self, label, *, target: Role = 'mass', k=50.0, m=2.0):
        super().__init__(label)
        self.target = target
        self.k = k
        self.m = m

    def _prepare(self, context=None):
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            x = context.target.x
            self.laws_ = [ObjectiveLaw(context.k * x**2 / 2 - context.m * 9.81 * x, x)]


mass = Element('mass', values={'x': 0.0})
hanging = System(elements=[mass], interactions=[HangingInteraction('spring')]).prepare()
round(float(hanging.simulate(until=1.0).get_values('mass.x')[0]), 4)
```

The mass rests 0.3924 below the hook, which is `2 * 9.81 / 50`.

The same law allocates a budget. A household splits one unit of money between two goods to maximise the utility
`log(a) + 2 log(b)`, which is the same as minimising its negative under the budget `a + b = 1`.

```python continuation
from sage.all__sagemath_symbolics import log


def spend(s):
    utility = log(s.a.x) + 2 * log(s.b.x)
    return [ObjectiveLaw(-utility, [s.a.x, s.b.x]), ConstraintLaw(s.a.x + s.b.x == 1, target=s.b.x)]


goods = [Element('a', values={'x': 0.5}), Element('b', values={'x': 0.5})]
spending = Interaction('spending', roles={'a': 'a', 'b': 'b'}, laws=spend)
budget = System(elements=goods, interactions=[spending]).prepare().simulate(until=1.0)
round(float(budget.get_values('a.x')[0]), 4), round(float(budget.get_values('b.x')[0]), 4)
```

The household spends a third on the first good and two thirds on the second, in the ratio of the weights.

**How the engine runs it.** An objective makes its system static, so the system needs no time of its own. The
compiler adds the objectives of every component into one, and `MinimizeSolver`, the default solver of such a system,
minimises it with `scipy.optimize.minimize` and the exact gradient. A constraint the compiler solves explicitly fixes
its state from the others, and a constraint it cannot solve holds as an equality constraint of the minimisation.

**How it combines.** Objectives add up, so several interactions can each add a cost to the same states. Constraint
laws hold at the minimum.

**Limits and errors.** The objective is minimised with its exact gradient, so it must be a symbolic expression, and
it must name the states it varies.

```python continuation
from sage.all__sagemath_symbolics import var

from skcomplex.core.errors import ValidationError

try:
    ObjectiveLaw(lambda s: 0.0, var('x')).prepare()
except ValidationError as error:
    message = str(error)
message
```

The message says that an objective is minimised with its exact gradient, so its value is a symbolic expression.

**A field realization.** The hanging mass is a realization of mechanics and the household one of economics. The loss of
a statistical model, minimised over its weights, is another.

See its [fact sheet](../../../components/laws.md#objectivelaw)
and its [API entry][skcomplex.core.components.laws.ObjectiveLaw].

## JumpLaw

A `JumpLaw` is a random event that changes states by whole amounts. With `rate`, events arrive as a Poisson process of
that propensity. Here a small colony of bacteria divides at a rate of 0.5 per cell and dies at a rate of 0.4 per cell.

```python
from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import JumpLaw
from skcomplex.core.systems import System


def divide_and_die(s):
    n = s.target.n
    return [JumpLaw(s.target.b * n, {n: 1}), JumpLaw(s.target.d * n, {n: -1})]


colony = Element('colony', values={'n': 10.0, 'b': 0.5, 'd': 0.4})
life = Interaction('life', roles={'target': 'colony'}, laws=divide_and_die)
System(elements=[colony], interactions=[life]).prepare().simulate(until=10, dt=5.0, random_state=0).get_values(
    'colony.n'
).tolist()
```

This run counts 10, 6, and 25 cells. The mean grows as `10 exp(0.1 t)`, but a small colony fluctuates far from it.

The same law runs a queue. Customers arrive at a rate of 2 per minute, and the desk serves them at a rate of 3 per
minute while anyone waits.

```python continuation
from skcomplex.core.components.laws.utils import where


def arrive_and_leave(s):
    q = s.target.q
    return [JumpLaw(s.target.arrival, {q: 1}), JumpLaw(where(q > 0, s.target.service, 0), {q: -1})]


desk = Element('desk', values={'q': 0.0, 'arrival': 2.0, 'service': 3.0})
queue = Interaction('queue', roles={'target': 'desk'}, laws=arrive_and_leave)
lengths = System(elements=[desk], interactions=[queue]).prepare().simulate(until=1000, dt=1.0, random_state=0)
round(float(lengths.get_values('desk.q').mean()), 1)
```

The mean length of the queue over 1000 minutes is 2.1. Queueing theory gives 2 when customers arrive at 2/3 of the rate
of service.

**How the engine runs it.** The law needs continuous time. Jump laws give a Markov chain in continuous time, and
`GillespieSolver` draws the time of the next event and which event fires, an exact stochastic simulation. With
`waiting` in place of `rate`, each firing waits a drawn time and applies `reset`, and `RenewalSolver` runs it.

**How it combines.** `RateLaw`s and `NoiseLaw`s move the states between jumps. An interaction that binds several
elements returns jump laws that change the states of all of them at once, as a reaction does.

**Limits and errors.** The changes are fixed numbers or functions of the states. A symbol in a change would make its
size unknown before the run, and `prepare` raises a `ValidationError`.

```python continuation
from sage.all__sagemath_symbolics import var

from skcomplex.core.errors import ValidationError

n, k = var('n k')
try:
    JumpLaw(n, {n: k}).prepare()
except ValidationError as error:
    message = str(error)
message
```

The message says that the jump of `n` has the increment `k`, which holds a symbol, and asks for a number or a function
`f(t, values)`.

**A field realization.** The colony above is a realization of population biology, and the queue one of operations
research. Chemistry writes each reaction as a jump law whose changes are the stoichiometric numbers.

See its [fact sheet](../../../components/laws.md#jumplaw)
and its [API entry][skcomplex.core.components.laws.JumpLaw].

## NoiseLaw

A `NoiseLaw` adds the amplitude of a random drive, `value dW`, to the highest derivative of a state, where `dW` is the
increment of a Wiener process, the random walk of continuous time. With a `RateLaw` it forms a stochastic differential
equation. A stock price that follows a geometric Brownian motion has the drift `0.05 S` and the amplitude `0.2 S`.

```python
import numpy as np

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import NoiseLaw, RateLaw
from skcomplex.core.systems import System


def brownian_motion(s):
    price = s.target.price
    return [RateLaw(price.diff(s.t), s.target.mu * price), NoiseLaw(price, s.target.sigma * price)]


stocks = Element('stocks', values={'price': 100.0, 'mu': 0.05, 'sigma': 0.2}, size=2000)
market = Interaction('market', roles={'target': 'stocks'}, laws=brownian_motion)
system = System(elements=[stocks], interactions=[market]).prepare()
prices = system.simulate(until=1, dt=0.01, random_state=0).get_values('stocks.price')
round(float(prices[-1].mean()), 1), round(float(np.log(prices[-1] / 100).std()), 2)
```

Over 2000 paths the mean price after one year is 105, which is `100 exp(0.05)`, and the standard deviation of the log
return is 0.2, the volatility.

The same law moves a particle that diffuses in a fluid. With the amplitude `sqrt(2 D)` alone, the variance of the
position grows as `2 D t`.

```python continuation
particles = Element('particles', values={'x': 0.0, 'D': 0.5}, size=2000)
diffusion = Interaction(
    'diffusion',
    roles={'target': 'particles'},
    laws=lambda s: [NoiseLaw(s.target.x, (2 * s.target.D) ** 0.5)],
)
system = System(elements=[particles], interactions=[diffusion]).prepare()
positions = system.simulate(until=2, dt=0.1, random_state=0).get_values('particles.x')
round(float(positions[-1].var()), 1)
```

The variance at time 2 is 2.0, which is `2 D t` for `D = 0.5`.

**How the engine runs it.** The law needs continuous time. The compiler builds the stochastic form, and
`EulerMaruyamaSolver` draws one increment per process and step. The `convention`, Itô, Stratonovich, or anti-Itô, adds
its drift correction at compile time, so every solver integrates the Itô form.

**How it combines.** It adds to the `RateLaw`s of the same state. Every member and component gets its own process,
unless laws name the same `process`, which correlates them. A `conserved` noise on a lattice field adds the divergence
of a random flux, and `hurst` makes the process a fractional Brownian motion.

**Limits and errors.** A Hurst index lies strictly between 0 and 1, and an index of 1/2 gives the Wiener process.

```python continuation
from sage.all__sagemath_symbolics import var

from skcomplex.core.errors import ValidationError

try:
    NoiseLaw(var('x'), 1.0, hurst=1.0).prepare()
except ValidationError as error:
    message = str(error)
message
```

The message reads "The Hurst index of the noise of x is 1.0. Give a number in (0, 1)."

**A field realization.** The stock above is a realization of finance, and the particle one of physics. Biology writes
the molecular noise of gene expression with the same law.

See its [fact sheet](../../../components/laws.md#noiselaw)
and its [API entry][skcomplex.core.components.laws.NoiseLaw].

## RateLaw

A `RateLaw` adds a rate to a time derivative of a state, so that a system with only rate laws is a system of ordinary
differential equations. A reactant consumed by a reaction of second order has the rate `-k c^2`.

```python
import numpy as np

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import RateLaw
from skcomplex.core.systems import System

reactant = Element('reactant', values={'c': 1.0, 'k': 0.5})
reaction = Interaction(
    'reaction',
    roles={'target': 'reactant'},
    laws=lambda s: [RateLaw(s.target.c.diff(s.t), -s.target.k * s.target.c**2)],
)
system = System(elements=[reactant], interactions=[reaction]).prepare()
system.simulate(until=4, dt=2.0).get_values('reactant.c').round(3).tolist()
```

The concentration falls to 0.5 and 0.333, which is `1 / (1 + 0.5 t)`.

The same law gives the acceleration of a mass on a spring. On the second derivative the rate is an acceleration, so
`-k x` is Hooke's law for a unit mass.

```python continuation
spring = Element('spring', values={'x': 1.0, 'k': 4.0})
hooke = Interaction(
    'hooke',
    roles={'target': 'spring'},
    laws=lambda s: [RateLaw(s.target.x.diff(s.t, 2), -s.target.k * s.target.x)],
)
system = System(elements=[spring], interactions=[hooke]).prepare()
(system.simulate(until=np.pi, dt=np.pi / 4).get_values('spring.x').round(3) + 0.0).tolist()
```

The position reads 1, 0, -1, 0, and 1 over one period, which is `cos(2 t)`.

**How the engine runs it.** The law needs continuous time. Rates on the same state add, and the compiler writes a
state whose laws reach its derivative of order `k` as `k` states of order 1. `ScipyODESolver` integrates the result by
default, and fixed step solvers such as `RK4Solver` take the same problem.

**How it combines.** Every mechanism adds its own rate law, so a growth and a predation superpose on one state.
Interactions of one element, of neighbours, and of pairs return rate laws. A `NoiseLaw` adds a random part.

**Limits and errors.** A rate is deterministic, so a random draw in it raises an `IncompatibleLawsError`. NumPy builds
an array when a NumPy scalar multiplies a symbolic vector, and `prepare` rejects that array.

```python continuation
from sage.all__sagemath_symbolics import var, vector

from skcomplex.core.errors import ValidationError

x, y = var('x y')
try:
    RateLaw(vector([x, y]), np.float64(0.5) * vector([-y, x])).prepare()
except ValidationError as error:
    message = str(error)
message
```

The message says that the value of the rate law is a NumPy array and asks to convert the NumPy scalar with
`float(...)` first. The fix is `float(np.float64(0.5)) * vector([-y, x])`.

**A field realization.** A field names a rate in an interaction class of its own, such as the recovery of an epidemic.

```python continuation
from types import SimpleNamespace

from skcomplex.core.types import Role


class RecoveryInteraction(Interaction):
    def __init__(self, label, *, infected: Role, gamma: float = 0.1):
        super().__init__(label)
        self.infected = infected
        self.gamma = gamma

    def _prepare(self, context=None):
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            x = context.infected.x
            self.laws_ = [RateLaw(x.diff(context.t), -context.gamma * x)]


infected = Element('infected', values={'x': 100.0})
epidemic = System(elements=[infected], interactions=[RecoveryInteraction('recovery', infected='infected')])
epidemic.prepare().simulate(until=10, dt=10.0).get_values('infected.x').round(1).tolist()
```

The infected fall from 100 to 36.8 in ten days.

See its [fact sheet](../../../components/laws.md#ratelaw)
and its [API entry][skcomplex.core.components.laws.RateLaw].

## UpdateLaw

An `UpdateLaw` gives the next value of a state after one step of discrete time. The logistic map follows a population
of insects from year to year, `x' = r x (1 - x)`.

```python
from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import UpdateLaw
from skcomplex.core.systems import System

insects = Element('insects', values={'x': 0.2, 'r': 2.5})
breeding = Interaction(
    'breeding',
    roles={'target': 'insects'},
    laws=lambda s: [UpdateLaw(s.target.x, s.target.r * s.target.x * (1 - s.target.x))],
)
system = System(elements=[insects], interactions=[breeding]).prepare()
system.simulate(until=30).get_values('insects.x')[-3:].round(3).tolist()
```

At `r = 2.5` the population settles at the fixed point `1 - 1 / r = 0.6`.

The same law compounds interest monthly. A balance of 1000 at a yearly rate of 6 percent grows by half a percent each
month.

```python continuation
account = Element('account', values={'balance': 1000.0, 'rate': 0.06})
interest = Interaction(
    'interest',
    roles={'target': 'account'},
    laws=lambda s: [UpdateLaw(s.target.balance, s.target.balance * (1 + s.target.rate / 12))],
)
system = System(elements=[account], interactions=[interest]).prepare()
round(float(system.simulate(until=12).get_values('account.balance')[-1]), 2)
```

After twelve months the balance is 1061.68, which is `1000 * 1.005^12`.

**How the engine runs it.** The law needs discrete time. `IterationSolver` computes every update from the states at
the start of the step, so the order of the laws does not matter. With `order`, the members of a group update one after
the other, and with `phase`, the law runs in a phase of the step. With `neighborhood`, the law is a local rule of the
cells of a lattice or a network, such as `UpdateLaw(s.target.alive, life, neighborhood=MOORE)` for the Game of Life,
which the lattice solver runs with lookup tables and fused loops.

**How it combines.** A state takes one update law, unless each of its laws says how it combines: an `increment` law
adds a change, and a law with a `source` changes only members that hold that value. Update laws take random draws,
such as `normal`.

**Limits and errors.** An increment changes every value, so it takes no source, and `prepare` raises a
`ValidationError` for both.

```python continuation
from sage.all__sagemath_symbolics import var

from skcomplex.core.errors import ValidationError

x = var('x')
try:
    UpdateLaw(x, x + 1, increment=True, source=0.0).prepare()
except ValidationError as error:
    message = str(error)
message
```

The message says that the update law is an increment with the source 0.0 and asks to give it no source.

**A field realization.** The insects above are a realization of ecology, and the account one of finance.

See its [fact sheet](../../../components/laws.md#updatelaw)
and its [API entry][skcomplex.core.components.laws.UpdateLaw].

## BaseLaw

`BaseLaw` is the abstract base of every law. A new kind of dynamics that needs a lowering of its own derives from it,
and every other model chooses an existing law, such as `RateLaw` or `UpdateLaw`. Its public method `prepare` sets the
states a law changes and the symbols it reads.

```python
from sage.all__sagemath_symbolics import var

from skcomplex.core.components.laws import BaseLaw, RateLaw, UpdateLaw

x, k = var('x k')
law = RateLaw(x, -k * x).prepare('decay')
issubclass(UpdateLaw, BaseLaw), law.targets_, sorted(map(str, law.reads_))
```

Every law derives from `BaseLaw`. The rate law above changes `x` and reads `k` and `x`.

See its [fact sheet](../../../components/laws.md#baselaw)
and its [API entry][skcomplex.core.components.laws.BaseLaw].
