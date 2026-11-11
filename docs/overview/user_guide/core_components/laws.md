# Laws

A law is one equation or rule by which states change, such as a rate, the next value at a step, a random jump, or an
algebraic equation. Elements and interactions return laws from their hook `_get_laws`, and the compiler lowers the laws
of a system to one problem that a solver runs. The kind of law decides the form of that problem: rate laws give ordinary
differential equations, update laws give a map, and jump laws give a Markov chain in continuous time. [How the
components fit](index.md) shows the families together, and the [laws page](../../../components/laws.md)
compares every law class and holds its fact sheet.

## ArrayUpdateLaw

An `ArrayUpdateLaw` hands the whole array of a state to a Python function at every step and stores the array it
returns. Here a sandpile of 5 by 5 sites receives one grain at a random site per step.

```python
from typing import Annotated

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.laws import ArrayUpdateLaw
from skcomplex.core.components.spaces import LatticeSpace
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.specs import StateSpec


def drop(values, *, random):
    following = values.copy()
    following[tuple(random.integers(values.shape))] += 1
    return following


class SandElement(Element):
    law_classes = (ArrayUpdateLaw,)

    def __init__(self, label, visual=None, *, height: Annotated[float, StateSpec()] = 0.0):
        super().__init__(label, visual=visual)
        self.height = height

    def _get_laws(self, namespace):
        return [ArrayUpdateLaw(namespace.height, drop)]


pile = GroupElement('pile', element=SandElement, space=LatticeSpace((5, 5)), on='sites')
heights = simulate(CompositeSystem(elements=[pile]), until=10, random_state=0).get_lattice('pile.height')
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


class NucleotideElement(Element):
    law_classes = (ArrayUpdateLaw,)

    def __init__(self, label, visual=None, *, base: Annotated[float, StateSpec()] = 0.0):
        super().__init__(label, visual=visual)
        self.base = base

    def _get_laws(self, namespace):
        return [ArrayUpdateLaw(namespace.base, mutate)]


genome = GroupElement('genome', element=NucleotideElement, space=LatticeSpace(30), on='sites')
bases = simulate(CompositeSystem(elements=[genome]), until=20, random_state=1).get_values('genome.base')
int((bases[-1] != 0).sum())
```

After 20 mutations, 9 sites differ from the first base. Some mutations hit a site twice or draw the base it already
holds.

**How the engine runs it.** The law needs discrete time and a lattice or a network. The lattice solver calls the
function once per step with the array of the state, shaped like the lattice or with one row per edge, and the array it
returns becomes the state. The keyword `random` is the generator of the run, so a fixed `random_state` repeats the run.

**How it combines.** It is the drive of a `CascadeLaw`, which adds grains before the lattice relaxes, and it runs in
the same step as `LocalRuleLaw`s on other states. A function that names `time` reads the time of the step, and one
that names `space` reads the space of the group.

**Limits and errors.** The law changes one whole state at once, so it cannot read the ordered neighbourhood of a site
the way a local rule does. The function must be callable, and `prepare` raises a `ValidationError` otherwise.

```python continuation
from sage.all__sagemath_symbolics import var

from skcomplex.core.errors import ValidationError

try:
    ArrayUpdateLaw(var('height'), 1.0).prepare()
except ValidationError as error:
    message = str(error)
message
```

The message reads "The function of an array update must be callable, but it is 1.0." The fix is a function of the
array, such as `drop` above.

**A field realization.** A field names the drive in an interaction of its own, which binds the pile through a role.

```python continuation
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.specs import RoleSpec


class GrainDropInteraction(Interaction):
    law_classes = (ArrayUpdateLaw,)

    def __init__(self, label, *, pile: Annotated[str, RoleSpec()]):
        super().__init__(label)
        self.pile = pile

    def _get_laws(self, namespace):
        return [ArrayUpdateLaw(namespace.pile.height, drop)]


class HeightElement(Element):
    def __init__(self, label, visual=None, *, height: Annotated[float, StateSpec()] = 0.0):
        super().__init__(label, visual=visual)
        self.height = height


table = GroupElement('table', element=HeightElement, space=LatticeSpace((3, 3)), on='sites')
dropping = CompositeSystem(elements=[table], interactions=[GrainDropInteraction('rain', pile='table')])
int(simulate(dropping, until=4, random_state=0).get_lattice('table.height')[-1].sum())
```

Four steps drop 4 grains on the table.

See its [fact sheet](../../../components/laws.md#arrayupdatelaw)
and its [API entry][skcomplex.core.components.laws.ArrayUpdateLaw].

## BilinearLaw

A `BilinearLaw` changes a distribution at the rate of meetings between the members of two distributions. A
distribution here is a group with one member per class of a trait, such as the classes of wealth, and each member holds
the amount of the population in its class. Immune cells that meet tumour cells destroy them at the rate `0.5 f g`,
where `f` and `g` are the two densities.

```python
from typing import Annotated

import numpy as np

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import BilinearLaw
from skcomplex.core.components.spaces import LatticeSpace
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.specs import ParameterSpec, RoleSpec, StateSpec


class DensityElement(Element):
    def __init__(self, label, visual=None, *, f: Annotated[float, StateSpec()] = 0.0):
        super().__init__(label, visual=visual)
        self.f = f


class KillingInteraction(Interaction):
    law_classes = (BilinearLaw,)

    def __init__(
        self,
        label,
        *,
        prey: Annotated[str, RoleSpec()],
        hunter: Annotated[str, RoleSpec()],
        k: Annotated[float, ParameterSpec()] = 0.5,
    ):
        super().__init__(label)
        self.prey = prey
        self.hunter = hunter
        self.k = k

    def _get_laws(self, namespace):
        return [BilinearLaw(namespace.prey.f, namespace.prey.f, namespace.hunter.f, -namespace.k)]


tumour = GroupElement('tumour', element=DensityElement, size=1, on='sites', values={'f': 1.0})
immune = GroupElement('immune', element=DensityElement, size=1, on='sites', values={'f': 0.4})
killing = KillingInteraction('killing', prey='tumour', hunter='immune')
body = CompositeSystem(elements=[tumour, immune], interactions=[killing], space=LatticeSpace(1))
simulate(body, until=5, dt=5.0).get_values('tumour.f').round(3).tolist()
```

No law changes the immune cells, so they act as a fixed input, and the tumour falls from 1 to 0.368 after five units of
time, which is `exp(-0.5 * 0.4 * 5)`.

The same law moves traders between two classes of wealth, poor and rich. The array `outcomes[h, k]` gives the
probabilities of the class a trader of class `h` ends in after it meets a trader of class `k`. A poor trader who meets a
rich one becomes rich with probability 0.3, and a rich trader who meets a poor one becomes poor with probability 0.5.
The law is conserved, so every trader who leaves a class enters another.

```python continuation
class ExchangeInteraction(Interaction):
    law_classes = (BilinearLaw,)

    def __init__(self, label, *, traders: Annotated[str, RoleSpec()]):
        super().__init__(label)
        self.traders = traders

    def _get_laws(self, namespace):
        outcomes = np.zeros((2, 2, 2))
        outcomes[0, 0] = outcomes[1, 1] = [0.5, 0.5]
        outcomes[0, 1] = [0.7, 0.3]
        outcomes[1, 0] = [0.5, 0.5]
        f = namespace.traders.f
        return [BilinearLaw(f, f, f, 1.0, outcomes, conserved=True)]


wealth = GroupElement('wealth', element=DensityElement, size=2, on='sites', values={'f': [0.9, 0.1]})
market = CompositeSystem(
    elements=[wealth], interactions=[ExchangeInteraction('exchange', traders='wealth')], space=LatticeSpace(2)
)
classes = simulate(market, until=20, dt=20.0).get_values('wealth.f')
classes.round(3).tolist(), classes.sum(axis=1).round(12).tolist()
```

The classes move from 0.9 and 0.1 to 0.55 and 0.45, and the total stays 1 at both frames.

**How the engine runs it.** The law needs continuous time and a lattice of classes. The compiler sums
`rate[h, k] outcomes[h, k, i] f_h g_k` over every pair of classes into the rate of class `i` of the target. A conserved
law also subtracts the matching loss from class `h`. The result is a system of ordinary differential equations, which
`ScipyODESolver` integrates.

**How it combines.** Its rates add to those of the `RateLaw`s on the same states, so a distribution can meet another
and also grow or decay. A second distribution that no law changes is a fixed input, such as a policy that acts on a
population from outside.

**Limits and errors.** Rates and outcomes are numbers or parameter expressions, so the probabilities cannot depend on
the states. A conserved law needs its table of outcomes, because without one its gain and loss cancel, and `prepare`
raises a `ValidationError`.

```python continuation
from sage.all__sagemath_symbolics import var

from skcomplex.core.errors import ValidationError

f, g = var('f g')
try:
    BilinearLaw(f, f, g, 1.0, conserved=True).prepare()
except ValidationError as error:
    message = str(error)
message
```

The message says that the conserved law of `f` has no outcome table, so its gain and loss cancel, and asks for the
outcome probabilities `B[h, k, i]`. The fix is an array `outcomes` of shape `(classes, classes, classes)`.

**A field realization.** The two interactions above are realizations, of immunology and of the kinetic theory of
markets. A field gives its interaction the names of its own populations and parameters, as `KillingInteraction` names
the prey, the hunter, and the killing rate `k`.

See its [fact sheet](../../../components/laws.md#bilinearlaw)
and its [API entry][skcomplex.core.components.laws.BilinearLaw].

## BoundaryLaw

A `BoundaryLaw` sets the value a state takes after every step of the solver. Here a particle drifts at a speed of 0.3
along a periodic box of length 1, and the law wraps its position back into the box.

```python
from typing import Annotated

from sage.all__sagemath_symbolics import floor

from skcomplex.core.components.elements import Element
from skcomplex.core.components.laws import BoundaryLaw, RateLaw
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.specs import ParameterSpec, StateSpec


class DriftElement(Element):
    law_classes = (RateLaw, BoundaryLaw)

    def __init__(
        self,
        label,
        visual=None,
        *,
        x: Annotated[float, StateSpec()] = 0.2,
        v: Annotated[float, ParameterSpec()] = 0.3,
    ):
        super().__init__(label, visual=visual)
        self.x = x
        self.v = v

    def _get_laws(self, namespace):
        return [RateLaw(namespace.x, namespace.v), BoundaryLaw(namespace.x, namespace.x - floor(namespace.x))]


particle = DriftElement('particle')
simulate(CompositeSystem(elements=[particle]), until=4, dt=1.0).get_values('particle.x').round(2).tolist()
```

The positions are 0.2, 0.5, 0.8, 0.1, and 0.4. At the third step the particle leaves the box at 1.1 and reenters at
0.1.

The same law wraps the angle of a rotor into the interval from minus pi to pi.

```python continuation
from sage.all__sagemath_symbolics import pi


class RotorElement(Element):
    law_classes = (RateLaw, BoundaryLaw)

    def __init__(
        self,
        label,
        visual=None,
        *,
        theta: Annotated[float, StateSpec()] = 0.0,
        omega: Annotated[float, ParameterSpec()] = 2.0,
    ):
        super().__init__(label, visual=visual)
        self.theta = theta
        self.omega = omega

    def _get_laws(self, namespace):
        theta = namespace.theta
        return [RateLaw(theta, namespace.omega), BoundaryLaw(theta, theta - 2 * pi * floor((theta + pi) / (2 * pi)))]


rotor = RotorElement('rotor')
simulate(CompositeSystem(elements=[rotor]), until=3, dt=1.0).get_values('rotor.theta').round(3).tolist()
```

The angle reads 0, 2, -2.283, and -0.283. The value 4 lies past pi, so it wraps to `4 - 2 pi`.

**How the engine runs it.** The law leaves the equations of the system as they are. After each step the solver
evaluates `value` at the new states and writes it into the target. A step that crosses the edge is corrected once, at
the end of the step, not at the time of contact.

**How it combines.** It runs beside rate laws in continuous time and beside update laws in discrete time, since it
requires no kind of time. A system of particles in a box with reflecting walls uses one boundary law for the position
and one for the velocity.

**Limits and errors.** The law corrects a state once per step, so a time of contact needs a `CrossingLaw`. Its value is
deterministic, and a random draw in it raises an `IncompatibleLawsError` when the system compiles.

```python continuation
from skcomplex.core.components.laws.utils import normal
from skcomplex.core.errors import IncompatibleLawsError


class ShakenElement(DriftElement):
    def _get_laws(self, namespace):
        return [RateLaw(namespace.x, namespace.v), BoundaryLaw(namespace.x, namespace.x + normal('kick', 0, 0.1))]


try:
    simulate(CompositeSystem(elements=[ShakenElement('particle')]), until=1)
except IncompatibleLawsError as error:
    message = str(error)
message
```

The message says that the boundary law holds the random draw `normal_kick(0, 0.1)` and that only update and renewal
laws take random draws. The fix is a `NoiseLaw` for the random part.

**A field realization.** The rotor above is a realization of mechanics. A field of soft matter writes the same law as
the reflection of a diffusing particle at the walls of a channel.

See its [fact sheet](../../../components/laws.md#boundarylaw)
and its [API entry][skcomplex.core.components.laws.BoundaryLaw].

## CascadeLaw

A `CascadeLaw` drives a lattice once per step and then relaxes it until it is stable, so one step holds a whole
avalanche. In the sandpile of Bak, Tang, and Wiesenfeld, a site with 4 grains or more topples and passes one grain to
each of its four neighbours, and grains that fall off the edge leave the pile.

```python
from typing import Annotated

import numpy as np

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.laws import ArrayUpdateLaw, CascadeLaw, LocalRuleLaw
from skcomplex.core.components.spaces import BallNeighborhood, LatticeSpace
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.specs import StateSpec

cross = BallNeighborhood(metric='manhattan', center='first')


def drop(values, *, random):
    following = values.copy()
    following[tuple(random.integers(values.shape))] += 1
    return following


def topple(z, *around):
    return z - 4 * (z >= 4) + sum(neighbour >= 4 for neighbour in around)


class SandpileElement(Element):
    law_classes = (CascadeLaw,)

    def __init__(self, label, visual=None, *, height: Annotated[float, StateSpec()] = 0.0):
        super().__init__(label, visual=visual)
        self.height = height

    def _get_laws(self, namespace):
        drive = ArrayUpdateLaw(namespace.height, drop)
        relax = LocalRuleLaw(namespace.height, cross, topple)
        return [CascadeLaw([drive], [relax], lambda z: bool((z < 4).all()), label='avalanche')]


pile = GroupElement('pile', element=SandpileElement, space=LatticeSpace((6, 6), boundary='fixed'), on='sites')
trajectory = simulate(CompositeSystem(elements=[pile]), until=300, dt=100, random_state=0)
len(trajectory.events), max(event.size for event in trajectory.events)
```

In 300 steps the pile records 91 avalanches, and the largest updates 114 sites.

The same law runs the earthquake model of Olami, Feder, and Christensen. The drive loads every site of a fault until
the most stressed one reaches 1. A site at 1 or above slips to 0 and passes a fifth of its stress to each neighbour.

```python continuation
def load(values, *, random):
    return values + (1.0 - values.max())


def slip(z, *around):
    return np.where(z >= 1.0, 0.0, z) + 0.2 * sum(np.where(neighbour >= 1.0, neighbour, 0.0) for neighbour in around)


class FaultElement(Element):
    law_classes = (CascadeLaw,)

    def __init__(self, label, visual=None, *, stress: Annotated[float, StateSpec()] = 0.0):
        super().__init__(label, visual=visual)
        self.stress = stress

    def _get_laws(self, namespace):
        drive = ArrayUpdateLaw(namespace.stress, load)
        relax = LocalRuleLaw(namespace.stress, cross, slip)
        return [CascadeLaw([drive], [relax], lambda z: bool((z < 1.0).all()), label='earthquake')]


stress = np.random.default_rng(0).random((6, 6))
fault = GroupElement(
    'fault', element=FaultElement, space=LatticeSpace((6, 6), boundary='fixed'), on='sites', values={'stress': stress}
)
quakes = simulate(CompositeSystem(elements=[fault]), until=200, dt=100, random_state=0).events
len(quakes), max(event.size for event in quakes)
```

Every step triggers one earthquake, so 200 steps record 200 events, and the largest updates 66 sites.

**How the engine runs it.** The law needs discrete time and a lattice or a network. The lattice solver applies the
drive laws once, then applies the relax laws again and again until `stable` holds. Each cascade that changes the
lattice becomes an event in `trajectory.events` with its size, the number of site updates, its duration, the number of
iterations, and its area, the number of distinct sites updated.

**How it combines.** Its drive laws are `ArrayUpdateLaw`s or `LocalRuleLaw`s, and its relax laws are `LocalRuleLaw`s,
or `ArrayUpdateLaw`s for a relaxation that sends parcels to random neighbours. The argument `activity` counts the sites
that topple, in place of the sites that change.

**Limits and errors.** The relaxation must end. A condition `stable` that never holds stops the step after
`max_iterations` iterations with a `SolverError`.

```python continuation
from skcomplex.core.errors import SolverError


class StuckElement(SandpileElement):
    def _get_laws(self, namespace):
        drive = ArrayUpdateLaw(namespace.height, drop)
        relax = LocalRuleLaw(namespace.height, cross, topple)
        return [CascadeLaw([drive], [relax], lambda z: bool((z < 0).all()), max_iterations=50)]


stuck = GroupElement('stuck', element=StuckElement, space=LatticeSpace((3, 3), boundary='fixed'), on='sites')
try:
    simulate(CompositeSystem(elements=[stuck]), until=1, random_state=0)
except SolverError as error:
    message = str(error)
message
```

The message reads "The cascade stuck is not stable after 50 iterations. Check its stability condition or raise
max_iterations." The condition `z < 0` never holds for heights, and the fix is `z < 4`.

**A field realization.** The two elements above are realizations of the sandpile and of seismology. A field also wraps
the drive and the relaxation in one interaction, with the threshold and the share as its arguments.

See its [fact sheet](../../../components/laws.md#cascadelaw)
and its [API entry][skcomplex.core.components.laws.CascadeLaw].

## CrossingLaw

A `CrossingLaw` resets states, or ends the run, when an expression of the states, its guard, crosses zero. A ball
dropped from a height of 1 metre bounces off the floor and keeps 80 percent of its speed.

```python
from typing import Annotated

from skcomplex.core.components.elements import Element
from skcomplex.core.components.laws import CrossingLaw, RateLaw
from skcomplex.core.components.laws.utils import derivative
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.specs import ParameterSpec, StateSpec


class BallElement(Element):
    law_classes = (RateLaw, CrossingLaw)

    def __init__(
        self,
        label,
        visual=None,
        *,
        h: Annotated[float, StateSpec(order=2)] = 1.0,
        e: Annotated[float, ParameterSpec()] = 0.8,
    ):
        super().__init__(label, visual=visual)
        self.h = h
        self.e = e

    def _get_laws(self, namespace):
        speed = derivative(namespace.h, 1)
        return [RateLaw(namespace.h, -9.81), CrossingLaw(namespace.h, {speed: -namespace.e * speed}, direction=-1)]


trajectory = simulate(CompositeSystem(elements=[BallElement('ball')]), until=2, dt=0.01)
[round(event.time, 3) for event in trajectory.events]
```

The ball hits the floor at 0.452, 1.174, and 1.752 seconds. The first time is `sqrt(2 / 9.81)`, and each flight after
a bounce lasts 0.8 times the one before.

The same law fires a neuron. The potential of a leaky integrate-and-fire neuron rises toward the input current, and
when it reaches the threshold 1 it resets to 0.

```python continuation
class NeuronElement(Element):
    law_classes = (RateLaw, CrossingLaw)

    def __init__(
        self,
        label,
        visual=None,
        *,
        v: Annotated[float, StateSpec()] = 0.0,
        current: Annotated[float, ParameterSpec()] = 1.5,
        tau: Annotated[float, ParameterSpec()] = 10.0,
    ):
        super().__init__(label, visual=visual)
        self.v = v
        self.current = current
        self.tau = tau

    def _get_laws(self, namespace):
        v = namespace.v
        return [RateLaw(v, (namespace.current - v) / namespace.tau), CrossingLaw(v - 1, {v: 0}, direction=1)]


spikes = simulate(CompositeSystem(elements=[NeuronElement('neuron')]), until=50, dt=0.1).events
[round(event.time, 2) for event in spikes]
```

The neuron fires at the times 10.99, 21.97, 32.96, and 43.94, once every `10 ln 3`.

**How the engine runs it.** The law needs continuous time. The solver watches the sign of the guard, locates the
crossing on its interpolant within its tolerance, applies every reset at once, and restarts the integration from
there. The trajectory keeps each crossing with its time and the states before and after it, and a `terminal` crossing
ends the run.

**How it combines.** It leaves the equations as they are, so it combines with `RateLaw`s and `NoiseLaw`s. A
`CrossingInteraction` returns it from an interaction, so a threshold can read the states of several elements.

**Limits and errors.** A guard that crosses zero again right after its reset raises an error that names the chattering
event. The direction is `1` for upward crossings, `-1` for downward ones, or `0` for both, and `prepare` raises a
`ValidationError` for any other value.

```python continuation
from sage.all__sagemath_symbolics import var

from skcomplex.core.errors import ValidationError

try:
    CrossingLaw(var('h'), direction=2).prepare()
except ValidationError as error:
    message = str(error)
message
```

The message reads "The direction of a crossing is -1, 0, or 1, but got 2." The fix is `direction=-1` for a fall.

**A field realization.** The neuron above is a realization of neuroscience, and a thermostat of engineering switches a
heater with the same law when the temperature crosses a set point.

See its [fact sheet](../../../components/laws.md#crossinglaw)
and its [API entry][skcomplex.core.components.laws.CrossingLaw].

## DissipatorLaw

A `DissipatorLaw` adds one Lindblad term to the evolution of an open quantum system, the loss of coherence or energy to
its surroundings. The term has a collapse operator `L` and a rate. Here a two-level atom decays from its excited state
by spontaneous emission, with `L` the lowering operator and the rate 0.5.

```python
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import CC

from skcomplex.core.components.elements import Element
from skcomplex.core.components.laws import DissipatorLaw
from skcomplex.core.components.laws.utils import local_operator
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.specs import HilbertSpec, ObservableSpec, ParameterSpec, StateSpec

QUBIT = HilbertSpec(
    2,
    {'minus': (np.array([[0.0, 0.0], [1.0, 0.0]]), False), 'excited': (np.diag([1.0, 0.0]), True)},
    basis={'excited': 0, 'ground': 1},
    display='excited',
)


class AtomElement(Element):
    law_classes = (DissipatorLaw,)

    def __init__(
        self,
        label,
        visual=None,
        *,
        psi: Annotated[object, StateSpec(domain=CC)] = 'excited',
        gamma: Annotated[float, ParameterSpec()] = 0.5,
        hilbert=QUBIT,
    ):
        super().__init__(label, visual=visual)
        self.psi = psi
        self.gamma = gamma
        self.hilbert = hilbert

    def _get_laws(self, namespace):
        return [DissipatorLaw(local_operator(namespace.psi, 'minus'), namespace.gamma)]


def excited(namespace):
    return local_operator(namespace.atom.psi, 'excited')


observables = [ObservableSpec('excited', excited)]
trajectory = simulate(CompositeSystem(elements=[AtomElement('atom')]), until=2, dt=1.0, observables=observables)
np.real(trajectory.get_values('excited')).round(3).tolist()
```

The probability of the excited state falls from 1 to 0.607 and 0.368, which is `exp(-0.5 t)`.

The same law makes a cavity of optics leak photons. The mode starts with two photons, and the annihilation operator
removes them at the rate 0.1.

```python continuation
CAVITY = HilbertSpec(
    4,
    {'annihilate': (np.diag(np.sqrt([1.0, 2.0, 3.0]), k=1), False), 'number': (np.diag([0.0, 1.0, 2.0, 3.0]), True)},
    basis={'two': 2},
    display='number',
)


class CavityElement(Element):
    law_classes = (DissipatorLaw,)

    def __init__(
        self,
        label,
        visual=None,
        *,
        psi: Annotated[object, StateSpec(domain=CC)] = 'two',
        kappa: Annotated[float, ParameterSpec()] = 0.1,
        hilbert=CAVITY,
    ):
        super().__init__(label, visual=visual)
        self.psi = psi
        self.kappa = kappa
        self.hilbert = hilbert

    def _get_laws(self, namespace):
        return [DissipatorLaw(local_operator(namespace.psi, 'annihilate'), namespace.kappa)]


def photons(namespace):
    return local_operator(namespace.cavity.psi, 'number')


observed = [ObservableSpec('photons', photons)]
trajectory = simulate(CompositeSystem(elements=[CavityElement('cavity')]), until=10, dt=5.0, observables=observed)
np.real(trajectory.get_values('photons')).round(3).tolist()
```

The mean number of photons falls from 2 to 1.213 and 0.736, which is `2 exp(-0.1 t)`.

**How the engine runs it.** The law needs continuous time. The compiler builds the operator form of the system, whose
state is a density matrix, and `LindbladSolver`, the default, integrates the master equation through QuTiP. A run of an
open system also records its purity.

**How it combines.** It adds to the `GeneratorLaw`s of the same system, which give the reversible part of the
evolution. Several dissipators add their terms, such as decay and loss of phase on one qubit.

**Limits and errors.** The operator and the rate are symbolic, because the compiler builds matrices from them, so
neither can be a function.

```python continuation
from skcomplex.core.errors import ValidationError

try:
    DissipatorLaw(lambda t, values: 0.0).prepare()
except ValidationError as error:
    message = str(error)
message
```

The message says that dissipators need symbolic operators and rates, and asks for the rate as a `series` of time. The
fix is an expression of local operators, such as `local_operator(namespace.psi, 'minus')`.

**A field realization.** The atom and the cavity above are realizations of quantum optics. A qubit of quantum
computing loses its phase through a dissipator whose operator is the Pauli matrix `z`.

See its [fact sheet](../../../components/laws.md#dissipatorlaw)
and its [API entry][skcomplex.core.components.laws.DissipatorLaw].

## EquationLaw

An `EquationLaw` states a relation that holds the highest time derivatives of its states linearly, as Newton's second
law holds an acceleration. The pendulum `m l θ'' = -m g sin θ` is one such relation, written as it reads.

```python
from typing import Annotated

from sage.all__sagemath_symbolics import sin

from skcomplex.core.components.elements import Element
from skcomplex.core.components.laws import EquationLaw
from skcomplex.core.components.laws.utils import derivative
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.specs import ParameterSpec, StateSpec


class PendulumElement(Element):
    law_classes = (EquationLaw,)

    def __init__(
        self,
        label,
        visual=None,
        *,
        theta: Annotated[float, StateSpec(order=2)] = 0.1,
        m: Annotated[float, ParameterSpec()] = 1.0,
        l: Annotated[float, ParameterSpec()] = 1.0,
    ):
        super().__init__(label, visual=visual)
        self.theta = theta
        self.m = m
        self.l = l

    def _get_laws(self, namespace):
        theta, m, l = namespace.theta, namespace.m, namespace.l
        return [EquationLaw(m * l * derivative(theta, 2) == -m * 9.81 * sin(theta))]


bob = PendulumElement('bob')
simulate(CompositeSystem(elements=[bob]), until=2, dt=1.0).get_values('bob.theta').round(3).tolist()
```

The angle swings from 0.1 to -0.1 after one second and back to 0.1 after two, close to the small period
`2 pi sqrt(l / g)` of 2.006 seconds.

The same law gives the charge `q` of a series circuit of an inductor, a resistor, and a capacitor,
`L q'' + R q' + q / C = 0`.

```python continuation
class CircuitElement(Element):
    law_classes = (EquationLaw,)

    def __init__(
        self,
        label,
        visual=None,
        *,
        q: Annotated[float, StateSpec(order=2)] = 1.0,
        L: Annotated[float, ParameterSpec()] = 1.0,
        R: Annotated[float, ParameterSpec()] = 0.5,
        C: Annotated[float, ParameterSpec()] = 0.25,
    ):
        super().__init__(label, visual=visual)
        self.q = q
        self.L = L
        self.R = R
        self.C = C

    def _get_laws(self, namespace):
        q, L, R, C = namespace.q, namespace.L, namespace.R, namespace.C
        return [EquationLaw(L * derivative(q, 2) + R * derivative(q, 1) + q / C == 0)]


circuit = CircuitElement('rlc')
simulate(CompositeSystem(elements=[circuit]), until=3, dt=1.0).get_values('rlc.q').round(3).tolist()
```

The charge oscillates and decays: 1, -0.223, -0.467, and 0.428.

**How the engine runs it.** The law needs continuous or static time. The compiler collects every equation of the
system, solves them together for the highest derivatives, adds the rate laws on the same states, and reduces the
result to first order. `ScipyODESolver` then integrates it. Equations that share a `target` add into one before the
solve.

**How it combines.** Equations of several components couple through their derivatives, as the two angles of a double
pendulum do through a mass matrix. `EulerLagrangeInteraction` and `ConnectionInteraction` return equation laws, and a
`RateLaw` on the same state adds a force.

**Limits and errors.** The relation must be linear in the highest derivatives. It is solved symbolically, so neither
side can be a function.

```python continuation
from skcomplex.core.errors import ValidationError

try:
    EquationLaw(lambda t, values: 0.0, 0.0).prepare()
except ValidationError as error:
    message = str(error)
message
```

The message says that equations are solved symbolically, so a side cannot be a function, and asks for a `RateLaw` of
the highest derivative with the function as its value.

**A field realization.** The circuit above is a realization of electrical engineering, and the pendulum one of
mechanics. A robot arm writes one equation per joint in the same way.

See its [fact sheet](../../../components/laws.md#equationlaw)
and its [API entry][skcomplex.core.components.laws.EquationLaw].

## ExtremalLaw

An `ExtremalLaw` updates the site where a state is smallest or largest, together with its neighbours. In the evolution
model of Bak and Sneppen, the least fit species of a ring and its two neighbours take new random fitnesses at every
step.

```python
from typing import Annotated

import numpy as np

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.laws import ExtremalLaw
from skcomplex.core.components.spaces import LatticeNeighborhood, LatticeSpace
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.specs import StateSpec

beside = LatticeNeighborhood(((-1,), (1,)))


def renew(fitness, random):
    return random


class SpeciesElement(Element):
    law_classes = (ExtremalLaw,)

    def __init__(self, label, visual=None, *, fitness: Annotated[float, StateSpec()] = 0.5):
        super().__init__(label, visual=visual)
        self.fitness = fitness

    def _get_laws(self, namespace):
        return [ExtremalLaw(namespace.fitness, beside, renew, label='extinction')]


fitness = np.random.default_rng(0).random(20)
species = GroupElement(
    'species', element=SpeciesElement, space=LatticeSpace(20), on='sites', values={'fitness': fitness}
)
trajectory = simulate(CompositeSystem(elements=[species]), until=200, dt=100, random_state=0)
values = trajectory.get_values('species.fitness')
len(trajectory.events), round(float(values[0].mean()), 2), round(float(values[-1].mean()), 2)
```

The 200 steps record 200 extinctions, and the mean fitness rises from 0.51 to 0.7 as the weakest species are replaced.

The same law breaks the most stressed fibre of a bundle, a model of fracture. The fibre drops to zero stress and passes
half of its load to each neighbour, so the total load stays the same.

```python continuation
def fail(stress, random):
    following = stress.copy()
    following[0] = 0.0
    following[1:] += stress[0] / 2
    return following


class FibreElement(Element):
    law_classes = (ExtremalLaw,)

    def __init__(self, label, visual=None, *, stress: Annotated[float, StateSpec()] = 0.0):
        super().__init__(label, visual=visual)
        self.stress = stress

    def _get_laws(self, namespace):
        return [ExtremalLaw(namespace.stress, beside, fail, extremum='max', label='failure')]


load = np.random.default_rng(1).random(10)
bundle = GroupElement('bundle', element=FibreElement, space=LatticeSpace(10), on='sites', values={'stress': load})
stresses = simulate(CompositeSystem(elements=[bundle]), until=5, random_state=0).get_values('bundle.stress')
round(float(load.sum()), 6), round(float(stresses[-1].sum()), 6)
```

The total stress is 5.104307 before and after five failures.

**How the engine runs it.** The law needs discrete time and a lattice or a network. At each application the lattice
solver finds the extremal site of the key state, breaks ties with the random stream of the run, gathers the site first
and then its neighbours, and calls the rule with one array per target and a uniform random array. It applies the law
`repeat` times per step and records each application as an event with its time, site, and extremal value.

**How it combines.** Its rule can update several states together, and `key` names the state whose extremum is
searched when it is not the first target. Other lattice laws of the element run in the same step.

**Limits and errors.** One application changes one site and its neighbours only. The extremum is `'min'` or `'max'`,
and `prepare` raises a `ValidationError` for any other value.

```python continuation
from sage.all__sagemath_symbolics import var

from skcomplex.core.errors import ValidationError

try:
    ExtremalLaw(var('fitness'), beside, renew, extremum='median').prepare()
except ValidationError as error:
    message = str(error)
message
```

The message reads "Unknown extremum 'median'. Use one of min, max."

**A field realization.** The species above are a realization of evolutionary biology, and the fibres one of the
physics of materials.

See its [fact sheet](../../../components/laws.md#extremallaw)
and its [API entry][skcomplex.core.components.laws.ExtremalLaw].

## FluxLaw

A `FluxLaw` gives the flux of a conserved state of a field, so that `du/dt + divergence(flux) = 0`. The density of
cars on a road has the flux `ρ (1 - ρ)` of Lighthill, Whitham, and Richards. Here light traffic at density 0.2 runs into
a jam at density 0.9.

```python
from typing import Annotated

import numpy as np

from skcomplex.core.components.elements import Element, FieldElement
from skcomplex.core.components.laws import FluxLaw
from skcomplex.core.components.spaces import LatticeSpace
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.solvers import SSPRKSolver
from skcomplex.core.specs import StateSpec


class TrafficElement(Element):
    law_classes = (FluxLaw,)

    def __init__(self, label, visual=None, *, rho: Annotated[float, StateSpec()] = 0.0):
        super().__init__(label, visual=visual)
        self.rho = rho

    def _get_laws(self, namespace):
        return [FluxLaw(namespace.rho, namespace.rho * (1 - namespace.rho))]


density = np.where(np.arange(50) < 25, 0.2, 0.9)
road = FieldElement(
    'road',
    element=TrafficElement,
    space=LatticeSpace(50, spacing=0.1),
    values={'rho': density},
    discretization='finite-volume',
)
cars = simulate(CompositeSystem(elements=[road]), until=1, dt=1.0, solver=SSPRKSolver()).get_values('road.rho')
cars.sum(axis=1).round(6).tolist(), int(np.argmax(np.diff(cars[-1])))
```

The number of cars stays 27.5, and the back of the jam moves from cell 24 to cell 23. The shock travels backwards at
the speed `1 - 0.2 - 0.9 = -0.1`, one cell of length 0.1 in one unit of time.

The same law forms a shock wave in a gas. The flux `u^2 / 2` of the Burgers equation steepens a smooth wave until it
breaks.

```python continuation
class BurgersElement(Element):
    law_classes = (FluxLaw,)

    def __init__(self, label, visual=None, *, u: Annotated[float, StateSpec()] = 0.0):
        super().__init__(label, visual=visual)
        self.u = u

    def _get_laws(self, namespace):
        return [FluxLaw(namespace.u, namespace.u**2 / 2, riemann='godunov')]


cells = (np.arange(64) + 0.5) * 2 * np.pi / 64
gas = FieldElement(
    'gas',
    element=BurgersElement,
    space=LatticeSpace(64, spacing=2 * np.pi / 64),
    values={'u': 1.5 + np.sin(cells)},
    discretization='finite-volume',
)
speeds = simulate(CompositeSystem(elements=[gas]), until=2, dt=2.0, solver=SSPRKSolver()).get_values('gas.u')
speeds.sum(axis=1).round(6).tolist(), round(float(np.abs(np.diff(speeds[-1])).max()), 2)
```

The total stays 96, and the largest jump between neighbouring cells grows from about 0.1 to 0.87, a shock.

**How the engine runs it.** The law needs continuous time and a field of finite volumes on a lattice or a mesh. The
scheme reconstructs the state in each cell with the slope limiter `limiter` and computes the flux between cells with
the numerical flux `riemann`. The wave speeds come from the derivative of the flux. `SSPRKSolver` steps the field by
the CFL condition and keeps its total.

**How it combines.** Rate laws on the same state add sources, such as a diffusion `RateLaw(u, nu * laplacian(u))` for
a viscous gas. The fluxes of several states form a system of conservation laws. `ConservationInteraction` returns
flux laws from an interaction.

**Limits and errors.** The flux is symbolic, because its derivative gives the wave speeds. The numerical flux is
`'rusanov'` or `'godunov'`, and `prepare` raises a `ValidationError` for any other name.

```python continuation
from sage.all__sagemath_symbolics import var

from skcomplex.core.errors import ValidationError

rho = var('rho')
try:
    FluxLaw(rho, rho * (1 - rho), riemann='roe').prepare()
except ValidationError as error:
    message = str(error)
message
```

The message reads "Unknown numerical flux 'roe'. Use 'rusanov' or 'godunov'."

**A field realization.** The road above is a realization of traffic engineering, and a river channel of hydrology
writes the flux of a flood wave in the same way.

See its [fact sheet](../../../components/laws.md#fluxlaw)
and its [API entry][skcomplex.core.components.laws.FluxLaw].

## GeneratorLaw

A `GeneratorLaw` adds a Hermitian operator `H` to the generator of the evolution `dx/dt = -i H x`, which keeps the norm
of the state. In quantum mechanics `H` is the Hamiltonian. Here a spin one half precesses in a field along `x`.

```python
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import CC

from skcomplex.core.components.elements import Element
from skcomplex.core.components.laws import GeneratorLaw
from skcomplex.core.components.laws.utils import local_operator
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.specs import HilbertSpec, ObservableSpec, ParameterSpec, StateSpec

SPIN = HilbertSpec(
    2,
    {'x': (np.array([[0.0, 0.5], [0.5, 0.0]]), True), 'z': (np.diag([0.5, -0.5]), True)},
    basis={'up': 0, 'down': 1},
)


class SpinElement(Element):
    law_classes = (GeneratorLaw,)

    def __init__(
        self,
        label,
        visual=None,
        *,
        psi: Annotated[object, StateSpec(domain=CC)] = 'up',
        omega: Annotated[float, ParameterSpec()] = 1.0,
        hilbert=SPIN,
    ):
        super().__init__(label, visual=visual)
        self.psi = psi
        self.omega = omega
        self.hilbert = hilbert

    def _get_laws(self, namespace):
        return [GeneratorLaw(namespace.omega * local_operator(namespace.psi, 'x'))]


def sz(namespace):
    return local_operator(namespace.spin.psi, 'z')


observables = [ObservableSpec('sz', sz)]
precession = CompositeSystem(elements=[SpinElement('spin')])
trajectory = simulate(precession, until=np.pi, dt=np.pi / 2, observables=observables)
(np.real(trajectory.get_values('sz')).round(3) + 0.0).tolist()
```

The expectation of the spin along `z` goes from 0.5 to 0 and to -0.5 at time pi, which is `0.5 cos(t)`.

The same law runs a quantum walk on a path of three nodes, with `H` minus the adjacency matrix of the path. A walker
that starts on the first node reaches the last one with certainty at the time `pi / sqrt(2)`.

```python continuation
PATH = HilbertSpec(
    3,
    {
        'adjacency': (np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]), True),
        'end': (np.diag([0.0, 0.0, 1.0]), True),
    },
    basis={'start': 0},
    display='end',
)


class WalkerElement(Element):
    law_classes = (GeneratorLaw,)

    def __init__(
        self,
        label,
        visual=None,
        *,
        psi: Annotated[object, StateSpec(domain=CC)] = 'start',
        hop: Annotated[float, ParameterSpec()] = 1.0,
        hilbert=PATH,
    ):
        super().__init__(label, visual=visual)
        self.psi = psi
        self.hop = hop
        self.hilbert = hilbert

    def _get_laws(self, namespace):
        return [GeneratorLaw(-namespace.hop * local_operator(namespace.psi, 'adjacency'))]


def arrived(namespace):
    return local_operator(namespace.walker.psi, 'end')


observed = [ObservableSpec('end', arrived)]
crossing = np.pi / np.sqrt(2)
walker = CompositeSystem(elements=[WalkerElement('walker')])
walk = simulate(walker, until=crossing, dt=crossing / 2, observables=observed)
np.real(walk.get_values('end')).round(3).tolist()
```

The probability on the last node goes from 0 to 0.25 and to 1.

**How the engine runs it.** The law needs continuous time. The compiler sums the operators of every generator law into
one matrix on the tensor product of the subsystems and checks that it is Hermitian. The default `LindbladSolver`
integrates the evolution through QuTiP, and `UnitarySolver` solves the Schrödinger equation of a closed system. A run
records the expectation of the generator as `energy`.

**How it combines.** A `NeighborInteraction` returns one generator law per pair of neighbours, so a chain of spins
couples through its bonds. `DissipatorLaw`s add the irreversible part of an open system.

**Limits and errors.** The operator must be built from local operators, which give the compiler its matrices.

```python continuation
from sage.all__sagemath_symbolics import var

from skcomplex.core.errors import ValidationError

try:
    GeneratorLaw(var('h')).prepare()
except ValidationError as error:
    message = str(error)
message
```

The message reads "A generator takes an operator built from local operators, but got h." The fix is a product such as
`var('h') * local_operator(namespace.psi, 'z')`.

**A field realization.** The spin above is a realization of physics, and the walker one of quantum search on a graph.

See its [fact sheet](../../../components/laws.md#generatorlaw)
and its [API entry][skcomplex.core.components.laws.GeneratorLaw].

## GradientStepLaw

A `GradientStepLaw` takes optimizer steps on the parameters of a nested system, so that a component of a model learns
from data during the run. One step of the outer system is one epoch, and `TrainingInteraction` returns the law. Here the
factor `a` of the map `x' = a x` is fitted to a series that decays by 0.8 per step.

```python
from typing import Annotated

import numpy as np

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import TrainingInteraction
from skcomplex.core.components.laws import UpdateLaw
from skcomplex.core.components.spaces import DiscreteTime
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.specs import DatasetSpec, ObservableSpec, ParameterSpec, StateSpec


class ScaleElement(Element):
    law_classes = (UpdateLaw,)

    def __init__(
        self,
        label,
        visual=None,
        *,
        x: Annotated[float, StateSpec()] = 1.0,
        a: Annotated[float, ParameterSpec(adaptive=True)] = 0.5,
        observed: Annotated[float, ParameterSpec()] = 0.0,
    ):
        super().__init__(label, visual=visual)
        self.x = x
        self.a = a
        self.observed = observed

    def _get_laws(self, namespace):
        return [UpdateLaw(namespace.x, namespace.a * namespace.x)]


def squared_error(namespace):
    return (namespace.unit.x - namespace.unit.observed) ** 2


decay = CompositeSystem('decay', elements=[ScaleElement('unit')], time=DiscreteTime())
training = TrainingInteraction(
    'fit',
    system='decay',
    trained=['unit.a'],
    loss=ObservableSpec('error', squared_error),
    data=DatasetSpec({'unit.observed': 0.8 ** np.arange(1, 11)[np.newaxis]}),
    learning_rate=0.05,
    batch_size=1,
)
fitting = CompositeSystem('fitting', systems=[decay], interactions=[training], time=DiscreteTime())
trajectory = simulate(fitting, until=100, random_state=0)
round(float(trajectory.get_values('decay/unit.a')[-1]), 3)
```

After 100 epochs the factor moves from 0.5 to 0.8, the decay of the data. A real study fits noisy data over more
epochs and checks the loss on held-out sequences.

The same law fits the interest of a deposit to the balances of ten years that grew by 4 percent a year.

```python continuation
balances = 1.04 ** np.arange(1, 11)[np.newaxis]
deposit = CompositeSystem('deposit', elements=[ScaleElement('unit', a=1.0)], time=DiscreteTime())
interest = TrainingInteraction(
    'fit',
    system='deposit',
    trained=['unit.a'],
    loss=ObservableSpec('error', squared_error),
    data=DatasetSpec({'unit.observed': balances}),
    learning_rate=0.01,
    batch_size=1,
)
fitting = CompositeSystem('fitting', systems=[deposit], interactions=[interest], time=DiscreteTime())
trajectory = simulate(fitting, until=100, random_state=0)
round(float(trajectory.get_values('deposit/unit.a')[-1]), 3)
```

The yearly factor reaches 1.04.

**How the engine runs it.** The law needs discrete time, and the outer system has the learning form, which
`GradientDescentSolver` runs. In an epoch the solver shuffles the sequences of the dataset into batches. For each batch
it runs the nested system over every sequence, averages the loss over the steps, and changes the trained parameters by
the gradient of that mean. JAX computes the gradient exactly by automatic differentiation of the runs.

**How it combines.** The trained parameters are declared `ParameterSpec(adaptive=True)` in the nested system, which
stays an ordinary system that runs and is analysed alone. The metrics of the law are observables recorded at the end of
every epoch.

**Limits and errors.** The nested system must be written as symbolic laws, so that JAX can differentiate its runs. The
optimizer is one of `'sgd'`, `'momentum'`, `'rmsprop'`, and `'adam'`, and the interaction raises a `ValidationError`
for any other name.

```python continuation
from skcomplex.core.errors import ValidationError

try:
    TrainingInteraction(
        'fit', system='decay', trained=['unit.a'], loss=training.loss, data=training.data, optimizer='newton'
    ).prepare()
except ValidationError as error:
    message = str(error)
message
```

The message says that the optimizer `'newton'` of `'fit'` is not supported and lists the four that are.

**A field realization.** A field of machine learning names the nested system a recurrent network and the trained
parameters its weights, and a reservoir computer trains only its readout.

See its [fact sheet](../../../components/laws.md#gradientsteplaw)
and its [API entry][skcomplex.core.components.laws.GradientStepLaw].

## JumpLaw

A `JumpLaw` is a random event that changes states by whole amounts. With `rate`, events arrive as a Poisson process of
that propensity. Here a small colony of bacteria divides at a rate of 0.5 per cell and dies at a rate of 0.4 per cell.

```python
from typing import Annotated

from skcomplex.core.components.elements import Element
from skcomplex.core.components.laws import JumpLaw
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.specs import ParameterSpec, StateSpec


class ColonyElement(Element):
    law_classes = (JumpLaw,)

    def __init__(
        self,
        label,
        visual=None,
        *,
        n: Annotated[float, StateSpec()] = 10.0,
        b: Annotated[float, ParameterSpec()] = 0.5,
        d: Annotated[float, ParameterSpec()] = 0.4,
    ):
        super().__init__(label, visual=visual)
        self.n = n
        self.b = b
        self.d = d

    def _get_laws(self, namespace):
        n = namespace.n
        return [JumpLaw(namespace.b * n, {n: 1}), JumpLaw(namespace.d * n, {n: -1})]


colony = CompositeSystem(elements=[ColonyElement('colony')])
simulate(colony, until=10, dt=5.0, random_state=0).get_values('colony.n').tolist()
```

This run counts 10, 6, and 25 cells. The mean grows as `10 exp(0.1 t)`, but a small colony fluctuates far from it.

The same law runs a queue. Customers arrive at a rate of 2 per minute, and the desk serves them at a rate of 3 per
minute while anyone waits.

```python continuation
from skcomplex.core.components.laws.utils import where


class QueueElement(Element):
    law_classes = (JumpLaw,)

    def __init__(
        self,
        label,
        visual=None,
        *,
        q: Annotated[float, StateSpec()] = 0.0,
        arrival: Annotated[float, ParameterSpec()] = 2.0,
        service: Annotated[float, ParameterSpec()] = 3.0,
    ):
        super().__init__(label, visual=visual)
        self.q = q
        self.arrival = arrival
        self.service = service

    def _get_laws(self, namespace):
        q = namespace.q
        return [JumpLaw(namespace.arrival, {q: 1}), JumpLaw(where(q > 0, namespace.service, 0), {q: -1})]


desk = CompositeSystem(elements=[QueueElement('desk')])
lengths = simulate(desk, until=1000, dt=1.0, random_state=0).get_values('desk.q')
round(float(lengths.mean()), 1)
```

The mean length of the queue over 1000 minutes is 2.1. Queueing theory gives 2 when customers arrive at 2/3 of the rate
of service.

**How the engine runs it.** The law needs continuous time. Jump laws give a Markov chain in continuous time, and
`GillespieSolver` draws the time of the next event and which event fires, an exact stochastic simulation. With
`waiting` in place of `rate`, each firing waits a drawn time and applies `reset`, and `RenewalSolver` runs it.

**How it combines.** `RateLaw`s and `NoiseLaw`s move the states between jumps. `JumpInteraction` returns jump laws
from an interaction, so an event can change the states of several elements at once, as a reaction does.

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

## LinearProgramLaw

A `LinearProgramLaw` sets algebraic states, which no derivative governs, to the optimum of a linear program. In flux
balance analysis a cell takes up at most 10 units of a nutrient, spends at least 2 on its upkeep, and grows on the rest.

```python
from typing import Annotated

from skcomplex.core.components.elements import Element
from skcomplex.core.components.laws import LinearProgramLaw
from skcomplex.core.components.spaces import StaticTime
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.specs import LinearObjectiveSpec, ParameterSpec, StateSpec


class CellElement(Element):
    law_classes = (LinearProgramLaw,)

    def __init__(
        self,
        label,
        visual=None,
        *,
        uptake: Annotated[float, StateSpec(algebraic=True)] = 0.0,
        upkeep: Annotated[float, StateSpec(algebraic=True)] = 0.0,
        growth: Annotated[float, StateSpec(algebraic=True)] = 0.0,
        supply: Annotated[float, ParameterSpec()] = 10.0,
    ):
        super().__init__(label, visual=visual)
        self.uptake = uptake
        self.upkeep = upkeep
        self.growth = growth
        self.supply = supply

    def _get_laws(self, namespace):
        uptake, upkeep, growth = namespace.uptake, namespace.upkeep, namespace.growth
        constraints = [uptake == upkeep + growth, uptake <= namespace.supply, upkeep >= 2, growth >= 0]
        return [LinearProgramLaw([uptake, upkeep, growth], constraints, [LinearObjectiveSpec(growth)])]


cell = CompositeSystem(elements=[CellElement('cell')], time=StaticTime())
trajectory = simulate(cell, until=0.0)
float(trajectory.get_values('cell.growth')[0]), float(trajectory.get_values('cell.uptake')[0])
```

The cell grows at 8 and takes up the whole supply of 10.

The same law dispatches power plants at least cost. Coal costs 20 per unit and runs up to 100 units, gas costs 50 and
runs up to 50, and together they meet a demand of 120.

```python continuation
class GridElement(Element):
    law_classes = (LinearProgramLaw,)

    def __init__(
        self,
        label,
        visual=None,
        *,
        coal: Annotated[float, StateSpec(algebraic=True)] = 0.0,
        gas: Annotated[float, StateSpec(algebraic=True)] = 0.0,
        demand: Annotated[float, ParameterSpec()] = 120.0,
    ):
        super().__init__(label, visual=visual)
        self.coal = coal
        self.gas = gas
        self.demand = demand

    def _get_laws(self, namespace):
        coal, gas = namespace.coal, namespace.gas
        constraints = [coal + gas == namespace.demand, coal <= 100, gas <= 50, coal >= 0, gas >= 0]
        cost = LinearObjectiveSpec(20 * coal + 50 * gas, sense='min')
        return [LinearProgramLaw([coal, gas], constraints, [cost])]


grid = CompositeSystem(elements=[GridElement('grid')], time=StaticTime())
dispatch = simulate(grid, until=0.0)
[float(dispatch.get_values(name)[0]) for name in ('grid.coal', 'grid.gas', 'grid.objective_0')]
```

Coal runs at its limit of 100, gas supplies the remaining 20, and the cost is 3000.

**How the engine runs it.** The law runs in continuous, discrete, or static time. HiGHS solves the program through
`highspy` whenever the form of the system needs its variables: at each frame of a static system, at each step of a
map, and at each evaluation of the rates of an ordinary differential equation. A run records the variables under their
paths and the optimum under `<path>.objective_0`.

**How it combines.** The laws whose variables overlap form one program, so an interaction adds a bound to the program
of an element by returning a law over the variable it bounds. Beside rate laws the program sets the fluxes that the
rates read at every time.

**Limits and errors.** Every constraint is linear in the variables. A program with no feasible point raises an
`InfeasibleError` by default.

```python continuation
from skcomplex.core.errors import InfeasibleError

short = CompositeSystem(elements=[GridElement('grid', demand=300.0)], time=StaticTime())
try:
    simulate(short, until=0.0)
except InfeasibleError as error:
    message = str(error)
message
```

The message says that the program of `grid` has no solution at time 0, names the demand constraint and the two
variables that cannot hold together, and suggests `infeasible='zero'` or `infeasible='hold'`. The fix is a demand
the plants can meet, or one of those policies.

**A field realization.** The cell above is a realization of systems biology, and the grid one of energy economics.

See its [fact sheet](../../../components/laws.md#linearprogramlaw)
and its [API entry][skcomplex.core.components.laws.LinearProgramLaw].

## LocalRuleLaw

A `LocalRuleLaw` sets the next value of every cell of a lattice from the ordered values of its neighbourhood. The
elementary cellular automaton of rule 90 sets each cell to the sum of its two neighbours modulo 2, and it grows the
Sierpinski triangle from one live cell.

```python
from typing import Annotated

import numpy as np

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.laws import LocalRuleLaw
from skcomplex.core.components.spaces import LatticeNeighborhood, LatticeSpace
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.specs import LinearRuleSpec, StateSpec


class CellElement(Element):
    law_classes = (LocalRuleLaw,)

    def __init__(self, label, visual=None, *, cell: Annotated[float, StateSpec()] = 0.0):
        super().__init__(label, visual=visual)
        self.cell = cell

    def _get_laws(self, namespace):
        return [LocalRuleLaw(namespace.cell, LatticeNeighborhood(((-1,), (1,))), LinearRuleSpec((1, 1), 'GF(2)'))]


seed = np.zeros(15)
seed[7] = 1
row = GroupElement('row', element=CellElement, space=LatticeSpace(15), on='sites', values={'cell': seed})
cells = simulate(CompositeSystem(elements=[row]), until=4).get_values('row.cell')
cells.sum(axis=1).astype(int).tolist()
```

The numbers of live cells at each step are 1, 2, 2, 4, and 2, the rows of the Sierpinski triangle.

The same law runs Conway's Game of Life on a torus of 8 by 8 cells. A glider moves one cell down and one cell right
every four steps.

```python continuation
from skcomplex.core.components.spaces import BallNeighborhood
from skcomplex.core.specs import OuterTotalisticSpec

life = OuterTotalisticSpec({(0, 3): 1, (1, 2): 1, (1, 3): 1})


class LifeElement(Element):
    law_classes = (LocalRuleLaw,)

    def __init__(self, label, visual=None, *, alive: Annotated[float, StateSpec()] = 0.0):
        super().__init__(label, visual=visual)
        self.alive = alive

    def _get_laws(self, namespace):
        return [LocalRuleLaw(namespace.alive, BallNeighborhood(), life)]


glider = np.zeros((8, 8))
glider[1, 2] = glider[2, 3] = glider[3, 1] = glider[3, 2] = glider[3, 3] = 1
board = GroupElement('board', element=LifeElement, space=LatticeSpace((8, 8)), on='sites', values={'alive': glider})
frames = simulate(CompositeSystem(elements=[board]), until=4).get_lattice('board.alive')
bool(np.array_equal(frames[-1], np.roll(glider, (1, 1), axis=(0, 1))))
```

The last frame equals the first one shifted by one cell along both axes.

**How the engine runs it.** The law needs discrete time and a lattice, a network, or the Cayley graph of a group. The
lattice solver applies the rule to every site from the previous configuration by default. Tables and totalistic rules
run as lookups, which the Numba backend fuses into one compiled loop. The schedules `'random-sequential'`,
`'random-site'`, `'checkerboard'`, and `'colored'` update the sites one at a time or one class at a time.

**How it combines.** A rule may read several states and set several targets together. `LocalRuleInteraction` returns
it from an interaction, and a `CascadeLaw` uses it to relax a lattice.

**Limits and errors.** The rule reads a fixed neighbourhood of each cell, so a change of the whole array at once needs
an `ArrayUpdateLaw`. An unknown schedule raises a `ValidationError` from `prepare`.

```python continuation
from sage.all__sagemath_symbolics import var

from skcomplex.core.errors import ValidationError

try:
    LocalRuleLaw(var('cell'), BallNeighborhood(), life, schedule='parallel').prepare()
except ValidationError as error:
    message = str(error)
message
```

The message names the unknown schedule `'parallel'` and lists the five schedules. The fix for an update of every cell
at once is `'synchronous'`, the default.

**A field realization.** The board above is a realization of computer science. A model of excitable tissue writes the
states of resting, excited, and refractory cells as one table rule.

See its [fact sheet](../../../components/laws.md#localrulelaw)
and its [API entry][skcomplex.core.components.laws.LocalRuleLaw].

## MatchLaw

A `MatchLaw` pairs the members of one or two groups of agents at every step, and the rules of the step act on the
pairs. Here 5 buyers meet 3 sellers in a market, paired at random, and each trade counts a deal for both.

```python
from typing import Annotated

import numpy as np

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import MatchLaw, RuleLaw
from skcomplex.core.components.laws.utils import matches
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.specs import StateSpec


class TraderElement(Element):
    def __init__(self, label, visual=None, *, deals: Annotated[float, StateSpec()] = 0.0):
        super().__init__(label, visual=visual)
        self.deals = deals


class MarketInteraction(Interaction):
    law_classes = (MatchLaw, RuleLaw)

    def _get_laws(self, namespace):
        return [MatchLaw('buyers', 'sellers', label='trades'), RuleLaw(self.trade)]

    def trade(self, system):
        for buyer, seller in matches(system, 'trades'):
            buyer.deals += 1
            seller.deals += 1


market = CompositeSystem(
    elements=[
        GroupElement('buyers', element=TraderElement, size=5, variable=True),
        GroupElement('sellers', element=TraderElement, size=3, variable=True),
    ],
    interactions=[MarketInteraction('market')],
)
trajectory = simulate(market, until=4, random_state=0)
np.asarray(trajectory.get_agents('buyers', 4).values['deals']).tolist()
```

After four steps the buyers hold 1, 3, 4, 2, and 2 deals. That is 12 deals, three per step, since each seller trades
once per step.

The same law pairs the members of one group, as birds pair in a model of mating. Seven birds form three pairs, and one
bird stays alone.

```python continuation
class CourtshipInteraction(Interaction):
    law_classes = (MatchLaw, RuleLaw)

    def _get_laws(self, namespace):
        return [MatchLaw('birds', None, label='pairs'), RuleLaw(self.mate)]

    def mate(self, system):
        for first, second in matches(system, 'pairs'):
            first.deals += 1
            second.deals += 1


flock = CompositeSystem(
    elements=[GroupElement('birds', element=TraderElement, size=7, variable=True)],
    interactions=[CourtshipInteraction('courtship')],
)
np.asarray(simulate(flock, until=1, random_state=0).get_agents('birds', 1).values['deals']).tolist()
```

Six birds count one pairing, and the fifth bird counts none.

**How the engine runs it.** The law needs discrete time, and a system with it is agent based, so `MesaSolver` runs it.
The pairing runs as a rule of the step, before the rules of its phase, and stores the pairs, which
`matches(system, label)` reads.

**How it combines.** It pairs agents for `RuleLaw`s, which act on the pairs. A `pairing` function gives a stable
matching, an order book, or an auction, and a `PartnerSpec` chooses partners within reach or among neighbours.
`TransferInteraction` and `GroupingInteraction` return match laws.

**Limits and errors.** A member takes at most `capacity` pairs per step, a positive whole number.

```python continuation
from skcomplex.core.errors import ValidationError

try:
    MatchLaw('buyers', 'sellers', capacity=0).prepare()
except ValidationError as error:
    message = str(error)
message
```

The message reads "The match 'match' has capacity 0. Give a positive integer."

**A field realization.** The market above is a realization of economics. A labour market matches workers with firms
by a `pairing` function, and a school choice model matches students with schools.

See its [fact sheet](../../../components/laws.md#matchlaw)
and its [API entry][skcomplex.core.components.laws.MatchLaw].

## NoiseLaw

A `NoiseLaw` adds the amplitude of a random drive, `value dW`, to the highest derivative of a state, where `dW` is the
increment of a Wiener process, the random walk of continuous time. With a `RateLaw` it forms a stochastic differential
equation. A stock price that follows a geometric Brownian motion has the drift `0.05 S` and the amplitude `0.2 S`.

```python
from typing import Annotated

import numpy as np

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.laws import NoiseLaw, RateLaw
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.specs import ParameterSpec, StateSpec


class StockElement(Element):
    law_classes = (RateLaw, NoiseLaw)

    def __init__(
        self,
        label,
        visual=None,
        *,
        price: Annotated[float, StateSpec()] = 100.0,
        mu: Annotated[float, ParameterSpec()] = 0.05,
        sigma: Annotated[float, ParameterSpec()] = 0.2,
    ):
        super().__init__(label, visual=visual)
        self.price = price
        self.mu = mu
        self.sigma = sigma

    def _get_laws(self, namespace):
        price = namespace.price
        return [RateLaw(price, namespace.mu * price), NoiseLaw(price, namespace.sigma * price)]


stocks = GroupElement('stocks', element=StockElement, size=2000)
prices = simulate(CompositeSystem(elements=[stocks]), until=1, dt=0.01, random_state=0).get_values('stocks.price')
round(float(prices[-1].mean()), 1), round(float(np.log(prices[-1] / 100).std()), 2)
```

Over 2000 paths the mean price after one year is 105, which is `100 exp(0.05)`, and the standard deviation of the log
return is 0.2, the volatility.

The same law moves a particle that diffuses in a fluid. With the amplitude `sqrt(2 D)` alone, the variance of the
position grows as `2 D t`.

```python continuation
class ParticleElement(Element):
    law_classes = (NoiseLaw,)

    def __init__(
        self,
        label,
        visual=None,
        *,
        x: Annotated[float, StateSpec()] = 0.0,
        D: Annotated[float, ParameterSpec()] = 0.5,
    ):
        super().__init__(label, visual=visual)
        self.x = x
        self.D = D

    def _get_laws(self, namespace):
        return [NoiseLaw(namespace.x, (2 * namespace.D) ** 0.5)]


particles = GroupElement('particles', element=ParticleElement, size=2000)
positions = simulate(CompositeSystem(elements=[particles]), until=2, dt=0.1, random_state=0).get_values('particles.x')
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

## ProductionLaw

A `ProductionLaw` rewrites every matching symbol of a word by a string at every generation, as an L-system does. The
Koch curve replaces each segment `F` by `F+F--F+F`, where `+` and `-` turn the drawing pen.

```python
from skcomplex.core.components.elements import WordElement
from skcomplex.core.components.laws import ProductionLaw
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate


class KochElement(WordElement):
    law_classes = (ProductionLaw,)

    def _get_laws(self, namespace):
        return [ProductionLaw('F', 'F+F--F+F')]


koch = KochElement('koch', axiom='F')
simulate(CompositeSystem(elements=[koch]), until=3).get_values('koch.size').tolist()
```

The word holds 1, 8, 36, and 148 symbols, since every segment becomes four segments and four turns.

The same law runs a grammar of linguistics. A sentence `S` becomes a noun phrase `N` and a verb phrase `V`, a noun
phrase becomes a determiner `D` and a word `W`, and a verb phrase takes a noun phrase as its object.

```python continuation
class SentenceElement(WordElement):
    law_classes = (ProductionLaw,)

    def _get_laws(self, namespace):
        return [ProductionLaw('S', 'NV'), ProductionLaw('N', 'DW'), ProductionLaw('V', 'VN')]


sentence = SentenceElement('sentence', axiom='S')
simulate(CompositeSystem(elements=[sentence]), until=3).get_values('sentence.size').tolist()
```

The structure holds 1, 2, 4, and 6 symbols.

**How the engine runs it.** The rewriting solver applies every production to every matching symbol at once, one
generation per unit of time, and records the length of the word as the state `size`.

**How it combines.** A `WordElement` returns its productions, which act on its word. A production may read a left or
a right context, a condition on the parameters of a module, or a probability, so stochastic and parametric L-systems
use the same law.

**Limits and errors.** A word grows at every generation, so the element bounds it with `max_size`. The probability of a
stochastic production lies in `[0, 1]`, and `prepare` raises a `RewritingError` otherwise.

```python continuation
from skcomplex.core.errors import RewritingError

try:
    ProductionLaw('F', 'FF', probability=1.5).prepare()
except RewritingError as error:
    message = str(error)
message
```

The message reads "The probability of a production is in [0, 1], not 1.5."

**A field realization.** The [WordElement](elements.md#wordelement) section holds the algae of Lindenmayer, a
realization of botany.

See its [fact sheet](../../../components/laws.md#productionlaw)
and its [API entry][skcomplex.core.components.laws.ProductionLaw].

## RateLaw

A `RateLaw` adds a rate to the highest time derivative of a state, so that a system with only rate laws is a system of
ordinary differential equations. A reactant consumed by a reaction of second order has the rate `-k c^2`.

```python
from typing import Annotated

import numpy as np

from skcomplex.core.components.elements import Element
from skcomplex.core.components.laws import RateLaw
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.specs import ParameterSpec, StateSpec


class ReactantElement(Element):
    law_classes = (RateLaw,)

    def __init__(
        self,
        label,
        visual=None,
        *,
        c: Annotated[float, StateSpec()] = 1.0,
        k: Annotated[float, ParameterSpec()] = 0.5,
    ):
        super().__init__(label, visual=visual)
        self.c = c
        self.k = k

    def _get_laws(self, namespace):
        return [RateLaw(namespace.c, -namespace.k * namespace.c**2)]


reactant = ReactantElement('reactant')
simulate(CompositeSystem(elements=[reactant]), until=4, dt=2.0).get_values('reactant.c').round(3).tolist()
```

The concentration falls to 0.5 and 0.333, which is `1 / (1 + 0.5 t)`.

The same law gives the acceleration of a mass on a spring. On a state of order 2 the rate is the second derivative, so
`-k x` is Hooke's law for a unit mass.

```python continuation
class SpringElement(Element):
    law_classes = (RateLaw,)

    def __init__(
        self,
        label,
        visual=None,
        *,
        x: Annotated[float, StateSpec(order=2)] = 1.0,
        k: Annotated[float, ParameterSpec()] = 4.0,
    ):
        super().__init__(label, visual=visual)
        self.x = x
        self.k = k

    def _get_laws(self, namespace):
        return [RateLaw(namespace.x, -namespace.k * namespace.x)]


spring = SpringElement('spring')
(simulate(CompositeSystem(elements=[spring]), until=np.pi, dt=np.pi / 4).get_values('spring.x').round(3) + 0.0).tolist()
```

The position reads 1, 0, -1, 0, and 1 over one period, which is `cos(2 t)`.

**How the engine runs it.** The law needs continuous time. Rates on the same state add, and the compiler writes a
state of order `k` as `k` states of order 1. `ScipyODESolver` integrates the result by default, and fixed step solvers
such as `RK4Solver` take the same problem.

**How it combines.** Every mechanism adds its own rate law, so a growth from an element and a predation from an
interaction superpose on one state. `RateInteraction`, `CouplingInteraction`, `PairInteraction`, and
`GradientInteraction` return rate laws. A `NoiseLaw` adds a random part.

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

**A field realization.** A field names a rate in an interaction of its own, such as the recovery of an epidemic.

```python continuation
from skcomplex.core.components.elements import DeclaredElement
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.specs import DeclaredVariableSpec, RoleSpec


class RecoveryInteraction(Interaction):
    law_classes = (RateLaw,)

    def __init__(
        self,
        label,
        *,
        infected: Annotated[str, RoleSpec()],
        gamma: Annotated[float, ParameterSpec()] = 0.1,
    ):
        super().__init__(label)
        self.infected = infected
        self.gamma = gamma

    def _get_laws(self, namespace):
        return [RateLaw(namespace.infected.x, -namespace.gamma * namespace.infected.x)]


infected = DeclaredElement('infected', variables={'x': DeclaredVariableSpec(100.0, StateSpec())})
epidemic = CompositeSystem(elements=[infected], interactions=[RecoveryInteraction('recovery', infected='infected')])
simulate(epidemic, until=10, dt=10.0).get_values('infected.x').round(1).tolist()
```

The infected fall from 100 to 36.8 in ten days.

See its [fact sheet](../../../components/laws.md#ratelaw)
and its [API entry][skcomplex.core.components.laws.RateLaw].

## RuleLaw

A `RuleLaw` runs Python code for every member of a group of agents once per step, so behaviour can branch, remember,
and search. Here 20 sheep wander a pasture of 10 by 10 cells. Each step a sheep moves to a neighbouring cell, keeps or
loses one unit of energy at random, and dies when its energy runs out.

```python
from typing import Annotated

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.laws import RuleLaw
from skcomplex.core.components.spaces import LatticeSpace
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.specs import StateSpec


class SheepElement(Element):
    law_classes = (RuleLaw,)

    def __init__(self, label, visual=None, *, energy: Annotated[float, StateSpec()] = 3.0):
        super().__init__(label, visual=visual)
        self.energy = energy

    def _get_laws(self, namespace):
        return [RuleLaw(self.graze)]

    def graze(self, agent, system):
        cells = agent.get_neighbor_cells()
        agent.move_to(cells[int(system.rng.integers(len(cells)))])
        agent.energy += system.rng.integers(2) - 1
        if agent.energy <= 0:
            agent.remove()


flock = GroupElement('flock', element=SheepElement, size=20, variable=True)
pasture = CompositeSystem(elements=[flock], space=LatticeSpace((10, 10), boundary='periodic'))
simulate(pasture, until=10, random_state=0).get_values('flock.count').astype(int).tolist()
```

The flock counts 20 sheep for three steps, then falls to 15, 12, 8, 5, 5, 3, 1, and 0, since no sheep finds food.

The same law runs a market of ten traders. Each step every trader with at least one unit of money gives one unit to a
trader drawn at random.

```python continuation
class TraderElement(Element):
    law_classes = (RuleLaw,)

    def __init__(self, label, visual=None, *, cash: Annotated[float, StateSpec()] = 10.0):
        super().__init__(label, visual=visual)
        self.cash = cash

    def _get_laws(self, namespace):
        return [RuleLaw(self.give)]

    def give(self, agent, system):
        if agent.cash >= 1:
            others = system.get_agents('traders')
            other = others[int(system.rng.integers(len(others)))]
            agent.cash -= 1
            other.cash += 1


traders = GroupElement('traders', element=TraderElement, size=10)
trajectory = simulate(CompositeSystem(elements=[traders]), until=50, random_state=0)
cash = trajectory.get_agents('traders', 50).values['cash']
float(sum(cash)), float(max(cash))
```

After 50 steps the total is still 100, and the richest trader holds 18, nearly twice the even share.

**How the engine runs it.** The law needs discrete time, and a system with rules is agent based. `MesaSolver` runs
one Mesa step per unit of time and activates every agent in random order. A rule of an element receives the agent and
the system, and a rule of an interaction receives the system.

**How it combines.** An element whose laws include rules is an agent. Rules read the pairs of a `MatchLaw`, order
themselves by `phase` within a step, and run beside `UpdateLaw`s of other components. The system gives a rule its random
generator `system.rng` and the agents of other groups.

**Limits and errors.** A rule runs arbitrary Python, so the compiler can neither lower nor differentiate it. The
function must be callable.

```python continuation
from skcomplex.core.errors import ValidationError

try:
    RuleLaw('graze').prepare()
except ValidationError as error:
    message = str(error)
message
```

The message reads "The function of a rule must be callable, but it is 'graze'." The fix is the method itself,
`RuleLaw(self.graze)`.

**A field realization.** The sheep above are a realization of ecology, and the traders one of economics.

See its [fact sheet](../../../components/laws.md#rulelaw)
and its [API entry][skcomplex.core.components.laws.RuleLaw].

## SampleLaw

A `SampleLaw` makes one Monte Carlo sweep of a lattice per unit of time. It proposes moves at the sites and accepts
them by the change of an objective, so the states follow a Boltzmann distribution. Below the critical temperature the
spins of an Ising magnet align.

```python
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import ZZ

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.laws import SampleLaw
from skcomplex.core.components.laws.utils import neighbors
from skcomplex.core.components.spaces import BallNeighborhood, LatticeSpace
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.specs import MoveSpec, ObjectiveSpec, ParameterSpec, SamplerSpec, StateSpec

nearest = BallNeighborhood(metric='manhattan')


class SpinElement(Element):
    law_classes = (SampleLaw,)

    def __init__(
        self,
        label,
        visual=None,
        *,
        spin: Annotated[float, StateSpec(domain=ZZ)] = 1.0,
        temperature: Annotated[float, ParameterSpec()] = 1.5,
    ):
        super().__init__(label, visual=visual)
        self.spin = spin
        self.temperature = temperature

    def _get_laws(self, namespace):
        energy = -namespace.spin * sum(neighbors(namespace.spin, nearest, reduce='each'))
        sampler = SamplerSpec('metropolis', 1 / namespace.temperature)
        return [SampleLaw(ObjectiveSpec(energy, nearest), MoveSpec(namespace.spin, 'flip'), sampler)]


magnet = GroupElement('magnet', element=SpinElement, space=LatticeSpace((16, 16)), on='sites')
spins = simulate(CompositeSystem(elements=[magnet]), until=50, dt=50, random_state=0).get_values('magnet.spin')
round(float(spins[-1].mean()), 2)
```

After 50 sweeps at temperature 1.5 the magnetisation is 0.99.

The same law samples cells of three kinds in a tissue, the adhesion part of the cellular Potts model of biology. A site
whose kind matches a neighbour lowers the energy, so the kinds sort into domains. The full model adds a term that keeps
every cell near a target area.

```python continuation
from skcomplex.core.components.laws.utils import where


class TissueElement(Element):
    law_classes = (SampleLaw,)

    def __init__(self, label, visual=None, *, kind: Annotated[float, StateSpec(domain=ZZ)] = 0.0):
        super().__init__(label, visual=visual)
        self.kind = kind

    def _get_laws(self, namespace):
        kind = namespace.kind
        energy = -neighbors(where(namespace.other.kind == kind, 1, 0), over=nearest)
        move = MoveSpec(kind, 'random-value', values=(0.0, 1.0, 2.0))
        return [SampleLaw(ObjectiveSpec(energy, nearest), move, SamplerSpec('metropolis', 2.0))]


def alike(grid):
    return float(((grid == np.roll(grid, 1, axis=0)).mean() + (grid == np.roll(grid, 1, axis=1)).mean()) / 2)


kinds = np.random.default_rng(0).integers(3, size=(16, 16)).astype(float)
tissue = GroupElement('tissue', element=TissueElement, space=LatticeSpace((16, 16)), on='sites', values={'kind': kinds})
cells = simulate(CompositeSystem(elements=[tissue]), until=50, dt=50, random_state=0).get_lattice('tissue.kind')
round(alike(cells[0]), 2), round(alike(cells[-1]), 2)
```

The share of neighbouring sites of the same kind rises from 0.35, about one third as in a random tissue, to 0.93.

**How the engine runs it.** The law needs discrete time and a lattice or a network. `MonteCarloSolver` makes one sweep
per unit of time. The change of the objective is derived symbolically from the local objective of a site, and the
sweep updates the two sublattices of a square lattice in turn by default. The intensity of the sampler is an
expression of parameters, so a schedule of `simulate` can anneal it.

**How it combines.** `SamplingInteraction` returns it from an interaction. `MoveSpec` chooses the move, a flip, a new
value, a rotation, a Gaussian step, or a cluster, and `SamplerSpec` the acceptance, Metropolis, Glauber, or heat bath.

**Limits and errors.** A sweep samples a distribution, so the frames are not a physical time course. An unknown move
raises a `ValidationError` from `prepare`.

```python continuation
from sage.all__sagemath_symbolics import var

from skcomplex.core.errors import ValidationError

spin = var('spin')
try:
    SampleLaw(ObjectiveSpec(-spin, nearest), MoveSpec(spin, 'swap')).prepare()
except ValidationError as error:
    message = str(error)
message
```

The message names the unknown move `'swap'` and lists the six moves.

**A field realization.** The magnet above is a realization of statistical physics, and the tissue one of biology.

See its [fact sheet](../../../components/laws.md#samplelaw)
and its [API entry][skcomplex.core.components.laws.SampleLaw].

## Sink

A `Sink` is a holder outside the system that a transfer releases an amount to. It has no value and never fills. Here a
tank drains at the rate `0.2` times its water, and the drain receives what the tank loses.

```python
from typing import Annotated

from skcomplex.core.analysis import loops
from skcomplex.core.components.elements import Element
from skcomplex.core.components.laws import Sink, TransferLaw
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.specs import ParameterSpec, StateSpec


class TankElement(Element):
    law_classes = (TransferLaw,)

    def __init__(
        self,
        label,
        visual=None,
        *,
        water: Annotated[float, StateSpec()] = 100.0,
        k: Annotated[float, ParameterSpec()] = 0.2,
    ):
        super().__init__(label, visual=visual)
        self.water = water
        self.k = k

    def _get_laws(self, namespace):
        return [TransferLaw(namespace.water, Sink('drain'), namespace.k * namespace.water, label='outflow')]


trajectory = simulate(CompositeSystem(elements=[TankElement('tank')]), until=5, dt=5.0)
flows = loops(trajectory, kind='flow-matrix', frame=0)
round(float(trajectory.get_values('tank.water')[-1]), 2), flows.holders, flows.matrix.tolist()
```

The tank holds 36.79 after five units of time, which is `100 exp(-1)`. The transaction-flow matrix of the first frame
has the tank and the drain as columns, and its one row moves 20 per unit of time from the tank to the drain.

The same holder takes the deaths out of a population. A cohort of 1000 people loses 5 percent a year.

```python continuation
class CohortElement(Element):
    law_classes = (TransferLaw,)

    def __init__(
        self,
        label,
        visual=None,
        *,
        n: Annotated[float, StateSpec()] = 1000.0,
        d: Annotated[float, ParameterSpec()] = 0.05,
    ):
        super().__init__(label, visual=visual)
        self.n = n
        self.d = d

    def _get_laws(self, namespace):
        return [TransferLaw(namespace.n, Sink('deaths'), namespace.d * namespace.n, label='dying')]


cohort = simulate(CompositeSystem(elements=[CohortElement('cohort')]), until=10, dt=10.0)
round(float(cohort.get_values('cohort.n')[-1]), 1)
```

After ten years 606.5 people remain, which is `1000 exp(-0.5)`.

**How the engine runs it.** A sink holds no state, so the transfer lowers to a loss of its source alone. Its label
names the column of the environment in a transaction-flow matrix.

**How it combines.** It is the sink of a `TransferLaw` or of a leg of a `TransferInteraction`, and a `Source` is its
counterpart for what enters.

**Limits and errors.** A sink receives and never gives, so it cannot be the source of a transfer.

```python continuation
from sage.all__sagemath_symbolics import var

from skcomplex.core.errors import ValidationError

try:
    TransferLaw(Sink(), var('water'), 1.0).prepare()
except ValidationError as error:
    message = str(error)
message
```

The message says that the source of a transfer must be the symbol of a scalar state, a symbol read at a member offset,
or `Source()`. The fix is `Source()` for an inflow from outside.

**A field realization.** A field names its sinks by their meaning, such as `Sink('atmosphere')` for the emissions of a
carbon cycle.

See its [API entry][skcomplex.core.components.laws.Sink]. It is a spec, not a component, so it has no fact sheet.

## Source

A `Source` is a holder outside the system that a transfer takes an amount from. It has no value and never runs out.
Here a river feeds a lake at 3 units per day, and the lake releases a tenth of its volume per day to the sea.

```python
from typing import Annotated

from skcomplex.core.components.elements import Element
from skcomplex.core.components.laws import Sink, Source, TransferLaw
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.specs import ParameterSpec, StateSpec


class ReservoirElement(Element):
    law_classes = (TransferLaw,)

    def __init__(
        self,
        label,
        visual=None,
        *,
        volume: Annotated[float, StateSpec()] = 0.0,
        inflow: Annotated[float, ParameterSpec()] = 3.0,
        k: Annotated[float, ParameterSpec()] = 0.1,
    ):
        super().__init__(label, visual=visual)
        self.volume = volume
        self.inflow = inflow
        self.k = k

    def _get_laws(self, namespace):
        volume = namespace.volume
        return [
            TransferLaw(Source('river'), volume, namespace.inflow, label='inflow'),
            TransferLaw(volume, Sink('sea'), namespace.k * volume, label='release'),
        ]


lake = simulate(CompositeSystem(elements=[ReservoirElement('lake')]), until=50, dt=25.0)
lake.get_values('lake.volume').round(2).tolist()
```

The volume grows from 0 to 27.54 and 29.8, toward the balance `3 / 0.1 = 30`.

The same holder is a bank that creates money. In discrete time the amount of a transfer is moved once per step, so the
bank lends 50 every step.

```python continuation
from skcomplex.core.components.spaces import DiscreteTime


class LedgerElement(Element):
    law_classes = (TransferLaw,)

    def __init__(self, label, visual=None, *, deposits: Annotated[float, StateSpec()] = 0.0):
        super().__init__(label, visual=visual)
        self.deposits = deposits

    def _get_laws(self, namespace):
        return [TransferLaw(Source('bank'), namespace.deposits, 50.0, label='lending')]


ledger = CompositeSystem(elements=[LedgerElement('ledger')], time=DiscreteTime())
simulate(ledger, until=4).get_values('ledger.deposits').tolist()
```

The deposits grow by 50 per step, from 0 to 200.

**How the engine runs it.** A source holds no state, so the transfer lowers to a gain of its sink alone. Its label
names the column of the environment in a transaction-flow matrix.

**How it combines.** It is the source of a `TransferLaw` or of a leg of a `TransferInteraction`, and a `Sink` is its
counterpart for what leaves.

**Limits and errors.** A source gives and never receives, so it cannot be the sink of a transfer.

```python continuation
from sage.all__sagemath_symbolics import var

from skcomplex.core.errors import ValidationError

try:
    TransferLaw(var('volume'), Source(), 1.0).prepare()
except ValidationError as error:
    message = str(error)
message
```

The message says that the sink of a transfer must be the symbol of a scalar state, a symbol read at a member offset, or
`Sink()`. The fix is `Sink()` for an outflow.

**A field realization.** A field names its sources by their meaning, such as `Source('births')` for a population.

See its [API entry][skcomplex.core.components.laws.Source]. It is a spec, not a component, so it has no fact sheet.

## SubstitutionLaw

A `SubstitutionLaw` replaces every tile of a prototile, a shape of tile, by smaller placed pieces at every generation.
The Sierpinski carpet keeps eight of the nine thirds of every square.

```python
from skcomplex.core.components.elements import PatchElement
from skcomplex.core.components.laws import SubstitutionLaw
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.specs import AffineSpec, PrototileSpec


class CarpetElement(PatchElement):
    law_classes = (SubstitutionLaw,)

    def _get_laws(self, namespace):
        corners = [(i / 3, j / 3) for i in range(3) for j in range(3) if (i, j) != (1, 1)]
        pieces = [('square', AffineSpec(scale=1 / 3, translation=corner)) for corner in corners]
        return [SubstitutionLaw('square', pieces)]


square = PrototileSpec('square', [(0, 0), (1, 0), (1, 1), (0, 1)])
carpet = CarpetElement('carpet', prototiles=[square])
simulate(CompositeSystem(elements=[carpet]), until=3).get_values('carpet.size').tolist()
```

The carpet holds 1, 8, 64, and 512 squares.

The same law subdivides the square tiling of the plane, which is its own substitution tiling. Every square becomes four
squares of half the size.

```python continuation
class RefinementElement(PatchElement):
    law_classes = (SubstitutionLaw,)

    def _get_laws(self, namespace):
        corners = [(0, 0), (1 / 2, 0), (0, 1 / 2), (1 / 2, 1 / 2)]
        pieces = [('square', AffineSpec(scale=1 / 2, translation=corner)) for corner in corners]
        return [SubstitutionLaw('square', pieces)]


grid = RefinementElement('grid', prototiles=[square])
simulate(CompositeSystem(elements=[grid]), until=3).get_values('grid.size').tolist()
```

The tiling holds 1, 4, 16, and 64 squares.

**How the engine runs it.** The rewriting solver applies every substitution to every tile of its prototile at once,
one generation per unit of time, and records the number of tiles as the state `size`.

**How it combines.** A `PatchElement` returns its substitutions, and several prototiles with one law each build a
tiling of several shapes, as the Penrose tiling does.

**Limits and errors.** Each piece pairs a prototile name with an `AffineSpec`, the map from the frame of the parent
tile, and `prepare` raises a `RewritingError` for any other piece.

```python continuation
from skcomplex.core.errors import RewritingError

try:
    SubstitutionLaw('square', [('square', 1 / 3)]).prepare()
except RewritingError as error:
    message = str(error)
message
```

The message says that a piece is a prototile name and an `AffineSpec`, not `('square', 0.3333333333333333)`. The fix
is `('square', AffineSpec(scale=1 / 3))`.

**A field realization.** The [PatchElement](elements.md#patchelement) section holds the Cantor set and the Sierpinski
triangle, realizations of mathematics.

See its [fact sheet](../../../components/laws.md#substitutionlaw)
and its [API entry][skcomplex.core.components.laws.SubstitutionLaw].

## TransferLaw

A `TransferLaw` moves an amount from a source holder to a sink holder, so what one loses the other gains, and the run
records the flow. In an epidemic, people move from susceptible to infected and from infected to recovered.

```python
from typing import Annotated

from skcomplex.core.components.elements import Element
from skcomplex.core.components.laws import TransferLaw
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.specs import ParameterSpec, StateSpec


class EpidemicElement(Element):
    law_classes = (TransferLaw,)

    def __init__(
        self,
        label,
        visual=None,
        *,
        s: Annotated[float, StateSpec()] = 990.0,
        i: Annotated[float, StateSpec()] = 10.0,
        r: Annotated[float, StateSpec()] = 0.0,
        beta: Annotated[float, ParameterSpec()] = 0.0003,
        gamma: Annotated[float, ParameterSpec()] = 0.1,
    ):
        super().__init__(label, visual=visual)
        self.s = s
        self.i = i
        self.r = r
        self.beta = beta
        self.gamma = gamma

    def _get_laws(self, namespace):
        s, i, r = namespace.s, namespace.i, namespace.r
        return [
            TransferLaw(s, i, namespace.beta * s * i, label='infections'),
            TransferLaw(i, r, namespace.gamma * i, label='recoveries'),
        ]


town = CompositeSystem(elements=[EpidemicElement('town')])
trajectory = simulate(town, until=60, dt=30.0)
states = [trajectory.get_values(f'town.{name}')[-1] for name in ('s', 'i', 'r')]
[round(float(value)) for value in states], round(float(sum(states)), 6)
```

After 60 days, in rounded numbers, 68 people are susceptible, 40 infected, and 891 recovered. The three states still
sum to 1000, since every transfer conserves what it moves.

The same law saves money from a checking account to a savings account. In discrete time the amount, a tenth of the
checking account, moves once per month.

```python continuation
from skcomplex.core.components.spaces import DiscreteTime


class AccountsElement(Element):
    law_classes = (TransferLaw,)

    def __init__(
        self,
        label,
        visual=None,
        *,
        checking: Annotated[float, StateSpec()] = 1000.0,
        savings: Annotated[float, StateSpec()] = 0.0,
        share: Annotated[float, ParameterSpec()] = 0.1,
    ):
        super().__init__(label, visual=visual)
        self.checking = checking
        self.savings = savings
        self.share = share

    def _get_laws(self, namespace):
        amount = namespace.share * namespace.checking
        return [TransferLaw(namespace.checking, namespace.savings, amount, label='saving')]


household = CompositeSystem(elements=[AccountsElement('bank')], time=DiscreteTime())
months = simulate(household, until=3)
months.get_values('bank.savings').round(2).tolist(), months.get_values('bank.saving').round(2).tolist()
```

The savings reach 100, 190, and 271. The recorded flow `bank.saving` holds the amount of each month, 100, 90, 81, and
72.9 at the frames.

**How the engine runs it.** The law lowers to a loss of its source and an equal gain of its sink. The amount is a rate
in continuous time, an amount per step in discrete time, and the size of each jump when `rate` is given. The run
records the amount at every frame under `<path>.<label>`.

**How it combines.** `Source` and `Sink` stand for the environment on either side. A holder read by `shift_member`
moves amounts along a group, such as stock along a supply chain. `TransferInteraction` returns transfer laws from an
interaction, and the analysis `loops` reads their flows as a transaction-flow matrix.

**Limits and errors.** A transfer changes a state on at least one side, so a transfer from a `Source` to a `Sink`
raises a `ValidationError`.

```python continuation
from skcomplex.core.components.laws import Sink, Source
from skcomplex.core.errors import ValidationError

try:
    TransferLaw(Source(), Sink(), 1.0).prepare()
except ValidationError as error:
    message = str(error)
message
```

The message reads "A transfer moves a quantity into or out of a state, but both of its holders are the environment."

**A field realization.** The epidemic above is a realization of epidemiology, and the accounts one of finance.

See its [fact sheet](../../../components/laws.md#transferlaw)
and its [API entry][skcomplex.core.components.laws.TransferLaw].

## TransitionLaw

A `TransitionLaw` draws the next value of a discrete state from the row of a transition matrix that its current value
selects, the step of a Markov chain. Weather moves between dry and wet days. A dry day stays dry with probability 0.8,
and a wet day turns dry with probability 0.4.

```python
from typing import Annotated

from sage.sets.finite_enumerated_set import FiniteEnumeratedSet

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.laws import TransitionLaw
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.specs import ParameterSpec, StateSpec


class WeatherElement(Element):
    law_classes = (TransitionLaw,)

    def __init__(
        self,
        label,
        visual=None,
        *,
        sky: Annotated[str, StateSpec(domain=FiniteEnumeratedSet(['dry', 'wet']))] = 'dry',
        stay: Annotated[float, ParameterSpec()] = 0.8,
    ):
        super().__init__(label, visual=visual)
        self.sky = sky
        self.stay = stay

    def _get_laws(self, namespace):
        return [TransitionLaw(namespace.sky, [[namespace.stay, 1 - namespace.stay], [0.4, 0.6]])]


days = simulate(CompositeSystem(elements=[WeatherElement('weather')]), until=5000, random_state=0)
round(float((days.get_values('weather.sky') == 'wet').mean()), 2)
```

Over 5000 days 0.34 of them are wet, close to the stationary share `0.2 / (0.2 + 0.4) = 1/3`.

The same law moves the credit rating of loans between grades. Default absorbs a loan, so its row keeps it there.

```python continuation
grades = FiniteEnumeratedSet(['good', 'poor', 'default'])


class RatingElement(Element):
    law_classes = (TransitionLaw,)

    def __init__(self, label, visual=None, *, grade: Annotated[str, StateSpec(domain=grades)] = 'good'):
        super().__init__(label, visual=visual)
        self.grade = grade

    def _get_laws(self, namespace):
        return [TransitionLaw(namespace.grade, [[0.9, 0.08, 0.02], [0.2, 0.7, 0.1], [0.0, 0.0, 1.0]])]


loans = GroupElement('loans', element=RatingElement, size=1000)
years = simulate(CompositeSystem(elements=[loans]), until=10, random_state=0).get_values('loans.grade')
int((years[-1] == 'default').sum())
```

Of 1000 loans that start in good standing, 291 are in default after ten years.

**How the engine runs it.** The law needs discrete time. A transition is a kind of update, so `IterationSolver` draws
the next value of every member from its row at each step, with the random stream of the run. With `source`, the law
draws an observed state from the row that a hidden state selects, as a hidden Markov model does.

**How it combines.** It runs beside `UpdateLaw`s of other states in the same step. `TransitionInteraction` returns it
from an interaction, and a matrix may hold expressions of parameters, such as `stay` above.

**Limits and errors.** The target takes a finite set of values. The matrix is a nonempty list of rows of equal length,
and `prepare` raises a `ValidationError` for any other shape.

```python continuation
from sage.all__sagemath_symbolics import var

from skcomplex.core.errors import ValidationError

try:
    TransitionLaw(var('sky'), [[0.8, 0.2], [0.4]]).prepare()
except ValidationError as error:
    message = str(error)
message
```

The message says that the matrix must be a nonempty sequence of rows of equal length and shows the ragged matrix. The
fix is the full second row `[0.4, 0.6]`.

**A field realization.** The weather above is a realization of meteorology, and the ratings one of finance.

See its [fact sheet](../../../components/laws.md#transitionlaw)
and its [API entry][skcomplex.core.components.laws.TransitionLaw].

## UpdateLaw

An `UpdateLaw` gives the next value of a state after one step of discrete time. The logistic map follows a population
of insects from year to year, `x' = r x (1 - x)`.

```python
from typing import Annotated

from skcomplex.core.components.elements import Element
from skcomplex.core.components.laws import UpdateLaw
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.specs import ParameterSpec, StateSpec


class LogisticElement(Element):
    law_classes = (UpdateLaw,)

    def __init__(
        self,
        label,
        visual=None,
        *,
        x: Annotated[float, StateSpec()] = 0.2,
        r: Annotated[float, ParameterSpec()] = 2.5,
    ):
        super().__init__(label, visual=visual)
        self.x = x
        self.r = r

    def _get_laws(self, namespace):
        return [UpdateLaw(namespace.x, namespace.r * namespace.x * (1 - namespace.x))]


insects = LogisticElement('insects')
simulate(CompositeSystem(elements=[insects]), until=30).get_values('insects.x')[-3:].round(3).tolist()
```

At `r = 2.5` the population settles at the fixed point `1 - 1 / r = 0.6`.

The same law compounds interest monthly. A balance of 1000 at a yearly rate of 6 percent grows by half a percent each
month.

```python continuation
class AccountElement(Element):
    law_classes = (UpdateLaw,)

    def __init__(
        self,
        label,
        visual=None,
        *,
        balance: Annotated[float, StateSpec()] = 1000.0,
        rate: Annotated[float, ParameterSpec()] = 0.06,
    ):
        super().__init__(label, visual=visual)
        self.balance = balance
        self.rate = rate

    def _get_laws(self, namespace):
        return [UpdateLaw(namespace.balance, namespace.balance * (1 + namespace.rate / 12))]


account = AccountElement('account')
round(float(simulate(CompositeSystem(elements=[account]), until=12).get_values('account.balance')[-1]), 2)
```

After twelve months the balance is 1061.68, which is `1000 * 1.005^12`.

**How the engine runs it.** The law needs discrete time. `IterationSolver` computes every update from the states at
the start of the step, so the order of the laws does not matter. With `order`, the members of a group update one after
the other, and with `phase`, the law runs in a phase of the step.

**How it combines.** A state takes one update law, unless each of its laws says how it combines: an `increment` law
adds a change, and a law with a `source` changes only members that hold that value. `CouplingInteraction`,
`MapInteraction`, and `JumpInteraction` return update laws, and update laws take random draws, such as `normal`.

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
and every other model chooses an existing law, such as `RateLaw` or `UpdateLaw`. The main public methods `prepare` and
`run` set the states a law changes, the symbols it reads, and its value at given values of those symbols.

```python
from sage.all__sagemath_symbolics import var

from skcomplex.core.components.laws import BaseLaw, RateLaw, UpdateLaw

x, k = var('x k')
law = RateLaw(x, -k * x).prepare('decay')
issubclass(UpdateLaw, BaseLaw), law.targets_, sorted(map(str, law.reads_)), law.run({'x': 2.0, 'k': 0.5}).value_
```

Every law derives from `BaseLaw`. The rate law above changes `x`, reads `k` and `x`, and has the value -1.0 at
`x = 2` and `k = 0.5`.

See its [fact sheet](../../../components/laws.md#baselaw)
and its [API entry][skcomplex.core.components.laws.BaseLaw].
