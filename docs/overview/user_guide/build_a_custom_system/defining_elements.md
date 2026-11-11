# Defining elements

This page writes the element and interaction classes of a system of your own. A class declares its states,
parameters, and [roles](../glossary.md#role) as arguments of its `__init__`, and returns its
[laws](../glossary.md#law) from `_get_laws`. A state is a number that changes as the system runs, a parameter is a
number that stays fixed during a run, and a role is a slot of an interaction that names the element it acts on.

Here a deer population is a subclass of [`Element`](../core_components/elements.md#element) with one nonnegative state
`n`, and births are a subclass of [`Interaction`](../core_components/interactions.md#interaction) that adds the rate
`rate * n` through a [`RateLaw`](../core_components/laws.md#ratelaw).

```python
from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet
from sklearn.base import clone

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import RateLaw
from skcomplex.core.specs import ParameterSpec, RoleSpec, StateSpec


class SpeciesElement(Element):
    def __init__(self, label, *, n: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 10.0):
        super().__init__(label)
        self.n = n


class BirthInteraction(Interaction):
    def __init__(
        self,
        label,
        *,
        target: Annotated[str, RoleSpec()],
        rate: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
    ):
        super().__init__(label)
        self.target = target
        self.rate = rate

    def _get_laws(self, s):
        return [RateLaw(s.target.n, s.rate * s.target.n)]


deer = SpeciesElement('deer', n=40.0)
birth = BirthInteraction('birth', target='deer', rate=0.2)
deer.n, clone(birth).rate
```

The deer start at 40.0, and the clone of the births keeps the rate 0.2. Every element class derives from `Element`,
and every interaction class from `Interaction`. Both, like laws and systems, derive from `BaseComponent`, which is a
scikit-learn estimator. A class annotates each variable with `typing.Annotated` and a spec from
`skcomplex.core.specs`. The class follows these rules.

- `StateSpec` marks a state, `ParameterSpec` a parameter, and `RoleSpec` a role of an interaction. The default of the
  argument is the initial value of a state or the default of a parameter. A role has no default.
- The constructor takes the label first and every other argument by keyword. It stores each value unmodified under its
  own name.
- The domain of a state or a parameter is a Sage set, such as `RR`, `ZZ`, `CC`, `RealSet.closed_open(0, oo)`, or
  `FiniteEnumeratedSet(['S', 'I', 'R'])`.
- `_get_laws(s)` returns the laws of the class, and returns none by default.
- `get_params`, `set_params`, `clone`, and `repr` follow the scikit-learn estimator contract, so every part clones with
  its values.

Inside `_get_laws`, `s` is the namespace of symbols. `s.rate` is the symbol of the parameter, and `s.target.n` is the
symbol of the state of the element bound to the role `target`. Parameters stay symbols, so the `schedule` of
`simulate` can change them during a run without compiling again.

## Constructor settings

This section adds a setting to a class, an argument that is neither a state, a parameter, nor a role. It is an
argument of `__init__` without a spec, such as a switch between two sets of laws, a radius, or a layer. Here a switch
turns the births into random events of a [`JumpLaw`](../core_components/laws.md#jumplaw).

```python continuation
from skcomplex.core.components.laws import JumpLaw


class SwitchedBirthInteraction(BirthInteraction):
    def __init__(self, label, *, target, rate=0.5, stochastic=False):
        super().__init__(label, target=target, rate=rate)
        self.stochastic = stochastic

    def _get_laws(self, s):
        if self.stochastic:
            return [JumpLaw(s.rate * s.target.n, {s.target.n: 1})]
        return super()._get_laws(s)


clone(SwitchedBirthInteraction('birth', target='deer', stochastic=True))
```

The clone shows `SwitchedBirthInteraction(label='birth', stochastic=True, target='deer')`, so `get_params` sees the
setting beside the role. A setting follows these rules.

- The setting is a keyword-only argument with a default.
- `__init__` stores it unmodified under its own name, like every other argument.
- A subclass that adds a setting repeats the arguments of its base in its own `__init__`, so that `get_params` sees them
  all. An argument repeated without its annotation keeps the spec of the base.

## States and parameters from data

This section builds an element whose variables come from data, such as a fitted surrogate or a metabolic network.
Such an element cannot name its variables in its signature.
[`DeclaredElement`](../core_components/elements.md#declaredelement) takes them in one argument, `variables`, instead.
Here two states `a` and `b` decay at the rate `k`.

```python continuation
from skcomplex.core.components.elements import DeclaredElement
from skcomplex.core.components.spaces import ContinuousTime
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.specs import DeclaredVariableSpec


class DecayElement(DeclaredElement):
    def _get_laws(self, s):
        return [RateLaw(getattr(s, name), -s.k * getattr(s, name)) for name in ('a', 'b')]


declared = {
    'a': DeclaredVariableSpec(1.0, StateSpec()),
    'b': DeclaredVariableSpec(2.0, StateSpec()),
    'k': DeclaredVariableSpec(0.5, ParameterSpec()),
}
decay = DecayElement('decay', variables=declared)
faster = clone(decay).set_params(k=1.0)
decay.get_value('k'), faster.get_value('k')
```

The element keeps the rate 0.5, and its clone takes the rate 1.0 under the name of the declared variable. A
[`CompositeSystem`](../core_components/systems.md#compositesystem) in
[`ContinuousTime`](../core_components/spaces.md#continuoustime) runs the element like any other.

```python continuation
system = CompositeSystem(elements=[decay], time=ContinuousTime())
simulate(system, until=1.0, dt=0.1).get_values('decay.b')[-1]
```

The state `b` falls from 2.0 to 1.213 at time 1, which is `2 exp(-0.5)`. Declared variables follow these rules.

- `variables` maps every name to a `DeclaredVariableSpec`, which holds the initial value and a `StateSpec` or a
  `ParameterSpec`. The compiler treats a declared variable as it treats an annotated argument.
- A law reads a declared variable from its namespace, and `get_value` reads its value from the element.
- `set_params` takes the name of a declared variable and changes its value, and `clone` copies the variables.
- A group or a field takes the element class and gives `variables` in its `values`, next to the values per member.
- A subclass that derives its variables from other arguments overrides `_declare_variables`.

## Writing laws

This section prepares and runs a law on its own, to see what it changes, what it reads, and what value it returns. A
law says how the states it targets change. Every law class derives from
[`BaseLaw`](../core_components/laws.md#baselaw).

```python
from sage.all__sagemath_symbolics import var

from skcomplex.core.components.laws import BaseLaw, RateLaw

x, k = var('x k')
law = RateLaw(x, -k * x).prepare('decay')
issubclass(RateLaw, BaseLaw), law.targets_, sorted(map(str, law.reads_)), law.owner_
```

The law is a `BaseLaw`, targets `x`, reads `k` and `x`, and belongs to `decay`.

```python continuation
law.run({'x': 2.0, 'k': 0.5}).value_
```

At `x = 2` and `k = 0.5` its value is -1.0. A law takes the states it changes and the expressions it reads as arguments
of its `__init__`, and an optional `label` by keyword. It has two main public methods.

- `prepare(owner)` validates the arguments, such as a value that is a NumPy array instead of an expression. It sets
  `targets_`, the states the law changes, `reads_`, the symbols it reads, and `owner_`, the path of the element or
  interaction that returns it.
- `run(values)` sets `value_`, the value of the law at the values of the symbols it reads, given by name.

Every law states what it requires of a system, such as continuous time for `RateLaw` and discrete time for
[`UpdateLaw`](../core_components/laws.md#updatelaw). Validation reports a system whose laws require different times.

A law of one state, such as `RateLaw`, takes the state as its first argument. A law that updates several states
together, such as [`LocalRuleLaw`](../core_components/laws.md#localrulelaw), takes a tuple of them. Laws that name their states in other values, such as the
increments of a `JumpLaw`, change the states they name.

A system is defined by its elements, its interactions, and the laws they return from `_get_laws(s)`, and by nothing
else. Two kinds of value stay outside it.

- A value that only names an expression is a Python expression in the laws, such as `leverage = s.debt / s.equity`. A
  function of the namespace shares it between laws.
- What a run records or checks is an argument of the run. `simulate(system, observables=[...])` records named values,
  and `simulate(system, invariants={name: tolerance})` checks that the observables it names stay constant.

A law can also be a Python function of time and the current values, which covers rules that have no closed form. The
values are keyed by dotted state and parameter name. Here a cull removes one deer a year while the herd holds more than
50.

```python
from typing import Annotated

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import RateLaw
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.specs import RoleSpec, StateSpec


class PopulationElement(Element):
    def __init__(self, label, *, n: Annotated[float, StateSpec()] = 1.0):
        super().__init__(label)
        self.n = n


class CullInteraction(Interaction):
    def __init__(self, label, *, target: Annotated[str, RoleSpec()]):
        super().__init__(label)
        self.target = target

    def _get_laws(self, s):
        name = f'{self.target}.n'
        return [RateLaw(s.target.n, lambda t, values: -1.0 if values[name] > 50 else 0.0)]


culled = CompositeSystem(
    elements=[PopulationElement('deer', n=60.0)],
    interactions=[CullInteraction('cull', target='deer')],
)
round(float(simulate(culled, until=20, dt=1.0).get_values('deer.n')[-1]))
```

The herd falls from 60 to 50 and stays there, since the cull stops at 50.

## Functions as laws

This section writes laws as Python functions, for rules that have no symbolic form. Every law whose value the engine
evaluates numerically takes a function, and `CallSpec` sets how the function is called. Here a tank fills at a rate
proportional to its level, and a [`CrossingLaw`](../core_components/laws.md#crossinglaw) resets the level to 1.0 when
it crosses 2.0 upward. One tank sits at the top of the system and one inside a nested system `plant`.

```python
from typing import Annotated

import numpy as np

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.laws import CrossingLaw, JumpLaw, RateLaw, UpdateLaw
from skcomplex.core.components.laws.utils import reduction
from skcomplex.core.components.spaces import DiscreteTime
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.specs import CallSpec, ObservableSpec, ParameterSpec, StateSpec


class TankElement(Element):
    def __init__(
        self,
        label,
        visual=None,
        *,
        level: Annotated[float, StateSpec()] = 1.0,
        rate: Annotated[float, ParameterSpec()] = 0.5,
    ):
        super().__init__(label, visual=visual)
        self.level = level
        self.rate = rate

    def _get_laws(self, s):
        grow = CallSpec(lambda t, values: values['rate'] * values['level'], relative=True)
        full = CallSpec(lambda t, values: values['level'] - 2.0, relative=True)
        return [RateLaw(s.level, grow), CrossingLaw(full, {s.level: lambda t, values: 1.0}, direction=1)]


tanks = CompositeSystem(elements=[TankElement('a')], systems=[CompositeSystem('plant', elements=[TankElement('b')])])
trajectory = simulate(tanks, until=2, dt=0.01)
len(trajectory.events) > 0, trajectory.get_values('a.level')[-1] == trajectory.get_values('plant/b.level')[-1]
```

The run records crossing events, one per tank, and the two tanks end at the same level, so one class works at either
place. A function runs in Python, so on the Numba [backend](../glossary.md#backend) the laws of its kind fall back to
Python with a `BackendWarning`. This table lists the laws that take a function, and the form the function takes.

| Law | Argument | Function |
| --- | --- | --- |
| `RateLaw`, `NoiseLaw`, `UpdateLaw`, `BoundaryLaw` | `value` | `f(t, values)` or `CallSpec` |
| `JumpLaw` | `rate`, and each value of `changes` | `f(t, values)` or `CallSpec` |
| `JumpLaw` on a clock of waiting times | `waiting` | `f(rng, size=, t=, **values)` |
| `CrossingLaw` | `guard`, and each value of `reset` | `f(t, values)` or `CallSpec(f, relative=True)` |
| `FluxLaw` | `speed` | `f(t, values)`, which gives the wave speed of the Rusanov flux |
| `TransitionLaw` | `matrix` | `f(t, values)` or `CallSpec(f, relative=True)` |
| `reduction(function, expression)` | `function` | `f(array)` of the values of the members |
| `ObservableSpec` | `function` | `f(t, values)` |

`CallSpec(f, (s.x, s.k))` calls `f(t, x, k)` once per member of a group. Three options change that.

- `relative=True` calls `f(t, values)` with the names of the component of the law, such as `x` for a state of the
  element and `target.x` for a state of the element bound to the role `target`. One element class then works under any
  label and inside any system, as the tanks `a` and `plant/b` above show.
- `random=True` passes the random stream of the run as the keyword `rng`, so one seed repeats the run.
- `vectorized=True` calls the function once per evaluation with one array per argument, instead of once per member.

`arguments=None` calls `f(t, values)` with the dotted names, like a plain function, and takes the options `random` and
`vectorized`.

A random function draws from `rng`, so two runs with the same seed agree. Here a walker in
[`DiscreteTime`](../core_components/spaces.md#discretetime) adds a normal step to its level on every step.

```python continuation
class WalkerElement(TankElement):
    def _get_laws(self, s):
        return [UpdateLaw(s.level, CallSpec(lambda t, level, rng: level + rng.normal(), (s.level,), random=True))]


def walk(seed):
    system = CompositeSystem(elements=[WalkerElement('a')], time=DiscreteTime())
    return simulate(system, until=5, random_state=seed).get_values('a.level')


np.array_equal(walk(1), walk(1))
```

The two runs with the seed 1 are equal. A vectorized function takes the arrays of a whole group at once, and
`reduction` applies a function of an array to the values of the members. Here a
[`GroupElement`](../core_components/elements.md#groupelement) of three tanks runs once with each.

```python continuation
class HerdElement(TankElement):
    def _get_laws(self, s):
        return [RateLaw(s.level, CallSpec(lambda t, level, rate: rate * level, (s.level, s.rate), vectorized=True))]


class MedianElement(TankElement):
    def _get_laws(self, s):
        return [RateLaw(s.level, reduction(np.median, s.level) - s.level)]


shapes = []
for element in (HerdElement, MedianElement):
    herd = CompositeSystem(elements=[GroupElement('herd', element=element, size=3, values={'level': [1.0, 2.0, 4.0]})])
    shapes.append(simulate(herd, until=1, dt=0.01).get_values('herd.level').shape)
shapes
```

Both runs return 101 frames of the three members. The change of a jump can be a function too. Here every jump doubles
the level.

```python continuation
class DoublerElement(TankElement):
    def _get_laws(self, s):
        return [JumpLaw(s.rate, {s.level: CallSpec(lambda t, values: values['level'], relative=True)})]


doubling = simulate(CompositeSystem(elements=[DoublerElement('a')]), until=10, random_state=0)
float(np.log2(doubling.get_values('a.level')[-1])).is_integer()
```

The final level is a power of 2, so every change was a doubling. An observable can be a function of the time and the
values too. The run records it at every frame. The function lives at module level, because the run pickles it.

```python continuation
def total_level(t, values):
    return values['a.level'] + values['a.rate']


observables = [ObservableSpec('sum', function=total_level)]
observed = simulate(CompositeSystem(elements=[TankElement('a')]), until=1, dt=0.5, observables=observables)
observed.get_values('sum').shape
```

The observable holds three values, one per frame at the times 0, 0.5, and 1. Some laws need symbols by nature, and
each says so when given a function.

- [`EquationLaw`](../core_components/laws.md#equationlaw) is solved symbolically, so a function goes in a `RateLaw` of
  the highest derivative.
- [`GeneratorLaw`](../core_components/laws.md#generatorlaw) and
  [`DissipatorLaw`](../core_components/laws.md#dissipatorlaw) build quantum operators from symbols.
- A [`FluxLaw`](../core_components/laws.md#fluxlaw) takes the derivative of its flux, for the wave speeds and for the critical points of the Godunov flux, so
  a function goes in `speed=` or in a `RateLaw`.
- A crossing, a matrix, and a jump change read no delays.
- A reset of a renewal stays symbolic, because it shares its draws with the waiting time.

## Symbolic expressions

This section writes laws as Sage expressions, which the library can inspect, differentiate, and solve. Here crowding
slows the growth of a herd below a capacity and shrinks it above, with a condition that picks the branch.

```python
from typing import Annotated

from sage.all__sagemath_symbolics import exp, oo, sqrt
from sage.sets.real_set import RealSet

from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import RateLaw
from skcomplex.core.components.laws.utils import where
from skcomplex.core.specs import ParameterSpec, RoleSpec


class CrowdingInteraction(Interaction):
    def __init__(
        self,
        label,
        *,
        target: Annotated[str, RoleSpec()],
        capacity: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 100.0,
    ):
        super().__init__(label)
        self.target = target
        self.capacity = capacity

    def _get_laws(self, s):
        n = s.target.n
        return [RateLaw(n, where(n > s.capacity, -sqrt(n - s.capacity), 0.1 * n * exp(-n / s.capacity)))]


CrowdingInteraction('crowding', target='deer')
```

The interaction builds with the default capacity 100.0, and its representation lists only the label and the role.
The symbols in `s` are Sage symbols that carry the domains of their variables, vector states are Sage vectors, and
mathematical functions come from Sage. Sage has no condition-based piecewise expression or symbolic remainder. So
`skcomplex.core.components.laws.utils` adds `where`, `either`, `both`, and `mod`, which every backend lowers, LAMMPS
included.

Systems written with SymPy move to Sage as the table below shows. A SymPy expression in a law raises a `ValidationError`, and
`expression._sage_()` converts one that is already built.

| SymPy | Sage and `skcomplex` |
| --- | --- |
| `sympy.sqrt`, `sympy.exp`, `sympy.sin`, and other functions | `sqrt`, `exp`, `sin`, and others from `sage.all__sagemath_symbolics` |
| `sympy.Symbol('x', positive=True)` | `var('x', domain='positive')` from `sage.all__sagemath_symbolics` |
| `sympy.Matrix([...])` and `a.dot(b)` | `vector([...])` from `sage.all__sagemath_symbolics` and `a.dot_product(b)` |
| `sympy.Piecewise((a, condition), (b, True))` | `where(condition, a, b)` |
| `condition1 \| condition2` and `condition1 & condition2` | `either(condition1, condition2)` and `both(condition1, condition2)` |
| `sympy.Mod(x, m)` | `mod(x, m)` |
| `sympy.diff(f, x)` and `f.subs(x, y)` | `f.diff(x)` and `f.subs({x: y})` |
| `sympy.Eq(a, b)`, `.lhs`, and `.rhs` | `a == b`, `.lhs()`, and `.rhs()` |

## State domains

This section gives a state labels or angles instead of plain numbers. Besides real, nonnegative, and integer values,
a state takes the labels of a finite alphabet, a `FiniteEnumeratedSet`, or angles, a state with
`StateSpec(period=2 * pi)`.

```python
from typing import Annotated

from sage.all__sagemath_symbolics import pi
from sage.sets.finite_enumerated_set import FiniteEnumeratedSet

from skcomplex.core.components.elements import Element
from skcomplex.core.specs import StateSpec

HEALTH = FiniteEnumeratedSet(['susceptible', 'infected', 'recovered'])


class PersonElement(Element):
    def __init__(self, label, *, health: Annotated[str, StateSpec(domain=HEALTH)] = 'susceptible'):
        super().__init__(label)
        self.health = health


class RotorElement(Element):
    def __init__(self, label, *, theta: Annotated[float, StateSpec(period=2 * pi)] = 0.0):
        super().__init__(label)
        self.theta = theta


PersonElement('ada').health, RotorElement('rotor').theta
```

The person starts as `'susceptible'` and the rotor at the angle 0.0. Values are stored as floats. Labels are stored as their index, and angles are wrapped into one turn after every frame.
Trajectories return labels, so `trajectory.get_values('ada.health')` gives the names of the health states over time.

## Complex states

This section gives a state complex values, such as the amplitude of an oscillation, with the domain `CC`. Here the
amplitude turns around the unit circle at the angular frequency 2.

```python
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import CC, I

from skcomplex.core.components.elements import Element
from skcomplex.core.components.laws import RateLaw
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.specs import ParameterSpec, StateSpec


class PhaseElement(Element):
    def __init__(
        self,
        label,
        *,
        z: Annotated[complex, StateSpec(domain=CC)] = 1.0,
        omega: Annotated[float, ParameterSpec()] = 2.0,
    ):
        super().__init__(label)
        self.z = z
        self.omega = omega

    def _get_laws(self, s):
        return [RateLaw(s.z, I * s.omega * s.z)]


z = simulate(CompositeSystem(elements=[PhaseElement('osc')]), until=10, dt=0.5).get_values('osc.z')
z.dtype == np.complex128, bool(np.allclose(np.abs(z), 1.0, atol=1e-4))
```

The trajectory is a complex array, and the modulus stays 1 to within `1e-4` over the run. A state or parameter of the domain `CC` takes complex values. Its laws use the imaginary unit `I`, arithmetic, the
elementary functions over the complex plane, and `real_part`, `imag_part`, `conjugate`, `abs`, and `arg`.

- The compiler writes every complex law as the laws of its real and imaginary parts, which every solver, stepper, and
  backend runs as a real system.
- A real state takes only real values, so `abs(z)` or `real_part(z)` turns a complex value into one.
- A condition compares the modulus or a part, since complex values have no order.
- Trajectories and frames return complex arrays. Their `.real`, `.imag`, `np.abs`, and `np.angle` are the real views
  that the analyses take.
- Agent states stay real, and the LAMMPS solver raises an error naming a complex state.

## Equations of any order

This section writes an equation of motion as it reads, with second or higher derivatives. Newton's second law for a
pendulum, `m l θ'' = -m g sin θ`, is an `EquationLaw` on a state of order 2.

```python
from typing import Annotated

from sage.all__sagemath_symbolics import pi, sin

from skcomplex.core.components.elements import Element
from skcomplex.core.components.laws import EquationLaw
from skcomplex.core.components.laws.utils import derivative
from skcomplex.core.specs import ParameterSpec, StateSpec


class PendulumElement(Element):
    def __init__(
        self,
        label,
        *,
        theta: Annotated[float, StateSpec(period=2 * pi, order=2)] = 0.5,
        theta_d1: float = 0.0,
        m: Annotated[float, ParameterSpec()] = 1.0,
        g: Annotated[float, ParameterSpec()] = 9.81,
        l: Annotated[float, ParameterSpec()] = 1.0,
    ):
        super().__init__(label)
        self.theta = theta
        self.theta_d1 = theta_d1
        self.m = m
        self.g = g
        self.l = l

    def _get_laws(self, s):
        return [EquationLaw(s.m * s.l * derivative(s.theta, 2) == -s.m * s.g * sin(s.theta))]


bob = PendulumElement('bob', theta_d1=0.5)
list(bob.get_params()), bob.theta_d1
```

The parameters list `g`, `l`, `label`, `m`, `theta`, and `theta_d1`, and the pendulum starts with the angular velocity
0.5. Laws give the highest time derivative of a state, and a state declares how high that is with `StateSpec(order=k)`.
The angular velocity is the companion state `theta_d1`, and the compiler reduces the equation to first order for the
solvers.

- A companion is a constructor argument without a spec, named `<state>_d<j>`. Its value is the initial value of the
  derivative of order `j`, so `PendulumElement('bob', theta_d1=0.5)` starts the pendulum with an angular velocity.
- `StateSpec(initial=...)` gives the initial values of the derivatives instead, for a type without companion arguments.
- On a state of order `k`, rate and noise laws contribute to the `k`-th derivative and add up. So separate interactions
  supply separate forces, voltages, or growth terms. `RateLaw(s.target.x, -s.k * s.target.x / s.target.m)` from a
  spring and `RateLaw(s.target.x, -s.c * derivative(s.target.x, 1) / s.target.m)` from a damper superpose.
- `derivative(x, j)` reads the companion `x_d<j>` below the state's order. An `EquationLaw` may contain the derivative
  of the state's own order, linearly. The compiler solves all equations of a system jointly for those derivatives,
  which couples, for example, the two angles of a double pendulum.
- Any order works the same way, such as Sprott's chaotic jerk equation `x''' = -A x'' - x' + |x| - 1` on a state of
  order 3.
- Difference equations keep past values with `StateSpec(lags=m)`. `lag(x, j)` reads the value `j` steps before the
  current one, and an update law gives the next value. So `UpdateLaw(s.x, s.x + lag(s.x, 1))` with one lag is the
  Fibonacci recurrence.

`symbolic(system)` of `skcomplex.core.analysis` returns these equations in the order they were written.
`symbolic(system, kind='solve', use_initial_state=True)` takes the initial values of every derivative and returns a
solution only when it has a closed form.
