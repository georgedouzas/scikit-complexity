# Defining elements

This page writes the element and interaction classes of a system of your own. An element class is a plain Python
class whose arguments are its values. An interaction class names its [roles](../glossary.md#role) and parameters as
arguments of its `__init__`, and its `_prepare` sets its [laws](../glossary.md#law) in `laws_`. A value that some law
changes is a state, a value that no law changes is a parameter, and a role is a slot of an interaction that names the
element it acts on.

Here a deer population is a subclass of [`Element`](../core_components/elements.md#element) with one nonnegative value
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
from skcomplex.core.systems import System
from skcomplex.core.types import Role


class Species(Element):
    def __init__(self, label, *, n: Annotated[float, RealSet.closed_open(0, oo)] = 10.0):
        super().__init__(label)
        self.n = n


class Birth(Interaction):
    def __init__(self, label, *, target: Role, rate: Annotated[float, RealSet.closed_open(0, oo)] = 0.5):
        super().__init__(label)
        self.target = target
        self.rate = rate

    def _prepare(self, s):
        super()._prepare(s)
        n = s.target.n
        self.laws_ = [RateLaw(n.diff(s.t), s.rate * n)]


deer = Species('deer', n=40.0)
birth = Birth('birth', target='deer', rate=0.2)
deer.n, clone(birth).rate
```

The deer start at 40.0, and the clone of the births keeps the rate 0.2. Every element class derives from `Element`,
and every interaction class from `Interaction`. Both, like systems, derive from `BaseComponent`, which is a
scikit-learn estimator. The classes follow these rules.

- The constructor takes the label first and every other argument by keyword. It stores each value unmodified under its
  own name.
- An argument typed `Role` is a role of an interaction, and has no default unless the element it names is fixed.
- The kind of a value comes from its type hint or its value: `float` is real, `int` an integer, `bool` a boolean, and
  `str` a category. Its allowed values are written in `Annotated` only where they are needed, as a Sage set such as
  `RealSet.closed_open(0, oo)` or a Python set of labels.
- `_prepare(s)` calls the method of its parent first and then sets `laws_`.
- `get_params`, `set_params`, `clone`, and `repr` follow the scikit-learn estimator contract, so every component clones
  with its values.

Inside `_prepare`, `s` is the namespace of symbols. `s.rate` is the symbol of the parameter, `s.target.n` is the
symbol of the value of the element bound to the role `target`, and `s.t` is the time. A value is a function of time,
so `n.diff(s.t)` is its derivative. Preparing a system with a value outside its allowed values raises an error that
names it.

```python continuation
from skcomplex.core.errors import ValidationError

herd = System(elements=[deer], interactions=[birth]).prepare()
try:
    System(elements=[deer], interactions=[Birth('birth', target='deer', rate=-1.0)]).prepare()
except ValidationError as error:
    message = str(error)
herd.states_, herd.parameters_, message
```

The count `deer.n` is the state, since the births change it, and the rate `birth.rate` is a parameter. A negative rate
raises "birth.rate = -1.0 is not in its domain [0, +oo)." Parameters stay symbols, so the `schedule` of `simulate` can
change them during a run without compiling again.

## Constructor settings

This section adds a setting to a class, an argument that is neither a value, a parameter of a law, nor a role, such as
a switch between two sets of laws, a radius, or a layer. Here a switch turns the births into random events of a
[`JumpLaw`](../core_components/laws.md#jumplaw).

```python continuation
from skcomplex.core.components.laws import JumpLaw


class SwitchedBirth(Birth):
    def __init__(self, label, *, target, rate=0.5, stochastic=False):
        super().__init__(label, target=target, rate=rate)
        self.stochastic = stochastic

    def _prepare(self, s):
        super()._prepare(s)
        if self.stochastic:
            n = s.target.n
            self.laws_ = [JumpLaw(s.rate * n, {n: 1})]


clone(SwitchedBirth('birth', target='deer', stochastic=True))
```

The clone shows `SwitchedBirth(label='birth', stochastic=True, target='deer')`, so `get_params` sees the setting
beside the role. A setting follows these rules.

- The setting is a keyword-only argument with a default.
- `__init__` stores it unmodified under its own name, like every other argument.
- A subclass that adds a setting repeats the arguments of its base in its own `__init__`, so that `get_params` sees them
  all. An argument repeated without its annotation keeps the annotation of the base.
- A setting that changes the laws changes the problem too. With `stochastic=True` the system is a jump process, solved
  by Gillespie's method.

## Values from data

This section builds an element whose values come from data, such as a fitted surrogate or a metabolic network. Such an
element cannot name its values in a signature, so a plain `Element` takes them in one argument, `values`. Here two
values `a` and `b` decay at the rate `k` of an interaction that reads them by name.

```python continuation
class Decay(Interaction):
    def __init__(self, label, *, target: Role, names=('a', 'b'), k=0.5):
        super().__init__(label)
        self.target = target
        self.names = names
        self.k = k

    def _prepare(self, s):
        super()._prepare(s)
        values = [getattr(s.target, name) for name in self.names]
        self.laws_ = [RateLaw(x.diff(s.t), -s.k * x) for x in values]


data = {'a': 1.0, 'b': 2.0}
decay = System(elements=[Element('decay', values=data)], interactions=[Decay('decay_rate', target='decay')])
faster = clone(decay).set_params(decay_rate__k=1.0)
decay.get_params()['decay_rate__k'], faster.get_params()['decay_rate__k']
```

The system keeps the rate 0.5, and its clone takes the rate 1.0 under the nested name `decay_rate__k`. The system runs
like any other.

```python continuation
round(float(decay.prepare().simulate(until=1.0, dt=0.1).get_values('decay.b')[-1]), 3)
```

The value `b` falls from 2.0 to 1.213 at time 1, which is `2 exp(-0.5)`. Values given as data follow these rules.

- `values` maps every name to its starting value, and the values keep the order in which they are given.
- A law reads a value from its namespace by name, such as `getattr(s.target, 'b')`.
- `set_params` changes a value with the nested name `<element>__values`, and `clone` copies the values.
- `size=N` gives one value per member as an array, and `space=` one per site of a space.

## Writing laws

This section looks at what the laws of a system change. A law says how the values it targets change, and the
interaction that sets it in `laws_` binds it to its elements. After `prepare`, `equations_` gives one readable
equation per state, summed over the laws.

```python continuation
decay.equations_
```

The equations read `'-decay.a*decay_rate.k'` for `decay.a` and `'-decay.b*decay_rate.k'` for `decay.b`. A law takes
the values it changes and the expressions it reads as arguments, and an optional `label` by keyword. It has no
parameters of its own and no methods to call, since it is the mathematics an interaction states.

Each kind of law implies a kind of time. Rates such as `RateLaw` flow in continuous time, updates such as
[`UpdateLaw`](../core_components/laws.md#updatelaw) step in discrete time, and objectives and constraints alone are
static. `prepare` infers the time from the kinds present and raises an error that lists kinds that do not combine.

A law of one value, such as `RateLaw`, takes the value as its first argument. A local rule that updates several values
together takes a tuple of them, and laws that name their values in other arguments, such as the changes of a
`JumpLaw`, change the values they name.

A system is defined by its elements, its interactions, and the laws they set in `laws_`, and by nothing else. Two kinds
of value stay outside it.

- A value that only names an expression is a Python expression in the laws, such as `leverage = s.debt / s.equity`. A
  function of the namespace shares it between laws.
- What a run records or checks is an argument of the run. `system.simulate(observables=[...])` records named values,
  and `system.simulate(invariants={name: tolerance})` checks that the observables it names stay constant.

A law can also be a Python function of time and the current values, which covers rules that have no closed form. The
values are keyed by dotted name. Here a cull removes one deer a year while the herd holds more than 50.

```python
from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import RateLaw
from skcomplex.core.systems import System
from skcomplex.core.types import Role


class Cull(Interaction):
    def __init__(self, label, *, target: Role):
        super().__init__(label)
        self.target = target

    def _prepare(self, s):
        super()._prepare(s)
        name = f'{self.target}.n'
        self.laws_ = [RateLaw(s.target.n.diff(s.t), lambda t, values: -1.0 if values[name] > 50 else 0.0)]


culled = System(elements=[Element('deer', values={'n': 60.0})], interactions=[Cull('cull', target='deer')])
round(float(culled.prepare().simulate(until=20, dt=1.0).get_values('deer.n')[-1]))
```

The herd falls from 60 to 50 and stays there, since the cull stops at 50.

## Functions as laws

This section writes laws as Python functions, for rules that have no symbolic form. Every law whose value the engine
evaluates numerically takes a function, and `CallSpec` sets how the function is called. Here a tank fills at a rate
proportional to its level, and an [`EventLaw`](../core_components/laws.md#eventlaw) resets the level to 1.0 when it
crosses 2.0 upward. One tank sits at the top of the system and one inside a nested system `plant`.

```python
import numpy as np

from skcomplex.core.components import Observable
from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import CallSpec, EventLaw, JumpLaw, RateLaw, UpdateLaw
from skcomplex.core.components.laws.utils import reduction
from skcomplex.core.systems import System
from skcomplex.core.types import Role


class Fill(Interaction):
    def __init__(self, label, *, target: Role, rate=0.5):
        super().__init__(label)
        self.target = target
        self.rate = rate

    def _prepare(self, s):
        super()._prepare(s)
        grow = CallSpec(lambda t, values: values['rate'] * values['target.level'], relative=True)
        full = CallSpec(lambda t, values: values['target.level'] - 2.0, relative=True)
        level = s.target.level
        self.laws_ = [RateLaw(level.diff(s.t), grow), EventLaw(full, {level: lambda t, values: 1.0}, direction=1)]


plant = System('plant', elements=[Element('b', values={'level': 1.0})], interactions=[Fill('fill', target='b')])
tanks = System(elements=[Element('a', values={'level': 1.0})], interactions=[Fill('fill', target='a')], systems=[plant])
trajectory = tanks.prepare().simulate(until=2, dt=0.01)
len(trajectory.events), trajectory.get_values('a.level')[-1] == trajectory.get_values('plant/b.level')[-1]
```

The run records two crossing events, one per tank, and the two tanks end at the same level, so one class works at
either place. A function runs in Python, so on the Numba [backend](../glossary.md#backend) the laws of its kind fall
back to Python with a `BackendWarning`. This table lists the laws that take a function, and the form the function
takes.

| Law | Argument | Function |
| --- | --- | --- |
| `RateLaw`, `NoiseLaw`, `UpdateLaw` | `value` | `f(t, values)` or `CallSpec` |
| `JumpLaw` | `rate`, and each value of `changes` | `f(t, values)` or `CallSpec` |
| `JumpLaw` on a clock of waiting times | `waiting` | `f(rng, size=, t=, **values)` |
| `EventLaw` | `guard`, and each value of `reset` | `f(t, values)` or `CallSpec(f, relative=True)` |
| `FiniteVolumeSpec` | `speed` | `f(t, values)`, which gives the wave speed of the Rusanov flux |
| `reduction(function, expression)` | `function` | `f(array)` of the values of the members |
| `Observable` | `function` | `f(t, values)` |

`CallSpec(f, (s.x, s.k))` calls `f(t, x, k)` once per member of a group. Three options change that.

- `relative=True` calls `f(t, values)` with the names of the interaction of the law, such as `rate` for its own
  parameter and `target.level` for a value of the element bound to the role `target`. One interaction class then works
  under any label and inside any system, as the tanks `a` and `plant/b` above show.
- `random=True` passes the random stream of the run as the keyword `rng`, so one seed repeats the run.
- `vectorized=True` calls the function once per evaluation with one array per argument, instead of once per member.

`arguments=None` calls `f(t, values)` with the dotted names, like a plain function, and takes the options `random` and
`vectorized`.

A random function draws from `rng`, so two runs with the same seed agree. Here a walker adds a normal step to its
level on every step of discrete time, which its `UpdateLaw` implies.

```python continuation
class Walk(Interaction):
    def __init__(self, label, *, target: Role):
        super().__init__(label)
        self.target = target

    def _prepare(self, s):
        super()._prepare(s)
        level = s.target.level
        self.laws_ = [UpdateLaw(level, CallSpec(lambda t, level, rng: level + rng.normal(), (level,), random=True))]


def walk(seed):
    system = System(elements=[Element('a', values={'level': 1.0})], interactions=[Walk('walk', target='a')])
    return system.prepare().simulate(until=5, random_state=seed).get_values('a.level')


np.array_equal(walk(1), walk(1))
```

The two runs with the seed 1 are equal. A vectorized function takes the arrays of a whole group at once, and
`reduction` applies a function of an array to the values of the members. Here an element of three tanks, given with
`size=3`, runs once with each.

```python continuation
class Herd(Interaction):
    def __init__(self, label, *, target: Role, rate=0.5, median=False):
        super().__init__(label)
        self.target = target
        self.rate = rate
        self.median = median

    def _prepare(self, s):
        super()._prepare(s)
        level = s.target.level
        if self.median:
            value = reduction(np.median, level) - level
        else:
            value = CallSpec(lambda t, level, rate: rate * level, (level, s.rate), vectorized=True)
        self.laws_ = [RateLaw(level.diff(s.t), value)]


shapes = []
for median in (False, True):
    herd = Element('herd', values={'level': [1.0, 2.0, 4.0]}, size=3)
    system = System(elements=[herd], interactions=[Herd('herding', target='herd', median=median)])
    shapes.append(system.prepare().simulate(until=1, dt=0.01).get_values('herd.level').shape)
shapes
```

Both runs return 101 frames of the three members. The change of a jump can be a function too. Here every jump doubles
the level.

```python continuation
class Double(Interaction):
    def __init__(self, label, *, target: Role, rate=0.5):
        super().__init__(label)
        self.target = target
        self.rate = rate

    def _prepare(self, s):
        super()._prepare(s)
        doubled = CallSpec(lambda t, values: values['target.level'], relative=True)
        self.laws_ = [JumpLaw(s.rate, {s.target.level: doubled})]


doubler = System(elements=[Element('a', values={'level': 1.0})], interactions=[Double('double', target='a')])
doubling = doubler.prepare().simulate(until=10, random_state=0)
float(np.log2(doubling.get_values('a.level')[-1])).is_integer()
```

The final level is a power of 2, so every change was a doubling. An observable can be a function of the time and the
values too. The run records it at every frame. The function lives at module level, because the run pickles it.

```python continuation
def total_level(t, values):
    return values['a.level'] + values['fill.rate']


observables = [Observable('sum', function=total_level)]
tank = System(elements=[Element('a', values={'level': 1.0})], interactions=[Fill('fill', target='a')])
observed = tank.prepare().simulate(until=1, dt=0.5, observables=observables)
observed.get_values('sum').shape
```

The observable holds three values, one per frame at the times 0, 0.5, and 1. Some laws need symbols by nature, and
each says so when given a function.

- [`ConstraintLaw`](../core_components/laws.md#constraintlaw) is solved symbolically, so a function goes in a `RateLaw`
  of the highest derivative.
- `GeneratorLaw` and `DissipatorLaw` of the [quantum package](quantum_systems.md) build quantum operators from
  symbols.
- A conservation law `RateLaw(u, -divergence(flux))` on a finite-volume field takes the derivative of its flux, for the
  wave speeds and for the critical points of the Godunov flux, so a function goes in the `speed` of the
  `FiniteVolumeSpec` of the field.
- A crossing and a jump change read no delays.
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
from skcomplex.core.types import Role


class Crowding(Interaction):
    def __init__(self, label, *, target: Role, capacity: Annotated[float, RealSet.open(0, oo)] = 100.0):
        super().__init__(label)
        self.target = target
        self.capacity = capacity

    def _prepare(self, s):
        super()._prepare(s)
        n = s.target.n
        crowded = where(n > s.capacity, -sqrt(n - s.capacity), 0.1 * n * exp(-n / s.capacity))
        self.laws_ = [RateLaw(n.diff(s.t), crowded)]


Crowding('crowding', target='deer')
```

The interaction builds with the default capacity 100.0, and its representation lists only the label and the role.
The symbols in `s` are Sage symbols that carry the allowed values of their values, vector values are Sage vectors, and
mathematical functions come from Sage. Sage has no condition-based piecewise expression or symbolic remainder. So
`skcomplex.core.components.laws.utils` adds `where`, `either`, `both`, and `mod`, which every backend lowers, LAMMPS
included.

Systems written with SymPy move to Sage as the table below shows. A SymPy expression in a law raises a
`ValidationError`, and `expression._sage_()` converts one that is already built.

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

## Value domains

This section gives a value labels or angles instead of plain numbers. Besides real, nonnegative, and integer values, a
value takes the labels of a finite set, written as its allowed values in `Annotated`, or angles, written with
[`Period`](../core_components/elements.md#period).

```python
from typing import Annotated

from sage.all__sagemath_symbolics import pi

from skcomplex.core.components.elements import Element, Period

HEALTH = {'susceptible', 'infected', 'recovered'}


class Person(Element):
    def __init__(self, label, *, health: Annotated[str, HEALTH] = 'susceptible'):
        super().__init__(label)
        self.health = health


class Rotor(Element):
    def __init__(self, label, *, theta: Annotated[float, Period(2 * pi)] = 0.0):
        super().__init__(label)
        self.theta = theta


Person('ada').health, Rotor('rotor').theta
```

The person starts as `'susceptible'` and the rotor at the angle 0.0. Values are stored as floats. Labels are stored as
their index, and angles are wrapped into one turn after every step. Trajectories return labels, so
`trajectory.get_values('ada.health')` gives the names of the health states over time.

## Complex values

This section gives a value complex numbers, such as the amplitude of an oscillation. A value given as a complex
number is complex. Here the amplitude turns around the unit circle at the angular frequency 2.

```python
import numpy as np
from sage.all__sagemath_symbolics import I

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import RateLaw
from skcomplex.core.systems import System
from skcomplex.core.types import Role


class Rotation(Interaction):
    def __init__(self, label, *, target: Role, omega=2.0):
        super().__init__(label)
        self.target = target
        self.omega = omega

    def _prepare(self, s):
        super()._prepare(s)
        z = s.target.z
        self.laws_ = [RateLaw(z.diff(s.t), I * s.omega * z)]


phase = System(elements=[Element('osc', values={'z': 1.0 + 0j})], interactions=[Rotation('rotation', target='osc')])
z = phase.prepare().simulate(until=10, dt=0.5).get_values('osc.z')
z.dtype == np.complex128, bool(np.allclose(np.abs(z), 1.0, atol=1e-4))
```

The trajectory is a complex array, and the modulus stays 1 to within `1e-4` over the run. The laws of a complex value
use the imaginary unit `I`, arithmetic, the elementary functions over the complex plane, and `real_part`,
`imag_part`, `conjugate`, `abs`, and `arg`.

- The compiler writes every complex law as the laws of its real and imaginary parts, which every solver, stepper, and
  backend runs as a real system.
- A real value takes only real numbers, so `abs(z)` or `real_part(z)` turns a complex value into one.
- A condition compares the modulus or a part, since complex values have no order.
- Trajectories and frames return complex arrays. Their `.real`, `.imag`, `np.abs`, and `np.angle` are the real views
  that the analyses take.
- Agent values stay real, and the LAMMPS solver raises an error naming a complex value.

## Equations of any order

This section writes an equation of motion as it reads, with second or higher derivatives. Newton's second law for a
pendulum, `m l θ'' = -m g sin θ`, is a [`ConstraintLaw`](../core_components/laws.md#constraintlaw) on the second
derivative `theta.diff(t, 2)`.

```python
from sage.all__sagemath_symbolics import sin

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import ConstraintLaw
from skcomplex.core.systems import System
from skcomplex.core.types import Role


class Gravity(Interaction):
    def __init__(self, label, *, target: Role, g=9.81, l=1.0):
        super().__init__(label)
        self.target = target
        self.g = g
        self.l = l

    def _prepare(self, s):
        super()._prepare(s)
        theta, m = s.target.theta, s.target.m
        self.laws_ = [ConstraintLaw(m * s.l * theta.diff(s.t, 2) == -m * s.g * sin(theta))]


bob = Element('bob', values={'theta': 0.5, "theta'": 0.5, 'm': 1.0})
pendulum = System(elements=[bob], interactions=[Gravity('gravity', target='bob')]).prepare()
pendulum.states_, pendulum.parameters_
```

The states are `bob.theta` and its angular velocity `bob.theta'`, and the parameters are `bob.m`, `gravity.g`, and
`gravity.l`. The pendulum starts with the angular velocity 0.5. Laws give the highest time derivative of a value, and
the compiler reads the order of every state from them and reduces the equation to first order for the solvers.

```python continuation
pendulum.equations_
```

The equations read `"bob.theta'"` for `bob.theta` and `'-gravity.g*sin(bob.theta)/gravity.l'` for `bob.theta'`.

- A derivative is named with primes, such as `"theta'"` and `"theta''"`, in the values, in `initial_state`, and in
  trajectories. Its starting value defaults to zero, so `values={"theta'": 0.5}` starts the pendulum with an angular
  velocity.
- Rate and noise laws on the same derivative add up. So separate interactions supply separate forces, voltages, or
  growth terms. `RateLaw(x.diff(t, 2), -k * x / m)` from a spring and `RateLaw(x.diff(t, 2), -c * x.diff(t) / m)` from
  a damper superpose.
- A law reads a lower derivative as `x.diff(t)`. A `ConstraintLaw` may contain the highest derivative linearly. The
  compiler solves all equations of a system jointly for those derivatives, which couples, for example, the two angles
  of a double pendulum.
- Any order works the same way, such as Sprott's chaotic jerk equation `x''' = -A x'' - x' + |x| - 1` with the third
  derivative `x.diff(t, 3)`.
- Difference equations read past values as `x(t=t - j)`, the value `j` steps before the current one, and an update law
  gives the next value. Past values are named with an index, such as `'x[-1]'`. So
  `UpdateLaw(x, x + x(t=t - 1))` with `values={'x': 1, 'x[-1]': 0}` is the Fibonacci recurrence.

```python continuation
from skcomplex.core.components.laws import UpdateLaw

fibonacci = Interaction(
    'fibonacci',
    roles={'target': 'fib'},
    laws=lambda s: [UpdateLaw(s.target.x, s.target.x + s.target.x(t=s.t - 1))],
)
numbers = System(elements=[Element('fib', values={'x': 1, 'x[-1]': 0})], interactions=[fibonacci]).prepare()
numbers.simulate(until=6).get_values('fib.x').tolist()
```

The recurrence gives 1, 1, 2, 3, 5, 8, and 13.

`symbolic(system)` of `skcomplex.core.analysis` returns these equations in the order they were written.
`symbolic(system, kind='solve', use_initial_state=True)` takes the initial values of every derivative and returns a
solution only when it has a closed form.
