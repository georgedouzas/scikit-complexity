# User guide

## Concepts

`scikit-complexity` separates the description of a model from the way it is solved.

- A system is made of **parts**: its elements, its interactions, the laws they return, and the system itself.
    - An **element** is an entity that carries state, such as a particle, a population, a compartment, or a market.
    - A **group** holds many identical elements of one type as arrays.
    - An **interaction** couples elements through **roles** and describes its effect with **laws**.
    - A **system** holds elements, interactions, nested systems, an optional space, and an optional time.
- Every part speaks three verbs. `check()` validates it, `prepare(...)` prepares it against what it works on, and
  `run(...)` returns what it contributes: an element, an interaction, or a system returns its laws, and a law returns
  its value at given values of the symbols it reads, such as the rate of a `RateLaw`. `check` and `prepare` return
  the part itself, and a part must be checked before it is prepared or run.
- A run records **observables**, named values computed from the states at every frame, and checks the
  **invariants** it is given.
- A **law** says how a state changes. Every law is a class, and the four most common laws are these.
    - `RateLaw` adds a term to the time derivative of a state. Rates on one state add up.
    - `NoiseLaw` drives a state with a Wiener process, independent unless laws name a shared or correlated one.
    - `JumpLaw` fires a random event that changes states by fixed increments.
    - `UpdateLaw` gives the next value of a state after one discrete step.
- A **space** gives vector states, such as positions, their dimension, and a **time** says whether the system evolves
  continuously, in discrete steps, or not at all.

These concepts carry no field-specific meaning. A population, a particle, a firm, and a cell are all elements, so a
model from any science is a configuration of the same interface, and every ready-made model in `skcomplex.models` is a
factory that builds such a configuration.

`skcomplex.core` holds these parts. `skcomplex.engine` compiles a system, infers its mathematical form from its laws,
and builds fast numeric functions that its solvers advance.

| Laws | Form | Default solver |
| --- | --- | --- |
| `RateLaw` | ordinary differential equations | `ScipyODESolver` with `LSODA` |
| `RateLaw` and `NoiseLaw` | stochastic differential equations | `EulerMaruyamaSolver` |
| `JumpLaw` | jump process | `GillespieSolver` |
| `UpdateLaw` | discrete map | `IterationSolver` |

## Defining elements

Every element type derives from `Element`, and every interaction type from `Interaction`; both, like laws and
systems, derive from `BasePart`, which is a scikit-learn estimator. A type declares its variables as arguments of its
`__init__`, annotated with `typing.Annotated` and a spec from `skcomplex.core.specs`.

- `StateSpec` marks a state, `ParameterSpec` a parameter, and `RoleSpec` a role of an interaction. The default of the
  argument is the initial value of a state or the default of a parameter, and a role has no default.
- The constructor takes the label first and every other argument by keyword, and stores each value unmodified under
  its own name.
- The domain of a state or a parameter is a Sage set, such as `RR`, `ZZ`, `CC`, `RealSet.closed_open(0, oo)`, or
  `FiniteEnumeratedSet(['S', 'I', 'R'])`.
- `_get_laws(s)` returns the laws of the type, and returns none by default.
- `get_params`, `set_params`, `clone`, and `repr` follow the scikit-learn estimator contract, so every part clones
  with its values.

```python
from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet
from sklearn.base import clone

from skcomplex.core.elements import Element
from skcomplex.core.interactions import Interaction
from skcomplex.core.laws import RateLaw
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
print(deer.n, clone(birth).rate)
```

Inside `_get_laws`, `s.rate` is the symbol of the parameter and `s.target.n` is the symbol of the state of the element
bound to the role `target`. Because parameters stay symbols, a running simulation can change them without compiling
again.

### Constructor settings

A setting that is neither a state, a parameter, nor a role, such as a switch between two sets of laws, a radius, or a
layer, is an argument of `__init__` without a spec.

- The setting is a keyword-only argument with a default.
- `__init__` stores it unmodified under its own name, like every other argument.
- A subclass that adds a setting repeats the arguments of its base in its own `__init__`, so that `get_params` sees
  them all. An argument repeated without its annotation keeps the spec of the base.

```python continuation
from skcomplex.core.laws import JumpLaw


class SwitchedBirthInteraction(BirthInteraction):
    def __init__(self, label, *, target, rate=0.5, stochastic=False):
        super().__init__(label, target=target, rate=rate)
        self.stochastic = stochastic

    def _get_laws(self, s):
        if self.stochastic:
            return [JumpLaw(s.rate * s.target.n, {s.target.n: 1})]
        return super()._get_laws(s)


print(clone(SwitchedBirthInteraction('birth', target='deer', stochastic=True)))
```

### Laws

Every law class derives from `BaseLaw`. A law takes the states it changes and the expressions it reads as arguments of
its `__init__`, and an optional `label` by keyword.

- `check()` validates the arguments, such as a value that is a NumPy array instead of an expression.
- `prepare(owner)` sets `targets_`, the states the law changes, `reads_`, the symbols it reads, and `owner_`, the path
  of the element or interaction that returns it.
- `run(values)` returns the value of the law at the values of the symbols it reads, given by name.
- Every law states what it requires of a system, such as continuous time for `RateLaw` and discrete time for
  `UpdateLaw`, and validation reports a system whose laws require different times.

A law of one state, such as `RateLaw`, takes the state as its first argument. A law that updates several states
together, such as `LocalRuleLaw`, takes a tuple of them. Laws that name their states in other values, such as the
increments of a `JumpLaw`, change the states they name.

A system is defined by its elements, its interactions, and the laws they return from `_get_laws(s)`, and by nothing
else. A value that only names an expression is a Python expression in the laws, such as `leverage = s.debt / s.equity`,
and a function of the namespace shares it between laws. What a run records or checks is an argument of the run:
`Simulation(observables=[...])` records named values, and `Simulation(invariants={name: tolerance})` checks that the
observables it names stay constant.

```python
from sage.all__sagemath_symbolics import var

from skcomplex.core.laws import BaseLaw, RateLaw

x, k = var('x k')
law = RateLaw(x, -k * x).check().prepare('decay')
print(issubclass(RateLaw, BaseLaw), law.targets_, sorted(map(str, law.reads_)), law.owner_)
print(law.run({'x': 2.0, 'k': 0.5}))
```

A law can also be a Python function of time and current values, which covers rules that have no closed form. The
values are keyed by dotted state and parameter name.

```python
from typing import Annotated

from skcomplex.core.interactions import Interaction
from skcomplex.core.laws import RateLaw
from skcomplex.core.specs import RoleSpec
from skcomplex.core.system import System
from skcomplex.engine.runtime import Simulation
from skcomplex.models.biology.ecology import PopulationElement


class ThresholdInteraction(Interaction):
    def __init__(self, label, *, target: Annotated[str, RoleSpec()]):
        super().__init__(label)
        self.target = target

    def _get_laws(self, s):
        name = f'{self.target}.n'
        return [RateLaw(s.target.n, lambda t, values: -1.0 if values[name] > 50 else 0.0)]


culled = System(
    elements=[PopulationElement('deer', n=60.0)],
    interactions=[ThresholdInteraction('cull', target='deer')],
)
print(round(float(Simulation(culled, dt=1.0).run(until=20).trajectory_.values('deer.n')[-1])))
```

### Symbolic expressions

Laws are Sage expressions. The symbols in `s` are Sage symbols that carry the domains of their variables, vector states
are Sage vectors, and mathematical functions come from Sage. Sage has no condition-based piecewise expression or
symbolic remainder, so `skcomplex.core.laws.utils` adds `where`, `either`, `both`, and `mod`, which every backend
lowers, LAMMPS included.

```python
from typing import Annotated

from sage.all__sagemath_symbolics import exp, oo, sqrt
from sage.sets.real_set import RealSet

from skcomplex.core.interactions import Interaction
from skcomplex.core.laws import RateLaw
from skcomplex.core.laws.utils import where
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
```

Models written with SymPy move to Sage as follows. A SymPy expression in a law raises a `ValidationError`, and
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

### State domains

Besides real, nonnegative, and integer values, a state takes the labels of a finite alphabet, a
`FiniteEnumeratedSet`, or angles, a state with `StateSpec(period=2 * pi)`. Values are stored as floats: labels as
their index, and angles wrapped into one turn after every frame. Trajectories return labels, and
`Simulation(..., checked=True)` raises an error naming any state that leaves its domain.

```python
from typing import Annotated

from sage.all__sagemath_symbolics import pi
from sage.sets.finite_enumerated_set import FiniteEnumeratedSet

from skcomplex.core.elements import Element
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


print(PersonElement('ada').check().health, RotorElement('rotor').check().theta)
```

### Complex states

A state or parameter of the domain `CC` takes complex values, and its laws use the imaginary unit `I`, arithmetic,
the elementary functions over the complex plane, and `real_part`, `imag_part`, `conjugate`, `abs`, and `arg`. The
compiler writes every complex law as the laws of its real and imaginary parts, which every solver, stepper, and backend
runs as a real system. A real state takes only real values, so `abs(z)` or `real_part(z)` turns a complex value into
one, and a condition compares the modulus or a part, since complex values have no order. Trajectories and frames return
complex arrays, whose `.real`, `.imag`, `np.abs`, and `np.angle` are the real views that the analyses take.

```python
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import CC, I

from skcomplex.core.elements import Element
from skcomplex.core.laws import RateLaw
from skcomplex.core.specs import ParameterSpec, StateSpec
from skcomplex.core.system import System
from skcomplex.engine.runtime import Simulation


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


z = Simulation(System(elements=[PhaseElement('osc')]), dt=0.5).run(until=10).trajectory_.values('osc.z')
print(z.dtype == np.complex128, bool(np.allclose(np.abs(z), 1.0, atol=1e-4)))
```

Agent states stay real, and the LAMMPS solver raises an error naming a complex state.

### Equations of any order

Laws give the highest time derivative of a state, and a state declares how high that is with `StateSpec(order=k)`.
Newton's second law for a pendulum, `m l θ'' = -m g sin θ`, is written as it reads with an `EquationLaw`. The angular
velocity is the companion state `theta_d1`, and the compiler reduces the equation to first order for the solvers.

- A companion is a constructor argument without a spec, named `<state>_d<j>`, whose value is the initial value of the
  derivative of order `j`, so `PendulumElement('bob', theta_d1=0.5)` starts the pendulum with an angular velocity.
- `StateSpec(initial=...)` gives the initial values of the derivatives instead, for a type without companion arguments.

```python
from typing import Annotated

from sage.all__sagemath_symbolics import pi, sin

from skcomplex.core.elements import Element
from skcomplex.core.laws import EquationLaw
from skcomplex.core.laws.utils import derivative
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


bob = PendulumElement('bob', theta_d1=0.5).check()
print(list(bob.get_params()), bob.theta_d1)
```

- On a state of order `k`, rate and noise laws contribute to the `k`-th derivative and add up, so separate interactions
  supply separate forces, voltages, or growth terms: `RateLaw(s.target.x, -s.k * s.target.x / s.target.m)` from a spring
  and `RateLaw(s.target.x, -s.c * derivative(s.target.x, 1) / s.target.m)` from a damper superpose.
- `derivative(x, j)` reads the companion `x_d<j>` below the state's order. An `EquationLaw` may contain the derivative
  of the state's own order, linearly, and the compiler solves all equations of a system jointly for those derivatives,
  which couples, for example, the two angles of a double pendulum.
- Any order works the same way, such as Sprott's chaotic jerk equation `x''' = -A x'' - x' + |x| - 1` on a state of
  order 3.
- Difference equations keep past values with `StateSpec(lags=m)`. `lag(x, j)` reads the value `j` steps before the
  current one, and an update law gives the next value, so `UpdateLaw(s.x, s.x + lag(s.x, 1))` with one lag is the
  Fibonacci recurrence.

`SymbolicAnalysis('equations')` returns these equations in the order they were written, and
`SymbolicAnalysis('solve', use_initial_state=True)` takes the initial values of every derivative and returns a solution
only when it has a closed form.

## Composing systems

Systems nest. A system holds nested systems in `systems`, and an interaction refers to an element of a nested system by
a slash-separated path.

```python
from skcomplex.core.system import System
from skcomplex.models.biology.ecology import PopulationElement, PredationInteraction

landscape = System(
    systems=[
        System('forest', elements=[PopulationElement('foxes', n=5)]),
        System('meadow', elements=[PopulationElement('rabbits', n=50)]),
    ],
    interactions=[PredationInteraction('hunt', predator='forest/foxes', prey='meadow/rabbits', a=0.02, b=0.01)],
)
```

States and parameters are named by their path, such as `meadow/rabbits.n` and `hunt.a`.

## Simulating

`Simulation` compiles the system, picks the solver of its form, and advances it in frames of `dt`. The default ODE
solver is `ScipyODESolver` with SciPy's `LSODA`, which starts with Adams methods and switches to BDF methods while the
system is stiff, so stiff systems also run with the default solver.

```python continuation
from skcomplex.engine.runtime import Simulation

trajectory = Simulation(landscape, dt=0.1).run(until=10).trajectory_
print(round(float(trajectory.values('meadow/rabbits.n')[-1]), 3))

simulation = Simulation(landscape, dt=0.1).run(until=10)
simulation.update_parameters({'hunt.a': 0.0})
print(len(simulation.extend(until=20).trajectory_.times))
```

- `dt` defaults to 1 for discrete maps and agents. For other forms it defaults to 0.01, raised to the smallest of 0.02,
  0.05, 0.1, and so on, in steps of 1, 2, and 5 times a power of ten, that keeps the run at most 1000 frames long, so
  `run(until=1000)` takes frames of 1.
- `clock=RealTimeClock(speed)` paces frames so that simulated time advances `speed` times faster than wall time.
- `start(until)` runs in a background thread, and `pause`, `resume`, `stop`, and `wait` control it.
- `update_parameters({'hunt.a': 0.05})` changes parameters before the next frame. Values set between runs stay in
  force for the later runs of the same simulation, and `clone(simulation).run(until)` starts afresh.
- `random_state` makes stochastic runs reproducible. A fixed-step solver splits every frame into
  `step_count(t, until, dt)` equal steps on every backend, so the same `random_state` gives the same path on the NumPy
  and Numba backends.
- `extend(until)` continues a finished run from its last frame with the same stepper, clock, and observers, and
  appends the new frames to the trajectory, while `run` starts again from time 0.
- `schedule={'hunt.a': lambda t: 0.02 + 0.001 * t}` sets parameters from functions of time at every frame, such as a
  slow ramp of a control parameter.
- `initial_state={'meadow/rabbits.n': 20}` replaces initial values of the elements for one simulation.

## Delays, events, drivers, and noise

Laws can read the past, change states at discrete events, follow inputs that switch in time, and take noise in the
convention of the field.

**Delays.** `delayed(x, tau)` from `skcomplex.core.laws.utils` is the state `x` at `t - tau`, for a constant or a
state-dependent delay. The solvers keep a history of dense output and step no further than the smallest delay, by the
method of steps. Before the start the history is the initial value, or what the state declares with
`StateSpec(history=...)`, a number or a function of time.

```python
from typing import Annotated

import numpy as np

from skcomplex.core.elements import Element
from skcomplex.core.laws import RateLaw
from skcomplex.core.laws.utils import delayed
from skcomplex.core.specs import ParameterSpec, StateSpec
from skcomplex.core.system import System
from skcomplex.engine.runtime import Simulation


def wavy_history(t):
    return 0.5 + 0.1 * np.sin(t)


class MackeyGlassElement(Element):
    def __init__(
        self,
        label,
        *,
        x: Annotated[float, StateSpec(history=wavy_history)] = 0.5,
        beta: Annotated[float, ParameterSpec()] = 0.2,
        gamma: Annotated[float, ParameterSpec()] = 0.1,
        n: Annotated[float, ParameterSpec()] = 10.0,
        tau: Annotated[float, ParameterSpec()] = 17.0,
    ):
        super().__init__(label)
        self.x = x
        self.beta = beta
        self.gamma = gamma
        self.n = n
        self.tau = tau

    def _get_laws(self, s):
        past = delayed(s.x, s.tau)
        return [RateLaw(s.x, s.beta * past / (1 + past**s.n) - s.gamma * s.x)]


cells = System(elements=[MackeyGlassElement('cells')])
print(round(float(Simulation(cells, dt=0.5).run(until=300).trajectory_.values('cells.x')[-1]), 3))
```

**Events.** `CrossingLaw(guard, reset, direction, terminal, label='')` fires when the guard crosses zero, upward for
`direction=1`, downward for `-1`, and either way for `0`. The solver locates the crossing on its interpolant to the
precision of the root finder, applies the reset to every named state at once, and restarts from there. A terminal
event ends the run at the crossing. Every frame carries the events since the previous one, and the trajectory keeps
them all with their times and the states before and after the reset. A guard that fires again right after its reset
raises an error that names the chattering event.

```python
from typing import Annotated

from skcomplex.core.elements import Element
from skcomplex.core.laws import CrossingLaw, RateLaw
from skcomplex.core.laws.utils import derivative
from skcomplex.core.specs import ParameterSpec, StateSpec
from skcomplex.core.system import System
from skcomplex.engine.runtime import Simulation


class BallElement(Element):
    def __init__(
        self,
        label,
        *,
        h: Annotated[float, StateSpec(order=2)] = 1.0,
        g: Annotated[float, ParameterSpec()] = 9.81,
        e: Annotated[float, ParameterSpec()] = 0.8,
    ):
        super().__init__(label)
        self.h = h
        self.g = g
        self.e = e

    def _get_laws(self, s):
        bounce = {derivative(s.h, 1): -s.e * derivative(s.h, 1)}
        return [RateLaw(s.h, -s.g), CrossingLaw(s.h, bounce, direction=-1, label='impact')]


trajectory = Simulation(System(elements=[BallElement('ball')]), dt=0.01).run(until=3).trajectory_
print([round(event.time, 6) for event in trajectory.events])
```

**Drivers.** `step`, `ramp`, `pulse`, `pulse_train`, and `series` with `'linear'` or `'previous'` interpolation, from
`skcomplex.core.laws.utils`, are inputs of time `s.t`. The compiler finds the times where they jump or bend, including
the periodic edges of pulse trains, and the solvers stop at every one and evaluate the inputs on the correct side of
it, so a step response is exact to the solver tolerance.

```python
from typing import Annotated

from skcomplex.core.elements import Element
from skcomplex.core.laws import RateLaw
from skcomplex.core.laws.utils import step
from skcomplex.core.specs import ParameterSpec, StateSpec
from skcomplex.core.system import System
from skcomplex.engine.runtime import Simulation


class FilterElement(Element):
    def __init__(
        self,
        label,
        *,
        y: Annotated[float, StateSpec()] = 0.0,
        k: Annotated[float, ParameterSpec()] = 1.0,
    ):
        super().__init__(label)
        self.y = y
        self.k = k

    def _get_laws(self, s):
        return [RateLaw(s.y, -s.k * s.y + step(s.t, 5.0))]


filtered = Simulation(System(elements=[FilterElement('filter')]), dt=0.5).run(until=10).trajectory_
print(round(float(filtered.values('filter.y')[-1]), 4))
```

**Noise conventions.** `NoiseLaw(..., convention='stratonovich')` or `'anti-ito'` declares how a noise law is read. The
compiler adds the drift correction `α g ∂g/∂x`, with `α = 1/2` and `1`, so every SDE solver integrates the equivalent
Itô equation. `EulerMaruyamaSolver` has strong order 1/2. `MilsteinSolver`, in Platen's derivative-free form, has strong
order 1 for diagonal noise, and `HeunSolver` averages the drift over every step.

```python
from typing import Annotated

from skcomplex.core.elements import Element
from skcomplex.core.laws import NoiseLaw, RateLaw
from skcomplex.core.specs import ParameterSpec, StateSpec
from skcomplex.core.system import System
from skcomplex.engine.runtime import Simulation
from skcomplex.engine.solvers import MilsteinSolver


class GrowthElement(Element):
    def __init__(
        self,
        label,
        *,
        x: Annotated[float, StateSpec()] = 1.0,
        mu: Annotated[float, ParameterSpec()] = 0.5,
        sigma: Annotated[float, ParameterSpec()] = 0.8,
    ):
        super().__init__(label)
        self.x = x
        self.mu = mu
        self.sigma = sigma

    def _get_laws(self, s):
        return [RateLaw(s.x, s.mu * s.x), NoiseLaw(s.x, s.sigma * s.x, convention='stratonovich')]


growth = System(elements=[GrowthElement('asset')])
print(round(float(Simulation(growth, solver=MilsteinSolver(dt=0.01), random_state=0).run(until=1).state_[0]), 3))
```

**Shared and correlated noise.** `NoiseLaw(..., process=...)` names the Wiener process a law draws. Without a process,
every element and every group member takes its own independent process. Laws of one element, or of one member, that
name the same label share one increment, so a reaction can drive every species it changes with one process. A
`WienerProcessSpec(label, dimension, covariance)` from `skcomplex.core.specs` declares processes whose increments over
a step `h` have the covariance `h C`. It is no element: an interaction binds it through a role annotated with
`RoleSpec(noise=True)`, and its noise laws name process `k` with `s.<role>.dW[k]`, which every element, and every
member of a group, can share. Interactions of one system that bind processes of one label share them and their
correlations. The covariance is fixed when the system compiles, and NumPy factors it once, so the steppers scale
standard normal increments drawn from the run's random generator.

```python
from typing import Annotated

import numpy as np

from skcomplex.core.elements import Element
from skcomplex.core.interactions import Interaction
from skcomplex.core.laws import NoiseLaw
from skcomplex.core.specs import RoleSpec, StateSpec, WienerProcessSpec
from skcomplex.core.system import System
from skcomplex.engine.runtime import Simulation
from skcomplex.engine.solvers import EulerMaruyamaSolver


class StockElement(Element):
    def __init__(self, label, *, n: Annotated[float, StateSpec()] = 0.0):
        super().__init__(label)
        self.n = n


class MarketInteraction(Interaction):
    def __init__(
        self,
        label,
        *,
        first: Annotated[str, RoleSpec()],
        second: Annotated[str, RoleSpec()],
        noise: Annotated[WienerProcessSpec, RoleSpec(noise=True)],
    ):
        super().__init__(label)
        self.first = first
        self.second = second
        self.noise = noise

    def _get_laws(self, s):
        return [NoiseLaw(s.first.n, 1.0, process=s.noise.dW[0]), NoiseLaw(s.second.n, 1.0, process=s.noise.dW[1])]


processes = WienerProcessSpec('market', dimension=2, covariance=[[1.0, 0.8], [0.8, 1.0]])
market = System(
    elements=[StockElement('a'), StockElement('b')],
    interactions=[MarketInteraction('drive', first='a', second='b', noise=processes)],
)
solver = EulerMaruyamaSolver(dt=1.0)
trajectory = Simulation(market, solver=solver, dt=1.0, random_state=0).run(until=2000).trajectory_
print(round(float(np.corrcoef(np.diff(trajectory.values('a.n')), np.diff(trajectory.values('b.n')))[0, 1]), 1))
```

The Stratonovich and anti-Itô corrections of each law use the full diffusion matrix and the covariance, so a state
that several laws drive and laws that share a process are corrected exactly. Laws that share a process declare one
convention, and validation reports laws of different conventions on one process, a Stratonovich or anti-Itô amplitude
that reads a state another element drives through its process, a covariance that is not symmetric positive
semidefinite, and a handle `dW` used anywhere but as the process of a noise law.

**Conserved noise.** `NoiseLaw(target, value, conserved=True)` on a state of a field on a lattice adds the
divergence of a random flux, `div(value eta)`, the thermal noise of Model B and of active field theories. Every link
between neighbouring sites draws one Wiener process, and site `i` gains `value / h` times the increment of the link
from its previous neighbour and loses the same amount on the link to its next one, along every dimension of spacing
`h`. The total of the field then changes only by its rate laws, exactly to rounding. The links that leave a lattice
through a zero-flux or fixed edge carry no process. The value is a number or an expression of parameters, the same
on every link, and the field uses finite differences on a square lattice. Validation reports a conserved noise law on
any other state with the code `noise-unsupported`.

```python
from typing import Annotated

from sage.all__sagemath_symbolics import sqrt

from skcomplex.core.elements import Element, FieldElement
from skcomplex.core.laws import NoiseLaw, RateLaw
from skcomplex.core.laws.utils import laplacian
from skcomplex.core.spaces import LatticeSpace
from skcomplex.core.specs import ParameterSpec, StateSpec
from skcomplex.core.system import System
from skcomplex.engine.runtime import Simulation
from skcomplex.engine.solvers import EulerMaruyamaSolver


class DensityElement(Element):
    def __init__(
        self,
        label,
        *,
        rho: Annotated[float, StateSpec()] = 1.0,
        D: Annotated[float, ParameterSpec()] = 1.0,
        T: Annotated[float, ParameterSpec()] = 0.1,
    ):
        super().__init__(label)
        self.rho = rho
        self.D = D
        self.T = T

    def _get_laws(self, s):
        return [RateLaw(s.rho, s.D * laplacian(s.rho)), NoiseLaw(s.rho, sqrt(2 * s.T), conserved=True)]


gas = System(elements=[FieldElement('gas', element=DensityElement, space=LatticeSpace((16, 16)))])
solver = EulerMaruyamaSolver(dt=0.01)
density = Simulation(gas, solver=solver, random_state=0).run(until=5).trajectory_.values('gas.rho')
print(round(float(density[-1].sum()), 6), bool(density[-1].std() > 0))
```

Solvers and backends declare the noise structures they draw: `'diagonal'`, one process per state, `'shared'`, a process
that drives several states, `'correlated'`, the processes of a `WienerProcessSpec` with a covariance, and `'conserved'`,
the link processes of conserved noise. `EulerMaruyamaSolver`, `HeunSolver`, and the NumPy and Numba backends support
all four. `MilsteinSolver`, whose derivative-free correction holds for diagonal noise only, and `LammpsSolver` support
diagonal noise, and validation reports a system with shared, correlated, or conserved noise for them before compiling
it. The compatibility page lists the declarations. In the library, `BindingInteraction` and `DissociationInteraction`
take `langevin=True` for chemical Langevin equations, as in `chemical_langevin_network`,
`correlated_geometric_brownian_motion` drives two assets with correlated processes, and `active_model_b` and
`cahn_hilliard` with a `temperature` take conserved noise.

SciPy, RK4, Euler-Maruyama, Milstein, and Heun handle delays, events, and drivers. Velocity Verlet, Gillespie, and map
iteration reject delays and events with an error that names the solvers to use. The library has these models ready:
`mackey_glass`, `delayed_kuramoto`, and `stochastic_resonance` in nonlinear dynamics, `hutchinson` in ecology,
`bouncing_ball`, `newtons_cradle`, `circular_billiard`, and `brownian_near_wall` in physics, and
`integrate_and_fire` and `izhikevich` in neuroscience.

## Stochastic maps

Update laws can draw random numbers, so a discrete map can be a discrete-time stochastic process. The draws come from
the simulation's random generator, so two runs with the same `random_state` give the same trajectory.

### Random draws

`uniform(name, low, high)`, `normal(name, mean, deviation)`, `bernoulli(name, p)`, and `categorical(name, weights)`
are symbolic values that update laws use like any other expression. A draw takes a fresh value at every step, and
every member of a group draws its own value. The arguments of a draw can be expressions of states and parameters, and
they take the values from before the step.

Every draw carries a name. Every use of the same name in one element reads the same value within a step, and draws
with different names are independent. Two uses of one name with different distributions or arguments fail to
compile. Within a step, the draws are taken by element path, then draw name, then member index, so the trajectory
does not depend on the order of the laws. The NumPy and Numba backends give equal trajectories for uniform, Bernoulli,
and categorical draws.

```python
from typing import Annotated

from skcomplex.core.elements import Element, GroupElement
from skcomplex.core.laws import UpdateLaw
from skcomplex.core.laws.utils import bernoulli, normal, where
from skcomplex.core.specs import ParameterSpec, StateSpec
from skcomplex.core.system import System
from skcomplex.engine.runtime import Simulation


class PersistentElement(Element):
    def __init__(
        self,
        label,
        *,
        d: Annotated[float, StateSpec()] = 1.0,
        x: Annotated[float, StateSpec()] = 0.0,
        q: Annotated[float, ParameterSpec()] = 0.8,
    ):
        super().__init__(label)
        self.d = d
        self.x = x
        self.q = q

    def _get_laws(self, s):
        turn = where(bernoulli('turn', 1 - s.q) > 0, -s.d, s.d)
        return [UpdateLaw(s.d, turn), UpdateLaw(s.x, s.x + turn + normal('jitter', 0, 0.1))]


walkers = System(elements=[GroupElement('walkers', element=PersistentElement, size=500)])
final = Simulation(walkers, dt=1.0, random_state=0).run(until=200).trajectory_.values('walkers.x')[-1]
print(round(float(final.var() / 200), 1))
```

Only update laws and renewal laws take draws. A draw in a rate, noise, jump, boundary, equation, or crossing law raises
`IncompatibleLawsError`, since noise laws and jump laws give randomness in continuous time. Draws are not states, so
the trajectory does not record them, and a model that wants to keep a draw assigns it to a state.

### Markov chains and hidden Markov sources

`TransitionLaw(targets, matrix)` is the step of a discrete-time Markov chain. The target is a state over a finite
alphabet, a `FiniteEnumeratedSet` of labels or the integers from 0 to `n - 1`, and its current value selects the row of
the transition matrix that the next value is drawn from. `EmissionLaw(targets, source, matrix)` draws an observed state
from the row of an emission matrix that the hidden state `source` selects, reading the hidden state after the same step,
so the two sequences align frame by frame. Both are update laws, so they combine with other update laws.

Matrix entries are numbers or expressions of parameters. Compiling checks that the entries are nonnegative and that
every row sums to 1, and `update_parameters` checks again every matrix a changed parameter enters, rejecting the change
with a `ParameterError` before it applies.

```python
from typing import Annotated

from sage.sets.finite_enumerated_set import FiniteEnumeratedSet

from skcomplex.core.elements import Element
from skcomplex.core.laws import EmissionLaw, TransitionLaw
from skcomplex.core.specs import ParameterSpec, StateSpec
from skcomplex.core.system import System
from skcomplex.engine.runtime import Simulation


class SourceElement(Element):
    def __init__(
        self,
        label,
        *,
        hidden: Annotated[str, StateSpec(domain=FiniteEnumeratedSet(['calm', 'storm']))] = 'calm',
        symbol: Annotated[str, StateSpec(domain=FiniteEnumeratedSet(['a', 'b']))] = 'a',
        stay: Annotated[float, ParameterSpec()] = 0.9,
    ):
        super().__init__(label)
        self.hidden = hidden
        self.symbol = symbol
        self.stay = stay

    def _get_laws(self, s):
        return [
            TransitionLaw(s.hidden, [[s.stay, 1 - s.stay], [0.4, 0.6]]),
            EmissionLaw(s.symbol, s.hidden, [[0.7, 0.3], [0.2, 0.8]]),
        ]


simulation = Simulation(System(elements=[SourceElement('source')]), dt=1.0, random_state=0).run(until=10000)
print(round(float((simulation.trajectory_.values('source.hidden') == 'calm').mean()), 1))
simulation.update_parameters({'source.stay': 0.5})
```

The library has these models ready: `persistent_random_walk`, `markov_source`, `hidden_markov_source`, and
`random_walk_with_memory` in stochastic processes, `chaos_game` in fractals, `probabilistic_boolean_network` and
`intrinsic_instability` in biology, and `diluted_hopfield` in neural networks.

## Anomalous processes

Jumps on clocks of waiting times and fractional noise laws write processes with memory: events that fire after waiting
times from any distribution, and noise whose increments are correlated over the whole run.

### Draws from SciPy distributions

`draw(name, distribution, **parameters)` draws from any distribution of `scipy.stats`, such as `'pareto'`, `'lomax'`,
`'expon'`, `'beta'`, or `'levy_stable'`, with its shape parameters, `loc`, and `scale` as keywords. The keywords can be
expressions of states and parameters. It follows the rules of the other named draws, and it samples with the
distribution's `rvs` and the simulation's random generator. Compiling rejects a name `scipy.stats` does not provide,
listing the closest names, and keywords the distribution does not take. In update laws, a SciPy draw is sampled for
every member before each step, and the map then steps in Python with the backend's compiled update.

```python
from typing import Annotated

from skcomplex.core.elements import Element, GroupElement
from skcomplex.core.laws import UpdateLaw
from skcomplex.core.laws.utils import draw
from skcomplex.core.specs import StateSpec
from skcomplex.core.system import System
from skcomplex.engine.runtime import Simulation


class FlierElement(Element):
    def __init__(self, label, *, x: Annotated[float, StateSpec()] = 0.0):
        super().__init__(label)
        self.x = x

    def _get_laws(self, s):
        return [UpdateLaw(s.x, s.x + draw('jump', 'levy_stable', alpha=1.5, beta=0.0))]


fliers = System(elements=[GroupElement('fliers', element=FlierElement, size=100)])
flight = Simulation(fliers, dt=1.0, random_state=0).run(until=20)
print(flight.trajectory_.values('fliers.x').shape)
```

### Jumps on clocks of waiting times

`JumpLaw(waiting=..., reset=..., label=...)` fires after a waiting time, applies its reset to the states all at once,
and draws the next waiting time from the states after the reset. Every member of a group renews on its own clock. Draws
in the waiting time and the reset take a fresh value at every renewal of every member, and every use of one name in a
renewal reads the same value. A waiting time that is a single draw must come from a distribution with nonnegative
support, and compiling rejects one that can be negative, such as `normal`.

A system or partition whose only laws are such jumps compiles to the renewal form, which `RenewalSolver` runs exactly,
from renewal to renewal with no time grid. A frame holds the states the last renewal at or before its time set. These
jumps also combine with the rate and noise laws of the same element, which evolve the states between renewals.
`RK4Solver` and `EulerMaruyamaSolver` stop every member at its own renewals when the members do not interact, and split
the step at every renewal time for all members when a law sums over the group. Other steppers reject these jumps.

```python
from typing import Annotated

from skcomplex.core.elements import Element, GroupElement
from skcomplex.core.laws import JumpLaw
from skcomplex.core.laws.utils import draw, normal
from skcomplex.core.specs import ParameterSpec, StateSpec
from skcomplex.core.system import System
from skcomplex.engine.runtime import Simulation


class WalkerElement(Element):
    def __init__(
        self,
        label,
        *,
        x: Annotated[float, StateSpec()] = 0.0,
        alpha: Annotated[float, ParameterSpec()] = 0.5,
    ):
        super().__init__(label)
        self.x = x
        self.alpha = alpha

    def _get_laws(self, s):
        waiting = draw('wait', 'pareto', b=s.alpha)
        return [JumpLaw(waiting=waiting, reset={s.x: s.x + normal('jump', 0, 1)}, label='jumps')]


walkers = System(elements=[GroupElement('walkers', element=WalkerElement, size=1000)])
walk = Simulation(walkers, dt=100.0, random_state=0).run(until=1000)
print(walk.problem_.form.value, bool(walk.trajectory_.values('jumps.renewals')[-1] > 1000))
```

The trajectory records every renewal pass as a `RenewalRecord` among its events, with the time and the member of every
renewal and the member's states before and after it. The observable `<label>.renewals` counts the renewals up to every
frame, and the Rerun observer streams it as a time series and the renewals as text entries under `renewals/<label>`.
The compiler keeps the count of every member in a hidden state, `<group>.<label>~renewals`.

### Fractional noise

`FractionalNoiseLaw(targets, value, hurst=hurst)` drives a state with fractional Brownian motion of Hurst index
`0 < H < 1`, whose increments anticorrelate for `H < 1/2` and correlate for `H > 1/2`. A Hurst index of 1/2 compiles to
an ordinary noise law. For other indices the amplitude must not depend on any state.
`FractionalEulerMaruyamaSolver(dt, horizon)` is the stepper of systems with fractional noise. It draws the increments
of the whole run on its grid when it first advances, with the circulant embedding of Davies and Harte over
`scipy.fft`, so frames must fall on multiples of `dt`. A simulation passes the end of its run as the horizon,
advancing beyond it raises `HorizonError`, and other SDE steppers reject fractional noise with an error that names
`FractionalEulerMaruyamaSolver`. `fractional_gaussian_noise` returns the noise paths themselves.

```python
from typing import Annotated

from skcomplex.core.elements import Element, GroupElement
from skcomplex.core.laws import FractionalNoiseLaw
from skcomplex.core.specs import StateSpec
from skcomplex.core.system import System
from skcomplex.engine.runtime import Simulation
from skcomplex.engine.solvers import FractionalEulerMaruyamaSolver


class RoughElement(Element):
    def __init__(self, label, *, x: Annotated[float, StateSpec()] = 0.0):
        super().__init__(label)
        self.x = x

    def _get_laws(self, s):
        return [FractionalNoiseLaw(s.x, 1, hurst=0.3)]


rough = Simulation(
    System(elements=[GroupElement('walkers', element=RoughElement, size=500)]),
    solver=FractionalEulerMaruyamaSolver(dt=0.01),
    dt=0.1,
    random_state=0,
).run(until=1)
print(round(float(rough.trajectory_.values('walkers.x')[-1].var()), 1))
```

The library has these models ready: `continuous_time_random_walk`, `levy_walk`, `annealed_transient_time_motion`,
`fractional_brownian_motion`, and `levy_flight` in soft matter, and `trap_model` in statistical mechanics.

### Fractional derivatives

A state declared with a fractional order, `StateSpec(order=0.6)`, has a Caputo derivative of that order. The rate
laws on it give that derivative, `D^alpha x = f(t, x)`, and superpose as for any state, so no other law kind is needed.
Order 1 is the ordinary rate law. The derivative of order `alpha` at a time depends on the whole past of the state, so
the solution relaxes like a Mittag-Leffler function, slower than any exponential.

A system with a state of a fractional order compiles to the fractional form, with the order of every state component
in `problem.orders`. The states of order 1 of the same system keep their ordinary equations. Three steppers integrate
the fractional form on a fixed grid from the start of the run, so frames must fall on multiples of their `dt`:

- `GrunwaldLetnikovSolver(dt, memory=None)` weighs the past states with the binomial weights
  `(-1)^j binomial(alpha, j)`.
- `L1Solver(dt, memory=None)` interpolates the state linearly between grid points inside the Caputo integral.
- `FractionalABMSolver(dt)` is the predictor-corrector of Diethelm, Ford, and Freed, of order `1 + alpha` for smooth
  solutions, and the default.

The two explicit schemes are of order 1. All three keep the whole history, so a run of `n` steps costs `n^2` operations.
`memory` bounds that cost: the history of the last `memory` time units enters with its exact weights, and the older
history enters through about 40 exponential modes fitted to the weights, so every step costs the same. For fractional
relaxation of order 0.6, a window of 10 time units stays within `1e-8` of the whole history at time 20. A change of a
fractional state between frames, such as one the simulation's interventions make, starts a new history from the
changed state.

```python
from typing import Annotated

from skcomplex.analysis import SymbolicAnalysis
from skcomplex.core.elements import Element
from skcomplex.core.laws import RateLaw
from skcomplex.core.specs import StateSpec
from skcomplex.core.system import System
from skcomplex.engine.runtime import Simulation
from skcomplex.engine.solvers import L1Solver


class RelaxingElement(Element):
    def __init__(self, label, *, x: Annotated[float, StateSpec(order=0.6)] = 1.0):
        super().__init__(label)
        self.x = x

    def _get_laws(self, s):
        return [RateLaw(s.x, -s.x)]


system = System(elements=[RelaxingElement('a')])
relaxing = Simulation(system, solver=L1Solver(dt=0.01, memory=10.0), dt=1.0).run(until=20)
solution = SymbolicAnalysis('solve', use_initial_state=True).check().prepare(system).run().solutions_['a.x'].rhs()
exact = solution.subs({solution.variables()[0]: 20.0})
print(relaxing.problem_.form.value, abs(float(relaxing.trajectory_.values('a.x')[-1]) - float(exact)) < 1e-3)
```

Validation reports the misuses of fractional orders before compiling, each with the path of the state and a fix: an
order that is neither an integer nor in `(0, 1)` (`fractional-order`), a fractional state in a system whose declared
time is not continuous (`fractional-time`), a chosen solver or backend that does not integrate fractional orders
(`order-unsupported`), and a law on a fractional state that no fractional solver runs, such as a noise law
(`solver-unsupported`). Solvers and backends declare the kinds of order they integrate in the `orders` field of their
`SupportSpec`, `'integer'` or `'fractional'`. The NumPy and Numba backends run the fractional form.

`SymbolicAnalysis('solve')` solves one linear fractional relaxation `D^alpha x = c x` as `x(0) E_alpha(c t^alpha)`,
written with the symbolic function `mittag_leffler`, which evaluates numerically: it sums the power series in arbitrary
precision with Sage for moderate arguments and takes the asymptotic expansion for large ones, and
`SymbolicAnalysis('equations')` writes a Caputo derivative as `caputo(x(t), t, alpha)`.

The order belongs to the element type, so a sweep over orders builds one system per order. The library has these
models ready in fractional dynamics, each with the order as an argument: `fractional_relaxation`,
`fractional_oscillator`, `fractional_van_der_pol`, and `fractional_lorenz`.

## Observables

An `ObservableSpec` is a frozen record of a label and an expression or a group reduction, which a run computes at every
frame, records in trajectories under the system's path, and logs to Rerun. An observable is either an expression of the
system namespace, where earlier observables appear by label, or a reduction over a group's members of a variable or an
expression of one member: `count`, `sum`, `mean`, `min`, `max`, `var`, or a function. Expressions and reductions given
as functions must be module-level functions, so that systems pickle, and a lambda raises an error.

```python
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import I, exp, pi

from skcomplex.core.elements import Element, GroupElement
from skcomplex.core.laws import RateLaw
from skcomplex.core.specs import ObservableSpec, ParameterSpec, StateSpec
from skcomplex.core.system import System
from skcomplex.engine.runtime import Simulation


class OscillatorElement(Element):
    def __init__(
        self,
        label,
        *,
        theta: Annotated[float, StateSpec(period=2 * pi)] = 0.0,
        omega: Annotated[float, ParameterSpec()] = 1.0,
    ):
        super().__init__(label)
        self.theta = theta
        self.omega = omega

    def _get_laws(self, s):
        return [RateLaw(s.theta, s.omega)]


def phase_vector(member):
    return exp(I * member.theta)


def order_parameter(s):
    return abs(s.z)


values = {'theta': np.linspace(0, 1, 100), 'omega': np.linspace(0.9, 1.1, 100)}
oscillators = System(elements=[GroupElement('phases', element=OscillatorElement, size=100, values=values)])
observables = [ObservableSpec('z', phase_vector, group='phases'), ObservableSpec('order', order_parameter)]
order = Simulation(oscillators, dt=1.0, observables=observables).run(until=50).trajectory_.values('order')
print(round(float(order[0]), 3), round(float(order[-1]), 3))
```

A law reads an observable through a role bound to the observable's path, as `s.<role>.value`. A symbolic observable,
an expression or a reduction with a string `reduce` over an element group, is read at the current state by laws of its
own partition, and the compiled problem evaluates it before the law, so the value the law reads equals the recorded
value at every frame. A law of another partition reads the observable as held at the last synchronisation point, so
agents and equations exchange macroscopic quantities in a co-simulation. An observable with a Python function as
`reduce`, or one over agents, is read only from another partition, and reading it in its own partition raises an error.

```python continuation
from sage.all__sagemath_symbolics import cos, sin

from skcomplex.core.interactions import Interaction
from skcomplex.core.specs import RoleSpec


class MeanFieldInteraction(Interaction):
    def __init__(
        self,
        label,
        *,
        target: Annotated[str, RoleSpec()],
        X: Annotated[str, RoleSpec()],
        Y: Annotated[str, RoleSpec()],
        K: Annotated[float, ParameterSpec()] = 1.0,
    ):
        super().__init__(label)
        self.target = target
        self.X = X
        self.Y = Y
        self.K = K

    def _get_laws(self, s):
        return [RateLaw(s.target.theta, s.K * (s.Y.value * cos(s.target.theta) - s.X.value * sin(s.target.theta)))]


def cosine(member):
    return cos(member.theta)


def sine(member):
    return sin(member.theta)


values = {'theta': np.linspace(0, 6, 100), 'omega': np.linspace(0.9, 1.1, 100)}
coupled = System(
    elements=[GroupElement('phases', element=OscillatorElement, size=100, values=values)],
    interactions=[MeanFieldInteraction('coupling', target='phases', X='X', Y='Y', K=2.0)],
)
observables = [ObservableSpec('X', cosine, group='phases'), ObservableSpec('Y', sine, group='phases')]
print(round(float(Simulation(coupled, dt=1.0, observables=observables).run(until=20).trajectory_.values('X')[-1]), 2))
```

## Stocks and flows

A state tagged with a quantity, `StateSpec(quantity='water')`, is a stock that changes only through flows. A
`TransferLaw(source, sink, amount, label=...)` moves an amount of the quantity from one stock to another, or exchanges
it with the environment through `Source()` and `Sink()`. In a system of rate laws the amount is a rate, in a system of
update laws it is moved once per step, and with `rate=...` it is the size of a jump with that propensity. Trajectories
record every flow under `<element path>.<label>`. An intermediate expression, such as a birth rate, is a function of the
namespace that the laws call, and `ObservableSpec(label, function, element=element)` records the same function as the
observable `<element>.<label>`, which Rerun logs under the element. `lookup(x, points)` is a table function, piecewise
linear between its points and flat outside them.

```python
from typing import Annotated

from skcomplex.core.elements import Element
from skcomplex.core.laws import Sink, Source, TransferLaw
from skcomplex.core.laws.utils import lookup
from skcomplex.core.specs import ObservableSpec, ParameterSpec, StateSpec
from skcomplex.core.system import System
from skcomplex.engine.runtime import Simulation


def birth_rate(s):
    return s.b * s.n * lookup(s.n, [(0, 1.0), (100, 1.0), (200, 0.0)])


class PeopleElement(Element):
    def __init__(
        self,
        label,
        *,
        n: Annotated[float, StateSpec(quantity='people')] = 10.0,
        b: Annotated[float, ParameterSpec()] = 0.03,
        d: Annotated[float, ParameterSpec()] = 0.01,
    ):
        super().__init__(label)
        self.n = n
        self.b = b
        self.d = d

    def _get_laws(self, s):
        return [
            TransferLaw(Source(), s.n, birth_rate(s), label='births'),
            TransferLaw(s.n, Sink(), s.d * s.n, label='deaths'),
        ]


crowd = System(elements=[PeopleElement('crowd')])
observed = [ObservableSpec('birth_rate', birth_rate, element='crowd')]
trajectory = Simulation(crowd, dt=1.0, observables=observed).run(until=10).trajectory_
print(round(float(trajectory.values('crowd.births')[-1]), 3), round(float(trajectory.values('crowd.deaths')[-1]), 3))
print(round(float(trajectory.values('crowd.birth_rate')[-1]), 3))
```

A rate or update law outside flows that changes a tagged state raises `ConservationError` when the system compiles.
`SymbolicAnalysis('conservation')` returns the net inflow of every quantity from the environment, after checking that
the internal flows cancel. The compiler accumulates that inflow in the hidden state `__boundary.<quantity>`, and the
observer `ConservationObserver` checks at every frame that each total changed only by it.
`LoopAnalysis('flow-matrix', frame=frame)` gives the transaction-flow matrix of a frame of a trajectory, one row per
flow and one column per holder, whose rows sum to zero.

```python continuation
from skcomplex.analysis import LoopAnalysis, SymbolicAnalysis
from skcomplex.engine.runtime import ConservationObserver

print(SymbolicAnalysis('conservation').check().prepare(crowd).run().inflows_)
monitored = Simulation(crowd, dt=1.0, observers=[ConservationObserver()]).run(until=10).trajectory_
print(LoopAnalysis('flow-matrix').check().prepare(monitored).run().matrix_.sum(axis=1).tolist())
```

`distributed_delay(value, time, order, conserved=True)` is a material delay of any order, hidden stocks in series whose
outflow it returns, and with `conserved=False` an information smoothing. It adds hidden stocks
`<element path>.<kind>_delay<order>_<index>` to rate laws of single elements and interactions, starting in
equilibrium with their input. The system-dynamics names `delay1`, `delay3`, `smooth`, and `smooth3` are its
realizations in `skcomplex.models.engineering.processes`.

`LoopAnalysis('loops')` derives the signed dependency graph of a system from the laws, and lists its feedback loops
with their polarity, and `LoopAnalysis('dominance')` ranks the loops by their gains at every frame of a trajectory. An
edge goes from one state to another when the rate of the second depends on the first, with the sign of the entry of
the Jacobian, and through the flows of systems that declare them, from a state to every flow it drives and from a
flow to its sink, `+`, and its source, `-`.

```python continuation
from skcomplex.analysis import LoopAnalysis

analysis = LoopAnalysis().check().prepare(crowd).run()
for nodes, polarity in zip(analysis.loops_, analysis.polarities_, strict=True):
    print(polarity, ' -> '.join(nodes))
dominance = LoopAnalysis('dominance').check().prepare(trajectory).run()
print(dominance.polarities_[int(dominance.dominant_[-1])])
```

The graph is that of any system with a vector field or a map. At one prey and one predator, prey growth closes a
reinforcing self-loop, predator death a balancing one, and predation a balancing loop through both populations.

```python continuation
from skcomplex.models import create

predation = LoopAnalysis(at=[1.0, 1.0]).check().prepare(create('lotka_volterra')).run()
print(dict(zip(predation.loops_, predation.polarities_, strict=True)))
```

Agents move conserved quantities between each other inside rules. `transfer(source, sink, amount, state)` moves an
amount between two agents at once, so a negative holding is an amount owed and total holdings stay constant. A run
checks that an expression keeps its first value with `Simulation(invariants={name: tolerance})`, which names an
observable of the run. At every frame the run compares the observable with its value at the first frame and raises
`ConservationError` naming the observable and the first frame that breaks it. Removing an agent that still holds a
tagged state raises `ConservationError` until its holdings are settled with transfers.

```python
from typing import Annotated

from skcomplex.core.elements import Element, GroupElement
from skcomplex.core.laws import RuleLaw
from skcomplex.core.laws.utils import transfer
from skcomplex.core.specs import ObservableSpec, StateSpec
from skcomplex.core.system import System
from skcomplex.engine.runtime import Simulation


def net_worth(agent):
    return agent.cash + agent.claims - agent.equity


def lend(agent, model):
    borrower = model.agents('traders')[0]
    if borrower != agent:
        transfer(agent, borrower, 1.0, 'cash')
        transfer(borrower, agent, 1.0, 'claims')


class TraderElement(Element):
    def __init__(
        self,
        label,
        *,
        cash: Annotated[float, StateSpec(quantity='money')] = 10.0,
        claims: Annotated[float, StateSpec(quantity='claims')] = 0.0,
        equity: Annotated[float, StateSpec()] = 10.0,
    ):
        super().__init__(label)
        self.cash = cash
        self.claims = claims
        self.equity = equity

    def _get_laws(self, s):
        return [RuleLaw(lend)]


balance = ObservableSpec('net_worth', net_worth, group='traders', reduce='sum')
market = System(elements=[GroupElement('traders', element=TraderElement, size=3)])
traders = Simulation(market, observables=[balance], invariants={'net_worth': 1e-9}).run(until=3).trajectory_
print(float(traders.agents('traders', 3).values['cash'].sum()))
```

The library models built on flows are `bathtub`, `rice_milling`, `solid_waste`, `food_relief`, and `irrigation` in
engineering processes, `car_dealer`, `commodity_cycle`, and `beer_game` in markets, `bass_diffusion` in innovation,
`stock_flow_consistent_economy` in macroeconomics, and `success_to_the_successful`, `shifting_the_burden`, and
`escalation` in organizations.

## Control

A control input is an adaptive parameter with bounds, annotated with `ParameterSpec(adaptive=True, bounds=(lower,
upper))`. Laws read it like a parameter, trajectories record it like a state, and a schedule, an optimal control solver,
or a feedback controller supplies its value. Without one it keeps its initial value. `validate` reports an initial value
outside the bounds as an error at the parameter, whose fix names the bounds.

```python
from typing import Annotated

from skcomplex.core.elements import Element
from skcomplex.core.laws import RateLaw
from skcomplex.core.specs import ParameterSpec, StateSpec
from skcomplex.core.system import System
from skcomplex.engine.compile import validate
from skcomplex.engine.runtime import Simulation


class HeaterElement(Element):
    def __init__(
        self,
        label,
        *,
        T: Annotated[float, StateSpec()] = 0.0,
        power: Annotated[float, ParameterSpec(adaptive=True, bounds=(0, 1))] = 0.0,
    ):
        super().__init__(label)
        self.T = T
        self.power = power

    def _get_laws(self, s):
        return [RateLaw(s.T, s.power - s.T)]


room = System(elements=[HeaterElement('heater')])
print([issue.path for issue in validate(System(elements=[HeaterElement('heater', power=2.0)])).errors])
schedule = {'heater.power': lambda t: 1.0 if t >= 1.0 else 0.0}
heated = Simulation(room, dt=0.5, schedule=schedule).run(until=3).trajectory_
print(heated.values('heater.power').tolist())
```

### Objectives and optimal control

An objective defines a control problem, not a part of a system, so the optimal control solvers take it as arguments.
`objective` is the running cost, and `terminal` the terminal cost at the horizon, the time `until` of the run. Each is
a function of the namespace `s` of the states, parameters, and time of the system by dotted name, such as
`lambda s: s.plant.x**2`, or a number. The value of a run is the integral of the running cost over the run plus the
terminal cost, and `sense` is `'min'` to minimise it or `'max'` to maximise it.

Two solvers optimise the objective over the bounded adaptive parameters of the system. `PontryaginSolver(objective)`
builds the Hamiltonian `H = L + λ · f` of the running cost `L` and the rates `f`, derives the costate equations `λ' =
-∂H/∂x` and the stationarity condition `∂H/∂u = 0` symbolically, projects the controls onto their bounds, takes a bound
where `H` is linear in a control, and solves the two-point problem by a forward-backward sweep.
`CollocationSolver(objective, nodes=..., scheme=...)` transcribes the problem with the `'hermite-simpson'` or
`'trapezoidal'` scheme and solves it with SciPy's `minimize`, with the Jacobians derived symbolically. Collocation
handles bang-bang controls, singular arcs, and saddle paths on which the sweep diverges. Both solve the problem over the
whole run when it starts, and `simulation.stepper_.solution_` holds the solution on its grid, its cost, and, for
`PontryaginSolver`, the costates.

```python
from typing import Annotated

from skcomplex.core.elements import Element
from skcomplex.core.laws import RateLaw
from skcomplex.core.specs import ParameterSpec, StateSpec
from skcomplex.core.system import System
from skcomplex.engine.runtime import Simulation
from skcomplex.engine.solvers import CollocationSolver, PontryaginSolver


class PlantElement(Element):
    def __init__(
        self,
        label,
        *,
        x: Annotated[float, StateSpec()] = 1.0,
        u: Annotated[float, ParameterSpec(adaptive=True, bounds=(-10, 10))] = 0.0,
    ):
        super().__init__(label)
        self.x = x
        self.u = u

    def _get_laws(self, s):
        return [RateLaw(s.x, s.u)]


plant = System(elements=[PlantElement('plant')])


def cost(s):
    return (s.plant.x**2 + s.plant.u**2) / 2


optimal = Simulation(plant, solver=PontryaginSolver(cost), dt=0.1).run(until=1.0)
print(round(float(optimal.trajectory_.values('plant.u')[0]), 3), round(float(optimal.stepper_.solution_.cost), 4))
transcribed = Simulation(plant, solver=CollocationSolver(cost, nodes=21), dt=0.1).run(until=1.0)
print(round(float(transcribed.trajectory_.values('plant.u')[0]), 3))
```

Every solver declares what it supports, and the optimal control solvers declare continuous time, no space, rate and
equation laws, real and angle states, and elements. `PontryaginSolver` also applies boundary laws after every step of
its sweep, and `CollocationSolver` keeps the states of nonnegative domains nonnegative instead. `validate` reports a
system with noise, jumps, or updates run with them. It also reports a solver without an objective, and a system under
discrete time, since an objective needs static or continuous time, each as a `SolverError`.

| Solver | Laws | Method |
| --- | --- | --- |
| `PontryaginSolver` | rate, equation, boundary | symbolic costates and a forward-backward sweep |
| `CollocationSolver` | rate, equation | direct transcription solved by SLSQP or trust-constr |

```python continuation
from skcomplex.engine.compile import validate

print([issue.code for issue in validate(plant, solver=PontryaginSolver()).errors])
```

### Feedback controllers

Feedback controllers are realizations of the control models, in `skcomplex.models.engineering.control`, so the core
holds none. `PIDInteraction(label, plant=, integrator=, control=, measure=, kp=, ki=, kd=, setpoint=)` sets the
control input `control` of the plant to `kp e + ki ∫ e dt - kd y'` from the error `e = setpoint - y` of its state
`measure`, with the integral of the error in an `IntegratorElement` element. `StateFeedbackInteraction(label, plant=,
control=, states=, gain=)` sets it to `-K x`. Both set the control after every solver step, as boundary laws, so a
fixed-step solver with small steps makes them close to continuous controllers. The engineering models
`pid_inverted_pendulum` and `lqr_oscillator` use them.

### Linear analysis

`LinearizationAnalysis(controls, outputs, at)` finds the state-space matrices `A_`, `B_`, `C_`, and `D_` of a system
under continuous time near an operating point, from its symbolic Jacobians, with the controls named by their adaptive
parameters. Its `controllability_` and `observability_` are the ranks of the Kalman matrices, and with the weights
`state_weights` and `control_weights` its `gain_` is the gain of the linear-quadratic regulator from SciPy's solution
of the Riccati equation.

```python
from typing import Annotated

from skcomplex.analysis import LinearizationAnalysis
from skcomplex.core.elements import Element
from skcomplex.core.laws import RateLaw
from skcomplex.core.specs import ParameterSpec, StateSpec
from skcomplex.core.system import System


class CartElement(Element):
    def __init__(
        self,
        label,
        *,
        x: Annotated[float, StateSpec()] = 0.0,
        v: Annotated[float, StateSpec()] = 0.0,
        force: Annotated[float, ParameterSpec(adaptive=True, bounds=(-1, 1))] = 0.0,
    ):
        super().__init__(label)
        self.x = x
        self.v = v
        self.force = force

    def _get_laws(self, s):
        return [RateLaw(s.x, s.v), RateLaw(s.v, s.force)]


model = LinearizationAnalysis(outputs=['cart.x'], state_weights=[[1, 0], [0, 1]], control_weights=[[1]])
model.check().prepare(System(elements=[CartElement('cart')])).run()
print(model.controllability_, model.observability_)
print(model.gain_.round(3).tolist())
```

The library models with optimal controls are `optimal_harvesting` in resources, `optimal_consumption` in
macroeconomics, `sir_vaccination` in epidemiology, and `optimal_chemotherapy` and `optimal_immunotherapy` in
oncology.

## Package layout

The package holds five parts, layered so that each imports only the parts below it. `skcomplex.experiments` builds on
`skcomplex.analysis`, which builds on `skcomplex.engine`, which builds on `skcomplex.core`. `skcomplex.models` is the
library of ready-made models built from the other four, and none of them imports it.

### The core

`skcomplex.core` defines what a system is, and nothing about how it runs. Its modules are listed in their import
order, so a module imports only the modules above it.

- `errors`: the errors of the package, such as `ValidationError`, `RoleError`, and `SolverError`.
- `specs`: the specs that annotate states, parameters, and roles, such as `StateSpec`, `ParameterSpec`, and `RoleSpec`,
  and the specs of runs and structures, such as `ObservableSpec`, `WienerProcessSpec`, and `RequirementSpec`.
- `types`: the types that annotations share.
- `base`: `BasePart`, the base of every part, with the verbs `check`, `prepare`, and `run`.
- `spaces`: the continuous, lattice, network, mesh, and Cayley spaces, their neighbourhoods, and the kinds of time.
- `laws`: the `BaseLaw` base and every law, and in `laws.utils` what laws are written with besides Sage, such as
  `where`, reductions, draws, delays, and drivers.
- `elements` and `interactions`: `Element`, `GroupElement`, `FieldElement`, and the other elements, and `Interaction`,
  `PairInteraction`, and the other interactions.
- `system`: `System`.

### The engine

`skcomplex.engine` runs systems. Its packages are listed in their import order too.

- `problems`: the compiled problems, their layouts of states and parameters, and their numeric functions.
- `backends`: the NumPy and Numba backends, which lower expressions to numeric functions.
- `compile`: `validate`, `compile_system`, and the pipeline of passes that turns a system into a problem.
- `solvers`: the solver contract, `step_count`, and every solver.
- `runtime`: `Simulation`, clocks, observers, and trajectories.
- `viz`: the visuals and the Rerun observer.

`skcomplex.analysis` holds the analyses of systems, trajectories, and data, and `skcomplex.experiments` the experiments
over designs of parameters and replicates. Tests of the import contracts enforce the layering.

## Kinds of dynamics

Some kinds of dynamics, such as lattice dynamics or quantum dynamics, are needed by few models. Each is not a package of
its own: its laws, elements, and spaces are parts of the core, and the passes that compile it, its solvers, and its
visuals are parts of the engine, beside those of every other kind.

### Where each kind lives

| Kind | Core | Engine |
| --- | --- | --- |
| agents | `RuleLaw`, `MatchLaw` | the agents passes, `MesaSolver` |
| lattices | `LocalRuleLaw`, `DriveLaw`, `RelaxationLaw`, `ExtremalLaw`, `LatticeElement`, `LatticeSpace`, `EnergySpec`, `SamplerSpec` | `LatticeSolver`, `MonteCarloSolver` |
| finitely generated groups | `CayleySpace` | `CayleyVisual` |
| rewriting | `ProductionLaw`, `SubstitutionLaw`, `GraphProductionLaw`, `WordElement`, `PatchElement`, `HostGraphElement` | a system compiler, `RewritingSolver` |
| quantum | `GeneratorLaw`, `DissipatorLaw`, `HilbertSpec` | a system compiler, `UnitarySolver`, `LindbladSolver`, `QuantumTrajectoriesSolver` |
| fields | `FieldElement`, `FluxLaw`, `MeshSpace`, `laplacian` and the other operators | the fields passes, `IMEXEulerSolver`, `CrankNicolsonSolver`, `SSPRKSolver` |
| kinetics | `EncounterLaw`, `moment` | the encounter passes, `KineticSolver` |
| fractional | `StateSpec(order=...)` in `(0, 1)`, `FractionalNoiseLaw` | the fractional passes, `FractionalABMSolver`, `GrunwaldLetnikovSolver`, `FractionalEulerMaruyamaSolver` |
| topology | `TopologyRequestSpec` | the topology passes |
| particles | `ParticleElement`, `PotentialInteraction`, `PairPotentialInteraction` | `VelocityVerletSolver`, `LammpsSolver` |
| control | `ParameterSpec(adaptive=True, bounds=...)` | `PontryaginSolver`, `CollocationSolver` |

Most kinds compile through the passes of `compile_system`, which lower each law to the problem of its form. Quantum
and rewriting systems are compiled whole by a system compiler, a `BaseSystemCompiler`, and `claiming_compiler` returns
the compiler that claims a system, or `None` for the passes.

### Writing a law

A new law derives from `BaseLaw`. It stores its arguments under their own names, and it implements three methods.

- `_get_targets` returns the states it changes, which `prepare` sets as `targets_`.
- `_get_requirements` returns a `RequirementSpec` of the kinds of time and space it runs in.
- `_run` returns its value at given values of the symbols it reads, which `prepare` sets as `reads_`.

```python
from typing import Annotated

from sage.all__sagemath_symbolics import var

from skcomplex.core.elements import Element
from skcomplex.core.laws import BaseLaw
from skcomplex.core.spaces import ContinuousTime
from skcomplex.core.specs import ParameterSpec, RequirementSpec, StateSpec
from skcomplex.core.system import System
from skcomplex.engine.compile import validate


class DecayLaw(BaseLaw):
    def __init__(self, target, rate, *, label=''):
        super().__init__(label)
        self.target = target
        self.rate = rate

    def _check(self):
        return self

    def _get_targets(self):
        return (self.target,)

    def _get_requirements(self):
        return RequirementSpec(time=(ContinuousTime,))

    def _run(self, inputs):
        return self._evaluate(-self.rate * self.target, inputs)


n, k = var('n k')
law = DecayLaw(n, k).check().prepare('atom')
print(law.targets_, sorted(map(str, law.reads_)), law.owner_, law.run({'n': 2.0, 'k': 0.5}))


class NucleusElement(Element):
    def __init__(self, label, *, n: Annotated[float, StateSpec()] = 1.0, k: Annotated[float, ParameterSpec()] = 1.0):
        super().__init__(label)
        self.n = n
        self.k = k

    def _get_laws(self, s):
        return [DecayLaw(s.n, s.k)]


print([issue.code for issue in validate(System(elements=[NucleusElement('atom')])).errors])
```

The passes and the system compilers are fixed lists of the engine, so a law of a new class
checks, prepares, and runs on its own, but no form takes it in a system. `validate` reports it with the code
`laws-mixed`, and a new kind of dynamics is a change to the engine.

## The model library

`skcomplex.models` is the library of ready-made models, organised by scientific field and subfield as
`skcomplex.models.<field>.<subfield>`.

| Field | Subfields |
| --- | --- |
| `physics` | `mechanics`, `celestial_mechanics`, `nonlinear_oscillators`, `synchronization`, `statistical_mechanics`, `criticality`, `soft_matter`, `active_matter`, `fluids`, `waves`, `pattern_formation`, `heat`, `cosmology` |
| `chemistry` | `kinetics`, `pattern_formation` |
| `biology` | `ecology`, `epidemiology`, `evolution`, `neuroscience`, `systems_biology`, `physiology`, `collective_behaviour`, `oncology` |
| `earth` | `climate`, `oceanography`, `seismology` |
| `economics` | `macroeconomics`, `markets`, `finance`, `econophysics`, `resources`, `innovation` |
| `social` | `opinion_dynamics`, `segregation`, `cooperation`, `spreading`, `crowds`, `organizations` |
| `engineering` | `traffic`, `circuits`, `control`, `power_systems`, `robotics`, `processes` |
| `mathematics` | `dynamical_systems`, `stochastic_processes`, `networks`, `fractals` |
| `computer_science` | `automata`, `evolutionary_computation`, `neural_networks` |

A model lives where the science that poses it lives, and tags carry its reach into other fields. The core never
imports the models, every model is built only from the public names of the core, and every subfield package is
self-contained, so no model package imports another. A test of the package layout enforces these rules.

### The model catalog

Every ready-made model is a factory registered with its field path, description, tags, references, and default run
settings. `catalog` lists them, filtered by a field, such as `biology`, by a field path, such as `biology/ecology`, or
by a tag, `describe` gives a model's parameters and the formalism inferred by compiling it, `create` builds it by name,
and `run` simulates it with its defaults. The catalog belongs to the models library alone: the engine and experiments
take a system or a factory, never a model name, so a model of the library is created with `create` or passed as its
factory.

```python
from skcomplex.models import catalog, create, describe, run

print([entry.name for entry in catalog(field='biology/ecology')])
print(sorted({entry.field for entry in catalog(field='chemistry')}))
print(describe('lotka_volterra').formalism)
predation = create('lotka_volterra', a=0.2)
print(run('sir').trajectory_.values('I.n').max().round())
```

### Models as patterns

The models are examples of the core interface, not a separate one, so each is also the pattern for a model of your own.
The page of every field, under Models in the documentation, shows each model's factory, its element and interaction
types with their states, parameters, and laws as written, and, for differential equations and maps, its equations as
`SymbolicAnalysis('equations')` returns them. A model of your own follows the same steps: declare the element and
interaction types, write a factory that returns the system, and register it with `register_model`.

```python
from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from skcomplex.core.elements import Element
from skcomplex.core.laws import RateLaw
from skcomplex.core.specs import ParameterSpec, StateSpec
from skcomplex.core.system import System
from skcomplex.models import create, register_model


class TankElement(Element):
    def __init__(
        self,
        label,
        *,
        level: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        k: Annotated[float, ParameterSpec()] = 0.5,
    ):
        super().__init__(label)
        self.level = level
        self.k = k

    def _get_laws(self, s):
        return [RateLaw(s.level, -s.k * s.level)]


@register_model(field='examples/processes', references=('Torricelli 1643',), defaults={'until': 5})
def draining_tank(level: float = 2.0) -> System:
    """Return a tank that drains at a rate proportional to its level."""
    return System(elements=[TankElement('tank', level=level)])


print(create('draining_tank').elements[0].level)
```

Every registered factory needs defaults for all of its arguments, so the catalog can create, compile, simulate,
visualize, and clone every model through the same interface.

### Coverage

The model inventory records every model surveyed from the reference books and articles, with its sources and status:
implemented, pending a planned change, or blocked by a missing feature of the core. The coverage page, under Models in
the documentation, renders it by field and subfield. A model the core cannot express is recorded as blocked on the
OpenSpec change that specifies the missing general feature, never worked around inside the model package, and a test
checks the inventory against the catalog and the changes.

Models exist first to exercise the core, and ready-made models are a by-product. Every entry lists the core primitives
it exercises, such as `rates`, `jumps`, `groups`, `reductions`, `member-offsets`, `lattices`, `fields`, `agents`, or
`stocks-and-flows`, from a vocabulary the inventory defines, and a primitive that the core does not have yet names the
change that adds it. An entry's priority is `core` when it exercises a primitive or a combination of primitives that
no earlier implemented entry exercises, or needs a missing one, and `example` when it repeats a covered pattern. The
coverage page shows, for every primitive, the implemented models that exercise it, and flags the uncovered primitives
with the changes that add them, so new models go first where the core is least exercised.

## Groups of many elements

A group holds N identical elements of one type as arrays with one row per member. It compiles once, whatever its size,
and every law written for one member runs over all members with array operations.

```python
import numpy as np

from skcomplex.core.elements import GroupElement, ParticleElement
from skcomplex.core.spaces import ContinuousSpace
from skcomplex.core.system import System
from skcomplex.engine.runtime import Simulation

rng = np.random.default_rng(0)
values = {'x': rng.uniform(0, 1, (1_000, 2)), 'v': rng.normal(0, 1, (1_000, 2))}
gas = System(
    elements=[GroupElement('gas', element=ParticleElement, size=1_000, values=values)],
    space=ContinuousSpace(extent=(1.0, 1.0), boundary='reflecting'),
)
trajectory = Simulation(gas, dt=0.01).run(until=0.5).trajectory_
print(trajectory.values('gas.x').shape)
```

- A group is `GroupElement(label, element=..., size=..., values={...})`, with per-member values by state name. States
  and parameters of agents cannot take the names of the agent view, such as `location` or `alive`.
- A value is an array whose first dimension is the size of the group, or one value shared by every member.
- An interaction whose role binds to a group applies to every member. A single element bound next to the group
  receives the sum over the members, unless its law reads the group only through reductions.
- Group variables are named by the group path, such as `gas.x` with shape `(1000, 2)`, and a group parameter takes
  one value for every member or one value per member in `update_parameters`.
- `ideal_gas(n_molecules, kT=..., size=...)` builds a gas of non-interacting molecules in a reflecting box.

### Reductions over a group

`total`, `mean`, `maximum`, `minimum`, `variance`, and `count` of `skcomplex.core.laws.utils` reduce an expression over
the members of a group inside any law. Inside the reduction, the symbols of a member stand for the reduced member, and
outside it for the reading member, so `RateLaw(s.x, -s.x + tanh(total(s.x)))` reads the group total in every member's
own law. A reduction is evaluated once per evaluation, before the member laws, at a cost linear in the size of the
group, on NumPy and Numba.

- `where` selects the members with a condition on the reduced member, such as `count(where=s.x > 0.5)`. The reductions
  of no selected member are 0.
- `weights` gives `total` and `mean` a constant matrix, a NumPy array or a SciPy sparse matrix, with one row per
  reading member and one column per reduced member, so member `i` reads `sum_j W_ij a_j`, such as a payoff matrix. A
  vector with one weight per reduced member is a row that every reading member shares, such as the activities of a
  moment.
- `over` names the group by its namespace, such as `over=s.neurons`, when the expression reads no symbol of it.
- A reduction reads one group, and an argument that reads two groups raises an error. Reductions nest, so the mean
  fitness `total(s.x * total(s.x, weights=A))` of replicator dynamics evaluates the inner reduction first.
- An interaction reads groups member by member outside reductions, which needs groups of one size, and reads other
  groups through reductions. A law that sets a single element from reductions alone runs once, not once per member.
- Discrete maps read the reductions at step `n`, jump propensities after every event, and events at the state they
  receive.

```python
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import tanh

from skcomplex.core.elements import Element, GroupElement
from skcomplex.core.interactions import Interaction
from skcomplex.core.laws import RateLaw
from skcomplex.core.laws.utils import total
from skcomplex.core.specs import RoleSpec, StateSpec
from skcomplex.core.system import System
from skcomplex.engine.runtime import Simulation


class UnitElement(Element):
    def __init__(self, label, *, x: Annotated[float, StateSpec()] = 0.0):
        super().__init__(label)
        self.x = x

    def _get_laws(self, s):
        return [RateLaw(s.x, -s.x + tanh(total(s.x)))]


units = System(elements=[GroupElement('units', element=UnitElement, size=50, values={'x': np.linspace(-1, 1, 50)})])
print(Simulation(units, dt=0.5).run(until=2).trajectory_.values('units.x')[-1, :3].round(3).tolist())

payoff = np.array([[-0.5, 2.0], [0.0, 1.0]])


class StrategyElement(Element):
    def __init__(self, label, *, x: Annotated[float, StateSpec()] = 0.5):
        super().__init__(label)
        self.x = x


class ReplicationInteraction(Interaction):
    def __init__(self, label, *, population: Annotated[str, RoleSpec()]):
        super().__init__(label)
        self.population = population

    def _get_laws(self, s):
        x = s.population.x
        return [RateLaw(x, x * (total(x, weights=payoff) - total(x * total(x, weights=payoff))))]


game = System(
    elements=[GroupElement('population', element=StrategyElement, size=2, values={'x': [0.1, 0.9]})],
    interactions=[ReplicationInteraction('replication', population='population')],
)
print(Simulation(game, dt=1.0).run(until=40).trajectory_.values('population.x')[-1].round(3).tolist())
```

A role also binds to an element set, a list of single elements of one type. The interaction's laws apply to each
element, the elements keep their own state names, and reductions over the role reduce over the set.

```python
from typing import Annotated

from skcomplex.core.elements import Element
from skcomplex.core.interactions import Interaction
from skcomplex.core.laws import RateLaw
from skcomplex.core.laws.utils import total
from skcomplex.core.specs import ParameterSpec, RoleSpec, StateSpec
from skcomplex.core.system import System
from skcomplex.engine.runtime import Simulation


class ActivityElement(Element):
    def __init__(self, label, *, x: Annotated[float, StateSpec()] = 0.1):
        super().__init__(label)
        self.x = x


class InhibitionInteraction(Interaction):
    def __init__(self, label, *, units: Annotated[str, RoleSpec()], beta: Annotated[float, ParameterSpec()] = 2.0):
        super().__init__(label)
        self.units = units
        self.beta = beta

    def _get_laws(self, s):
        x = s.units.x
        return [RateLaw(x, x * (1 - x) - s.beta * x * (total(x) - x))]


network = System(
    elements=[ActivityElement(f'unit_{index}', x=0.1 * (index + 1)) for index in range(3)],
    interactions=[InhibitionInteraction('inhibition', units=['unit_0', 'unit_1', 'unit_2'])],
)
print((Simulation(network, dt=1.0).run(until=30).state_.round(3) + 0.0).tolist())
```

### Variadic roles

A role declared with `RoleSpec(variadic=True)` binds a list of element paths and group member references `<group>[<i>]`,
and laws read it as a tuple of namespaces in the given order, so a law can take any number of inputs. A parameter
declared with `ParameterSpec(per='<role>')` holds one value per entry, laws read it as a tuple of symbols, and each
value is the parameter component `<path>.<name>[<i>]`, which `update_parameters` and `schedule` set like any parameter.

```python
from typing import Annotated

from skcomplex.core.elements import Element, GroupElement
from skcomplex.core.interactions import Interaction
from skcomplex.core.laws import UpdateLaw
from skcomplex.core.specs import ParameterSpec, RoleSpec, StateSpec
from skcomplex.core.system import System
from skcomplex.engine.runtime import Simulation


class NodeElement(Element):
    def __init__(self, label, *, x: Annotated[float, StateSpec()] = 0.0):
        super().__init__(label)
        self.x = x


class WeightedSumInteraction(Interaction):
    def __init__(
        self,
        label,
        *,
        target: Annotated[str, RoleSpec()],
        inputs: Annotated[list[str], RoleSpec(variadic=True)],
        weights: Annotated[list[float], ParameterSpec(per='inputs')] = 1.0,
    ):
        super().__init__(label)
        self.target = target
        self.inputs = inputs
        self.weights = weights

    def _get_laws(self, s):
        total_input = sum(weight * item.x for weight, item in zip(s.weights, s.inputs, strict=True))
        return [UpdateLaw(s.target.x, total_input)]


summed = System(
    elements=[
        NodeElement('a', x=1.0),
        NodeElement('b', x=2.0),
        NodeElement('out'),
        GroupElement('nodes', element=NodeElement, size=5, values={'x': [0.0, 1.0, 2.0, 3.0, 4.0]}),
    ],
    interactions=[WeightedSumInteraction('sum', target='out', inputs=['a', 'b', 'nodes[3]'], weights=[1.0, 1.0, 10.0])],
)
simulation = Simulation(summed, dt=1).run(until=1)
simulation.update_parameters({'sum.weights[2]': 0.0})
print(simulation.trajectory_.values('out.x').tolist(), simulation.parameters_.tolist())
```

An element set and a variadic role both bind a list. An element set applies the laws once per element, each element
read like a single element, while a variadic role gives one law the whole sequence. `random_boolean_network` and
`yeast_cell_cycle` read the inputs of every node through one variadic role. A role declared with
`RoleSpec(noise=True)` binds a `WienerProcessSpec` instead of a path, as the section on shared and correlated noise
shows.

### Several groups and member offsets

Roles of one interaction bind several groups of one size, and the laws read row `i` of every group together. Groups of
different sizes raise a `RoleError`, since coupling every member of one group with every member of another is the work
of a pair interaction or a reduction.

`shift_member(expression, offset, boundary='periodic', fill=0.0)` reads an expression of a group's members at the
member `offset` places along the group order, the member index. Past the ends, `'periodic'` wraps around, `'reflecting'`
mirrors the order, and `'fixed'` reads `fill`. Nested offsets add, and an offset read of a single element raises an
error. The read compiles to an index table of the group, so one kernel runs over every member on the NumPy and the
Numba backends. The ring roads of `optimal_velocity` and `intelligent_driver_model` read the car ahead this way, with
`mod` for the headway across the point where positions wrap.

```python
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import tanh

from skcomplex.core.elements import Element, GroupElement
from skcomplex.core.interactions import Interaction
from skcomplex.core.laws import RateLaw
from skcomplex.core.laws.utils import mod, shift_member
from skcomplex.core.specs import ParameterSpec, RoleSpec, StateSpec
from skcomplex.core.system import System
from skcomplex.engine.runtime import Simulation


class CarElement(Element):
    def __init__(self, label, *, x: Annotated[float, StateSpec()] = 0.0, v: Annotated[float, StateSpec()] = 1.0):
        super().__init__(label)
        self.x = x
        self.v = v

    def _get_laws(self, s):
        return [RateLaw(s.x, s.v)]


class FollowingInteraction(Interaction):
    def __init__(self, label, *, cars: Annotated[str, RoleSpec()], length: Annotated[float, ParameterSpec()] = 40.0):
        super().__init__(label)
        self.cars = cars
        self.length = length

    def _get_laws(self, s):
        headway = mod(shift_member(s.cars.x, 1) - s.cars.x, s.length)
        return [RateLaw(s.cars.v, tanh(headway - 2.0) + tanh(2.0) - s.cars.v)]


positions = np.arange(20) * 2.0 + np.eye(20)[0] * 0.1
ring = System(
    elements=[GroupElement('cars', element=CarElement, size=20, values={'x': positions})],
    interactions=[FollowingInteraction('following', cars='cars', length=40.0)],
)
print(bool(Simulation(ring, dt=1.0).run(until=50).trajectory_.values('cars.v')[-1].std() > 0.1))
```

### Jacobians of reductions

A reduction makes every reading member depend on every reduced member, so the Jacobian of a drift with reductions has
the rank-one block `dF_i/dR dR/dx_j` of every reduction `R`. `problem.jacobian` of an ODE problem evaluates the exact
Jacobian by the chain rule through every reduction, as a SciPy sparse matrix, and `problem.jacobian.pattern()` states
its structure, which the implicit methods of `ScipyODESolver` receive (see [Discretizations and stiff
solvers](#discretizations-and-stiff-solvers)). When the dense blocks exceed 4 million entries, `pattern()` returns
`None` and an implicit solver estimates a dense Jacobian. Mean-field groups are rarely stiff, so an explicit method such
as `ScipyODESolver(method='RK45')` suits large groups with reductions.

```python
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import tanh

from skcomplex.core.elements import Element, GroupElement
from skcomplex.core.laws import RateLaw
from skcomplex.core.laws.utils import total
from skcomplex.core.specs import StateSpec
from skcomplex.core.system import System
from skcomplex.engine.compile import compile_system


class UnitElement(Element):
    def __init__(self, label, *, x: Annotated[float, StateSpec()] = 0.0):
        super().__init__(label)
        self.x = x

    def _get_laws(self, s):
        return [RateLaw(s.x, -s.x + tanh(total(s.x)))]


units = System(elements=[GroupElement('units', element=UnitElement, size=50, values={'x': np.linspace(-1, 1, 50)})])
problem = compile_system(units)
jacobian = problem.jacobian.evaluate(0.0, problem.initial_state, problem.initial_parameters)
print(jacobian.shape, problem.jacobian.pattern().nnz)
```

`StabilityAnalysis` and the other analyses of the vector field of `skcomplex.analysis` expand a group of at most 64
members into one equation per member, with every reduction written as an explicit sum, so the fixed points of
replicator dynamics are derived symbolically. A larger group raises an error that states the limit.

### Pair interactions

A pair interaction is written once for a pair of members and summed over every pair closer than its cutoff, inside one
group or between two groups. `PairPotentialInteraction` derives the forces from a pair energy.

```python
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from skcomplex.core.elements import GroupElement, ParticleElement
from skcomplex.core.interactions import PairPotentialInteraction
from skcomplex.core.spaces import ContinuousSpace
from skcomplex.core.specs import ParameterSpec
from skcomplex.core.system import System
from skcomplex.engine.runtime import Simulation


class LennardJonesInteraction(PairPotentialInteraction):
    def __init__(
        self,
        label,
        first,
        second,
        cutoff=2.5,
        *,
        epsilon: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        sigma: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
    ):
        super().__init__(label, first, second, cutoff=cutoff)
        self.epsilon = epsilon
        self.sigma = sigma

    def _get_pair_energy(self, s, r):
        return 4 * s.epsilon * ((s.sigma / r) ** 12 - (s.sigma / r) ** 6)


side = np.arange(10) * 1.2
positions = np.stack(np.meshgrid(side, side), axis=-1).reshape(-1, 2)
liquid = System(
    elements=[GroupElement('atoms', element=ParticleElement, size=100, values={'x': positions})],
    interactions=[LennardJonesInteraction('lj', first='atoms', second='atoms', cutoff=2.5)],
    space=ContinuousSpace(n_dim=2),
)
print(Simulation(liquid, dt=0.05).run(until=1).trajectory_.values('atoms.x').shape)
```

`period` gives the side of a periodic box, so pair distances use the nearest periodic image.

### Boundaries

A `BoundaryLaw` sets the value of a state after every internal solver step, however long the frame, so a frame never
reports a state that a boundary would change. Adaptive solvers find the first time in a step where a boundary acts and
restart there. Boundaries combine with every form, so walls and periodic boxes work for differential equations,
stochastic systems, jump processes, and discrete maps.
A `ContinuousSpace` with `boundary='reflecting'` mirrors the position states of its elements at the walls and reverses
the velocities declared with `StateSpec(velocity_of=...)`, and `boundary='periodic'` wraps positions around the box.

## Lattice models

Cellular automata, sandpiles, and spin models live on lattices. `LatticeSpace(shape, boundary=...)` is a box of `Z^d`
with `'periodic'`, `'fixed'`, or `'reflecting'` edges, and `LatticeElement(label, element=..., space=space,
values={...})` puts one member of an element type on every site, with values given as arrays of the lattice shape.
Laws are written for one cell.

A space is a part of its system and speaks the same verbs. `space.check().prepare()` checks its arguments and derives
its facts, which end in `_`: the number of sites `size_`, the dimension `n_dim_`, the shape of the arrays of its sites
`shape_`, the position of every site `positions_`, whether it wraps `periodic_`, and the visual of its groups
`default_visual_`. Compiling a system prepares its spaces, so a space needs `prepare()` only when its facts are read
before.

```python
from skcomplex.core.spaces import LatticeSpace

space = LatticeSpace((4, 3)).check().prepare()
print(space.size_, space.shape_, space.periodic_, space.positions_[4].tolist())
```

**Local rules.** `LocalRuleLaw(targets, over, rule)` computes a cell's next value from the ordered values of its
neighbourhood. `MooreNeighborhood`, `VonNeumannNeighborhood`, `BoxNeighborhood`, or `LatticeNeighborhood(offsets)`
give the offsets. A rule is a `Table` of patterns, a `Totalistic` or `OuterTotalistic` rule from
`skcomplex.core.laws.utils`, a rule code whose `table()` returns a table, such as `Wolfram` of the automata models, or
a function of NumPy arrays, one per neighbour, that also receives the parameters it names and, for stochastic rules,
uniform random numbers. Tables and totalistic rules run as lookups, fused into one compiled loop on the Numba backend.

```python
from typing import Annotated

import numpy as np
from sage.sets.finite_enumerated_set import FiniteEnumeratedSet

from skcomplex.core.elements import Element, LatticeElement
from skcomplex.core.laws import LocalRuleLaw
from skcomplex.core.laws.utils import OuterTotalistic
from skcomplex.core.spaces import LatticeSpace, MooreNeighborhood
from skcomplex.core.specs import StateSpec
from skcomplex.core.system import System
from skcomplex.engine.runtime import Simulation


class LifeElement(Element):
    def __init__(self, label, *, alive: Annotated[float, StateSpec(domain=FiniteEnumeratedSet([0, 1]))] = 0):
        super().__init__(label)
        self.alive = alive

    def _get_laws(self, s):
        return [LocalRuleLaw(s.alive, MooreNeighborhood(), OuterTotalistic({(0, 3): 1, (1, 2): 1, (1, 3): 1}))]


glider = np.zeros((20, 20))
glider[1, 2] = glider[2, 3] = glider[3, 1] = glider[3, 2] = glider[3, 3] = 1
space = LatticeSpace((20, 20))
cells = System(elements=[LatticeElement('cells', element=LifeElement, space=space, values={'alive': glider})])
trajectory = Simulation(cells, dt=1).run(until=4).trajectory_
print(np.array_equal(trajectory.lattice('cells.alive')[-1], np.roll(glider, (1, 1), axis=(0, 1))))
```

- Rules update synchronously by default, every cell from the previous configuration. `schedule='random-sequential'`,
  `'random-site'`, `'checkerboard'`, or `'colored'` update cells one at a time or one class of sites at a time.
- One unit of time is one application of every rule, so `dt` is an integer, and `trajectory.lattice(name)` returns
  the states in lattice shape.
- `DriveLaw(target, function)` changes a whole lattice at once, such as dropping a grain on a random site.

**Relaxation.** `RelaxationLaw(drive, relax, stable)` separates a slow drive from a fast relaxation, as in
self-organised criticality. Every step applies the drive laws once and the relax laws until `stable` holds, and each
relaxation is recorded among the trajectory's events with its size, duration, and area. `activity` marks the sites
that count as updates, such as the toppling sites of a sandpile.

```python
from skcomplex.engine.runtime import Simulation

from skcomplex.models.physics.criticality import sandpile

trajectory = Simulation(sandpile((20, 20)), dt=100, random_state=0).run(until=2000).trajectory_
print(max(event.size for event in trajectory.events))
```

**Monte Carlo.** A `SampleLaw(energy, move, sampler)` samples states. `EnergySpec(expression, over)` is the local
energy of a site, every term of the total energy that involves it, with neighbour values read through `neighbors`.
`MoveSpec(target, kind)` proposes changes, with the kinds `'flip'`, `'random-value'`, and `'rotate'`, and
`SamplerSpec('metropolis' | 'glauber' | 'heat-bath', temperature)` accepts them. The energy change of a move is derived
symbolically, sweeps update the two sublattices in turn, and the temperature is a parameter that changes live.

```python
from typing import Annotated

from sage.all__sagemath_symbolics import ZZ

from skcomplex.core.elements import Element, LatticeElement
from skcomplex.core.laws import SampleLaw
from skcomplex.core.laws.utils import neighbors
from skcomplex.core.spaces import LatticeSpace, VonNeumannNeighborhood
from skcomplex.core.specs import EnergySpec, MoveSpec, ParameterSpec, SamplerSpec, StateSpec
from skcomplex.core.system import System
from skcomplex.engine.runtime import Simulation

nearest = VonNeumannNeighborhood()


class SpinElement(Element):
    def __init__(
        self,
        label,
        *,
        spin: Annotated[float, StateSpec(domain=ZZ)] = 1.0,
        J: Annotated[float, ParameterSpec()] = 1.0,
        temperature: Annotated[float, ParameterSpec()] = 1.5,
    ):
        super().__init__(label)
        self.spin = spin
        self.J = J
        self.temperature = temperature

    def _get_laws(self, s):
        energy = -s.J * s.spin * sum(neighbors(s.spin, nearest, reduce='each'))
        sampler = SamplerSpec('metropolis', s.temperature)
        return [SampleLaw(EnergySpec(energy, nearest), MoveSpec(s.spin, 'flip'), sampler)]


spins = System(elements=[LatticeElement('spins', element=SpinElement, space=LatticeSpace((32, 32)))])
print(abs(Simulation(spins, dt=50, random_state=0).run(until=200).state_.mean()) > 0.9)
```

**Triangular lattices.** `LatticeSpace(shape, geometry='triangular')` keeps integer coordinates on a skewed grid, so
every boundary works as on a square lattice, and places site `(i, j)` at `(j + i / 2, i sqrt(3) / 2)` in the plane,
given by the fitted `positions_`. `TriangularNeighborhood()` holds its six neighbours counterclockwise from 0 degrees,
so direction `k + 3` modulo 6 is opposite to direction `k`, as lattice gases need. The cells of the sites are the
hexagons of a hexagonal tiling.

```python
from typing import Annotated

import numpy as np

from skcomplex.core.elements import Element, LatticeElement
from skcomplex.core.laws import LocalRuleLaw
from skcomplex.core.spaces import LatticeSpace, TriangularNeighborhood
from skcomplex.core.specs import StateSpec
from skcomplex.core.system import System
from skcomplex.engine.runtime import Simulation


def spread(*values):
    return np.max(values, axis=0)


class BurnElement(Element):
    def __init__(self, label, *, fire: Annotated[float, StateSpec()] = 0.0):
        super().__init__(label)
        self.fire = fire

    def _get_laws(self, s):
        return [LocalRuleLaw(s.fire, TriangularNeighborhood(center=True), spread)]


fire = np.zeros((40, 40))
fire[20, 20] = 1
space = LatticeSpace((40, 40), geometry='triangular')
forest = System(elements=[LatticeElement('forest', element=BurnElement, space=space, values={'fire': fire})])
burning = Simulation(forest, dt=1).run(until=3).trajectory_.lattice('forest.fire')
print([int(item.sum()) for item in burning])
```

**Lattices on graphs.** `LatticeElement` also takes a `NetworkSpace`, with one site per node in the order of
`graph.nodes`, and initial values as arrays in node order or mappings from node to value.
`AdjacencyNeighborhood(center=False)` is the neighbourhood of a node, its neighbours along its edges, padded to the
largest degree with the network's `fill` value. Local rules, drives, relaxations, Monte Carlo laws, and extremal laws
run on graphs as on lattices. Lookup rules on graphs take the totalistic forms, and a function rule that names the
keyword `degree` receives the number of real neighbours of every site. The padded table grows with the largest degree,
so graphs with hubs of very high degree, beyond scale-free graphs of about 10^5 nodes, use much memory.

**Neighbour sums and colored schedules.** `neighbors(expression, reduce='sum')` sums an expression of a site and one
neighbour, read through `s.other`, over the real neighbours of the site, so local energies are exact on sites of any
degree and outside fixed edges. `schedule='colored'` on local rules and samplers updates the classes of a proper
coloring of the neighbour graph in turn, which NetworkX computes once per law. The checkerboard schedule stays the
two-color case of square lattices.

```python
from typing import Annotated

import networkx as nx
from sage.all__sagemath_symbolics import ZZ

from skcomplex.core.elements import Element, LatticeElement
from skcomplex.core.laws import SampleLaw
from skcomplex.core.laws.utils import neighbors
from skcomplex.core.spaces import AdjacencyNeighborhood, NetworkSpace
from skcomplex.core.specs import EnergySpec, MoveSpec, ParameterSpec, SamplerSpec, StateSpec
from skcomplex.core.system import System
from skcomplex.engine.runtime import Simulation


class GraphSpinElement(Element):
    def __init__(
        self,
        label,
        *,
        spin: Annotated[float, StateSpec(domain=ZZ)] = 1.0,
        J: Annotated[float, ParameterSpec()] = 1.0,
        temperature: Annotated[float, ParameterSpec()] = 1.0,
    ):
        super().__init__(label)
        self.spin = spin
        self.J = J
        self.temperature = temperature

    def _get_laws(self, s):
        energy = -s.J * s.spin * neighbors(s.other.spin, reduce='sum')
        sampler = SamplerSpec('metropolis', s.temperature, schedule='colored')
        return [SampleLaw(EnergySpec(energy, AdjacencyNeighborhood()), MoveSpec(s.spin, 'flip'), sampler)]


graph = nx.random_regular_graph(4, 200, seed=0)
spins = System(elements=[LatticeElement('spins', element=GraphSpinElement, space=NetworkSpace(graph))])
print(abs(Simulation(spins, dt=50, random_state=0).run(until=200).state_.mean()) > 0.9)
```

**Extremal updates.** `ExtremalLaw(targets, over, rule, key=None, extremum='min', repeat=1)` finds the site with
the smallest or largest value of a key state, breaking ties with the simulation's random stream, and applies
`rule(*values, random, **parameters)` to the site and its real neighbours, `repeat` times per step. Every application
is recorded among the trajectory's events as an `ExtremalRecord` with its time, site, and extremal value. On the Numba
backend the applications of a step run in one compiled loop when Numba compiles the rule, which then takes its
arguments by position or keyword, with no keyword-only arguments, and both backends give the same events.

```python
from typing import Annotated

import numpy as np

from skcomplex.core.elements import Element, LatticeElement
from skcomplex.core.laws import ExtremalLaw
from skcomplex.core.spaces import LatticeNeighborhood, LatticeSpace
from skcomplex.core.specs import StateSpec
from skcomplex.core.system import System
from skcomplex.engine.runtime import Simulation


def renew(fitness, random):
    return random


class SpeciesElement(Element):
    def __init__(self, label, *, fitness: Annotated[float, StateSpec()] = 0.5):
        super().__init__(label)
        self.fitness = fitness

    def _get_laws(self, s):
        return [ExtremalLaw(s.fitness, LatticeNeighborhood(((-1,), (1,))), renew, repeat=100, label='extinction')]


fitness = np.random.default_rng(0).random(100)
space = LatticeSpace(100)
ring = System(elements=[LatticeElement('ring', element=SpeciesElement, space=space, values={'fitness': fitness})])
trajectory = Simulation(ring, dt=10, random_state=0).run(until=100).trajectory_
print(len(trajectory.events), trajectory.events[0].label)
```

Square lattice groups stream to Rerun as images colored by value, and triangular lattices and graph lattices stream as
sites with `SiteVisual`: hexagonal cells at the site positions, or nodes colored by state with their edges logged
once. The library has `elementary_automaton`, `game_of_life`, `majority_rule`, `langtons_ant`,
`nagel_schreckenberg`, `hodgepodge`, `sandpile`, `forest_fire`, `hexagonal_forest_fire`, `earthquakes`, `ising`,
`graph_ising`, `potts`, `xy`, `fhp_lattice_gas`, and `bak_sneppen`, each in the subfield of its science, such as
`skcomplex.models.computer_science.automata` for `game_of_life` and `skcomplex.models.physics.statistical_mechanics`
for `ising`.

## Networks

A `NetworkSpace` holds a NetworkX graph whose nodes carry the members of groups on its sites and whose edges carry the
members of groups on its edges. The states and the structure of a network can change together: interactions read and
change edge states, laws reduce over the neighbours of a node, and topology requests add and remove nodes and edges.

### Generators and layers

A `NetworkSpace` takes any NetworkX graph, so the generators of NetworkX build it, such as `nx.erdos_renyi_graph`,
`nx.watts_strogatz_graph`, or `nx.barabasi_albert_graph`. A generator takes a seed, so a network drawn for an
experiment is reproducible, and node positions in the node attribute `pos`, as `nx.random_geometric_graph` sets them,
become the fitted `positions_`. A `LayerSpec` names a second relation over the same nodes, such as work next to
contacts. Preparing a network lowers every layer to arrays by its label, `None` for the graph itself: `graphs_` holds
its graph, `edges_` the source and target of every edge, and `adjacency_` the neighbours of every node in compressed
sparse row form.

```python
import networkx as nx

from skcomplex.core.spaces import NetworkSpace
from skcomplex.core.specs import LayerSpec

first = NetworkSpace(nx.watts_strogatz_graph(100, 6, 0.1, seed=7))
second = NetworkSpace(nx.watts_strogatz_graph(100, 6, 0.1, seed=7))
print(sorted(first.graph.edges) == sorted(second.graph.edges))

space = NetworkSpace(nx.path_graph(4), layers=[LayerSpec('work', nx.complete_graph(4))]).check().prepare()
print(list(space.graphs_), [item.tolist() for item in space.edges_['work']])
```

### Interactions over edges

A group on the edges of a network, `GroupElement(..., on='edges')`, holds one member per edge of its layer. A
`NeighborInteraction` binds a group on the nodes and, optionally, a group on the edges of the same network. Its laws
run once per edge and read `s.source`, `s.target`, and `s.edge`: the members at both ends and the member of the edge.
Rate, noise, and jump laws may target any of them, and update laws the edge member. The argument `layer` chooses the
layer of an interaction without an edge group. Hebbian weights grow with the product of the activities at the ends:

```python
from typing import Annotated

import networkx as nx
import numpy as np

from skcomplex.core.elements import Element, GroupElement, LatticeElement
from skcomplex.core.interactions import NeighborInteraction
from skcomplex.core.laws import RateLaw
from skcomplex.core.spaces import NetworkSpace
from skcomplex.core.specs import ParameterSpec, RoleSpec, StateSpec
from skcomplex.core.system import System
from skcomplex.engine.runtime import Simulation


class NeuronElement(Element):
    def __init__(self, label, *, a: Annotated[float, StateSpec()] = 0.0):
        super().__init__(label)
        self.a = a


class SynapseElement(Element):
    def __init__(self, label, *, w: Annotated[float, StateSpec()] = 0.1):
        super().__init__(label)
        self.w = w


class HebbianInteraction(NeighborInteraction):
    def __init__(
        self,
        label,
        *,
        neurons: Annotated[str, RoleSpec()],
        synapses: Annotated[str, RoleSpec()],
        eta: Annotated[float, ParameterSpec()] = 1.0,
    ):
        super().__init__(label)
        self.neurons = neurons
        self.synapses = synapses
        self.eta = eta

    def _get_laws(self, s):
        return [RateLaw(s.edge.w, s.eta * s.source.a * s.target.a)]


space = NetworkSpace(nx.cycle_graph(6))
system = System(
    elements=[
        LatticeElement('neurons', element=NeuronElement, space=space, values={'a': [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]}),
        GroupElement('synapses', element=SynapseElement, space=space, on='edges'),
    ],
    interactions=[HebbianInteraction('hebb', neurons='neurons', synapses='synapses')],
)
trajectory = Simulation(system, dt=0.5).run(until=1.0).trajectory_
print(np.round(trajectory.values('synapses.w')[-1], 3).tolist())
```

### Reductions over neighbourhoods

`neighbors(expression, over=..., reduce=..., weight=...)` reduces an expression written for one neighbour, through
`s.other`, over the neighbours of every member: the neighbours of a network by default, a layer with
`AdjacencyNeighborhood(layer=label)`, the neighbourhood of a lattice, or in a continuous space `RadiusNeighborhood(r)`,
`NearestNeighborhood(k)`, and `ConeNeighborhood(r, angle)`. The reductions are `'sum'`, `'mean'`, `'count'`, `'max'`,
`'min'`, and `'normalized-mean'`. A weight may read the state of the edge group of the layer through a role of an
interaction, as the weight of the edge that joins the member to its neighbour. The Kuramoto coupling on a network is
`RateLaw(s.theta, s.omega + s.K * neighbors(sin(s.other.theta - s.theta), reduce='mean'))`.

```python
from typing import Annotated

import networkx as nx

from skcomplex.core.elements import Element, GroupElement, LatticeElement
from skcomplex.core.interactions import Interaction
from skcomplex.core.laws import RateLaw
from skcomplex.core.laws.utils import neighbors
from skcomplex.core.spaces import NetworkSpace
from skcomplex.core.specs import RoleSpec, StateSpec
from skcomplex.core.system import System
from skcomplex.engine.compile import compile_system


class NodeElement(Element):
    def __init__(self, label, *, u: Annotated[float, StateSpec()] = 0.0):
        super().__init__(label)
        self.u = u


class LinkElement(Element):
    def __init__(self, label, *, w: Annotated[float, StateSpec()] = 1.0):
        super().__init__(label)
        self.w = w


class DiffusionInteraction(Interaction):
    def __init__(self, label, *, nodes: Annotated[str, RoleSpec()], links: Annotated[str, RoleSpec()]):
        super().__init__(label)
        self.nodes = nodes
        self.links = links

    def _get_laws(self, s):
        return [RateLaw(s.nodes.u, neighbors(s.nodes.other.u - s.nodes.u, weight=s.links.w))]


space = NetworkSpace(nx.path_graph(3))
system = System(
    elements=[
        LatticeElement('nodes', element=NodeElement, space=space, values={'u': [0.0, 1.0, 4.0]}),
        GroupElement('links', element=LinkElement, space=space, on='edges', values={'w': [1.0, 10.0]}),
    ],
    interactions=[DiffusionInteraction('diffusion', nodes='nodes', links='links')],
)
problem = compile_system(system)
drift = problem.drift(0.0, problem.initial_state, problem.initial_parameters)
print(problem.states.read(drift, 'nodes.u').tolist())
```

Validation reports a reduction whose neighbourhood the space does not offer, such as `RadiusNeighborhood(1.0)` on a
network, with the code `law-space`, and a layer the network lacks with the code `binding`.

### Topology requests

A variable group on the sites or edges of a network, `GroupElement(..., on='sites', variable=True, capacity=n)`, holds
rows for up to `capacity` members, of which the members present are marked active. The NumPy and Numba backends run
such groups with fixed array shapes: laws leave the free rows unchanged, reductions read the members present, and
neighbourhoods and interactions over neighbours read the edges present. A `TopologyRequestSpec` of the kind
`'add-node'`, `'remove-node'`, `'add-edge'`, `'remove-edge'`, or `'rewire'` is the target of update and jump laws:
`UpdateLaw(request, probability)` fires the request with a probability at every step for every member or edge its
law runs for, and `JumpLaw(rate, {request: 1})` fires it at a rate. The iteration and Gillespie steppers apply the
requests between steps. A request that joins nodes draws the new ends with weights from `toward`, a module-level
function of the namespace of a candidate node, such as one that returns `neighbors(1)` for its degree. A request on a
group that is not variable gets the validation issue `group-fixed`, and a full capacity raises `CapacityError`.

Trajectories hold the values of absent members as NaN, and `trajectory.network(frame)` returns the nodes and edges
present at a frame. Rerun draws the nodes present and the edges at every frame where they change. Barabási-Albert growth
adds one node with two edges at every step:

```python
from typing import Annotated

import networkx as nx
import numpy as np

from skcomplex.core.elements import Element, GroupElement
from skcomplex.core.laws import UpdateLaw
from skcomplex.core.laws.utils import neighbors
from skcomplex.core.spaces import DiscreteTime, NetworkSpace
from skcomplex.core.specs import StateSpec, TopologyRequestSpec
from skcomplex.core.system import System
from skcomplex.engine.runtime import Simulation


class NodeElement(Element):
    def __init__(self, label, *, age: Annotated[float, StateSpec()] = 0.0):
        super().__init__(label)
        self.age = age

    def _get_laws(self, s):
        return [UpdateLaw(s.age, s.age + 1)]


class LinkElement(Element):
    pass


def degree(node):
    return neighbors(1)


class GrowthElement(Element):
    def _get_laws(self, s):
        return [UpdateLaw(TopologyRequestSpec('add-node', edges=2, toward=degree), 1)]


system = System(
    elements=[
        GroupElement('nodes', element=NodeElement, on='sites', variable=True, capacity=203),
        GroupElement('links', element=LinkElement, on='edges', variable=True, capacity=403),
        GrowthElement('growth'),
    ],
    space=NetworkSpace(nx.complete_graph(3)),
    time=DiscreteTime(),
)
trajectory = Simulation(system, dt=50.0, random_state=1).run(until=200).trajectory_
snapshot = trajectory.network(-1)
degrees = np.bincount(np.concatenate([snapshot.sources, snapshot.targets]))
print(snapshot.nodes.size, snapshot.sources.size, int(degrees.max()) > 10)
```

## Kinetic models of active particles

The kinetic theory of active particles of Bellomo et al. describes a population by the distribution of its members
over an activity variable, such as wealth, an opinion, or the progression of a tumour cell. Encounters between
particles change the distribution, and its moments give the macroscopic quantities.

### Distributions as site groups

A distribution over `n` activity values is a group on the sites of a lattice of `n` sites, one member per activity,
`GroupElement(label, element=E, size=n, on='sites')`. The element declares one real state, the density at the member's
activity, and the activity values are an array that the laws and the moments receive. No distribution component is
needed, and a population of several functional subsystems is several such groups.

```python
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from skcomplex.core.elements import Element, GroupElement
from skcomplex.core.specs import StateSpec


class WealthDensityElement(Element):
    def __init__(self, label, *, f: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0):
        super().__init__(label)
        self.f = f


classes = np.linspace(0.0, 1.0, 9)
profile = np.exp(-((classes - 0.5) ** 2) / 0.05)
wealth = GroupElement('wealth', element=WealthDensityElement, size=9, on='sites', values={'f': profile / profile.sum()})
```

### Encounters

`EncounterLaw(candidate, field, rate, transition, test=None)` declares that a candidate particle at activity `h` meets a
field particle at activity `k` at the rate `eta[h, k]`, and ends at activity `i` with the probability `A[h, k, i]`. An
interaction returns it with roles bound to the candidate and field groups, and the compiler derives the gain and loss
equations

`df_i/dt = sum_hk eta[h, k] A[h, k, i] f_h g_k - f_i sum_k eta[i, k] g_k`

without any hand derivation. The candidate moves into the `test` group when it is given, such as a cell that mutates
into another subsystem.

- `ProliferationLaw(candidate, field, rate, outcomes=None, test=None)` creates a particle at the rate `mu[h, k]` without
  removing the candidate, at the candidate's activity or at an activity drawn from `outcomes`.
- `DestructionLaw(candidate, field, rate)` removes the candidate at the rate `mu[h, k]`.
- Rates and probabilities are numbers or parameter expressions, scalars or arrays. Only the nonzero probabilities
  enter the lowering, as sparse contractions on NumPy and one fused loop on Numba.
- The three laws need continuous time, a lattice space, and real states, and validation reports a law whose system
  lacks them, with the fix to declare them.
- Validation reports the encounters whose probabilities do not sum to one over the outcomes, with the candidate and
  field activities and the sum, before compilation.
- A system with encounters has no exact Jacobian, so implicit solvers estimate it by finite differences.

```python continuation
from skcomplex.core.interactions import Interaction
from skcomplex.core.laws import EncounterLaw
from skcomplex.core.spaces import ContinuousTime, LatticeSpace
from skcomplex.core.specs import ParameterSpec, RoleSpec
from skcomplex.core.system import System
from skcomplex.engine.compile import validate
from skcomplex.engine.runtime import Simulation


def exchange_table(cooperation, competition):
    table = np.zeros((9, 9, 9))
    for candidate in range(9):
        for field in range(9):
            gap = int(np.sign(field - candidate))
            for outcome, probability in ((candidate + gap, cooperation), (candidate - gap, competition)):
                if gap and 0 <= outcome < 9:
                    table[candidate, field, outcome] += probability
            table[candidate, field, candidate] += 1 - table[candidate, field].sum()
    return table


class ExchangeInteraction(Interaction):
    def __init__(self, label, *, population: Annotated[str, RoleSpec()], eta: Annotated[float, ParameterSpec()] = 1.0):
        super().__init__(label)
        self.population = population
        self.eta = eta

    def _get_laws(self, s):
        return [EncounterLaw(s.population.f, s.population.f, s.eta, exchange_table(0.2, 0.3))]


economy = System(
    elements=[wealth],
    interactions=[ExchangeInteraction('exchange', population='wealth')],
    space=LatticeSpace(9, boundary='fixed'),
    time=ContinuousTime(),
)
densities = Simulation(economy, dt=10.0).run(until=100).trajectory_.values('wealth.f')
print(densities.sum(axis=1)[[0, -1]].round(12).tolist(), densities[-1].round(3).tolist())
print(validate(economy).ok)
```

### External actions

External actions need no primitive of their own.

- A micro-scale action, such as a welfare policy that acts on the transitions of the wealth classes, is an encounter
  whose field role binds a group with fixed states, a group no law changes.
- A macro-scale action, such as a recruitment of security forces proportional to the mean criminality, is a `RateLaw`
  that reads the moments of a distribution.

```python continuation
class PolicyInteraction(Interaction):
    def __init__(
        self,
        label,
        *,
        population: Annotated[str, RoleSpec()],
        government: Annotated[str, RoleSpec()],
        tax: Annotated[float, ParameterSpec()] = 0.2,
    ):
        super().__init__(label)
        self.population = population
        self.government = government
        self.tax = tax

    def _get_laws(self, s):
        rates, table = np.zeros((9, 1)), np.zeros((9, 1, 9))
        for candidate in range(9):
            outcome = candidate + 1 if candidate < 3 else candidate - 1 if candidate >= 6 else candidate
            rates[candidate, 0] = float(outcome != candidate)
            table[candidate, 0, outcome] = 1.0
        return [EncounterLaw(s.population.f, s.government.f, rates * s.tax, table)]


welfare = System(
    elements=[wealth, GroupElement('government', element=WealthDensityElement, size=1, values={'f': 1.0})],
    interactions=[
        ExchangeInteraction('exchange', population='wealth'),
        PolicyInteraction('policy', population='wealth', government='government'),
    ],
    space=LatticeSpace(9, boundary='fixed'),
    time=ContinuousTime(),
)
print(Simulation(welfare, dt=10.0).run(until=100).trajectory_.values('wealth.f')[-1].round(3).tolist())
```

### Moments

`moment(expression, order, coordinates)` is the reduction `sum_j c_j^r f_j` over the members of a group, with the
coordinate `c_j` of every member, such as its activity. The order-0 moment is the density. `normalized=True` divides by
the density, so the normalized order-1 moment is the mean activity, and `central=True` takes the moment about the mean,
so the central order-2 moment is the variance. Moments are expressions of `total` with a shared row of weights, a
vector with one weight per reduced member, so laws read them and observables record them. An observable of a group
receives the namespace of one member, and its function must be defined at module level so that it pickles.

```python continuation
from skcomplex.core.laws.utils import moment
from skcomplex.core.specs import ObservableSpec


def mean_wealth(s):
    return moment(s.f, 1, classes, normalized=True)


def wealth_variance(s):
    return moment(s.f, 2, classes, central=True)


observables = [
    ObservableSpec('mean_wealth', mean_wealth, group='wealth'),
    ObservableSpec('wealth_variance', wealth_variance, group='wealth'),
]
trajectory = Simulation(economy, dt=10.0, observables=observables).run(until=100).trajectory_
variance = trajectory.values('wealth_variance')
print(round(float(trajectory.values('mean_wealth')[-1]), 3), variance[[0, -1]].round(3).tolist())
```

### The particle solver

`KineticSolver(n, dt)` runs the same declarations by direct simulation Monte Carlo. It holds every density as counts
of particles of one mass, with `n` particles for the largest initial density. In every step a candidate particle
encounters with the probability `1 - exp(-lambda_h dt)`, where `lambda_h = sum_k eta[h, k] g_k`, and picks its field
activity and outcome in proportion to `eta[h, k] A[h, k, i] g_k`. A proliferation adds a Poisson number of particles
and a destruction removes a binomial number. The state holds the empirical densities, so trajectories match the ODE
form, and the error against the ODE solution falls as one over the square root of `n`. The solver supports continuous
time, lattice spaces, site groups, and encounter laws only, so validation reports a rate law with it and lists the
solvers that run it.

```python continuation
from skcomplex.engine.solvers import KineticSolver

sampled = Simulation(economy, dt=10.0, solver=KineticSolver(n=20_000), random_state=0).run(until=100)
print(round(float(np.abs(sampled.trajectory_.values('wealth.f')[-1] - densities[-1]).sum()), 2))
print(validate(welfare, solver=KineticSolver()).ok)
```

### Levels

`CoarseGrainAnalysis(over=values)` coarse-grains in state space: it bins the values of a state of a group's members,
from their initial values or from a frame of a trajectory, over given values, every member counting at the nearest
value, so the densities sum to 1 and start a distribution site group. Members drawn from a distribution are values
sampled with NumPy in proportion to the densities, and the group's constructor builds them. A model can then run one
level in a partition of a co-simulation and feed another level its moments.

```python continuation
from skcomplex.analysis import CoarseGrainAnalysis


class TraderElement(Element):
    def __init__(self, label, *, w: Annotated[float, StateSpec()] = 0.0):
        super().__init__(label)
        self.w = w


profile = densities[-1] / densities[-1].sum()
wealths = classes[np.random.default_rng(0).choice(classes.size, size=1000, p=profile)]
traders = GroupElement('traders', element=TraderElement, size=1000, values={'w': wealths}, variable=True)
density = CoarseGrainAnalysis(over=classes).check().prepare(traders.values['w']).run().field_
binned = GroupElement('wealth', element=WealthDensityElement, size=classes.size, on='sites', values={'f': density})
print(round(float(density.sum()), 6), round(float(density @ classes), 3), round(float(traders.values['w'].mean()), 3))
```

The library has `kinetic_wealth_welfare` in `skcomplex.models.economics.econophysics`, `immune_competition` in
`skcomplex.models.biology.oncology`, `kinetic_opinion` in `skcomplex.models.social.opinion_dynamics`, and `evacuation`
and `criminality` in `skcomplex.models.social.crowds`. `evacuation` holds a crowd on a network of rooms as a site group
on a two-dimensional lattice indexed by room and speed class, whose walking encounters read the doors of the building.

### Automata on groups

A cellular automaton needs only a group: its cells are the elements, and a cell at `g` reads the cells at `g s` for
the elements `s` of a finite memory set (Ceccherini-Silberstein and Coornaert 2023). `CayleySpace(group, ball=r)` is the
Cayley graph of a finitely generated group on the ball of radius `r` in its generators and their inverses, and
`CayleySpace(group)` without a ball is the whole of a finite group, such as a finite quotient, where configurations are
periodic. A group space is a space with sites, so `LocalRuleLaw`, `SampleLaw`, and the other lattice laws run on it
with the NumPy and Numba backends, and a run reads only its neighbour table. Preparing the space enumerates its sites,
numbered by word length: `generators_` names the generators, `words_` gives the shortest word of every site,
`ball_sizes_` gives the growth of the balls, and cells outside a ball read the space's `fill`. The memory set is a
`MemoryNeighborhood` of words such as `'a'`, `'a^-1'`, or `'a*b'`, with `'1'` for the cell itself, and
`GeneratorNeighborhood(generators)` gives the identity followed by every generator and its inverse.

The group is a native Sage or GAP group, so a group given by any presentation whose word problem a library solves
works:

- A free group, `libgap.FreeGroup(['a', 'b'])`, reduces its words directly, and a finitely presented group,
  `free / relators`, is solved by Knuth-Bendix completion with KBMAG in GAP. A presentation it cannot complete raises
  `SpaceError`, which asks for a finite quotient.
- An automaton group of the AutomGrp package, such as `libgap.AutomatonGroup('t = (1, t)(1, 2)')` for the adding
  machine, has its elements told apart by their action on a level of the tree, and `libgap.PermGroupOnLevel(group, n)`
  is its finite quotient on level `n`.
- A Sage `PermutationGroup` or `MatrixGroup` compares its elements directly.

GAP comes with the `passagemath-gap` dependency and starts on the first ball that needs it. A ball whose size bound
exceeds `max_sites`, 100,000 by default, raises `SpaceTooLargeError` before GAP is called, and `validate` reports it as
`space-too-large` with the largest radius that fits. A group space lacks the regular geometry of a square lattice and
an image layout, so validation reports a checkerboard schedule as `schedule-unsupported` and an image visual as
`visual-unsupported`. Groups on group spaces stream to Rerun with `CayleyVisual`: nodes on rings by word length,
colored by state, with the edges of the generators logged once.

```python
from typing import Annotated

import numpy as np
from sage.all__sagemath_gap import PermutationGroup, libgap
from sage.sets.finite_enumerated_set import FiniteEnumeratedSet

from skcomplex.core.elements import Element, LatticeElement
from skcomplex.core.laws import LocalRuleLaw
from skcomplex.core.laws.utils import Totalistic
from skcomplex.core.spaces import CayleySpace, GeneratorNeighborhood
from skcomplex.core.specs import StateSpec
from skcomplex.core.system import System
from skcomplex.engine.runtime import Simulation

tree = CayleySpace(libgap.FreeGroup(['a', 'b']), ball=4).check().prepare()
memory = GeneratorNeighborhood(tree.generators_)


class MajorityElement(Element):
    def __init__(self, label, *, alive: Annotated[float, StateSpec(domain=FiniteEnumeratedSet([0, 1]))] = 0):
        super().__init__(label)
        self.alive = alive

    def _get_laws(self, s):
        return [LocalRuleLaw(s.alive, memory, Totalistic({3: 1, 4: 1, 5: 1}))]


cells = (np.random.default_rng(0).random(tree.size_) < 0.5).astype(float)
majority = System(elements=[LatticeElement('cells', element=MajorityElement, space=tree, values={'alive': cells})])
print(tree.ball_sizes_, Simulation(majority, dt=1).run(until=3).trajectory_.values('cells.alive').shape)
print(CayleySpace(PermutationGroup([[(1, 2)], [(1, 2, 3)]])).check().prepare().words_)
```

Named groups are realizations of the automata models: `free_group`, `heisenberg`, `lamplighter`, `baumslag_solitar`,
and `grigorchuk_group` of `skcomplex.models.computer_science.automata`. The Grigorchuk group, which acts on the binary
tree by `a = (1, 1)(0 1)`, `b = (a, c)`, `c = (a, d)`, `d = (1, b)`, is not finitely presented, and its balls grow
faster than any polynomial and slower than any exponential. Its action on level 3 of the tree is a finite quotient.

```python
from sage.all__sagemath_gap import libgap

from skcomplex.core.spaces import CayleySpace
from skcomplex.models.computer_science.automata import grigorchuk_group

ball = CayleySpace(grigorchuk_group(), ball=6).check().prepare()
quotient = CayleySpace(libgap.PermGroupOnLevel(grigorchuk_group(), 3)).check().prepare()
print(ball.ball_sizes_, quotient.size_)
```

**Theory of automata.** `skcomplex.analysis` decides the properties the theory studies on finite universes, where
they are decidable: periodic configurations on a torus or a finite quotient, and patterns on windows. The checks do
not decide them on infinite groups. `Automaton(space, neighborhood, rule, states)` of
`skcomplex.models.computer_science.automata` is the map on configurations of a local rule, with `step` and `run`, and
`AutomatonAnalysis` prepares on it.

- The kinds `'injective'` and `'surjective'`, which agree on a finite universe, enumerate the configurations of small
  universes and solve a SAT encoding with PySAT on larger ones. `'pre-injective'` asks whether two configurations that
  differ only inside a `window` can share their image.
- `'garden-of-eden'` finds a pattern without a preimage, by enumeration or by SAT on random patterns.
- `'inverse'` finds the rule of the inverse of a reversible automaton with the smallest memory up to `max_radius`, and
  `'compose'` gives the automaton of its composition with a `second` one.

```python
from skcomplex.core.laws.utils import OuterTotalistic, Table
from skcomplex.core.spaces import LatticeNeighborhood, LatticeSpace, MooreNeighborhood

from skcomplex.analysis import AutomatonAnalysis
from skcomplex.models.computer_science.automata import Automaton

life = Automaton(LatticeSpace((6, 6)), MooreNeighborhood(), OuterTotalistic({(0, 3): 1, (1, 2): 1, (1, 3): 1}))
print(AutomatonAnalysis('surjective').check().prepare(life).run().surjective_)
print(AutomatonAnalysis('garden-of-eden', random_state=0).check().prepare(life).run().pattern_ is not None)
shift = Automaton(LatticeSpace(8), LatticeNeighborhood(((1,),)), Table({(1,): 1}))
back = AutomatonAnalysis('inverse').check().prepare(shift).run().inverse_
composition = AutomatonAnalysis('compose', second=back).check().prepare(shift).run().composition_
print(composition.step([1, 0, 1, 1, 0, 0, 0, 1]).tolist())
```

**Linear automata and subshifts.** `LinearAutomaton(space, neighborhood, coefficients, field)` is the element
`sum_s a_s s` of the group ring over `'GF(p)'` or `'QQ'`. On a finite universe it is a matrix whose rank decides
injectivity, surjectivity, and pre-injectivity together, and on `Z^d`, with `infinite=True`, its Laurent polynomial
decides them: it is invertible when the polynomial is a monomial and surjective when it is not zero.
`Subshift(neighborhood, forbidden)` holds the configurations without forbidden patterns: it checks configurations,
counts the admissible ones of a window, and samples them by a Markov chain of single-cell moves. Both are models of
`skcomplex.models.computer_science.automata`, and `AutomatonAnalysis` tells the three models apart by their methods.
`AutomatonAnalysis('entropy', windows=windows)` gives `log |X_F| / |F|` along windows `F`, and
`AutomatonAnalysis('difference')` the cells where two runs differ after one cell changes.
`EntropyAnalysis('rate')` of the frames of a run as columns gives the spatial entropy rate of every frame, which never
increases under an automaton.

```python
from skcomplex.core.spaces import LatticeNeighborhood, LatticeSpace

from skcomplex.analysis import AutomatonAnalysis
from skcomplex.models.computer_science.automata import LinearAutomaton, Subshift

laplacian = LinearAutomaton(LatticeSpace(5), LatticeNeighborhood(((-1,), (0,), (1,))), (1, -2, 1), 'GF(2)')
golden = Subshift(LatticeNeighborhood(((0,), (1,))), [(1, 1)])
print(AutomatonAnalysis('injective').check().prepare(laplacian).run().injective_)
print(AutomatonAnalysis('surjective', infinite=True).check().prepare(laplacian).run().surjective_)
print([golden.count(LatticeSpace(length, boundary='fixed')) for length in range(1, 8)])
windows = [LatticeSpace(24, boundary='fixed')]
print(AutomatonAnalysis('entropy', windows=windows).check().prepare(golden).run().entropy_.round(3))
```

The library has `free_group_automaton`, `lamplighter_automaton`, `heisenberg_automaton`, `baumslag_solitar_automaton`,
`grigorchuk_automaton`, the Laplacian `linear_ring_automaton`, and the golden mean shift `subshifts` in
`skcomplex.models.computer_science.automata`.

## Fields

A field holds one value per site of a lattice or per node of a network. `FieldElement(label, element=..., space=...,
boundary=..., values=...)` places a member of an element type on every site, and the laws of the type, written for one
site, read the neighbouring sites through differential operators. The method of lines turns a field into ordinary
differential equations, so every ODE solver and backend runs it.

```python
from typing import Annotated

import numpy as np

from skcomplex.core.elements import Element, FieldElement
from skcomplex.core.laws import RateLaw
from skcomplex.core.laws.utils import laplacian
from skcomplex.core.spaces import LatticeSpace
from skcomplex.core.specs import ParameterSpec, StateSpec
from skcomplex.core.system import System
from skcomplex.engine.runtime import Simulation


class HeatElement(Element):
    def __init__(
        self,
        label,
        *,
        u: Annotated[float, StateSpec()] = 0.0,
        D: Annotated[float, ParameterSpec()] = 1.0,
    ):
        super().__init__(label)
        self.u = u
        self.D = D

    def _get_laws(self, s):
        return [RateLaw(s.u, s.D * laplacian(s.u))]


initial = np.zeros(400)
initial[200] = 20.0
space = LatticeSpace(400, spacing=0.05)
rod = FieldElement('rod', element=HeatElement, space=space, boundary={'u': 'zero-flux'}, values={'u': initial})
trajectory = Simulation(System(elements=[rod]), dt=0.1).run(until=0.5).trajectory_
heat = trajectory.values('rod.u')
print(round(float(heat[-1].max()), 3), heat.sum(axis=1)[[0, -1]].round(6).tolist())
```

- `laplacian(u)` lowers to the `2d + 1`-point stencil divided by the squared spacing, and to the graph Laplacian on a
  `NetworkSpace`. `gradient(u, n_dim)` and `divergence(vector)` use central differences, and `advection(u, velocity)`
  upwind differences. Operators apply to any expression of the field's states and parameters, and nest.
- `boundary={'u': 'periodic' | 'zero-flux' | ('fixed', value)}` sets each state's edges. Zero flux mirrors the field
  and conserves its total, and fixed edges read the given value. A lattice's own `boundary` is the default.
- The equations keep the operators, `diff(u(t), t) == D*laplacian(u(t))`, while the backends evaluate the stencils,
  fused into one compiled loop on Numba for lattices.
- `problem.stable_step` is the diffusive limit `h^2 / (2 d D)` of explicit steps, and fixed-step solvers whose `dt`
  exceeds it warn before the run with the limit.

Agents on a grid of the same shape couple to a field in co-simulation. A rule reads the field at a cell with
`model.field('ground/pheromone.level', cell)` and adds to it with `model.deposit(name, cell, amount)`, and the
deposits reach the field at the next synchronisation point. Lattice fields stream to Rerun as images. The library has
`heat_diffusion`, `fisher_kpp`, `gray_scott`, `brusselator`, `fitzhugh_nagumo`, `schnakenberg`, and
`pheromone_foraging`, such as `gray_scott` in `skcomplex.models.chemistry.pattern_formation` and `heat_diffusion` in
`skcomplex.models.physics.heat`.

### Discretizations and stiff solvers

The same laws of a field run with another discretization by changing `FieldElement(..., discretization=...)`.

- `'spectral'` fields on periodic lattices multiply Fourier modes, with the two-thirds rule removing aliasing from
  every operator, so `laplacian(u)` is exact for smooth fields. `inverse_laplacian(u)` solves Poisson equations, and
  `symmetry={'u': (None, 'odd')}` keeps a state odd or even along an axis, which represents walls on a doubled
  lattice. `LatticeSpace(spacing=(dx, dy))` sets a spacing per axis.
- `'finite-volume'` fields take `FluxLaw(state, flux)`s, `du/dt + divergence(flux) = 0`, updated with MUSCL
  reconstruction, minmod or van Leer limiters, and Rusanov or exact Godunov fluxes. `SSPRKSolver` steps them by the CFL
  condition and by the stability limit of their Laplacian rates at the current parameters, and conserves their totals.
- `'finite-element'` fields live on a `MeshSpace` of the points and cells of a scikit-fem mesh, such as
  `MeshSpace(grid.p, grid.t)` for `grid = skfem.MeshTri.init_lshaped().refined(3)`, with stiffness and mass matrices
  from scikit-fem and boundary values that can be functions of the coordinates. They stream to Rerun as colored meshes.

The compiler splits the rate of every field into a linear part with constant coefficients, such as `D * laplacian(u)` or
`-laplacian(laplacian(u))`, and a nonlinear remainder. `IMEXEulerSolver`, `IMEX2Solver`, and `CrankNicolsonSolver` treat
the linear part implicitly, with sparse solves for finite differences and finite elements and solves per Fourier mode
for spectral fields. `ETDRK4Solver` integrates it exactly per mode, coupled states included, so stiff equations take
steps far beyond the explicit limit.

```python
from typing import Annotated

import numpy as np

from skcomplex.core.elements import Element, FieldElement
from skcomplex.core.laws import RateLaw
from skcomplex.core.laws.utils import divergence, laplacian
from skcomplex.core.spaces import LatticeSpace
from skcomplex.core.specs import StateSpec
from skcomplex.core.system import System
from skcomplex.engine.runtime import Simulation
from skcomplex.engine.solvers import ETDRK4Solver


class ChaosElement(Element):
    def __init__(self, label, *, u: Annotated[float, StateSpec()] = 0.0):
        super().__init__(label)
        self.u = u

    def _get_laws(self, s):
        return [RateLaw(s.u, -divergence([s.u**2 / 2]) - laplacian(s.u) - laplacian(laplacian(s.u)))]


positions = np.arange(1, 129) * 32 * np.pi / 128
line = FieldElement(
    'line',
    element=ChaosElement,
    space=LatticeSpace(128, spacing=32 * np.pi / 128),
    values={'u': np.cos(positions / 16) * (1 + np.sin(positions / 16))},
    discretization='spectral',
)
state = Simulation(System(elements=[line]), dt=10, solver=ETDRK4Solver(dt=0.25)).run(until=100).state_
print(round(float(np.abs(state).max()), 1))
```

The implicit methods of `ScipyODESolver` are the stiff path for finite-difference and finite element fields, with
adaptive error control that the fixed-step split steppers lack. The compiler derives the sparsity pattern of the
Jacobian from the laws: member and shared reads, the neighbour tables of stencils, the sparse matrices of finite
elements, the reconstruction stencils of finite volumes, and dense blocks where couplings are known only at run time,
such as pairs, spectral operators, and callables. `ScipyODESolver(method='BDF')` and `ScipyODESolver(method='Radau')`
receive the pattern, so SciPy estimates the Jacobian with one drift evaluation per group of independent columns and
factors it with sparse LU, and `LSODA` receives the half bandwidths of a pattern narrower than a tenth of the size. The
option `jacobian` selects the path: `'auto'` by default, `'exact'` for the sparse Jacobian of the laws differentiated by
Sage, which laws without a symbolic form cannot give, and `'dense'` for the dense estimate. Explicit methods ignore it,
and `RK45` warns when one call spans more than 1000 stable steps of the diffusion of a field.

```python
from skcomplex.engine.runtime import Simulation

from skcomplex.engine.solvers import ScipyODESolver
from skcomplex.models.biology.ecology import fisher_kpp

stiff = fisher_kpp(400, D=1.0, spacing=0.05)
simulation = Simulation(stiff, dt=1.0, solver=ScipyODESolver(method='BDF')).run(until=5)
print(simulation.stepper_.jacobian_pattern_.nnz, simulation.state_.max().round(3))
```

On Fisher-KPP on 4000 points, whose largest eigenvalue is about 1600, run to `t = 10`, BDF with the dense estimate took
35 to 50 s, against about 0.2 s with the pattern, 0.3 s with the exact Jacobian, 0.7 s for Radau with the pattern, and
0.8 s for `CrankNicolsonSolver(dt=0.01)`. Spectral fields keep `ETDRK4Solver` and the IMEX steppers, which win there
because their linear part is diagonal in Fourier space while its pattern is dense.

`skcomplex.models.physics.fluids.incompressible_flow(shape, size, nu=..., forcing=..., buoyancy=...)` builds
two-dimensional incompressible flow in vorticity-streamfunction form, with Boussinesq convection when `buoyancy` gives
`Ra` and `Pr`. The library has `kuramoto_sivashinsky`, `complex_ginzburg_landau`, `swift_hohenberg`, and `heat_l_shape`
in the subfields `fluids`, `pattern_formation`, and `heat` of `skcomplex.models.physics`, and `burgers`,
`shallow_water`, `decaying_turbulence`, and `rayleigh_benard` in `skcomplex.models.physics.fluids`.

## Backends and speed

Compiling a system produces numeric functions with a backend. `NumpyBackend` evaluates every law with NumPy arrays,
and `NumbaBackend` generates one compiled function per form, with compiled cell lists for pair interactions and
compiled loops for `RK4Solver`, `EulerMaruyamaSolver`, `GillespieSolver`, and `IterationSolver`. Without a backend,
`Simulation` picks Numba when the groups of a system hold at least 1000 members in total, because compiling takes a
second or two that small systems do not win back, and `problem_.backend` names the backend it used.

```python
from skcomplex.engine.backends import NumbaBackend
from skcomplex.engine.runtime import Simulation
from skcomplex.engine.solvers import RK4Solver
from skcomplex.models.physics.statistical_mechanics import ideal_gas

simulation = Simulation(ideal_gas(50_000, random_state=0), solver=RK4Solver(dt=0.01), backend=NumbaBackend(), dt=0.01)
print(simulation.run(until=0.5).problem_.backend)
```

Compiled kernels are cached on disk, under `~/.cache/skcomplex` or the `cache_dir` of `NumbaBackend(cache_dir=...)`,
so running the same system again skips compilation.

A law value can be a function called once per member with `Call(function, arguments)` of
`skcomplex.core.laws.utils`. A Numba-compiled function
runs inside the compiled kernels, and the NumPy backend calls it from Python.

```python
from typing import Annotated

import numba

from skcomplex.core.elements import GroupElement
from skcomplex.core.interactions import Interaction
from skcomplex.core.laws import RateLaw
from skcomplex.core.laws.utils import Call
from skcomplex.core.specs import ParameterSpec, RoleSpec
from skcomplex.core.system import System
from skcomplex.engine.runtime import Simulation
from skcomplex.models.biology.ecology import PopulationElement


@numba.njit
def saturating(t, n, r):
    return r * n / (1 + n)


class SaturationInteraction(Interaction):
    def __init__(self, label, *, target: Annotated[str, RoleSpec()], r: Annotated[float, ParameterSpec()] = 1.0):
        super().__init__(label)
        self.target = target
        self.r = r

    def _get_laws(self, s):
        return [RateLaw(s.target.n, Call(saturating, (s.target.n, s.r)))]


herds = System(
    elements=[GroupElement('herds', element=PopulationElement, size=2000, values={'n': 1.0})],
    interactions=[SaturationInteraction('growth', target='herds')],
)
print(round(float(Simulation(herds, dt=0.1).run(until=1).trajectory_.values('herds.n')[-1, 0]), 4))
```

A plain Python callable of the values, as in `RateLaw(s.target.n, lambda t, values: ...)`, also works with the Numba
backend, but runs in Python and turns off the compiled stepper loops for that kind of law.

## External backends

An external backend runs a whole system in a third-party solver library, and the simulation keeps its frames,
trajectories, live parameters, and Rerun visualization. `LammpsSolver` runs particle systems in LAMMPS, which
integrates them with velocity Verlet. Its `pairs` map pair potential classes to native LAMMPS pair styles and its
`fixes` map interaction classes to native fixes, and the particle realizations of physics give both as `LAMMPS_PAIRS`
and `LAMMPS_FIXES`.

```python
import numpy as np

from skcomplex.core.elements import GroupElement, ParticleElement
from skcomplex.core.spaces import ContinuousSpace
from skcomplex.core.system import System
from skcomplex.engine.runtime import Simulation
from skcomplex.engine.solvers import LammpsSolver
from skcomplex.models.physics.common.particles import (
    LAMMPS_FIXES,
    LAMMPS_PAIRS,
    LangevinThermostatInteraction,
    LennardJonesInteraction,
)

side = np.arange(10) * 1.2
grid = np.stack(np.meshgrid(side, side), axis=-1).reshape(-1, 2)
atoms = System(
    elements=[GroupElement('atoms', element=ParticleElement, size=100, values={'x': grid})],
    interactions=[
        LennardJonesInteraction('lj', first='atoms', second='atoms', cutoff=2.5),
        LangevinThermostatInteraction('bath', target='atoms', temperature=0.5, gamma=1.0),
    ],
    space=ContinuousSpace(n_dim=2),
)
solver = LammpsSolver(dt=0.005, pairs=LAMMPS_PAIRS, fixes=LAMMPS_FIXES)
trajectory = Simulation(atoms, solver=solver, dt=0.05, random_state=0).run(until=1).trajectory_
print(trajectory.values('atoms.x').shape)
```

Before the first frame the backend checks the system and raises one error that lists every element, interaction, and
law it cannot run. The translation is the following.

| Framework | LAMMPS |
| --- | --- |
| `ParticleElement` and particle groups | one atom type per element or group, with one mass per group |
| two or three dimensions | `dimension 2` with `fix enforce2d`, or `dimension 3` |
| `ContinuousSpace(extent, boundary='reflecting')` | `fix wall/reflect` at `0` and the extent |
| `ContinuousSpace(extent, boundary='periodic')` | periodic boundaries of the extent, shared by every group |
| a class in `pairs`, such as `LennardJonesInteraction` | its native pair style, such as `pair_style lj/cut` |
| other `PairPotentialInteraction` | `pair_style lepton` with the pair energy printed as a Lepton expression |
| a class in `fixes` | its native fix, such as `fix addforce`, `fix viscous`, or `fix langevin` |
| any other interaction that only changes velocities | `fix external`, evaluated by the framework every step |

`LangevinThermostatInteraction` becomes `fix langevin`, seeded from the simulation's random state.

## Agent-based models

Agents are ordinary elements whose laws are rules written as code. An `Element` declares the states every agent holds
and the parameters its group shares as annotated arguments of its `__init__`, and returns from `_get_laws` `RuleLaw`s
whose functions take the agent and the model. An element or a group whose laws include rules is an agent, and so is
every member of a variable group. In a system with rules, a group whose element type has no laws is a group of passive
agents that rules read and change, unless an interaction whose laws are not rules binds it. A system with rules is
agent-based, runs in discrete time, and runs on `MesaSolver`, one Mesa step per unit of time, with every agent
activated in random order. `MesaSolver` declares what it runs like any solver, so
`validate(system, solver=MesaSolver())` reports the laws, spaces, and times it does not support.

```python
from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from skcomplex.core.elements import Element, GroupElement
from skcomplex.core.laws import RuleLaw
from skcomplex.core.spaces import LatticeSpace
from skcomplex.core.specs import ParameterSpec, StateSpec, VisualSpec
from skcomplex.core.system import System
from skcomplex.engine.runtime import Simulation


class ForagerElement(Element):
    def __init__(
        self,
        label,
        visual=VisualSpec('agents', {'color_by': 'food'}),
        *,
        food: Annotated[float, StateSpec()] = 5.0,
        hunger: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
    ):
        super().__init__(label, visual=visual)
        self.food = food
        self.hunger = hunger

    def _get_laws(self, s):
        return [RuleLaw(self.forage)]

    def forage(self, agent, model):
        cells = agent.neighbor_cells()
        agent.move_to(cells[int(model.rng.integers(len(cells)))])
        agent.food += model.rng.poisson(0.8) - agent.hunger
        if agent.food <= 0:
            agent.remove()
        elif agent.food > 10:
            agent.food /= 2
            agent.create(food=agent.food)


foragers = System(
    elements=[GroupElement('foragers', element=ForagerElement, size=50, variable=True)],
    space=LatticeSpace((30, 30), boundary='periodic'),
)
trajectory = Simulation(foragers, random_state=0).run(until=20).trajectory_
print(int(trajectory.values('foragers.count')[-1]), len(trajectory.agents('foragers', 20).ids))
```

- An agent reads and sets its states as attributes and reads its group's parameters, which change live like any
  parameter. `neighbors`, `neighbor_cells`, `agents_here`, `move_to`, `create`, and `remove` act on its place and its
  population. Only the members of a group declared `variable=True` are created and removed during a run.
- `GroupElement(label, element=..., size=..., location=...)` places the agents at cell coordinates, network nodes, or
  positions, one per agent or one shared, and agents without a location take random places.
- The model gives the random generator `model.rng`, the other agents with `model.agents(name)` and
  `model.count(name)`, and empty cells.
- Every rule receives the model view scoped to the interaction or agent group that owns it. `model.parameter('name')`
  reads the owner's parameter, and `model.agents('role')` reads the group bound to a role of the owner, then a group by
  its path relative to the owner's system, then by its path from the root, so a rule works unchanged in a subsystem.
  Interactions with agent rules bind any number of agent groups, of any sizes, through their roles.
- `model.dimensions` gives the extent of the space per axis and `model.torus` whether it wraps.
  `agent.displacement_to(other)` gives the vector to another agent or a location, the shortest one on a torus, and
  `agent.distance_to(other)` its length, or the number of hops on a network. `agent.group` is the path of the agent's
  group and `agent.group_label` its label.
- `LatticeSpace` gives agents Moore or von Neumann neighbourhoods and an optional cell capacity, `NetworkSpace` places
  agents on the nodes of a NetworkX graph, and `ContinuousSpace` finds neighbours within a radius.
- Frames and trajectories hold every group's agents with their ids, states, and locations, through
  `trajectory.agents(group, frame)`, and the count of every group as the reporter `<group>.count`.
- `skcomplex.models.social` has Schelling segregation and the voter model, `skcomplex.models.economics.econophysics`
  has Boltzmann wealth exchange, and `skcomplex.models.biology.ecology` has the wolf and sheep model.

```python
from typing import Annotated

from skcomplex.core.elements import Element, GroupElement
from skcomplex.core.interactions import Interaction
from skcomplex.core.laws import RuleLaw
from skcomplex.core.spaces import ContinuousSpace
from skcomplex.core.specs import ParameterSpec, RoleSpec
from skcomplex.core.system import System
from skcomplex.engine.runtime import Simulation


class WalkerElement(Element):
    def _get_laws(self, s):
        return [RuleLaw(self.step)]

    def step(self, agent, model):
        width, height = model.dimensions
        x, y = agent.location
        agent.move_to(((x + model.rng.uniform(0, 2)) % width, y))


class SpreadInteraction(Interaction):
    def __init__(self, label, *, walkers: Annotated[str, RoleSpec()], reach: Annotated[float, ParameterSpec()] = 10.0):
        super().__init__(label)
        self.walkers = walkers
        self.reach = reach

    def _get_laws(self, s):
        return [RuleLaw(self.measure)]

    def measure(self, model):
        walkers = model.agents('walkers')
        close = [a for a in walkers for b in walkers if a != b and a.distance_to(b) < model.parameter('reach')]
        counts.append(len(close) // 2)


counts = []
places = [(10.0 * index, 0.5) for index in range(5)]
road = System(
    systems=[
        System(
            'road',
            elements=[GroupElement('walkers', element=WalkerElement, size=5, location=places)],
            interactions=[SpreadInteraction('spread', walkers='walkers')],
        ),
    ],
    space=ContinuousSpace(extent=(50.0, 1.0), boundary='periodic'),
)
Simulation(road, random_state=0).run(until=5)
print(counts)
```

### Phases, markets, institutions, and learning

Economic and social models order each step into phases, match agents in markets, let institutions set policy, and let
agents learn from their past.

- `DiscreteTime(schedule=PhasesScheduleSpec([...]))` lists the phases of a step. A `PhaseSpec(name, activation, repeat)`
  runs its agents `'random'`ly, `'sequential'`ly in order of creation, `'simultaneous'`ly, or `'by-type'`, for `repeat`
  rounds such as intraday rounds. In a simultaneous phase every rule reads the states of the start of the phase, and the
  writes apply at its end. `RuleLaw(function, phase=name)` puts a rule in a phase. Rules without a phase run in the
  phase `'default'`, which runs first unless the system lists it, so systems without phases keep one random activation
  per step.
- `MarketInteraction(label, buyers=..., sellers=..., protocol=...)`, from
  `skcomplex.models.economics.common.markets`, matches two groups. Buyers state their limit `bid` and
  `demand`, and sellers their `ask` and `supply`. The protocols are `BilateralSearch`, `SampleAndSwitch` (partner
  choice with switching), `RationedQueue`, `Walrasian` (tâtonnement), `OrderBook` (price-time priority), `CallAuction`
  (one uniform price), and `DeferredAcceptance` (stable matching by the `matching` library). Every trade moves
  `money` to the seller and `goods` to the buyer at once, within the buyer's budget. The market reports
  `<market>.price`, `<market>.volume`, and `<market>.unmatched`.
- An `InstitutionInteraction`, from `skcomplex.models.economics.common.institutions`, acts once per round of its
  phase with its method `act`. It reads `model.observable(name)`, reporters through `model.history(name, window)`, and
  agents, and it sets parameters of any element or interaction with `model.set_parameter(name, value)`. Rules see the
  change at once, and the trajectory records it. A subclass that declares its own parameters takes them as annotated
  arguments of its `__init__` next to `rule` and `phase`.
- Elements and interactions declare the values they publish in the class variable `reports` and publish them with
  `model.report(name, value)`. `model.storage` keeps private values of an element or an interaction across the steps of
  a run.
- Learning blocks: `agent.lags(state, window)`, a method of the agent view a rule receives, returns the past values of a
  state declared with lags, `StateSpec(lags=m)`. In `skcomplex.models.economics.common.learning`, `heuristics` has
  adaptive, trend-following, anchoring, and fundamentalist forecasts, `logit_choice` picks a strategy by past
  performance, and `evolve` breeds the next generation of strategies with the model's random stream. `imitate`, in
  `skcomplex.models.social.cooperation`, copies the best neighbour.

```python
from typing import Annotated, ClassVar

import numpy as np

from skcomplex.core.elements import Element, GroupElement
from skcomplex.core.laws import RuleLaw
from skcomplex.core.spaces import DiscreteTime
from skcomplex.core.specs import ParameterSpec, PhaseSpec, PhasesScheduleSpec, StateSpec
from skcomplex.core.system import System
from skcomplex.engine.runtime import Simulation
from skcomplex.models.economics.common.institutions import InstitutionInteraction
from skcomplex.models.economics.common.markets import CallAuction, MarketInteraction


class TraderElement(Element):
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

    def quote(self, agent, model):
        agent.bid = model.rng.uniform(0, agent.value) if agent.demand else 0.0
        agent.ask = model.rng.uniform(agent.value, 2.0) if agent.supply else 2.0


class ClerkInteraction(InstitutionInteraction):
    reports: ClassVar[tuple[str, ...]] = ('traded',)

    def __init__(self, label, *, rule=None, phase=None, fee: Annotated[float, ParameterSpec()] = 0.0):
        super().__init__(label, rule=rule, phase=phase)
        self.fee = fee

    def act(self, model):
        traded = sum(agent.goods for agent in model.agents('buyers'))
        model.report('traded', traded)
        model.set_parameter('fee', 0.01 * traded)


rng = np.random.default_rng(0)
buyers = {'value': rng.uniform(0.5, 2, 20), 'demand': 1.0, 'money': 2.0}
sellers = {'value': rng.uniform(0, 1.5, 20), 'supply': 1.0}
auction = System(
    elements=[
        GroupElement('buyers', element=TraderElement, size=20, values=buyers),
        GroupElement('sellers', element=TraderElement, size=20, values=sellers),
    ],
    interactions=[
        MarketInteraction('auction', buyers='buyers', sellers='sellers', protocol=CallAuction(), phase='trade'),
        ClerkInteraction('clerk', phase='trade'),
    ],
    time=DiscreteTime(schedule=PhasesScheduleSpec([PhaseSpec('quote'), PhaseSpec('trade')])),
)
trajectory = Simulation(auction, random_state=0).run(until=5).trajectory_
prices = [round(float(price), 3) for price in trajectory.values('auction.price')[1:]]
print(prices, float(trajectory.values('clerk.traded')[-1]), round(float(trajectory.values('clerk.fee')[-1]), 3))
print(trajectory.values('auction.volume').tolist())
```

### Porting Mesa models

A Mesa model is ported to a system: its agent class becomes an element whose step is a `RuleLaw`, its agent attributes
become states, and its model attributes become parameters, which then change live, are validated, and are recorded
like those of any system. Mesa still runs the steps, as the backend of agent systems.

```python
from typing import Annotated

from skcomplex.core.elements import Element, GroupElement
from skcomplex.core.laws import RuleLaw
from skcomplex.core.spaces import DiscreteTime
from skcomplex.core.specs import ParameterSpec, StateSpec
from skcomplex.core.system import System
from skcomplex.engine.runtime import Simulation


class CoinElement(Element):
    def __init__(
        self,
        label,
        *,
        heads: Annotated[float, StateSpec()] = 0.0,
        bias: Annotated[float, ParameterSpec()] = 0.5,
    ):
        super().__init__(label)
        self.heads = heads
        self.bias = bias

    def _get_laws(self, s):
        return [RuleLaw(self.flip)]

    def flip(self, agent, model):
        agent.heads += float(model.rng.random() < agent.bias)


coins = System('coins', elements=[GroupElement('coins', element=CoinElement, size=10)], time=DiscreteTime())
print(int(Simulation(coins, random_state=0).run(until=10).trajectory_.agents('coins', 10).values['heads'].sum()))
```

## Rewriting systems

A rewriting system grows by replacing its parts in parallel: a `WordElement` by string productions in the style of
L-systems, a `PatchElement` of tiles by affine substitutions, and a `HostGraphElement` by double-pushout graph
productions. It
compiles to the rewriting form, every step is one generation, and `run(until=n)` records the generations `0` to `n`.
The structure lives next to the state vector, whose states are the count of every symbol, prototile, or node label
that is a word, and the size `<element>.size`. `Trajectory.structure(path, frame)` returns the recorded structure,
which later generations do not change.

### Words and productions

A word, a patch, or a host graph rewrites itself by the rewriting laws it returns from `_get_laws`: a
`ProductionLaw` for every rule of a word, and every generation applies all the productions of an element at once.
Strings are words of one-character symbols, and `ModuleSpec` gives a symbol parameters.

```python
from skcomplex.core.elements import WordElement
from skcomplex.core.laws import ProductionLaw
from skcomplex.core.system import System
from skcomplex.engine.runtime import Simulation


class AlgaeElement(WordElement):
    def _get_laws(self, s):
        return [ProductionLaw('A', 'AB'), ProductionLaw('B', 'A')]


algae = System(elements=[AlgaeElement('algae', axiom='A')])
trajectory = Simulation(algae).run(until=5).trajectory_
print(trajectory.values('algae.size').tolist(), trajectory.structure('algae', 5).text)
```

A production can be stochastic, with probabilities that sum to one per predecessor and context and draws from the
simulation's random stream. It can depend on a left or a right context, which skips the word's `ignore` symbols and
treats bracketed branches as Prusinkiewicz and Lindenmayer (1990) do, and it then takes precedence over context-free
productions. A parametric production binds the predecessor's parameters to symbols, with a condition and successor
parameters written as expressions of them and of the parameters of the rewritten element, which `schedule` and
`update_parameters` change between generations. A word element with parameters declares them as annotated arguments of
its `__init__`.

```python
from typing import Annotated

from sage.all__sagemath_symbolics import var

from skcomplex.core.elements import WordElement
from skcomplex.core.laws import ProductionLaw
from skcomplex.core.specs import ModuleSpec, ParameterSpec
from skcomplex.core.system import System
from skcomplex.engine.runtime import Simulation

x = var('x')


class ShrinkingElement(WordElement):
    def __init__(self, label, axiom=(), *, alphabet=None, ratio: Annotated[float, ParameterSpec()] = 0.5):
        super().__init__(label, axiom, alphabet=alphabet)
        self.ratio = ratio

    def _get_laws(self, s):
        successor = [ModuleSpec('F', (x,)), ModuleSpec('A', (s.ratio * x,))]
        return [ProductionLaw(ModuleSpec('A', (x,)), successor, condition=x > 0.1)]


plant = System(elements=[ShrinkingElement('plant', axiom=[ModuleSpec('A', (1.0,))], alphabet={'A': 1, 'F': 1})])
print(Simulation(plant).run(until=5).trajectory_.structure('plant', 5).text)
```

A generation that would make a word longer than `max_size` raises `RewritingError` before allocating it, and the
frames before it stay recorded.

### Word geometries, patches, and measures

A `GeometrySpec`, from `skcomplex.core.specs`, draws a word as line segments in two or three dimensions, as the turtle
interpretation of L-systems does: `F` draws forward, `f` moves, `+` and `-` turn, `[` and `]` push and pop a branch, and
`contraction` scales the step at every generation. A `PatchElement` holds tiles, each a `PrototileSpec` segment or
polygon under an affine placement, and a `SubstitutionLaw` replaces every tile of a prototile by pieces placed by
`AffineSpec` maps relative to it. Structures with a geometry report `<element>.length` and `<element>.area` at every
frame, and give `bounds()` and `raster(resolution)`, a boolean grid that `ScalingAnalysis('box-counting')` reads.

```python
from skcomplex.analysis import ScalingAnalysis
from skcomplex.core.elements import PatchElement, WordElement
from skcomplex.core.laws import ProductionLaw, SubstitutionLaw
from skcomplex.core.specs import AffineSpec, GeometrySpec, PrototileSpec
from skcomplex.core.system import System
from skcomplex.engine.runtime import Simulation


class KochElement(WordElement):
    def _get_laws(self, s):
        return [ProductionLaw('F', 'F+F--F+F')]


koch = System(elements=[KochElement('koch', axiom='F', geometry=GeometrySpec(angle=60.0, contraction=1 / 3))])
curve = Simulation(koch).run(until=4).trajectory_.structure('koch', 4)
print(len(curve.segments()), round(curve.length(), 6))

interval = PrototileSpec('interval', [(0, 0), (1, 0)])
thirds = [('interval', AffineSpec(scale=1 / 3)), ('interval', AffineSpec(scale=1 / 3, translation=(2 / 3, 0)))]


class CantorElement(PatchElement):
    def _get_laws(self, s):
        return [SubstitutionLaw('interval', thirds)]


cantor = System(elements=[CantorElement('cantor', prototiles=[interval])])
trajectory = Simulation(cantor).run(until=5).trajectory_
print(trajectory.values('cantor.size')[-1], round(trajectory.values('cantor.length')[-1], 6))
grid = trajectory.structure('cantor', 5).raster(729)
print(round(ScalingAnalysis('box-counting', scales=[3, 9, 27]).check().prepare(grid).run().exponent_, 2))
```

The `geometry` visual logs segments and outlines as line strips, a patch with `filled=True` as an image of its raster,
and a host graph as graph nodes and edges, in the bounds the element declares or that its first tiles give.

### Graph rewriting

A `GraphProductionLaw` gives a left graph, an interface, and a right graph that share nodes by name, and matches by
their `label` attributes. Every generation applies it at a maximal set of parallel independent matches, chosen in an
order drawn from the random stream, and skips matches whose deleted nodes would leave dangling edges.

```python
import networkx as nx

from skcomplex.core.elements import HostGraphElement
from skcomplex.core.laws import GraphProductionLaw
from skcomplex.core.system import System
from skcomplex.engine.runtime import Simulation

interface = nx.Graph()
interface.add_nodes_from('ab')
subdivide = GraphProductionLaw(nx.Graph([('a', 'b')]), interface, nx.Graph([('a', 'c'), ('c', 'b')]))


class RingElement(HostGraphElement):
    def _get_laws(self, s):
        return [subdivide]


ring = System(elements=[RingElement('ring', graph=nx.cycle_graph(4))])
print(Simulation(ring, random_state=0).run(until=3).trajectory_.values('ring.size').tolist())
```

## Quantum systems

A quantum system is a set of quantum subsystems whose state is one vector of their joint Hilbert space. Every
subsystem holds its state `psi` in the Hilbert space of a `HilbertSpec`, which names its operators. The realizations of
`skcomplex.models.physics.common.quantum` hold the common subsystems: `SpinElement` has the components `x`, `y`,
`z` and the ladder operators `plus` and `minus` of spin `j`, `QubitElement` has the Pauli matrices, `ModeElement` has
`annihilate`, `create`, and `number` truncated at `quanta`, and `SiteElement` is a particle on the sites of a lattice
or a network with its `laplacian` and its `adjacency`. A group of subsystems, such as a `LatticeElement` of spins, is
the tensor product of its members' spaces in member order, and the initial state of every subsystem is a basis label,
an index, or a vector.

### Operators and laws

Laws name the operators of a subsystem with `local_operator(s.psi, name)`, from `skcomplex.core.laws.utils`, which
combine by sums, numbers, and symbolic coefficients of parameters and time, and `operator_product` and `dagger` keep
the order of products and adjoints. `GeneratorLaw(operator)` adds to the Hamiltonian, with `ħ = 1`, and
`DissipatorLaw(operator, rate)` adds a Lindblad dissipator, which makes the system open. A `NeighborInteraction` is
written once for a bond, with `s.source` and `s.target`, and acts on every pair of neighbours of the space of the
group its one role binds. Compiling checks that the Hamiltonian is Hermitian and that the dimension stays within the
`max_dimension` of the quantum solver. A whole-system initial state, such as an entangled vector or `'ground'`, is
given to the simulation as `initial_state={'<system label>.state': value}`.

```python
from typing import Annotated

import numpy as np

from skcomplex.core.elements import LatticeElement
from skcomplex.core.interactions import NeighborInteraction
from skcomplex.core.laws import GeneratorLaw
from skcomplex.core.laws.utils import local_operator
from skcomplex.core.spaces import LatticeSpace
from skcomplex.core.specs import ObservableSpec, ParameterSpec, RoleSpec
from skcomplex.core.system import System
from skcomplex.engine.runtime import Simulation
from skcomplex.models.physics.common.quantum import SpinElement


class CouplingInteraction(NeighborInteraction):
    def __init__(self, label, *, target: Annotated[str, RoleSpec()], J: Annotated[float, ParameterSpec()] = 1.0):
        super().__init__(label)
        self.target = target
        self.J = J

    def _get_laws(self, s):
        a, b = s.source.psi, s.target.psi
        exchange = sum(local_operator(a, name) * local_operator(b, name) for name in ('x', 'y', 'z'))
        return [GeneratorLaw(s.J * exchange)]


def sz(member):
    return local_operator(member.psi, 'z')


def sites(values):
    return values


chain = System(
    elements=[LatticeElement('chain', element=SpinElement, space=LatticeSpace(6), values={'psi': ['up', 'down'] * 3})],
    interactions=[CouplingInteraction('coupling', target='chain')],
)
observables = [ObservableSpec('sz', sz, group='chain', reduce=sites)]
trajectory = Simulation(chain, dt=0.5, observables=observables).run(until=2.0).trajectory_
print(trajectory.values('sz').shape, round(float(np.ptp(trajectory.values('energy'))), 8))
```

### Solvers, observables, and spectra

A quantum system compiles to the quantum form, whose state is stored as complex pairs under `<system>.state`, a pure
state for a closed system and a density matrix for an open one. Its solvers hand every frame to QuTiP: `UnitarySolver`
for the Schrödinger equation, `LindbladSolver`, the default, for the master equation, and `QuantumTrajectoriesSolver`
for quantum jumps drawn from the run's random stream. An `ObservableSpec` whose expression is an operator records its
expectation, and one over a group gives one value per member when its `reduce` keeps the values. Every quantum run also
records its `energy`, and its `purity` when it is open. Groups on a lattice stream the space-time image of their local
expectations to Rerun. `QuantumAnalysis` analyses quantum systems: its kinds `'spectrum'` and `'ground-state'`
diagonalise the Hamiltonian exactly, optionally in the sector of a conserved diagonal operator, and `'purity'` and
`'entanglement'` read trajectories. `SymbolicAnalysis('hamiltonian')` prints the Hamiltonian in operator notation.

```python
from skcomplex.analysis import QuantumAnalysis
from skcomplex.models import create

spectrum = QuantumAnalysis(n_states=2).check().prepare(create('quantum_heisenberg', n_sites=4)).run()
print(spectrum.energies_.round(6).tolist())
```

## Co-simulation

A system whose laws do not fit one form, such as agents next to differential equations, is split into partitions at
subsystem boundaries. Every subsystem whose laws fit one form becomes a partition, and a law joins the partition that
owns its target, so an interaction of the parent system may contribute to several partitions. A subsystem whose own
elements mix law kinds raises `IncompatibleLawsError` with its path and the suggestion to split it.

```python
from skcomplex.engine.runtime import Simulation

from skcomplex.engine.compile import compile_system
from skcomplex.models.economics.resources import harvesting

farming = harvesting(10, effort=0.05)
print([(partition.name, partition.problem.form.value) for partition in compile_system(farming).partitions])
trajectory = Simulation(farming, random_state=0).run(until=50).trajectory_
print(round(trajectory.values('land/stock.n')[-1]), trajectory.agents('village/farmers', 50).values['wealth'].size)
print(trajectory.values('village/farmers_count.value')[-1])
```

- Every partition holds the states of the other partitions as inputs, which keep their values from the last
  synchronisation point while it advances.
- A law of one partition reads a reduction of a group of another through a role bound to the group, such as
  `count(s.farmers)` or `total(s.farmers.wealth)`. The co-simulation computes every reduction of agents at each
  synchronisation point and holds it until the next, in the state `value` of an element next to the group, such as
  `village/farmers_count.value`. A reduction of an element group reads the states of the other partition, which keep
  their values from the last synchronisation point too.
- Agent rules read the states of other partitions through `model.inputs[name]`, such as
  `model.inputs['land/stock.n']`.
- The `CoSimulationSolver` advances every partition with its own solver, chosen per partition through `solvers={'land':
  ScipyODESolver()}`, and exchanges inputs every `sync_dt`, which defaults to the frame duration and must divide it, and
  must be a whole number of steps for discrete and agent partitions.
- The Gauss-Seidel scheme, the default, advances partitions in system order, each reading the values already advanced
  by those before it. `scheme='jacobi'` advances all of them from the values at the start of the interval.
- Splitting adds an error of first order in `sync_dt`, so halving `sync_dt` roughly halves it at the cost of more
  exchanges. `CoSimulationSolver(split=True)` splits a system that fits one form, which shows the error against the
  joint solution.

## Visualizing with Rerun

`RerunObserver` streams frames to a spawned viewer, a running viewer, or a `.rrd` file. Entity paths follow the system
tree, each element type chooses its visual, and parameters appear under `parameters/`.

```python
from skcomplex.engine.runtime import Simulation
from skcomplex.core.system import System

import tempfile
from pathlib import Path

from skcomplex.core.spaces import ContinuousSpace
from skcomplex.core.elements import ParticleElement
from skcomplex.models.physics.common.particles import SpringInteraction
from skcomplex.engine.viz import RerunObserver

molecule = System(
    elements=[ParticleElement('a', x=[0.0, 0.0, 0.0]), ParticleElement('b', x=[1.5, 0.0, 0.0], v=[0.0, 0.5, 0.0])],
    interactions=[SpringInteraction('bond', first='a', second='b', k=4.0, length=1.0)],
    space=ContinuousSpace(n_dim=3),
)
path = Path(tempfile.mkdtemp()) / 'molecule.rrd'
Simulation(molecule, dt=0.02, observers=[RerunObserver('molecule', save_path=path)]).run(until=5)
```

Particles appear as 3D points, populations as time series, agents on grids and in continuous spaces as points colored by
a state, agents on networks as graph nodes, and reporters under `reporters/`. A group is logged as one batch per state,
and `PointsVisual(position='x', color_by='v')` colors its points by speed. Points and agents in a continuous space
without a given `radius` take a share of the extent of the view, and a group of at most ten points draws the trail of
its recent positions, which `trail=0` turns off. A one-dimensional lattice also logs a space-time image, one row per
logged frame with the oldest at the top, holding at most `ImageVisual(history=...)` rows. A complex state of a lattice
or a mesh appears as two images, its modulus under `<path>/<state>/modulus` colored over the range and its phase under
`<path>/<state>/phase` colored by a cyclic palette over `(-π, π]`, and a complex scalar as the series
`<path>/<state>/real` and `<path>/<state>/imag`. An element or a group sets its own visual with its `visual` argument,
such as `GroupElement(..., visual=VisualSpec('points', {...}))`. A `VisualSpec(kind, options)` names a visual of the
visualization layer, such as `'points'`, `'image'`, or `'agents'`, and `resolve_visual(spec)` builds it. Every visual
subclasses `BaseVisual` and implements the hook `_log` for elements or `_log_agents` for agents.
`visual.prepare(element, problem)` prepares a visual for one element and sets its views `views_`, and `visual.run(frame,
recording)` logs the element at a frame.

### Layouts and blueprints

Every visual declares the Rerun views that show what it logs, as its fitted attribute `views_`. Scalars go in time
series views, points in 2D or 3D spatial views by their dimension, lattices and fields in one 2D view per state, after
the 2D views of their space-time images in one dimension, meshes in 3D views, and agents in a 2D view or, on a network,
a graph view. Agents without a space get no spatial view. `default_blueprint` arranges these views into a Rerun
blueprint:
- spatial views come first and side by side, with time series views below them;
- scalar elements of one type share a time series view with at most six series;
- parameters are left out unless `parameters=True`;
- the side panels are collapsed, the top panel is hidden, and the time panel follows the latest frame.

`primary_view` returns the first view, the one that shows the system's signature behaviour. `RerunObserver` sends a
blueprint with `blueprint='default'` or a blueprint of your own, and writes it next to a saved recording as a `.rbl`
file. `save_blueprint` writes any blueprint to a file.

```python continuation
from skcomplex.engine.runtime import Simulation
from skcomplex.engine.viz import RerunObserver

import rerun.blueprint as rrb

from skcomplex.engine.viz import default_blueprint, primary_view, save_blueprint
from skcomplex.models import create

predation = create('lotka_volterra')
print(type(primary_view(predation)).__name__)
observer = RerunObserver('lotka_volterra', save_path=path.with_name('lotka_volterra.rrd'), blueprint='default')
Simulation(predation, dt=0.1, observers=[observer]).run(until=20)
print(sorted(file.name for file in path.parent.glob('lotka_volterra.*')))
custom = rrb.Blueprint(rrb.TimeSeriesView(origin='/prey', name='prey'), collapse_panels=True)
save_blueprint(custom, path.with_name('prey.rbl'), application_id='lotka_volterra')
```

```text
TimeSeriesView
['lotka_volterra.rbl', 'lotka_volterra.rrd']
```

Open both files with `rerun lotka_volterra.rrd lotka_volterra.rbl` to see the layout.

## Experiments

Complex systems are studied through many runs. `Experiment` runs a model, a factory of systems or a system, over a
design of factory arguments and dotted paths, scenarios, and replicates. Every design row and replicate gets
its own random stream spawned from `random_state` in a fixed order, so results are the same serially, on threads, or on
processes, and every scenario of a row and replicate shares its stream. Parameter and initial-state overrides reuse one
compiled problem, so sweeps of thousands of runs compile once. An experiment is an estimator: `check` validates its
settings, `prepare` resolves the model and the runs, and `run` executes them and keeps the results in `result_`.

```python
from skcomplex.experiments import Experiment, GridDesign, Scenario
from skcomplex.models.biology.epidemiology import sir
from skcomplex.models.economics.resources import harvesting

bounds = {'stochastic': [True], 'gamma': [0.1, 0.2]}
ensemble = Experiment(sir, GridDesign(), bounds, replicates=10, until=60, random_state=0).check().check().prepare().run()
print(round(float(ensemble.result_.summary()[0]['I.n.mean']), 1))

policies = [Scenario('baseline'), Scenario('quota', schedule=[(50.0, 'harvest.effort', 0.025)])]
stocks = Experiment(harvesting, scenarios=policies, until=80, outputs=['land/stock.n']).check().check().prepare().run()
print([round(float(row['land/stock.n'])) for row in stocks.result_.rows])
```

- Designs are `GridDesign`, `RandomDesign`, `LatinHypercubeDesign`, `SobolDesign`, `RowsDesign`, and the designs of
  sensitivity analysis `OneAtATimeDesign`, `MorrisDesign`, and `SaltelliDesign`. A design stores its settings, such as
  `SobolDesign(n=64)`, and prepares on the `bounds` of the experiment: a pair or a SciPy distribution per name for a
  sampled design, a sequence of values per name for a grid, and nothing for a rows design. `prepare` sets their rows in
  `rows_` and `run` returns them.
- `outputs` keeps states and observables as series, after `warmup`, and functions of a trajectory as scalars.
  `analyses` are unprepared analyses: every run clones each one, prepares it on its trajectory, runs it, and keeps what
  its `run` sets as columns such as `entropyanalysis.entropy`. Results give one row per run, `series`, `ensemble` with
  one row per run on common times and the seeds of the runs, `summary` with means, standard deviations, and quantiles
  across replicates, and `to_frame` for pandas.
- `SensitivityAnalysis` prepares on the result of an experiment and ranks its inputs one at a time, by Morris screening,
  or by Sobol indices with SALib.
- `CalibrationExperiment` fits dotted paths to observed series by least squares or a Gaussian likelihood, and keeps the
  estimates and their covariance with R², RMSE, MAE, and MAPE.

```python
from skcomplex.experiments import CalibrationExperiment, Experiment, OneAtATimeDesign, SensitivityAnalysis
from skcomplex.models.biology.ecology import logistic_growth

import numpy as np

times = np.arange(0.0, 21.0)
observed = 500 / (1 + 49 * np.exp(-0.8 * times))
bounds = {'growth.r': (0.1, 2.0), 'growth.K': (100, 1000)}
fit = CalibrationExperiment(logistic_growth, {'population.n': (times, observed)}, bounds).check().check().prepare().run()
print(round(fit.estimates_['growth.r'], 2), round(fit.r2_, 4))

sweep = Experiment(logistic_growth, OneAtATimeDesign(), bounds, until=20, outputs=['population.n'])
sweep = sweep.check().prepare().run()
print(SensitivityAnalysis('oat', 'population.n').check().prepare(sweep.result_).run().ranking_)
```

## Analysing symbolically

`SymbolicAnalysis('equations')` returns the equations of a system as Sage relations, and `SymbolicAnalysis('solve')`
solves an ODE system exactly where Maxima can, with `desolve` for one equation and `desolve_system` for linear systems
with constant coefficients.

```python
from skcomplex.core.elements import ParticleElement
from skcomplex.core.spaces import ContinuousSpace
from skcomplex.core.system import System

from skcomplex.analysis import SymbolicAnalysis
from skcomplex.models.physics.common.particles import HarmonicPotentialInteraction

oscillator = System(
    elements=[ParticleElement('p', x=1.0)],
    interactions=[HarmonicPotentialInteraction('trap', target='p', k=4.0)],
    space=ContinuousSpace(n_dim=1),
)
print(SymbolicAnalysis().check().prepare(oscillator).run().equations_['p.v[0]'])
print(SymbolicAnalysis('solve', use_initial_state=True).check().prepare(oscillator).run().solutions_['p.x[0]'])
```


## Analysing dynamics and data

`skcomplex.analysis` measures complex dynamics in the same way for every field. Every measure takes a system, or plain
arrays from trajectories or measured data.

**Stability and chaos.** `StabilityAnalysis` finds the fixed points, Jacobian, stability, nullclines, and divergence of
the symbolic vector field of a flow or a map. `LyapunovAnalysis` integrates its tangent dynamics compiled with Numba,
and the experiments `OrbitDiagramExperiment` and `BasinStabilityExperiment` run a model over parameters or initial
conditions.

```python
from skcomplex.analysis import LyapunovAnalysis, StabilityAnalysis
from skcomplex.experiments import OrbitDiagramExperiment
from skcomplex.models.biology.ecology import lotka_volterra
from skcomplex.models.mathematics.dynamical_systems import logistic_map, lorenz

print(StabilityAnalysis().check().prepare(lotka_volterra()).run().classifications_)
print(LyapunovAnalysis(until=200).check().prepare(lorenz()).run().exponents_.round(2).tolist())
diagram = OrbitDiagramExperiment(logistic_map, 'map.r', [2.9, 3.2, 3.5], 'map.x', transient=500, keep=32, dt=1)
print(diagram.check().prepare().run().periods_.tolist())
```

**Sections.** `SectionAnalysis('poincare')` finds the states where a state crosses a value in a direction, located
like an event, and `SectionAnalysis('stroboscopic')` samples a periodically forced system once per period.

```python
import numpy as np

import math

from skcomplex.analysis import SectionAnalysis
from skcomplex.models.physics.nonlinear_oscillators import duffing

section = SectionAnalysis('stroboscopic', period=2 * math.pi / 1.2, n_points=200).check().prepare(duffing()).run()
print(len(np.unique(np.round(section.states_, 4), axis=0)))
```

**Distributions and events.** `DistributionAnalysis` describes heavy tails by its `kind`: a `'histogram'` with
logarithmic bins, the `'ccdf'`, the Hill `'tail-index'`, and the `'gini'` coefficient. `EventAnalysis` turns series
into avalanches or bursts with sizes, durations, and waiting times. `FitAnalysis(law='power')` adapts the `powerlaw`
library: it chooses the lower cutoff by Clauset, Shalizi, and Newman's Kolmogorov-Smirnov criterion, accepts an upper
cutoff, uses the exact discrete likelihood for integers, gives a bootstrap p-value, and compares the power law with the
exponential, log-normal, stretched exponential, and truncated power-law tails of `compare_with`.

```python
import numpy as np

from skcomplex.analysis import DistributionAnalysis, EventAnalysis, FitAnalysis

samples = (1 - np.random.default_rng(0).random(5000)) ** (-1 / 1.5)
fit = FitAnalysis('power', bootstrap=20, random_state=0).check().prepare(samples).run()
gini = DistributionAnalysis('gini').check().prepare(samples).run().gini_
print(round(fit.exponent_, 2), fit.p_value_, round(gini, 2))
print(EventAnalysis().check().prepare([0, 2, 3, 0, 0, 1, 0]).run().sizes_.tolist())
```

**Time series and information.** `CorrelationAnalysis`, `SpectrumAnalysis`, `DisplacementAnalysis`,
`ScalingAnalysis`, and `RollingAnalysis` describe memory and diffusion. The mean squared displacement of
`DisplacementAnalysis` comes from `tidynamics` over all lags of unwrapped positions, with its exponent and diffusion
coefficient fitted over a chosen range of lags. The fluctuation analyses come from `nolds`: detrended
fluctuation of any order, which gives `H + 1` for an integrated series such as a random walk, or the rescaled range.
`RollingAnalysis` gives the variance, the lag-one autocorrelation, or the skewness in rolling windows with its
trend, and `EarlyWarningAnalysis` of `skcomplex.models.earth.climate` gathers them with spectral reddening as early
warnings of tipping.
`InformationAnalysis` measures structure and information flow by its `kind`, such as `'mutual'` and `'transfer'`, as in
Lindgren's information theory of complex systems, and `EntropyAnalysis` gives the Shannon, block, rate, excess,
permutation, and Lempel-Ziv entropies by its `kind`. Its setting `symbols` turns real series into symbols first: bins of
equal width, equally populated bins, or ordinal patterns, coded by the lexicographic rank of their permutation with
`ordpy`, with a delay and of any length. The Lempel-Ziv parsing comes from `antropy`.

```python
import numpy as np

from skcomplex.analysis import EntropyAnalysis, InformationAnalysis, ScalingAnalysis

noise = np.random.default_rng(1).standard_normal(20_000)
print(ScalingAnalysis().check().prepare(np.column_stack([noise, np.cumsum(noise)])).run().exponent_.round(2).tolist())
for kind in ('rate', 'excess'):
    print(EntropyAnalysis(kind, dimension=4).check().prepare([0, 0, 1, 1] * 200).run().entropy_.round(3).tolist())
driver = np.random.default_rng(2).integers(0, 2, 5000)
transfer = InformationAnalysis('transfer').check().prepare(np.column_stack([driver, np.roll(driver, 1)])).run()
print(round(transfer.information_[0, 1], 2))
```

**Space and networks.** `ClusterAnalysis` labels clusters of a grid, with periodic edges, `CorrelationAnalysis`
gives radially averaged correlations, and `ScalingAnalysis` measures fractals. `NetworkAnalysis` measures the
degrees, the summary, and the centralities of NetworkX graphs and of the networks of trajectories.

```python
import numpy as np

import networkx as nx

from skcomplex.analysis import ClusterAnalysis, NetworkAnalysis, ScalingAnalysis

clusters = ClusterAnalysis('labels').check().prepare(np.random.default_rng(3).random((100, 100)) < 0.6).run()
print(clusters.spanning_, clusters.sizes_.max())
rows, columns = np.indices((512, 512))
print(round(ScalingAnalysis('box-counting').check().prepare((rows & columns) == columns).run().exponent_, 2))
print(NetworkAnalysis('summary').check().prepare(nx.watts_strogatz_graph(200, 6, 0.1, seed=0)).run().diameter_)
```

### Kinds of data and signatures

Every analysis consumes one kind of data, never a model or a field of science: a series, a jump path, a field, agent
snapshots, a network, or an ensemble of runs. The same `SpectrumAnalysis` serves an Ising lattice and a
reaction-diffusion field, and the same `EventAnalysis('excursions')` serves a sandpile and an earthquake model. Every
analysis takes plain arrays or graphs, so it works on measured data as well.

An analysis prepared on a trajectory extracts its kind of data itself: named series of states, reporters, or
observables, the jump path of a discrete state, the frames of a lattice, the agents of a group with their positions
unwrapped across periodic sides, or the network of every frame. An ensemble of runs is one row per run and one column
per common time, as `ExperimentResult.ensemble` returns it. `DiscoveryAnalysis`, prepared on a trajectory, a system, or
an array, keeps the data kinds the object holds in `kinds_`, and its run lists in `analyses_` the analyses of the
library that apply to it under each kind, with how they read it in `readings_`.

```python
import numpy as np

from skcomplex.analysis import DiscoveryAnalysis, SpectrumAnalysis
from skcomplex.models import run

spins = run('ising', size=32, settings={'until': 20}).trajectory_
print(DiscoveryAnalysis().check().prepare(spins).kinds_)
print(SpectrumAnalysis('structure-factor').check().prepare(spins).run().wavenumbers_[:3].round(3).tolist())
discovery = DiscoveryAnalysis().check().prepare(np.zeros((16, 16))).run()
print(discovery.readings_['series'], len(discovery.analyses_['field']) > 0)
```

`SignatureAnalysis` computes one default analysis per kind of data of a trajectory, with its curves, axis labels, title,
and measured values: the power spectrum and the autocorrelation time of series, or the return map of a map, the waiting
times and the dispersion index of jump paths, the structure factor of the last frame of fields with its correlation
length or peak wavelength, the polar and milling orders of moving agents or their nearest-neighbour distances, and the
degree distribution of networks. The gallery draws it for every example that names no analysis of its own.

```python
from skcomplex.analysis import SignatureAnalysis
from skcomplex.models import run

predation = run('lotka_volterra', settings={'until': 20.0}).trajectory_
for signature in SignatureAnalysis().check().prepare(predation).run().signatures_:
    print(signature.kind, signature.title, signature.xscale, list(signature.values))
```

### Complexity and information

`EntropyAnalysis` measures every entropy by its `kind`. The kinds `'renyi'` and `'tsallis'` generalize the Shannon
entropy of a distribution. The kinds `'sample'`, `'approximate'`, `'multiscale'`, and `'spectral'` come from `antropy`
and measure the regularity of a series. The kind `'permutation'` reads ordinal patterns of series with a delay, or of
two-dimensional fields in a window, in its Shannon, Rényi, or Tsallis `measure`. Discrete states such as spins have
many equal values, which `ties='random'` ranks at random. The kind `'ensemble'` follows the entropy of an ensemble and
its divergence from a `reference` law frame by frame, as in the H-theorem, and `ComplexityAnalysis` places data on
Rosso's complexity-entropy plane between its bounding curves with `ordpy`. The kind `'active-storage'` of
`InformationAnalysis` measures the information the past of a series holds about its next value, and the kind
`'transfer'` estimates the flow from every series to every other by plug-in counts of symbols or by the
nearest-neighbour estimator of `ennemi` on continuous values.

```python
import numpy as np

import math

from skcomplex.analysis import ComplexityAnalysis, EntropyAnalysis, InformationAnalysis

rng = np.random.default_rng(0)
renyi = EntropyAnalysis('renyi', order=2, probabilities=True).check().prepare(np.full(8, 1 / 8)).run()
print(renyi.entropy_.round(3).tolist())
sample = EntropyAnalysis('sample', dimension=2).check().prepare(rng.standard_normal(2000)).run()
print(sample.entropy_.round(1).tolist())
plane = ComplexityAnalysis(dimension=4).check().prepare(rng.standard_normal(20_000)).run()
print(plane.entropy_.round(2).tolist(), plane.complexity_.round(2).tolist())
x = rng.standard_normal(5000)
y = np.concatenate([[0.0], 0.8 * x[:-1]]) + 0.6 * rng.standard_normal(5000)
transfer = InformationAnalysis('transfer', estimator='knn', base=math.e).check().prepare(np.column_stack([x, y])).run()
print(transfer.information_[0, 1] > transfer.information_[1, 0])
```

### Higher-order information and emergence

The measures of many variables take one row per sample and one column per variable. Among the kinds of
`InformationAnalysis`, `'total-correlation'` counts the information shared at any order, `'dual-total-correlation'` the
information shared by two or more variables, and `'o-information'`, their difference, is positive when the variables
are redundant and negative when they are synergistic. `'o-information-gradients'` gives the share of every variable.
They take the `'plugin'` estimator of `dit` for symbols, `'gaussian'` for the closed form of a fitted Gaussian, or
`'knn'` for chains of `ennemi` estimates. `PartialInformationAnalysis` splits what two sources of symbols tell about a
target into redundant, unique, and synergistic parts with a redundancy measure of `dit`. `LocalInformationAnalysis`
returns the entropy, the mutual information, or the transfer entropy at every sample, and their means equal the global
measures. `SurrogateTestAnalysis` compares the attribute of any of these analyses with its values on copies of the
data whose variables are shuffled independently.

```python continuation
import numpy as np

from skcomplex.analysis import InformationAnalysis, PartialInformationAnalysis, SurrogateTestAnalysis

bits = rng.integers(0, 2, (5000, 2))
xor = np.column_stack([bits, bits[:, 0] ^ bits[:, 1]])
print(round(InformationAnalysis('o-information').check().prepare(xor).run().information_, 2))
gate = np.column_stack([bits, bits[:, 0] & bits[:, 1]])
parts = PartialInformationAnalysis().check().prepare(gate).run()
print(round(parts.redundancy_, 2), round(parts.synergy_, 2))
measure = InformationAnalysis('o-information')
test = SurrogateTestAnalysis(measure, 'information_', n=19, alternative='less', random_state=0)
print(test.check().prepare(xor).run().p_value_)
```

The kind `'criteria'` of `EmergenceAnalysis` computes the criteria of Rosas et al. (2020) for a macro feature of micro
variables, given by `partition`: Ψ is positive when the feature predicts its own future beyond what each micro variable
predicts, Δ is positive when it predicts a micro variable beyond the micro variables, and Γ is zero for a feature
decoupled from every micro variable. The kind `'closure'` takes a partition of micro states, as a label per state or a
function of the micro series, and compares how well the macro past and the micro past predict the macro future. A gap of
zero means the coarse-graining is closed. `EpsilonMachineAnalysis` reconstructs the causal states of a discrete series
by causal-state splitting, with their transitions, statistical complexity, and entropy rate.

```python continuation
import numpy as np

from skcomplex.analysis import EmergenceAnalysis, EpsilonMachineAnalysis

parity = rng.integers(0, 2, (5000, 4))
parity[:, -1] = (parity[:, :-1].sum(axis=1) + np.arange(5000) // 100) % 2
criteria = EmergenceAnalysis('criteria', partition=lambda micro: micro.sum(axis=1) % 2).check().prepare(parity).run()
print(criteria.psi_ > 0.5)
closure = EmergenceAnalysis('closure', partition=[0, 0, 1, 1]).check().prepare(rng.integers(0, 4, 5000)).run()
print(round(closure.gap_, 2))
golden = [1]
for _ in range(20_000):
    golden.append(1 if golden[-1] == 0 else int(rng.random() < 0.5))
machine = EpsilonMachineAnalysis(max_length=4).check().prepare(golden).run()
print(len(machine.histories_), round(machine.statistical_complexity_, 2))
```

### Invariants of dynamics

`EmbeddingAnalysis` builds delay coordinates of a series, with the delay at the first minimum of the auto mutual
information and the dimension where the false nearest neighbours fall below a threshold.
`ScalingAnalysis('correlation-dimension')` counts close pairs of any point set with a k-d tree, so it serves embedded
series and agent positions alike. The Kaplan-Yorke dimension and the Kolmogorov-Sinai entropy of `LyapunovAnalysis`
follow from a Lyapunov spectrum, `LyapunovAnalysis('largest')` estimates the largest exponent from one series with
Rosenstein's method of `nolds`, and `ErgodicityAnalysis` of kind `'invariant-density'` or `'ulam'` estimates the
invariant density of a map. `RecurrenceAnalysis` gives the recurrence matrix, rate, determinism, laminarity, line
lengths, trapping time, and line entropy of embedded series.

```python
import numpy as np

from skcomplex.analysis import (
    EmbeddingAnalysis,
    ErgodicityAnalysis,
    LyapunovAnalysis,
    RecurrenceAnalysis,
    ScalingAnalysis,
)

points = np.random.default_rng(1).random((3000, 2))
print(round(ScalingAnalysis('correlation-dimension').check().prepare(points).run().exponent_, 1))
print(round(LyapunovAnalysis().check().prepare([0.906, 0.0, -14.572]).run().kaplan_yorke_dimension_, 2))
chain = ErgodicityAnalysis('ulam', edges=np.linspace(0, 1, 21), samples=200)
print(chain.check().prepare(lambda x: 4 * x * (1 - x)).run().density_.round(1)[:3].tolist())
sine = EmbeddingAnalysis(delay=15, dimension=2).check().prepare(np.sin(np.arange(1000) * 0.1)).run().points_
print(round(RecurrenceAnalysis().check().prepare(sine).run().determinism_, 2))
```

### Surrogates fitted to data

`ModeDecompositionAnalysis`, `SparseIdentificationAnalysis`, and `ReservoirAnalysis` learn a system from data: they
prepare on one or more trajectories or arrays sampled at a uniform step `dt`, and their run sets `system_`, a `System`
of one surrogate element, which `validate` checks and `Simulation` runs like a hand-written system, `training_error_`
and `validation_error_`, the relative errors of the fit on the training samples and on the end of every series held out
by `validation`, and `forecast_`, the variables over `horizon` samples after the last one.

- `ModeDecompositionAnalysis` fits dynamic mode decomposition with PyDMD and sets the continuous eigenvalues
  `ln(mu) / dt` in `eigenvalues_`, the modes, their amplitudes, and a surrogate under continuous time with the matrix
  rate law `dg/dt = L g`. A scikit-learn transformer passed as `dictionary`, such as `PolynomialFeatures(2)`, lifts the
  variables to observables first, which is extended DMD.
- `SparseIdentificationAnalysis` fits sparse identification of nonlinear dynamics with PySINDy and sets the monomials of
  the variables in `features_`, their coefficients in `coefficients_`, and a surrogate under continuous time with one
  symbolic rate law per variable.
- `ReservoirAnalysis` fits a next-generation reservoir or an echo state network from ReservoirPy with a ridge readout
  of the increments over one sample. Its surrogate runs under discrete time, one step per sample, holds the memory of
  the reservoir as states, and starts at the last sample, so a run of `system_` is the autonomous forecast.

```python
import numpy as np

from skcomplex.analysis import ModeDecompositionAnalysis, ReservoirAnalysis, SparseIdentificationAnalysis
from skcomplex.engine.compile import validate

times = np.arange(0.0, 10.0, 0.01)
spiral = np.exp(-0.1 * times)[:, None] * np.column_stack([np.cos(times), np.sin(times)])
decomposition = ModeDecompositionAnalysis(['x', 'y'], dt=0.01).check().prepare(spiral).run()
print(np.sort(decomposition.eigenvalues_.imag).round(3).tolist(), validate(decomposition.system_).ok)
sparse = SparseIdentificationAnalysis(['x', 'y'], degree=1, threshold=0.05, horizon=50, dt=0.01)
sparse = sparse.check().prepare(spiral).run()
print(sparse.features_, sparse.coefficients_.round(2).tolist(), sparse.forecast_.shape)
reservoir = ReservoirAnalysis(['x', 'y'], horizon=50).check().prepare(spiral).run()
print(type(reservoir.system_.time).__name__, reservoir.forecast_.shape)
```

### Stochastic processes

`PassageAnalysis` measures first passages, absorption, and returns of paths, and the committor and mean first-passage
time of a transition matrix, by its `kind`, and `EventAnalysis('laminar-durations')` measures laminar phases.
`MarkovAnalysis` estimates transition matrices from paths, reads transition and rate matrices with `matrix`, and gives
their stationary laws, spectral gaps, and relaxation times, and Markov state models and implied timescales with
`deeptime`. `DistributionAnalysis('dispersion-index')`, `ScalingAnalysis('variance-mean')`, and
`EventAnalysis('counts')` describe fluctuations, and `KramersMoyalAnalysis` estimates the drift and diffusion of a
series with `kramersmoyal`.
`IrreversibilityAnalysis` measures entropy production from rates, jump paths, Langevin paths, or transition counts,
tests detailed balance and the fluctuation theorem, and checks the thermodynamic uncertainty relation: the squared
relative fluctuation of a current times the entropy produced is at least 2. It also gives the response to a field and
the fluctuation-dissipation ratio. `LargeDeviationAnalysis` gives the scaled cumulant generating function from a tilted
generator, from blocks of one run, or by cloning a population of trajectories of a model through the simulation's
steppers, for systems whose tilted generators are too large to write down, and its Legendre transform at `values`. It
also gives rate functions from histograms at several sizes and reweights samples. `CorrelationAnalysis('two-time')` and
`CollapseAnalysis` measure aging. `FitAnalysis` with the laws `'extreme-value'`, `'stable'`, and the named laws such as
`'exponential'`, and `DistributionAnalysis('records')` describe extremes and limit laws.

```python
import numpy as np

from skcomplex.analysis import IrreversibilityAnalysis, LargeDeviationAnalysis, MarkovAnalysis

ring = 2 * np.roll(np.eye(5), 1, axis=1) + np.roll(np.eye(5), -1, axis=1)
stationary = MarkovAnalysis('stationary', matrix='rate').check().prepare(ring).run().stationary_
production = IrreversibilityAnalysis().check().prepare(ring).run().entropy_production_
print(stationary.round(2).tolist(), round(production, 4))
gap = MarkovAnalysis('spectral-gap', matrix='transition').check().prepare([[0.9, 0.1], [0.2, 0.8]]).run()
print(round(gap.spectral_gap_, 2))
tilted = LargeDeviationAnalysis(weights=[[1.0]], tilts=np.linspace(-3, 3, 601), values=3.0)
print(round(float(tilted.check().prepare([[2.0]]).run().rate_function_[0]), 3))
```

Cloning advances every clone of a population by an interval, weights it by `exp(s A)` for the increment `A` of a
time-additive observable, and resamples the population by the weights. The observable is the dotted name of a state
whose increment is counted, or a function of the trajectory of each interval. A warning names a population whose
effective size collapses, a sign that the estimate is biased.

```python
from typing import Annotated

import numpy as np

from skcomplex.analysis import IrreversibilityAnalysis, LargeDeviationAnalysis
from skcomplex.core.elements import Element
from skcomplex.core.laws import JumpLaw
from skcomplex.core.specs import StateSpec
from skcomplex.core.system import System


class CounterElement(Element):
    def __init__(self, label, *, n: Annotated[float, StateSpec()] = 0.0):
        super().__init__(label)
        self.n = n

    def _get_laws(self, s):
        return [JumpLaw(1.0, {s.n: 1})]


counter = System(elements=[CounterElement('counter')])
cloning = LargeDeviationAnalysis(
    'cloning', observable='counter.n', tilts=0.5, population=200, until=10.0, random_state=0
)
print(cloning.check().prepare(counter).run().scgf_.round(2).tolist(), round(float(np.exp(0.5) - 1), 2))
currents = np.random.default_rng(0).normal(10.0, np.sqrt(30.0), 5000)
relation = IrreversibilityAnalysis('uncertainty-relation', entropy=10.0, random_state=0)
print(relation.check().prepare(currents).run().holds_)
```

### Emergence and ergodicity

`EmergenceAnalysis` reads a transition matrix for its kinds `'effective-information'` and `'causal'`. The effective
information measures how much a transition matrix constrains the next state: it is the mutual information in bits
between a uniform intervention on the current state and the next state, and it equals the determinism of the matrix
minus its degeneracy. Rows with no observed transition are left out with an `UnobservedStatesWarning` that names their
states. The kind `'causal'` reports the gain in effective information of a partition over the micro states, and without
a partition it searches for one, greedily by merging the pair of blocks with the largest gain, or exhaustively for at
most 10 states. The partition is an integer label per state or a function of the array of states, and the macro matrix
sums the transitions within every block, weighting the states of a block uniformly or by the stationary distribution. In
the chain of Hoel, Albantakis, and Tononi, seven noisy states merge into one deterministic macro state that carries more
effective information than the micro states.

```python
import numpy as np

from skcomplex.analysis import EmergenceAnalysis

noisy = np.zeros((8, 8))
noisy[:7, :7], noisy[7, 7] = 1 / 7, 1.0
print(round(EmergenceAnalysis('effective-information').check().prepare(noisy).run().information_, 3))
found = EmergenceAnalysis('causal').check().prepare(noisy).run()
print(found.partition_.tolist(), round(found.macro_information_, 3), round(found.gain_, 3))
print(found.macro_matrix_.tolist())
```

`ErgodicityAnalysis` of kind `'breaking'`, `'metric'`, or `'distance'` diagnoses whether the runs of an ensemble, one
row per run such as `ExperimentResult.ensemble` returns, explore the same states. The ergodicity-breaking parameter is
the variance of the time averages across runs over the variance of all values, which tends to 0 for ergodic runs and
stays near 1 for runs frozen in separate states. The fluctuation metric of Thirumalai and Mountain is the variance of
the running time averages, which decays as `1 / t` for ergodic runs and levels off for frozen ones. The ergodicity
distance is the median total-variation distance between the values of one run over time and the values of the whole
ensemble.

```python
import numpy as np

from skcomplex.analysis import ErgodicityAnalysis

rng = np.random.default_rng(0)
mixing = rng.standard_normal((100, 5000))
frozen = rng.choice([-1.0, 1.0], size=(100, 1)) + 0.1 * rng.standard_normal((100, 5000))
for runs in (mixing, frozen):
    breaking = ErgodicityAnalysis('breaking').check().prepare(runs).run().ergodicity_breaking_
    metric = ErgodicityAnalysis('metric').check().prepare(runs).run().exponent_
    distance = ErgodicityAnalysis('distance').check().prepare(runs).run().ergodicity_distance_
    print(round(breaking, 2), round(metric, 1), round(distance, 1))
```

### Fields and lattices

`SpectrumAnalysis('structure-factor')` averages the power of a field radially on a periodic grid of any dimension, and
`CorrelationAnalysis('length')` fits its Ornstein-Zernike form or the decay of the correlation function.
`DomainAnalysis` measures coarsening with the kinds `'size'` and `'growth'`, the wavelength and orientation of patterns
with `'wavelength'`, and the interfaces between domains with `'interface-density'`, and
`ScalingAnalysis('variance-mean')` the number fluctuations of density fields in boxes of growing side. `ClusterAnalysis`
with the kinds `'sizes'`, `'percolation'`, and `'spanning'` describes clusters of grids and components of graphs, and
`DistributionAnalysis('moment-ratio')` with `CollapseAnalysis` of the kinds `'crossings'` and `'scaling'` locate
transitions by finite-size scaling. `SusceptibilityAnalysis` of `skcomplex.models.physics.statistical_mechanics`
locates the peak of the susceptibility over a temperature sweep of the spin models.
`StabilityAnalysis('dispersion')` predicts patterns from a field model's laws before running it: it linearises them
around a uniform state, turns every Laplacian into its eigenvalue on each Fourier mode of the lattice, and returns the
growth rate and frequency of every mode, exact for the model's discretization. A band of growing modes is a Turing
instability, and a growing mode with a nonzero frequency travels.

```python
import numpy as np

from skcomplex.analysis import ClusterAnalysis, DistributionAnalysis, DomainAnalysis, StabilityAnalysis
from skcomplex.models.chemistry.pattern_formation import brusselator

relation = StabilityAnalysis('dispersion').check().prepare(brusselator((128,), b=1.9)).run()
print(relation.band_.min().round(2), relation.band_.max().round(2))

stripes = np.cos(2 * np.pi * np.arange(64) / 8)[:, None] * np.ones(64)
print(DomainAnalysis('wavelength').check().prepare(stripes).run().wavelength_.tolist())
grid = np.random.default_rng(2).random((128, 128)) < 0.7
print(ClusterAnalysis('percolation').check().prepare(grid).run().spanning_)
samples = np.random.default_rng(3).standard_normal(10_000)
print(DistributionAnalysis('moment-ratio').check().prepare(samples).run().cumulant_.round(1).tolist())
```

### Agents

`AlignmentAnalysis` measures the alignment of every frame of agents, of a field of angles or vectors, or of series of
phases, by the length of the mean vector, the mean rotation, the axial order, or the mean phase. The active matter
models name them the polar, milling, and nematic orders, and the synchronization models the Kuramoto order.
`DistributionAnalysis` and `FitAnalysis` read the speeds of moving agents, and `KineticTemperatureAnalysis` and
`MaxwellBoltzmannAnalysis`, next to the gases of `skcomplex.models.physics.statistical_mechanics`, compare velocities
with the Maxwell-Boltzmann law of their dimension. `NeighborAnalysis` describes point patterns in
open or periodic boxes by their nearest-neighbour `'distances'`, their `'clark-evans'` ratio, and their
`'radial-distribution'`, and `CoarseGrainAnalysis` turns agents into density or polarization fields for the field
analyses. `NeighborAnalysis('like-share')` and `DomainAnalysis('interface-density')` measure segregation on grids and
networks, and `DistributionAnalysis('total-variation')` gives the distance between the distributions of two groups
across units, the dissimilarity index of segregation studies.

```python
import numpy as np
from skcomplex.models import run

from skcomplex.analysis import AlignmentAnalysis, CoarseGrainAnalysis, NeighborAnalysis

flock = run('boids', settings={'until': 5}).trajectory_
print(AlignmentAnalysis('mean-vector').check().prepare(flock).run().order_.shape)
positions = np.random.default_rng(4).random((1000, 2))
ratio = NeighborAnalysis('clark-evans', box=[1.0, 1.0]).check().prepare(positions).run().ratio_
field = CoarseGrainAnalysis([8, 8], box=[1.0, 1.0]).check().prepare(positions).run().field_
print(ratio.round(1).tolist(), float(field.sum()))
```

### Criticality

`EventAnalysis('excursions')` extracts avalanches from activity series or relaxation records, and
`FitAnalysis(law='power')` fits their size and duration laws. `EventAnalysis('branching')` estimates the branching
ratio by the multistep regression of `mrestimator`, which stays unbiased under subsampling.
`CracklingNoiseAnalysis` of `skcomplex.models.physics.criticality` checks the scaling relation between the exponents of
the sandpiles' avalanches and collapses their mean shapes. `DistributionAnalysis('moments')` gives the
order parameter and its variance over a sweep, `ScalingAnalysis('power')` fits the order-parameter exponent, and
`CorrelationAnalysis('overlap')` compares replicas.

```python
import numpy as np

from skcomplex.analysis import EventAnalysis

found = EventAnalysis('excursions').check().prepare([0, 1, 3, 2, 0, 0, 4, 1, 0]).run()
print(found.sizes_.tolist(), found.durations_.tolist())
rng = np.random.default_rng(5)
activity = np.zeros(5000)
activity[0] = 50
for t in range(1, activity.size):
    activity[t] = rng.poisson(0.8 * activity[t - 1] + 10)
print(round(EventAnalysis('branching').check().prepare(activity).run().branching_ratio_, 1))
```

### Networks

`NetworkAnalysis` measures the network of a graph or of every frame of a trajectory by its `kind`: the
`'degrees'` with their moments and tail exponent, the `'assortativity'`, the `'centrality'`, the `'neighbor-degree'`,
the `'summary'`, and the `'communities'` by Louvain modularity maximization or label propagation with their agreement
with a reference partition. `MotifAnalysis` counts motifs of three and four nodes and scores them against
degree-preserving edge switches or the configuration model. The library
model `network_structure_analysis` records a network with these measures.

```python
import networkx as nx

from skcomplex.analysis import MotifAnalysis, NetworkAnalysis

club = nx.karate_club_graph()
communities = NetworkAnalysis('communities', random_state=0).check().prepare(club).run()
print(communities.modularity_.round(2).tolist(), MotifAnalysis().check().prepare(club).run().counts_[0]['triangle'])
```
