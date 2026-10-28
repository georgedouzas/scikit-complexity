# User guide

## Concepts

`scikit-complexity` separates the description of a model from the way it is solved.

- A system is made of **parts**: its elements, its interactions, the laws they return, and the system itself.
    - An **element** is an entity that carries state, such as a particle, a population, a compartment, or a market.
    - A **group** holds many identical elements of one type as arrays.
    - An **interaction** couples elements through **roles** and describes its effect with **laws**.
    - A **system** holds elements, interactions, child systems, and an optional space.
- Every part speaks three verbs. `check()` validates it, `prepare(...)` prepares it against what it works on, and
  `run(...)` returns what it contributes: an element, an interaction, or a system returns its laws, and a law returns
  its contribution at given values, such as the rate of a `RateLaw`. Elements, interactions, and systems return
  themselves from `check` and `prepare`; laws are values, so they return checked and prepared copies.
- A run records **observables**, named values computed from the states at every frame, and checks the
  **invariants** it is given.
- A **law** says how a state changes. Every law has a kind, and the four most common kinds are these.
    - `RateLaw` adds a term to the time derivative of a state. Rates on one state add up.
    - `NoiseLaw` drives a state with a Wiener process, independent unless laws name a shared or correlated one.
    - `JumpLaw` fires a random event that changes states by fixed increments.
    - `UpdateLaw` gives the next value of a state after one discrete step.
- A **space** gives vector states, such as positions, their dimension.
- A **formalism** adds the laws, components, compilation, and solvers of one kind of dynamics, such as lattice
  dynamics or quantum dynamics, to the shared core.

These concepts carry no field-specific meaning. A population, a particle, a firm, and a cell are all elements, so a
model from any science is a configuration of the same interface, and every ready-made model in `skcomplex.models` is a
factory that builds such a configuration.

Compiling a system infers its mathematical form from its law kinds and builds fast numeric functions.

| Law kinds | Form | Default solver |
| --- | --- | --- |
| `RateLaw` | ordinary differential equations | `ScipyODESolver` with `LSODA` |
| `RateLaw` and `NoiseLaw` | stochastic differential equations | `EulerMaruyamaSolver` |
| `JumpLaw` | jump process | `GillespieSolver` |
| `UpdateLaw` | discrete map | `IterationSolver` |

## Defining elements

Every element type derives from `Element`, and every interaction type from `BaseInteraction`; both, like laws and
systems, derive from `BasePart`. A type declares its variables as class attributes made by `state`, `parameter`, and `role`.

- The constructor takes the label first, and then the value of every declared name by keyword.
- The constructor stores each value unmodified under its own name, and the declared default for a name it is not given.
- A value for a name the type does not declare raises `ArgumentError`, and so does a missing role.
- `get_params`, `set_params`, `clone`, and `repr` follow the scikit-learn estimator contract, so every component clones
  with its values.

```python
from sklearn.base import clone

from skcomplex.core.base import NumericDomain, parameter, role, state
from skcomplex.core.components import Element
from skcomplex.core.base import BaseInteraction
from skcomplex.core.laws import JumpLaw, RateLaw


class SpeciesElement(Element):
    n: float = state(10.0, domain=NumericDomain.NONNEGATIVE)


class BirthInteraction(BaseInteraction):
    target: str = role()
    rate: float = parameter(0.5, domain=NumericDomain.NONNEGATIVE)

    def laws(self, s):
        return [RateLaw(s.target.n, s.rate * s.target.n)]


deer = SpeciesElement('deer', n=40.0)
birth = BirthInteraction('birth', target='deer', rate=0.2)
print(deer.n, clone(birth).rate)
```

Inside `laws`, `s.rate` is the symbol of the parameter and `s.target.n` is the symbol of the state of the element bound
to the role `target`. Because parameters stay symbols, a running simulation can change them without compiling again.

### Constructor settings

A setting that is neither a state, a parameter, nor a role, such as a switch between two sets of laws, a radius, or a
layer, needs an explicit `__init__` on its type.

- The `__init__` takes the label first and the setting as a keyword-only argument with a default.
- It passes the label and the remaining values to `super().__init__(label, **values)`.
- It stores the setting unmodified under its own name.

`get_params` reads the keyword arguments of these constructors along the class hierarchy, so a setting clones and
prints like a declared value.

```python continuation
from skcomplex.core.laws import JumpLaw
from sklearn.base import clone

class SwitchedBirthInteraction(BirthInteraction):
    def __init__(self, label, *, stochastic=False, **values):
        super().__init__(label, **values)
        self.stochastic = stochastic

    def laws(self, s):
        if self.stochastic:
            return [JumpLaw(s.rate * s.target.n, {s.target.n: 1})]
        return super().laws(s)


print(clone(SwitchedBirthInteraction('birth', target='deer', stochastic=True)))
```

### Laws

Every law class derives from `BaseLaw` and is a frozen dataclass. It declares what it is as class variables, which the
compiler and validation read instead of lists of their own.

- `law_kind` is the registered kind of the law, such as `LawKind.RATE`. Solvers and backends name the kinds they
  support.
- `time_class` is the time the law is valid under: continuous, event, discrete, constraint, or any.
- `traits` is a set of registered facts about the class.
    - `Trait.DECLARABLE` marks a law that the `laws` method of a component may return.
    - `Trait.FORMLESS` marks a law, such as `BoundaryLaw`, that joins the form of the other laws of its component and
      chooses no form of its own.
    - `Trait.KERNEL_FREE` marks a law, such as `TransferLaw`, that the compiler lowers into other laws and that needs no
      numeric function of its own.
- `requires` states what the law needs from the system: the time, the space, and the domains of the states it
  targets.

Every law also has the field `targets`, the states it changes, which `BaseLaw` defines. A law of one state, such as
`RateLaw`, takes the state as its first argument, alone or in a one-element tuple, and checking the law turns it into a
tuple. A law that updates several states together, such as `LocalRuleLaw`, takes a tuple of them. Laws that name their
states in other values, such as the increments of a `JumpLaw`, leave it empty.

A system holds only laws of registered classes. The shared core registers its general laws, which are `RateLaw`,
`NoiseLaw`, `JumpLaw`, `UpdateLaw`, `TransitionLaw`, `EmissionLaw`, `EventLaw`, `BoundaryLaw`, `EquationLaw`, and
`TransferLaw`. Each formalism registers its own law classes when it is imported. `law_classes(trait)` lists the
registered classes, optionally only those with a trait.

A system is defined by its elements, its interactions, and the laws they return from `laws(s)`, and by nothing else.
A value that only names an expression is a Python expression in the laws, such as `leverage = s.debt / s.equity`, and
a function of the namespace shares it between laws. What a run records or checks is an argument of the run:
`Simulation(observables=[...])` records named values, and `Simulation(invariants={name: tolerance})` checks that the
observables it names stay constant.

```python
from skcomplex.core.laws import RateLaw

from skcomplex.core.base import LawKind, Trait
from skcomplex.core.laws import BoundaryLaw
from skcomplex.core.base import BaseLaw, law_classes

print(issubclass(RateLaw, BaseLaw), RateLaw.law_kind == LawKind.RATE, RateLaw.time_class)
print(Trait.FORMLESS in BoundaryLaw.traits, RateLaw in law_classes(Trait.DECLARABLE))

from skcomplex.core.base import symbol

x = symbol('x')
print(RateLaw(x, -x).check().targets)
```

A law can also be a Python function of time and current values, which covers rules that have no closed form.

```python
from skcomplex.core.base import role
from skcomplex.core.base import BaseInteraction, parameter, role
from skcomplex.core.laws import RateLaw

class ThresholdInteraction(BaseInteraction):
    target: str = role()

    def laws(self, s):
        name = f'{self.target}.n'
        return [RateLaw(s.target.n, lambda t, values: -1.0 if values[name] > 50 else 0.0)]
```

### Symbolic expressions

Laws are Sage expressions. The symbols in `s` are Sage symbols that carry the domains of their variables, vector states
are Sage vectors, and mathematical functions come from Sage. Sage has no condition-based piecewise expression or
symbolic remainder, so `skcomplex.core.laws` adds `where`, `either`, `both`, and `mod`, which every backend lowers, LAMMPS
included.

```python
from skcomplex.core.base import NumericDomain, parameter, role
from skcomplex.core.base import BaseInteraction
from skcomplex.core.laws import RateLaw

from sage.all__sagemath_symbolics import exp, sqrt

from skcomplex.core.laws import where


class CrowdingInteraction(BaseInteraction):
    target: str = role()
    capacity: float = parameter(100.0, domain=NumericDomain.POSITIVE)

    def laws(self, s):
        n = s.target.n
        return [RateLaw(n, where(n > s.capacity, -sqrt(n - s.capacity), 0.1 * n * exp(-n / s.capacity)))]
```

Models written with SymPy move to Sage as follows. A SymPy expression in a law raises a `ValidationError`, and
`expression._sage_()` converts one that is already built.

| SymPy | Sage and `skcomplex` |
| --- | --- |
| `sympy.sqrt`, `sympy.exp`, `sympy.sin`, and other functions | `sqrt`, `exp`, `sin`, and others from `sage.all__sagemath_symbolics` |
| `sympy.Symbol('x', positive=True)` | `skcomplex.core.base.symbol('x', NumericDomain.POSITIVE)` |
| `sympy.Matrix([...])` and `a.dot(b)` | `skcomplex.core.base.vector([...])` and `a.dot_product(b)` |
| `sympy.Piecewise((a, condition), (b, True))` | `where(condition, a, b)` |
| `condition1 \| condition2` and `condition1 & condition2` | `either(condition1, condition2)` and `both(condition1, condition2)` |
| `sympy.Mod(x, m)` | `mod(x, m)` |
| `sympy.diff(f, x)` and `f.subs(x, y)` | `f.diff(x)` and `f.subs({x: y})` |
| `sympy.Eq(a, b)`, `.lhs`, and `.rhs` | `a == b`, `.lhs()`, and `.rhs()` |

### State domains

Besides real, positive, nonnegative, and integer values, a state takes booleans, angles, or the labels of a finite
alphabet. Values are stored as floats: booleans as 0 and 1, labels as their index, and angles wrapped into one turn
after every frame. Trajectories return booleans and labels, and `Simulation(..., checked=True)` raises an error naming
any state that leaves its domain.

```python
from skcomplex.core.base import NumericDomain, state
from skcomplex.core.components import Element

health = NumericDomain.categorical('susceptible', 'infected', 'recovered')
class PersonElement(Element):
    """Person element."""

    health = state('susceptible', domain=health)
class RotorElement(Element):
    """Rotor element."""

    theta = state(0.0, domain=NumericDomain.ANGLE)
```

### Complex states

A state or parameter of `NumericDomain.COMPLEX` takes complex values, and its laws use the imaginary unit `I`, arithmetic,
the elementary functions over the complex plane, and `real_part`, `imag_part`, `conjugate`, `abs`, and `arg`. The
compiler writes every complex law as the laws of its real and imaginary parts, which every solver, stepper, and backend
runs as a real system. A real state takes only real values, so `abs(z)` or `real_part(z)` turns a complex value into
one, and a condition compares the modulus or a part, since complex values have no order. Trajectories and frames return
complex arrays, whose `.real`, `.imag`, `np.abs`, and `np.angle` are the real views that the analysis functions take.

```python
import numpy as np
from sage.all__sagemath_symbolics import I

from skcomplex.core.base import NumericDomain, parameter, state
from skcomplex.core.components import Element
from skcomplex.core.laws import RateLaw
from skcomplex.core.system import System
from skcomplex.engine.runtime import Simulation

class PhaseElement(Element):
    """Phase element."""

    z = state(1.0, domain=NumericDomain.COMPLEX)
    omega = parameter(2.0)

    def laws(self, s):
        """Return the laws of the phase element."""
        return [RateLaw(s.z, I * s.omega * s.z)]
z = Simulation(System(elements=[PhaseElement('osc')]), dt=0.5).run(until=10).trajectory_.values('osc.z')
assert z.dtype == np.complex128 and np.allclose(np.abs(z), 1.0, atol=1e-4)
```

Agent states stay real, and the LAMMPS backend raises an error naming a complex state.

### Equations of any order

Laws give the highest time derivative of a state, and a state declares how high that is with `order`. Newton's second
law for a pendulum, `m l θ'' = -m g sin θ`, is written as it reads. The angular velocity is the companion state
`theta_d1`, which the type derives from the declaration of `theta`, and the compiler reduces the equation to first
order for the solvers.

- A companion is a constructor argument with a default of zero, so `PendulumElement('bob', theta_d1=0.5)` starts the
  pendulum with an angular velocity.
- A companion is not a class attribute. The fitted `declarations_` of a pendulum list it after its state.

```python
from skcomplex.core.base import NumericDomain, state
from skcomplex.core.components import Element

from sage.all__sagemath_symbolics import sin

from skcomplex.core.laws import EquationLaw, derivative
from skcomplex.core.declarations import parameter

class PendulumElement(Element):
    """Pendulum element."""

    theta = state(default=0.5, domain=NumericDomain.ANGLE, order=2)
    m = parameter(1.0)
    g = parameter(9.81)
    l = parameter(1.0)

    def laws(self, s):
        """Return the laws of the pendulum element."""
        return [EquationLaw(s.m * s.l * derivative(s.theta, 2) == -s.m * s.g * sin(s.theta))]
bob = PendulumElement('bob', theta_d1=0.5).prepare()
print(list(bob.declarations_), bob.theta_d1, hasattr(PendulumElement, 'theta_d1'))
```

- On a state of order `k`, rate and noise laws contribute to the `k`-th derivative and add up, so separate interactions
  supply separate forces, voltages, or growth terms: `RateLaw(s.target.x, -s.k * s.target.x / s.target.m)` from a spring
  and `RateLaw(s.target.x, -s.c * derivative(s.target.x, 1) / s.target.m)` from a damper superpose.
- `derivative(x, j)` reads the companion `x_d<j>` below the state's order. An `EquationLaw` may contain the derivative of
  the state's own order, linearly, and the compiler solves all equations of a system jointly for those derivatives,
  which couples, for example, the two angles of a double pendulum.
- Any order works the same way, such as Sprott's chaotic jerk equation `x''' = -A x'' - x' + |x| - 1` on a state of
  order 3.
- Difference equations keep past values with `lags`. `lag(x, j)` reads the value `j` steps before the current one, and
  an update law gives the next value, so `UpdateLaw(s.x, s.x + lag(s.x, 1))` with one lag is the Fibonacci recurrence.

`equations_of` returns these equations in the order they were written, and `solve_symbolically` takes the initial
values of every derivative and returns a solution only when it has a closed form.

## Composing systems

Systems nest. An interaction refers to an element of a child system by a slash-separated path.

```python
from skcomplex.core.system import System
from skcomplex.models.biology.ecology import PopulationElement, PredationInteraction

landscape = System(
    elements=[
        System('forest', components=[PopulationElement('foxes', n=5)]),
        System('meadow', components=[PopulationElement('rabbits', n=50)]),
    ],
    interactions=[PredationInteraction('hunt', predator='forest/foxes', prey='meadow/rabbits', a=0.02, b=0.01)],
)
```

States and parameters are named by their path, such as `meadow/rabbits.n` and `hunt.a`.

## Simulating

`Simulation` compiles the system, picks the solver of its form, and advances it in frames of `dt`. The default ODE
solver is `ScipyODESolver` with SciPy's `LSODA`, which starts with Adams methods and switches to BDF methods while the system
is stiff, so stiff systems also run with the default solver.

```python continuation
from skcomplex.engine.runtime import RealTimeClock, Simulation

trajectory = Simulation(landscape, dt=0.1).run(until=10).trajectory_
print(trajectory.values('meadow/rabbits.n')[-1])

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

## Delays, events, drivers, and noise

Laws can read the past, change states at discrete events, follow inputs that switch in time, and take noise in the
convention of the field.

**Delays.** `delayed(x, tau)` is the state `x` at `t - tau`, for a constant or a state-dependent delay. The solvers keep
a history of dense output and step no further than the smallest delay, by the method of steps. Before the start the
history is the initial value, or what the state declares with `state(..., history=value or function of time)`.

```python
from skcomplex.core.base import state
from skcomplex.core.components import Element
from skcomplex.engine.runtime import Simulation
from skcomplex.core.system import System

import numpy as np

from skcomplex.core.laws import EventLaw, NoiseLaw, RateLaw, delayed, step
from skcomplex.core.declarations import parameter

class MackeyGlassElement(Element):
    """Mackey glass element."""

    x = state(0.5, history=lambda t: 0.5 + 0.1 * np.sin(t))
    beta = parameter(0.2)
    gamma = parameter(0.1)
    n = parameter(10.0)
    tau = parameter(17.0)

    def laws(self, s):
        """Return the laws of the mackey glass element."""
        return [RateLaw(s.x, s.beta * delayed(s.x, s.tau) / (1 + delayed(s.x, s.tau) ** s.n) - s.gamma * s.x)]
cells = System(elements=[MackeyGlassElement('cells')])
print(Simulation(cells, dt=0.5).run(until=300).trajectory_.values('cells.x')[-1])
```

**Events.** `EventLaw(guard, reset, direction, terminal, label=None)` fires when the guard crosses zero, upward for
`direction=1`, downward for `-1`, and either way for `0`. The solver locates the crossing on its interpolant to the
precision of the root finder, applies the reset to every named state at once, and restarts from there. A terminal
event ends the run at the crossing. Every frame carries the events since the previous one, and the trajectory keeps
them all with their times and the states before and after the reset. A guard that fires again right after its reset
raises an error that names the chattering event.

```python
from skcomplex.core.base import state
from skcomplex.core.components import Element
from skcomplex.core.laws import EventLaw, RateLaw, derivative
from skcomplex.engine.runtime import Simulation
from skcomplex.core.system import System
from skcomplex.core.declarations import parameter

class BallElement(Element):
    """Ball element."""

    h = state(default=1.0, order=2)
    g = parameter(9.81)
    e = parameter(0.8)

    def laws(self, s):
        """Return the laws of the ball element."""
        return [
        RateLaw(s.h, -s.g),
        EventLaw(s.h, {derivative(s.h, 1): -s.e * derivative(s.h, 1)}, direction=-1, label='impact'),
    ]
trajectory = Simulation(System(elements=[BallElement('ball')]), dt=0.01).run(until=3).trajectory_
print([round(event.time, 6) for event in trajectory.events])
```

**Drivers.** `step`, `ramp`, `pulse`, `pulse_train`, and `series` with `'linear'` or `'previous'` interpolation are
inputs of time `s.t`. The compiler finds the times where they jump or bend, including the periodic edges of pulse
trains, and the solvers stop at every one and evaluate the inputs on the correct side of it, so a step response is
exact to the solver tolerance.

```python
from skcomplex.core.components import Element
from skcomplex.core.laws import RateLaw, step
from skcomplex.engine.runtime import Simulation
from skcomplex.core.system import System
from skcomplex.core.declarations import parameter, state

class FilterElement(Element):
    """Filter element."""

    y = state(0.0)
    k = parameter(1.0)

    def laws(self, s):
        """Return the laws of the filter element."""
        return [RateLaw(s.y, -s.k * s.y + step(s.t, 5.0))]
print(Simulation(System(elements=[FilterElement('filter')]), dt=0.5).run(until=10).trajectory_.values('filter.y')[-1])
```

**Noise conventions.** `NoiseLaw(..., convention='stratonovich')` or `'anti-ito'` declares how a noise law is read. The
compiler adds the drift correction `α g ∂g/∂x`, with `α = 1/2` and `1`, so every SDE solver integrates the equivalent
Itô equation. `EulerMaruyamaSolver` has strong order 1/2. `MilsteinSolver`, in Platen's derivative-free form, has strong order 1 for
diagonal noise, and `HeunSolver` averages the drift over every step.

```python
from skcomplex.core.components import Element
from skcomplex.core.laws import NoiseLaw, RateLaw
from skcomplex.engine.runtime import Simulation
from skcomplex.core.system import System

from skcomplex.engine.solvers import MilsteinSolver
from skcomplex.core.declarations import parameter, state

class GrowthElement(Element):
    """Growth element."""

    x = state(1.0)
    mu = parameter(0.5)
    sigma = parameter(0.8)

    def laws(self, s):
        """Return the laws of the growth element."""
        return [RateLaw(s.x, s.mu * s.x), NoiseLaw(s.x, s.sigma * s.x, convention='stratonovich')]
growth = System(elements=[GrowthElement('asset')])
print(Simulation(growth, solver=MilsteinSolver(dt=0.01), random_state=0).run(until=1).state_)
```

**Shared and correlated noise.** `NoiseLaw(..., process=...)` names the Wiener process a law draws. Without a process,
every component and every group member takes its own independent process. Laws of one component, or of one member,
that name the same label share one increment, so a reaction can drive every species it changes with one process. A
`WienerProcessSpec(label, dimension, covariance)` declares processes whose increments over a step `h` have the covariance
`h C`. It is no component: an interaction binds it through a role declared with `role(noise=True)`, and its noise laws
name process `k` with `s.<role>.dW[k]`, which every component, and every member of a group, can share. Interactions of
one system that bind processes of one label share them and their correlations. The covariance is fixed when the system
compiles, and NumPy factors it once, so the steppers scale standard normal increments drawn from the run's random
generator.

```python
import numpy as np
from skcomplex.core.components import Element
from skcomplex.core.laws import NoiseLaw
from skcomplex.engine.runtime import Simulation
from skcomplex.core.system import System

from skcomplex.core.base import role
from skcomplex.core.specs import WienerProcessSpec
from skcomplex.core.base import BaseInteraction
from skcomplex.engine.solvers import EulerMaruyamaSolver
from skcomplex.core.declarations import state

class StockElement(Element):
    """Stock element."""

    n = state(0.0)


class MarketInteraction(BaseInteraction):
    first: str = role()
    second: str = role()
    noise: WienerProcessSpec = role(noise=True)

    def laws(self, s):
        return [NoiseLaw(s.first.n, 1.0, process=s.noise.dW[0]), NoiseLaw(s.second.n, 1.0, process=s.noise.dW[1])]


processes = WienerProcessSpec('market', dimension=2, covariance=[[1.0, 0.8], [0.8, 1.0]])
market = System(
    elements=[StockElement('a'), StockElement('b')],
    interactions=[MarketInteraction('drive', first='a', second='b', noise=processes)],
)
trajectory = Simulation(market, solver=EulerMaruyamaSolver(dt=1.0), dt=1.0, random_state=0).run(until=2000).trajectory_
print(np.corrcoef(np.diff(trajectory.values('a.n')), np.diff(trajectory.values('b.n')))[0, 1])
```

The Stratonovich and anti-Itô corrections of each law use the full diffusion matrix and the covariance, so a state
that several laws drive and laws that share a process are corrected exactly. Laws that share a process declare one
convention, and validation reports laws of different conventions on one process, a Stratonovich or anti-Itô amplitude
that reads a state another component drives through its process, a covariance that is not symmetric positive
semidefinite, and a handle `dW` used anywhere but as the process of a noise law.

**Conserved noise.** `NoiseLaw(targets, value, conserved=True)` on a state of a field on a lattice adds the
divergence of a random flux, `div(value eta)`, the thermal noise of Model B and of active field theories. Every link
between neighbouring sites draws one Wiener process, and site `i` gains `value / h` times the increment of the link
from its previous neighbour and loses the same amount on the link to its next one, along every dimension of spacing
`h`. The total of the field then changes only by its rate laws, exactly to rounding. The links that leave a lattice
through a zero-flux or fixed edge carry no process. The value is a number or an expression of parameters, the same
on every link, and the field uses finite differences on a square lattice. Validation reports a conserved noise law on
any other state with the code `noise-unsupported`.

```python
from skcomplex.core.components import Element
from skcomplex.core.laws import NoiseLaw, RateLaw
from skcomplex.engine.runtime import Simulation
from skcomplex.engine.solvers import EulerMaruyamaSolver
from skcomplex.core.system import System

from sage.all__sagemath_symbolics import sqrt

from skcomplex.core.elements import FieldElement
from skcomplex.core.laws.utils import laplacian
from skcomplex.core.spaces import GeneratorNeighborhood, LatticeSpace, MooreNeighborhood, TriangularNeighborhood, VonNeumannNeighborhood
from skcomplex.core.declarations import parameter, state

class DensityElement(Element):
    """Density element."""

    rho = state(1.0)
    D = parameter(1.0)
    T = parameter(0.1)

    def laws(self, s):
        """Return the laws of the density element."""
        return [RateLaw(s.rho, s.D * laplacian(s.rho)), NoiseLaw(s.rho, sqrt(2 * s.T), conserved=True)]
gas = System(elements=[FieldElement('gas', DensityElement, LatticeSpace((16, 16)))])
density = Simulation(gas, solver=EulerMaruyamaSolver(dt=0.01), random_state=0).run(until=5).trajectory_.values('gas.rho')
print(density[-1].sum(), density[-1].std() > 0)
```

Solvers and backends declare the noise structures they draw: `'diagonal'`, one process per state, `'shared'`, a process
that drives several states, `'correlated'`, the processes of a `WienerProcessSpec` with a covariance, and `'conserved'`,
the link processes of conserved noise. `EulerMaruyamaSolver`, `HeunSolver`, and the NumPy and Numba backends support all four.
`MilsteinSolver`, whose derivative-free correction holds for diagonal noise only, and the LAMMPS backend support diagonal
noise, and validation reports a system with shared, correlated, or conserved noise for them before compiling it. The compatibility page lists the declarations. In the library,
`Binding` and `DissociationInteraction` take `langevin=True` for chemical Langevin equations, as in `chemical_langevin_network`,
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

Every draw carries a name. Every use of the same name in one component reads the same value within a step, and draws
with different names are independent. Two uses of one name with different distributions or arguments fail to
compile. Within a step, the draws are taken by component path, then draw name, then member index, so the trajectory
does not depend on the order of the laws. The NumPy and Numba backends give equal trajectories for uniform, Bernoulli,
and categorical draws.

```python
from skcomplex.core.components import Element
from skcomplex.engine.runtime import Simulation
from skcomplex.core.system import System

from skcomplex.core.laws import UpdateLaw, bernoulli, normal, where
from skcomplex.core.declarations import parameter, state
from skcomplex.core.elements import GroupElement

class PersistentElement(Element):
    """Persistent element."""

    d = state(1.0)
    x = state(0.0)
    q = parameter(0.8)

    def laws(self, s):
        """Return the laws of the persistent element."""
        return [
        UpdateLaw(s.d, where(bernoulli('turn', 1 - s.q) > 0, -s.d, s.d)),
        UpdateLaw(s.x, s.x + where(bernoulli('turn', 1 - s.q) > 0, -s.d, s.d) + normal('jitter', 0, 0.1)),
    ]
walkers = System(elements=[GroupElement('walkers', element=PersistentElement, size=500)])
final = Simulation(walkers, dt=1.0, random_state=0).run(until=200).trajectory_.values('walkers.x')[-1]
print(round(float(final.var() / 200), 1))
```

Only update laws and renewal laws take draws. A draw in a rate, noise, jump, boundary, equation, or event law raises
`IncompatibleLawsError`, since noise laws and jump laws give randomness in continuous time. Draws are not states, so
the trajectory does not record them, and a model that wants to keep a draw assigns it to a state.

### Markov chains and hidden Markov sources

`TransitionLaw(targets, matrix)` is the step of a discrete-time Markov chain. The target is a state over a finite alphabet,
a categorical domain or the integers from 0 to `n - 1`, and its current value selects the row of the transition matrix
that the next value is drawn from. `EmissionLaw(targets, source, matrix)` draws an observed state from the row of an
emission matrix that the hidden state `source` selects, reading the hidden state after the same step, so the two
sequences align frame by frame. Both are update laws, so they combine with other update laws.

Matrix entries are numbers or expressions of parameters. Compiling checks that the entries are nonnegative and that
every row sums to 1, and `update_parameters` checks again every matrix a changed parameter enters, rejecting the change
with a `ParameterError` before it applies.

```python
from skcomplex.core.base import NumericDomain, state
from skcomplex.core.components import Element
from skcomplex.engine.runtime import Simulation
from skcomplex.core.system import System

from skcomplex.core.laws import EmissionLaw, TransitionLaw
from skcomplex.core.declarations import parameter

class SourceElement(Element):
    """Source element."""

    hidden = state(default='calm', domain=NumericDomain.categorical('calm', 'storm'))
    symbol = state(default='a', domain=NumericDomain.categorical('a', 'b'))
    stay = parameter(0.9)

    def laws(self, s):
        """Return the laws of the source element."""
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

Jumps on clocks of waiting times and fractional noise laws write processes with memory: events that fire after waiting times from any
distribution, and noise whose increments are correlated over the whole run.

### Draws from SciPy distributions

`draw(name, distribution, **parameters)` draws from any distribution of `scipy.stats`, such as `'pareto'`, `'lomax'`,
`'expon'`, `'beta'`, or `'levy_stable'`, with its shape parameters, `loc`, and `scale` as keywords. The keywords can be
expressions of states and parameters. It follows the rules of the other named draws, and it samples with the
distribution's `rvs` and the simulation's random generator. Compiling rejects a name `scipy.stats` does not provide,
listing the closest names, and keywords the distribution does not take. In update laws, a SciPy draw is sampled for
every member before each step, and the map then steps in Python with the backend's compiled update.

```python
from skcomplex.core.components import Element
from skcomplex.core.laws import UpdateLaw
from skcomplex.engine.runtime import Simulation
from skcomplex.core.system import System

from skcomplex.core.laws import draw
from skcomplex.core.declarations import state
from skcomplex.core.elements import GroupElement

class FlierElement(Element):
    """Flier element."""

    x = state(0.0)

    def laws(self, s):
        """Return the laws of the flier element."""
        return [UpdateLaw(s.x, s.x + draw('jump', 'levy_stable', alpha=1.5, beta=0.0))]
flight = Simulation(System(elements=[GroupElement('fliers', element=FlierElement, size=100)]), dt=1.0, random_state=0).run(until=20)
print(flight.trajectory_.values('fliers.x').shape)
```

### Jumps on clocks of waiting times

`JumpLaw(waiting=..., reset=..., label=...)` fires after a waiting time, applies its reset to the states all at once, and draws the
next waiting time from the states after the reset. Every member of a group renews on its own clock. Draws in the
waiting time and the reset take a fresh value at every renewal of every member, and every use of one name in a
renewal reads the same value. A waiting time that is a single draw must come from a distribution with nonnegative
support, and compiling rejects one that can be negative, such as `normal`.

A system or partition whose only laws are such jumps compiles to the renewal form, which `RenewalSolver` runs
exactly, from renewal to renewal with no time grid. A frame holds the states the last renewal at or before its time
set. These jumps also combine with the rate and noise laws of the same component, which evolve the states between
renewals. `RK4Solver` and `EulerMaruyamaSolver` stop every member at its own renewals when the members do not interact, and split
the step at every renewal time for all members when a law sums over the group. Other steppers reject these jumps.

```python
from skcomplex.core.components import Element
from skcomplex.core.laws import draw, normal
from skcomplex.engine.runtime import Simulation
from skcomplex.core.system import System

from skcomplex.core.laws import JumpLaw
from skcomplex.core.declarations import parameter, state
from skcomplex.core.elements import GroupElement

class WalkerElement(Element):
    """Walker element."""

    x = state(0.0)
    alpha = parameter(0.5)

    def laws(self, s):
        """Return the laws of the walker element."""
        return [JumpLaw(waiting=draw('wait', 'pareto', b=s.alpha), reset={s.x: s.x + normal('jump', 0, 1)}, label='jumps')]
walk = Simulation(System(elements=[GroupElement('walkers', element=WalkerElement, size=1000)]), dt=100.0, random_state=0).run(until=1000)
print(walk.problem_.form.value, walk.trajectory_.values('jumps.renewals')[-1] > 1000)
```

The trajectory records every renewal pass as a `RenewalRecord` among its events, with the time and the member of every
renewal and the member's states before and after it. The observable `<label>.renewals` counts the renewals up to every
frame, and the Rerun observer streams it as a time series and the renewals as text entries under `renewals/<label>`.
The compiler keeps the count of every member in a hidden state, `<group>.<label>~renewals`.

### Fractional noise

`FractionalNoiseLaw(targets, value, hurst=hurst)` drives a state with fractional Brownian motion of Hurst index `0 < H < 1`,
whose increments anticorrelate for `H < 1/2` and correlate for `H > 1/2`. A Hurst index of 1/2 compiles to an ordinary
noise law. For other indices the amplitude must not depend on any state. `FractionalEulerMaruyamaSolver(dt, horizon)` is the
stepper of systems with fractional noise. It draws the increments of the whole run on its grid when it first advances,
with the circulant embedding of Davies and Harte over `scipy.fft`, so frames must fall on multiples of `dt`. A
simulation passes the end of its run as the horizon, advancing beyond it raises `HorizonError`, and other SDE steppers
reject fractional noise with an error that names `FractionalEulerMaruyamaSolver`. `fractional_gaussian_noise` returns the
noise paths themselves.

```python
from skcomplex.core.components import Element
from skcomplex.engine.runtime import Simulation
from skcomplex.core.system import System

from skcomplex.engine.solvers import FractionalEulerMaruyamaSolver
from skcomplex.core.laws import FractionalNoiseLaw
from skcomplex.core.declarations import state
from skcomplex.core.elements import GroupElement

class RoughElement(Element):
    """Rough element."""

    x = state(0.0)

    def laws(self, s):
        """Return the laws of the rough element."""
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

A state declared with a fractional order, `state(1.0, order=0.6)`, has a Caputo derivative of that order. The rate
laws on it give that derivative, `D^alpha x = f(t, x)`, and superpose as for any state, so no other law kind is needed.
Order 1 is the ordinary rate law. The derivative of order `alpha` at a time depends on the whole past of the state, so
the solution relaxes like a Mittag-Leffler function, slower than any exponential.

A system with a state of a fractional order compiles to the fractional form, with the order of every state component
in `problem.orders`. The states of order 1 of the same system keep their ordinary equations. Three steppers integrate
the fractional form on a fixed grid from the start of the run, so frames must fall on multiples of their `dt`:

- `GrunwaldLetnikovSolver(dt, memory=None)` weighs the past states with the binomial weights `(-1)^j binomial(alpha, j)`.
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
from skcomplex.core.components import Element
from skcomplex.core.laws import RateLaw
from skcomplex.engine.runtime import Simulation
from skcomplex.core.system import System

from skcomplex.analysis import mittag_leffler
from skcomplex.core.base import state
from skcomplex.engine.solvers import L1Solver

class RelaxingElement(Element):
    """Relaxing element."""

    x = state(1.0, order=0.6)

    def laws(self, s):
        """Return the laws of the relaxing element."""
        return [RateLaw(s.x, -s.x)]
relaxing = Simulation(System(elements=[RelaxingElement('a')]), solver=L1Solver(dt=0.01, memory=10.0), dt=1.0).run(until=20)
exact = mittag_leffler(0.6, 1.0, -(20.0**0.6))
print(relaxing.problem_.form.value, abs(float(relaxing.trajectory_.values('a.x')[-1]) - float(exact)) < 1e-3)
```

Validation reports the misuses of fractional orders before compiling, each with the path of the state and a fix: an
order that is neither an integer nor in `(0, 1)` (`fractional-order`), a fractional state in a system whose declared
time is not continuous (`fractional-time`), a chosen solver or backend that does not integrate fractional orders
(`order-unsupported`), and a law on a fractional state that no fractional solver runs, such as a noise law
(`solver-unsupported`). Solvers and backends declare the kinds of order they integrate in the `orders` field of their
`SupportSpec`, `'integer'` or `'fractional'`. The NumPy and Numba backends run the fractional form.

`mittag_leffler(alpha, beta, z)` evaluates the two-parameter Mittag-Leffler function. It sums the power series in
arbitrary precision with Sage for moderate arguments and takes the asymptotic expansion for large ones.
`solve_symbolically` solves one linear fractional relaxation `D^alpha x = c x` as `x(0) E_alpha(c t^alpha)`, written
with the symbolic function `mittag_leffler`, and `equations_of` writes a Caputo derivative as `caputo(x(t), t, alpha)`.

The order belongs to the element type, so a sweep over orders builds one system per order. The library has these
models ready in fractional dynamics, each with the order as an argument: `fractional_relaxation`,
`fractional_oscillator`, `fractional_van_der_pol`, and `fractional_lorenz`.

## Observables

An `Observable` is a frozen record of a label and an expression or a group reduction, which a run computes at every
frame, records in trajectories under the system's path, and logs to Rerun. An observable is either an expression of the system namespace, where earlier observables
appear by label, or a reduction over a group's members of a variable or an expression of one member: `count`, `sum`,
`mean`, `min`, `max`, `var`, or a function. Expressions and reductions given as functions must be module-level
functions, so that systems pickle, and a lambda raises an error.

```python
from skcomplex.core.base import NumericDomain, state
from skcomplex.core.components import Element
from skcomplex.core.laws import RateLaw
from skcomplex.engine.runtime import Simulation
from skcomplex.core.system import System

import numpy as np
from sage.all__sagemath_symbolics import I, exp

from skcomplex.core.specs import ObservableSpec
from skcomplex.core.declarations import parameter
from skcomplex.core.elements import GroupElement

class OscillatorElement(Element):
    """Oscillator element."""

    theta = state(0.0, domain=NumericDomain.ANGLE)
    omega = parameter(1.0)

    def laws(self, s):
        """Return the laws of the oscillator element."""
        return [RateLaw(s.theta, s.omega)]


def phase_vector(member):
    return exp(I * member.theta)


def order_parameter(s):
    return abs(s.z)


oscillators = System(
    elements=[GroupElement('phases', element=OscillatorElement, size=100, values={'theta': np.linspace(0, 1, 100), 'omega': np.linspace(0.9, 1.1, 100)})],
)
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
import numpy as np
from skcomplex.core.base import BaseInteraction
from skcomplex.core.laws import RateLaw
from skcomplex.engine.runtime import Simulation
from skcomplex.core.specs import ObservableSpec
from skcomplex.core.system import System

from sage.all__sagemath_symbolics import cos, sin

class MeanFieldInteraction(BaseInteraction):
    target = role()
    X = role()
    Y = role()
    K = parameter(1.0)

    def laws(self, s):
        return [RateLaw(s.target.theta, s.K * (s.Y.value * cos(s.target.theta) - s.X.value * sin(s.target.theta)))]


def cosine(member):
    return cos(member.theta)


def sine(member):
    return sin(member.theta)


coupled = System(
    elements=[GroupElement(
            'phases',
            element=OscillatorElement,
            size=100,
            values={'theta': np.linspace(0, 6, 100), 'omega': np.linspace(0.9, 1.1, 100)},
        )],
    interactions=[MeanFieldInteraction('coupling', target='phases', X='X', Y='Y', K=2.0)],
)
observables = [ObservableSpec('X', cosine, group='phases'), ObservableSpec('Y', sine, group='phases')]
print(Simulation(coupled, dt=1.0, observables=observables).run(until=20).trajectory_.values('X')[-1].round(2))
```

## Stocks and flows

A state tagged with a quantity, `state(..., quantity='water')`, is a stock that changes only through flows. A
`TransferLaw(source, sink, amount, label=...)` moves an amount of the quantity from one stock to another, or exchanges it with
the environment through `Source()` and `Sink()`. In a system of rate laws the amount is a rate, in a system of update
laws it is moved once per step, and with `rate=...` it is the size of a jump with that propensity. Trajectories record
every flow under `<component path>.<label>`. An intermediate expression, such as a birth rate, is a function of the
namespace that the laws call, and `observe_element(element, label, function)` records the same function as the
observable `<element>.<label>`, which Rerun logs under the element. `lookup(x, points)` is a table function, piecewise
linear between its points and flat outside them.

```python
from skcomplex.core.base import state
from skcomplex.core.components import Element
from skcomplex.engine.runtime import Simulation
from skcomplex.core.system import System

from skcomplex.core.laws import Sink, Source, TransferLaw, lookup
from skcomplex.core.specs import ObservableSpec
from skcomplex.core.declarations import parameter


def birth_rate(s):
    return s.b * s.n * lookup(s.n, [(0, 1.0), (100, 1.0), (200, 0.0)])


def population_laws(s):
    return [TransferLaw(Source(), s.n, birth_rate(s), label='births'), TransferLaw(s.n, Sink(), s.d * s.n, label='deaths')]


class PeopleElement(Element):
    """People element."""

    n = state(10.0, quantity='people')
    b = parameter(0.03)
    d = parameter(0.01)

    def laws(self, s):
        """Return the laws of the people element."""
        return population_laws(s)
crowd = System(elements=[PeopleElement('crowd')])
observed = [observe_element('crowd', 'birth_rate', birth_rate)]
trajectory = Simulation(crowd, dt=1.0, observables=observed).run(until=10).trajectory_
print(round(float(trajectory.values('crowd.births')[-1]), 3), round(float(trajectory.values('crowd.deaths')[-1]), 3))
print(round(float(trajectory.values('crowd.birth_rate')[-1]), 3))
```

A rate or update law outside flows that changes a tagged state raises `ConservationError` when the system compiles.
`conservation(system)` returns the net inflow of every quantity from the environment, after checking that the internal
flows cancel. The compiler accumulates that inflow in the hidden state `__boundary.<quantity>`, and the observer
`ConservationObserver` checks at every frame that each total changed only by it. `flow_matrix(trajectory, frame)` is
the transaction-flow matrix of a frame, one row per flow and one column per holder, whose rows sum to zero.

```python continuation
from skcomplex.engine.runtime import Simulation

from skcomplex.analysis import conservation, flow_matrix
from skcomplex.engine.runtime import ConservationObserver

print(conservation(crowd))
monitored = Simulation(crowd, dt=1.0, observers=[ConservationObserver()]).run(until=10).trajectory_
print(flow_matrix(monitored, -1).values.sum(axis=1))
```

`distributed_delay(value, time, order, conserved=True)` is a material delay of any order, hidden stocks in series whose
outflow it returns, and with `conserved=False` an information smoothing. It adds hidden stocks
`<component path>.<kind>_delay<order>_<index>` to rate laws of single elements and interactions, starting in
equilibrium with their input. The system-dynamics names `delay1`, `delay3`, `smooth`, and `smooth3` are its
realizations in `skcomplex.models.engineering.processes`.

`loops(system)` derives the signed dependency graph of states and flows from the laws, and lists its
feedback loops with their polarity, and `loop_dominance(trajectory)` ranks the loops by their gains at every frame.

```python continuation
from skcomplex.analysis import loop_dominance, loops

for loop in loops(crowd).loops:
    print(loop.polarity, ' -> '.join(loop.nodes))
dominance = loop_dominance(trajectory)
print(dominance.loops[int(dominance.dominant[-1])].polarity)
```

Agents move conserved quantities between each other inside rules. `transfer(source, sink, amount, state)` moves an
amount between two agents at once, so a negative holding is an amount owed and total holdings stay constant. A run
checks that an expression keeps its first value with `Simulation(invariants={name: tolerance})`, which names an
observable of the run. At every frame the run compares the observable with its value at the first frame and raises
`ConservationError` naming the observable and the first frame that breaks it. Removing an agent that still holds a
tagged state raises `ConservationError` until its holdings are settled with transfers.

```python
from skcomplex.core.base import state
from skcomplex.engine.runtime import Simulation
from skcomplex.core.specs import ObservableSpec
from skcomplex.core.system import System

from skcomplex.core.components import Element
from skcomplex.core.laws import RuleLaw
from skcomplex.core.laws import transfer
from skcomplex.core.elements import GroupElement


def net_worth(agent):
    return agent.cash + agent.claims - agent.equity


class TraderElement(Element):
    cash: float = state(10.0, quantity='money')
    claims: float = state(0.0, quantity='claims')
    equity: float = state(10.0)

    def laws(self, s):
        return [RuleLaw(self.lend)]

    def lend(self, agent, model):
        borrower = model.agents('traders')[0]
        if borrower != agent:
            transfer(agent, borrower, 1.0, 'cash')
            transfer(borrower, agent, 1.0, 'claims')


balance = ObservableSpec('net_worth', net_worth, group='traders', reduce='sum')
market = System(elements=[GroupElement('traders', element=TraderElement, size=3)])
traders = Simulation(market, observables=[balance], invariants={'net_worth': 1e-9}).run(until=3).trajectory_
print(traders.agents('traders', 3).values['cash'].sum())
```

The library models built on flows are `bathtub`, `rice_milling`, `solid_waste`, `food_relief`, and `irrigation` in
engineering processes, `car_dealer`, `commodity_cycle`, and `beer_game` in markets, `bass_diffusion` in innovation,
`stock_flow_consistent_economy` in macroeconomics, and `success_to_the_successful`, `shifting_the_burden`, and
`escalation` in organizations.

## Control

A control input is an adaptive parameter with bounds, `parameter(initial, adaptive=True, bounds=(lower, upper))`. Laws
read it like a parameter, trajectories record it like a state, and a schedule, an optimal control solver, or a
feedback controller supplies its value. Without one it keeps its initial value. `validate` reports an initial value
outside the bounds as an error at the parameter, whose fix names the bounds.

```python
from skcomplex.core.base import parameter
from skcomplex.core.components import Element
from skcomplex.core.laws import RateLaw
from skcomplex.engine.runtime import Simulation
from skcomplex.core.system import System

from skcomplex.engine.compile import validate
from skcomplex.core.declarations import state

class HeaterElement(Element):
    """Heater element."""

    T = state(0.0)
    power = parameter(0.0, adaptive=True, bounds=(0, 1))

    def laws(self, s):
        """Return the laws of the heater element."""
        return [RateLaw(s.T, s.power - s.T)]
room = System(elements=[HeaterElement('heater')])
print([issue.path for issue in validate(System(elements=[HeaterElement('heater', power=2.0)])).errors])
schedule = {'heater.power': lambda t: 1.0 if t >= 1.0 else 0.0}
heated = Simulation(room, dt=0.5, schedule=schedule).run(until=3).trajectory_
print(heated.values('heater.power'))
```

### Objectives and optimal control

An objective defines a control problem, not a part of a system, so the optimal control solvers take it as arguments.
`objective` is the running cost, and `terminal` the terminal cost at the horizon, the time `until` of the run. Each is
a function of the namespace `s` of the states, parameters, and time of the system by dotted name, such as
`lambda s: s.plant.x**2`, or a number. The value of a run is the integral of the running cost over the run plus the
terminal cost, and `sense` is `'min'` to minimise it or `'max'` to maximise it.

Two solvers optimise the objective over the bounded adaptive parameters of the system. `PontryaginSolver(objective)`
builds the Hamiltonian `H = L + λ · f` of the running cost `L` and the rates `f`, derives the costate equations
`λ' = -∂H/∂x` and the stationarity condition `∂H/∂u = 0` symbolically, projects the controls onto their bounds, takes
a bound where `H` is linear in a control, and solves the two-point problem by a forward-backward sweep.
`CollocationSolver(objective, nodes=..., scheme=...)` transcribes the problem with the `'hermite-simpson'` or
`'trapezoidal'` scheme and solves it with SciPy's `minimize`, with the Jacobians derived symbolically. Collocation
handles bang-bang controls, singular arcs, and saddle paths on which the sweep diverges. Both solve the problem over
the whole run when it starts, and `simulation.stepper_.solution_` holds the solution on its grid, its cost, and, for
`PontryaginSolver`, the costates.

```python
from skcomplex.core.base import parameter
from skcomplex.core.components import Element
from skcomplex.core.laws import RateLaw
from skcomplex.engine.runtime import Simulation
from skcomplex.core.system import System

from skcomplex.engine.solvers import CollocationSolver, PontryaginSolver
from skcomplex.core.declarations import state

class PlantElement(Element):
    """Plant element."""

    x = state(1.0)
    u = parameter(0.0, adaptive=True, bounds=(-10, 10))

    def laws(self, s):
        """Return the laws of the plant element."""
        return [RateLaw(s.x, s.u)]
plant = System(elements=[PlantElement('plant')])


def cost(s):
    return (s.plant.x**2 + s.plant.u**2) / 2


optimal = Simulation(plant, solver=PontryaginSolver(cost), dt=0.1).run(until=1.0)
print(round(optimal.trajectory_.values('plant.u')[0], 3), round(optimal.stepper_.solution_.cost, 4))
transcribed = Simulation(plant, solver=CollocationSolver(cost, nodes=21), dt=0.1).run(until=1.0)
print(round(transcribed.trajectory_.values('plant.u')[0], 3))
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

Feedback controllers are realizations of the engineering field, in `skcomplex.models.engineering.realizations`, so the
core holds none. `PIDInteraction(label, plant=, integrator=, control=, measure=, kp=, ki=, kd=, setpoint=)` sets the control
input `control` of the plant to `kp e + ki ∫ e dt - kd y'` from the error `e = setpoint - y` of its state `measure`,
with the integral of the error in an `IntegratorElement` element. `StateFeedbackInteraction(label, plant=, control=, states=, gain=)` sets
it to `-K x`. Both set the control after every solver step, as boundary laws, so a fixed-step solver with small steps
makes them close to continuous controllers. The engineering models `pid_inverted_pendulum` and `lqr_oscillator` use
them.

### Linear analysis

`linearize(system, controls, outputs, at)` returns the state-space matrices `A`, `B`, `C`, and `D` of a system under
continuous time near an operating point, from its symbolic Jacobians, with the controls named by their adaptive
parameters. `controllability(A, B)` and `observability(A, C)` return the ranks of the Kalman matrices, and
`lqr(A, B, Q, R)` returns the gain of the linear-quadratic regulator from SciPy's solution of the Riccati equation.

```python
from skcomplex.core.base import parameter
from skcomplex.core.components import Element
from skcomplex.core.laws import RateLaw
from skcomplex.core.system import System

from skcomplex.analysis import controllability, linearize, lqr, observability
from skcomplex.core.declarations import state

class CartElement(Element):
    """Cart element."""

    x = state(0.0)
    v = state(0.0)
    force = parameter(0.0, adaptive=True, bounds=(-1, 1))

    def laws(self, s):
        """Return the laws of the cart element."""
        return [RateLaw(s.x, s.v), RateLaw(s.v, s.force)]
model = linearize(System(elements=[CartElement('cart')]), outputs=['cart.x'])
print(controllability(model.A, model.B), observability(model.A, model.C))
print(lqr(model.A, model.B, [[1, 0], [0, 1]], [[1]]).round(3))
```

The library models with optimal controls are `optimal_harvesting` in resources, `optimal_consumption` in
macroeconomics, `sir_vaccination` in epidemiology, and `optimal_chemotherapy` and `optimal_immunotherapy` in
oncology.

## Package layout

The package holds two parts. `skcomplex.core` is the domain-agnostic framework, and `skcomplex.models` is the library
of ready-made models built from it.

### The shared core

The shared core holds the general concepts, one package for each part of a system or step of a run. The packages are
listed in their import order, so a package imports only the packages above it.

- `base`: symbols, domains, the declarations `state`, `parameter`, and `role`, errors, the bases of elements,
  interactions, and laws, and the open registries of law kinds, forms, space kinds, and traits.
- `space`: `BaseSpace` and its general kinds, the continuous, lattice, and network spaces, neighbourhoods, and time.
- `laws`: the `BaseLaw` base, the general laws, conditions such as `where`, random draws, reductions, and stocks and
  flows.
- `components`: `BaseElement`, `Element`, `GroupElement`, and `WienerProcessSpec`, which is no component.
- `interactions`: `BaseInteraction` and the bindings of its roles.
- `system`: `System`, `Observable`, `observe_element`, and ensembles.
- `compile`: validation and the pipeline of passes that turns a system into a problem.
- `solvers`: the stepper contract, `step_count`, and the general steppers by form.
- `backends`: the NumPy and Numba backends.
- `runtime`: `Simulation`, clocks, observers, and the registry of models.
- `viz`: the visuals and the Rerun viewer.
- `experiments`: experiments over grids of parameters and replicates.
- `analysis`: analyses of systems, trajectories, and data.
- `formalisms`: one package for each formalism, which may import every package above it.

No package of the shared core imports a formalism, so a system built only from the shared core compiles and simulates
with no formalism loaded. A test of the package layout enforces the import order.

## Formalisms

A formalism is one kind of dynamics that not every model needs, such as lattice dynamics or quantum dynamics. Each
formalism is a package of `skcomplex.formalisms` that holds its laws, components, compilation passes, solvers,
backends, visuals, and validation checks, so one kind of dynamics lives in one place.

### The formalisms

- `agents`: `RuleLaw` and `MatchLaw`, the phases of a step, agent memory, and the Mesa stepper.
- `lattices`: `LocalRuleLaw`, `Energy`, drives, relaxations, extremal updates, Monte Carlo sampling, and the lattice
  steppers.
- `groups`: finitely generated groups, their Cayley graphs as spaces of cellular automata, and the visual of those
  graphs.
- `rewriting`: productions on words, substitutions of tiles, graph productions, and their geometry.
- `quantum`: operators, `GeneratorLaw` and `DissipatorLaw`s, quantum elements, and the QuTiP solvers.
- `fields`: `FieldElement`, differential operators, `FluxLaw`s, meshes, discretizations, and stiff integrators.
- `kinetics`: encounter laws, moments, activity levels, and direct simulation Monte Carlo.
- `fractional`: states of fractional order, Caputo schemes, and fractional noise.
- `topology`: requests of laws that add and remove the nodes and edges of networks.
- `particles`: `ParticleElement`, potentials, pair potentials, and the LAMMPS backend.
- `control`: the optimal control solvers by collocation and by Pontryagin's principle.

`fields` builds on `lattices`, and the other formalisms build on the shared core alone.

### Importing a formalism

A name specific to one formalism is imported from the package of that formalism, as
`from skcomplex.formalisms.<formalism> import <Name>`. The shared packages do not export it.

- Importing any name of a formalism registers the formalism, with every law, solver, and backend it contributes.
- A system that holds a law of a class that no formalism registered fails validation with an issue that names the
  class.
- `registered_formalisms()` lists the registered formalisms in registration order.

```python
from skcomplex.core.laws import GeneratorLaw

quantum = next(item for item in registered_formalisms() if item.name == 'quantum')
print([law.__name__ for law in quantum.laws], sorted(quantum.solvers))
```

### Writing a formalism

A new formalism needs no change to the shared core. It declares its contributions in one `Formalism` and passes it to
`register` when its package is imported.

- A new law kind is a `LawKind`, and a new form is a `Form`. Both are registered when they are created.
- A law class derives from `BaseLaw` and declares its `law_kind`, `time_class`, `traits`, and `requires`.
- A `BaseSystemCompiler` claims the systems it compiles, setting `claims_` when it is fitted to a system, and its `run`
  turns each into a `Problem` of the new form.
- A solver is a `Solver` that declares the form it advances and what it supports. It implements `_advance`, which
  advances the state from one time to the next. The runtime calls `prepare(problem, rng, until)` before the first frame
  and `run(t, x, p, until)` at every frame, which returns the state and sets `events_`, `stopped_at_`, `snapshot_`,
  and `agents_`. The last run, `run(..., final=True)`, releases what fitting acquired, such as an external process.
- `depends` names the formalisms it builds on, which must be registered first.

The example below adds radioactive decay as a formalism with its own law, pass, and exact solver.

```python
from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from skcomplex.core.base import (
    ElementKind,
    NumericDomain,
    DomainKind,
    Expression,
    Form,
    LawClass,
    LawKind,
    OrderKind,
    RequirementSpec,
    SupportSpec,
    TimeKind,
    Trait,
    VisualSpec,
    parameter,
    state,
    symbol,
)
from skcomplex.engine.compile import BaseSystemCompiler
from skcomplex.engine.problems import CompiledElement, Problem
from skcomplex.engine.problems import Entry, Layout
from skcomplex.core.components import Element
from skcomplex.core.base import BaseLaw
from skcomplex.engine.runtime import Simulation
from skcomplex.engine.solvers import Solver
from skcomplex.core.system import System

DECAY = LawKind.register('DECAY', 'decay')
EXPONENTIAL_DECAY = Form.register('EXPONENTIAL_DECAY', 'exponential decay')


@dataclass(frozen=True)
class DecayLaw(BaseLaw):
    law_kind: ClassVar[LawKind] = DECAY
    time_class: ClassVar[LawClass] = LawClass.CONTINUOUS
    traits: ClassVar[frozenset[Trait]] = frozenset({Trait.DECLARABLE})
    requires: ClassVar[RequirementSpec] = RequirementSpec()

    targets: Expression | tuple[Expression, ...]
    rate: object


class NucleusElement(Element):
    n: float = state(1.0, domain=NumericDomain.NONNEGATIVE)
    k: float = parameter(1.0, domain=NumericDomain.POSITIVE)

    def laws(self, s):
        return [DecayLaw(s.n, s.k)]


@dataclass(frozen=True)
class DecayProblem(Problem):
    rates: tuple[int, ...] = ()


class DecayCompiler(BaseSystemCompiler):
    def _prepare(self, system):
        self.claims_ = bool(system.elements) and all(isinstance(item, NucleusElement) for item in system.elements)

    def _run(self, system, *, backend, solver, initial_state):
        nuclei = list(system.elements)
        states = [Entry(f'{item.label}.n', i, (), NumericDomain.NONNEGATIVE) for i, item in enumerate(nuclei)]
        rates = [Entry(f'{item.label}.k', i, (), NumericDomain.POSITIVE) for i, item in enumerate(nuclei)]
        return DecayProblem(
            form=EXPONENTIAL_DECAY,
            states=Layout(states),
            parameters=Layout(rates),
            initial_state=np.array([float(initial_state.get(f'{item.label}.n', item.n)) for item in nuclei]),
            initial_parameters=np.array([float(item.k) for item in nuclei]),
            time=symbol('t'),
            symbols={},
            equations={},
            equation_symbols={},
            has_callables=False,
            has_group_sums=False,
            elements=tuple(
                CompiledElement(item.label, item, {'n': f'{item.label}.n'}, {'k': f'{item.label}.k'}, VisualSpec())
                for item in nuclei
            ),
            system=system,
            rates=tuple(entry.start for entry in rates),
        )


class ExactDecaySolver(Solver):
    supports: ClassVar[SupportSpec] = SupportSpec(
        time={TimeKind.CONTINUOUS},
        laws={DECAY},
        domains={DomainKind.REAL},
        components={ElementKind.ELEMENT},
        orders={OrderKind.INTEGER},
    )
    form: ClassVar[Form] = EXPONENTIAL_DECAY

    def _advance(self, t, x, p, until):
        return x * np.exp(-p[list(self.problem_.rates)] * (until - t))


register(
    Formalism(
        'decay',
        laws=(DecayLaw,),
        components=(NucleusElement,),
        compilers=(DecayCompiler(),),
        forms=(EXPONENTIAL_DECAY,),
        solvers={'exact-decay': ExactDecaySolver},
        defaults=(ExactDecaySolver,),
    ),
)
decayed = Simulation(System(elements=[NucleusElement('atom', n=2.0, k=0.5)]), dt=0.5).run(until=2)
print(type(decayed.stepper_).__name__, decayed.trajectory_.values('atom.n')[-1].round(4))
```

The formalism packages of `skcomplex.formalisms` follow one internal layout, with the modules `_laws`,
`_components`, `_spaces`, `_passes`, `_solvers`, `_backends`, `_visuals`, and `_checks`, each present only when the
formalism needs it, and an `__init__` that registers the formalism.

### What stays in the shared core

Some parts look specific to one kind of model but stay in the shared core, because general laws or several formalisms
need them.

- **Renewals** stay because `JumpLaw` with a clock of waiting times is a general law. Its lowering and its stepper belong
  with the other jump processes, so a waiting-time jump never compiles as a rate jump when a formalism is missing.
- **Stocks and flows** stay because their lowering also lowers the general law `TransferLaw`, and `analysis` reads their
  feedback loops.
- **The `'pairs'` binding** stays because models without particles couple the pairs of members of a group closer than
  a cutoff.
- **The bodies visual** stays because models without particles draw their members as bodies.

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
and `run` simulates it with its defaults.

```python
from skcomplex.models import catalog, create, describe, run

print([entry.name for entry in catalog(field='biology/ecology')])
print(sorted({entry.field for entry in catalog(field='chemistry')}))
print(describe('lotka_volterra').formalism)
predation = create('lotka_volterra', a=0.2)
print(run('sir').trajectory_.values('I.n').max().round())
```

### Models as patterns

The models are examples of the core interface, not a separate one, so each is also the pattern for a model of your
own. The page of every field, under Models in the documentation, shows each model's factory, its component types with
their states, parameters, and laws as written, and, for differential equations and maps, its equations as
`equations_of` returns them. A model of your own follows the same steps: declare the component types, write a factory
that returns the system, and register it with `register_model`.

```python
from skcomplex.core.base import NumericDomain, state
from skcomplex.core.components import Element
from skcomplex.core.laws import RateLaw
from skcomplex.core.system import System
from skcomplex.engine.runtime import create, register_model
from skcomplex.core.declarations import parameter

class TankElement(Element):
    """Tank element."""

    level = state(default=1.0, domain=NumericDomain.NONNEGATIVE)
    k = parameter(0.5)

    def laws(self, s):
        """Return the laws of the tank element."""
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
from skcomplex.engine.runtime import Simulation
from skcomplex.core.system import System

import numpy as np

from skcomplex.core.elements import ParticleElement
from skcomplex.core.spaces import ContinuousSpace, GeneratorNeighborhood, MooreNeighborhood, TriangularNeighborhood, VonNeumannNeighborhood
from skcomplex.core.elements import GroupElement

rng = np.random.default_rng(0)
gas = System(
    elements=[GroupElement('gas', element=ParticleElement, size=1_000, values={'x': rng.uniform(0, 1, (1_000, 2)), 'v': rng.normal(0, 1, (1_000, 2))})],
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

`total`, `mean`, `maximum`, `minimum`, `variance`, and `count` reduce an expression over the members of a group inside
any law. Inside the reduction, the symbols of a member stand for the reduced member, and outside it for the reading
member, so `RateLaw(s.x, -s.x + tanh(total(s.x)))` reads the group total in every member's own law. A reduction is
evaluated once per evaluation, before the member laws, at a cost linear in the size of the group, on NumPy and Numba.

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
import numpy as np
from skcomplex.core.components import Element
from skcomplex.core.base import BaseInteraction
from skcomplex.core.laws import RateLaw
from skcomplex.engine.runtime import Simulation
from skcomplex.core.system import System

from sage.all__sagemath_symbolics import tanh

from skcomplex.core.laws import total
from skcomplex.core.declarations import role, state
from skcomplex.core.elements import GroupElement

class UnitElement(Element):
    """Unit element."""

    x = state(0.0)

    def laws(self, s):
        """Return the laws of the unit element."""
        return [RateLaw(s.x, -s.x + tanh(total(s.x)))]
units = System(elements=[GroupElement('units', element=UnitElement, size=50, values={'x': np.linspace(-1, 1, 50)})])
print(Simulation(units, dt=0.5).run(until=2).trajectory_.values('units.x')[-1, :3].round(3))

payoff = np.array([[-0.5, 2.0], [0.0, 1.0]])
class StrategyElement(Element):
    """Strategy element."""

    x = state(0.5)
class ReplicationInteraction(BaseInteraction):
    """Replication interaction."""

    population = role()

    def laws(self, s):
        """Return the laws of the replication interaction."""
        return [
        RateLaw(
            s.population.x,
            s.population.x
            * (total(s.population.x, weights=payoff) - total(s.population.x * total(s.population.x, weights=payoff))),
        ),
    ]
game = System(
    elements=[GroupElement('population', element=StrategyElement, size=2, values={'x': [0.1, 0.9]})],
    interactions=[ReplicationInteraction('replication', population='population')],
)
print(Simulation(game, dt=1.0).run(until=40).trajectory_.values('population.x')[-1].round(3))
```

A role also binds to an element set, a list of single elements of one type. The interaction's laws apply to each
element, the elements keep their own state names, and reductions over the role reduce over the set.

```python
from skcomplex.core.components import Element
from skcomplex.core.base import BaseInteraction
from skcomplex.core.laws import RateLaw, total
from skcomplex.engine.runtime import Simulation
from skcomplex.core.system import System
from skcomplex.core.declarations import parameter, role, state

class InhibitionInteraction(BaseInteraction):
    """Inhibition interaction."""

    units = role()
    beta = parameter(2.0)

    def laws(self, s):
        """Return the laws of the inhibition interaction."""
        return [RateLaw(s.units.x, s.units.x * (1 - s.units.x) - s.beta * s.units.x * (total(s.units.x) - s.units.x))]
class ActivityElement(Element):
    """Activity element."""

    x = state(0.1)
network = System(
    elements=[ActivityElement(f'unit_{index}', x=0.1 * (index + 1)) for index in range(3)],
    interactions=[InhibitionInteraction('inhibition', units=['unit_0', 'unit_1', 'unit_2'])],
)
print(Simulation(network, dt=1.0).run(until=30).state_.round(3))
```

### Variadic roles

A role declared with `role(variadic=True)` binds a list of element paths and group member references `<group>[<i>]`,
and laws read it as a tuple of namespaces in the given order, so a law can take any number of inputs. A parameter
declared with `parameter(per='<role>')` holds one value per entry, laws read it as a tuple of symbols, and each value is
the parameter component `<path>.<name>[<i>]`, which `update_parameters` and `schedule` set like any parameter.

```python
from skcomplex.core.base import parameter, role
from skcomplex.core.components import Element
from skcomplex.core.base import BaseInteraction
from skcomplex.core.laws import UpdateLaw
from skcomplex.core.system import System
from skcomplex.engine.runtime import Simulation
from skcomplex.core.declarations import state
from skcomplex.core.elements import GroupElement

class NodeElement(Element):
    """Node element."""

    x = state(0.0)


class WeightedSumInteraction(BaseInteraction):
    target: str = role()
    inputs: list[str] = role(variadic=True)
    weights: list[float] = parameter(1.0, per='inputs')

    def laws(self, s):
        total_input = sum(weight * item.x for weight, item in zip(s.weights, s.inputs, strict=True))
        return [UpdateLaw(s.target.x, total_input)]


summed = System(
    elements=[NodeElement('a', x=1.0), NodeElement('b', x=2.0), NodeElement('out'), GroupElement('nodes', element=NodeElement, size=5, values={'x': [0.0, 1.0, 2.0, 3.0, 4.0]})],
    interactions=[WeightedSumInteraction('sum', target='out', inputs=['a', 'b', 'nodes[3]'], weights=[1.0, 1.0, 10.0])],
)
simulation = Simulation(summed, dt=1).run(until=1)
simulation.update_parameters({'sum.weights[2]': 0.0})
print(simulation.trajectory_.values('out.x'), simulation.parameters_)
```

An element set and a variadic role both bind a list. An element set applies the laws once per element, each element
read like a single element, while a variadic role gives one law the whole sequence. `random_boolean_network` and
`yeast_cell_cycle` read the inputs of every node through one variadic role. A role declared with `role(noise=True)`
binds a `WienerProcessSpec` instead of a path, as the section on shared and correlated noise shows.

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
from skcomplex.core.components import Element
from skcomplex.core.base import BaseInteraction
from skcomplex.engine.runtime import Simulation
from skcomplex.core.system import System

import numpy as np
from sage.all__sagemath_symbolics import tanh

from skcomplex.core.laws import RateLaw, mod, shift_member
from skcomplex.core.declarations import parameter, role, state
from skcomplex.core.elements import GroupElement

class CarElement(Element):
    """Car element."""

    x = state(0.0)
    v = state(1.0)

    def laws(self, s):
        """Return the laws of the car element."""
        return [RateLaw(s.x, s.v)]
class FollowingComponentInteraction(BaseInteraction):
    """Following component interaction."""

    cars = role()
    length = parameter(40.0)

    def laws(self, s):
        """Return the laws of the following component interaction."""
        return [
        RateLaw(s.cars.v, tanh(mod(shift_member(s.cars.x, 1) - s.cars.x, s.length) - 2.0) + tanh(2.0) - s.cars.v),
    ]
ring = System(
    elements=[GroupElement('cars', element=CarElement, size=20, values={'x': np.arange(20) * 2.0 + np.eye(20)[0] * 0.1})],
    interactions=[FollowingComponentInteraction('following', cars='cars', length=40.0)],
)
print(Simulation(ring, dt=1.0).run(until=50).trajectory_.values('cars.v')[-1].std() > 0.1)
```

### Jacobians of reductions

A reduction makes every reading member depend on every reduced member, so the Jacobian of a drift with reductions has
the rank-one block `dF_i/dR dR/dx_j` of every reduction `R`. `problem.jacobian` of an ODE problem evaluates the exact
Jacobian by the chain rule through every reduction, as a SciPy sparse matrix, and `problem.jacobian.pattern()` states
its structure, which the implicit methods of `ScipyODESolver` receive (see
[Discretizations and stiff solvers](#discretizations-and-stiff-solvers)). When the dense blocks exceed 4 million
entries, `pattern()` returns `None` and an implicit solver estimates a dense Jacobian. Mean-field groups are rarely stiff, so explicit steppers such as `RK4Solver` and
`ScipyODESolver(method='RK45')` suit large groups with reductions.

```python
import numpy as np
from sage.all__sagemath_symbolics import tanh

from skcomplex.engine.compile import compile_system
from skcomplex.core.components import Element
from skcomplex.core.laws import RateLaw, total
from skcomplex.core.system import System
from skcomplex.core.declarations import state
from skcomplex.core.elements import GroupElement

class UnitElement(Element):
    """Unit element."""

    x = state(0.0)

    def laws(self, s):
        """Return the laws of the unit element."""
        return [RateLaw(s.x, -s.x + tanh(total(s.x)))]
units = System(elements=[GroupElement('units', element=UnitElement, size=50, values={'x': np.linspace(-1, 1, 50)})])
problem = compile_system(units)
jacobian = problem.jacobian.evaluate(0.0, problem.initial_state, problem.initial_parameters)
print(jacobian.shape, problem.jacobian.pattern().nnz)
```

`dynamics_of`, `fixed_points`, `stability`, and `jacobian` of `skcomplex.analysis` expand a group of at most 64
members into one equation per member, with every reduction written as an explicit sum, so the fixed points of
replicator dynamics are derived symbolically. A larger group raises an error that states the limit.

### Pair interactions

A pair interaction is written once for a pair of members and summed over every pair closer than its cutoff, inside one
group or between two groups. `PairPotentialInteraction` derives the forces from a pair energy.

```python
import numpy as np
from skcomplex.core.elements import ParticleElement
from skcomplex.engine.runtime import Simulation
from skcomplex.core.spaces import ContinuousSpace, GeneratorNeighborhood, MooreNeighborhood, TriangularNeighborhood, VonNeumannNeighborhood
from skcomplex.core.system import System

from skcomplex.core.base import NumericDomain, parameter
from skcomplex.core.interactions import PairPotentialInteraction
from skcomplex.core.elements import GroupElement


class LennardJonesInteraction(PairPotentialInteraction):
    epsilon: float = parameter(1.0, domain=NumericDomain.POSITIVE)
    sigma: float = parameter(1.0, domain=NumericDomain.POSITIVE)

    def pair_energy(self, s, r):
        return 4 * s.epsilon * ((s.sigma / r) ** 12 - (s.sigma / r) ** 6)


side = np.arange(10) * 1.2
liquid = System(
    elements=[GroupElement('atoms', element=ParticleElement, size=100, values={'x': np.stack(np.meshgrid(side, side), axis=-1).reshape(-1, 2)})],
    interactions=[LennardJonesInteraction('lj', first='atoms', second='atoms', cutoff=2.5)],
    space=ContinuousSpace(n_dim=2),
)
Simulation(liquid, dt=0.05).run(until=1)
```

`period` gives the side of a periodic box, so pair distances use the nearest periodic image.

### Boundaries

A `BoundaryLaw` sets the value of a state after every internal solver step, however long the frame, so a frame never
reports a state that a boundary would change. Adaptive solvers find the first time in a step where a boundary acts and
restart there. Boundaries combine with every form, so walls and periodic boxes work for differential equations,
stochastic systems, jump processes, and discrete maps.
`ReflectingBox` mirrors positions and velocities at the walls, and `PeriodicBox` wraps positions around the box.

## Lattice models

Cellular automata, sandpiles, and spin models live on lattices. `LatticeSpace(shape, boundary=...)` is a box of `Z^d`
with `'periodic'`, `'fixed'`, or `'reflecting'` edges, and `LatticeElement(label, element=..., space=space, values={...})` puts one member
of an element type on every site, with values given as arrays of the lattice shape. Laws are written for one cell.

Every space speaks the verbs of the framework. `space.prepare()` checks its arguments and derives its facts, which end in
`_`: the number of sites `size_`, the dimension `n_dim_`, the shape of the arrays of its sites `shape_`, the position
of every site `positions_`, whether it wraps `periodic_`, and the visual of its groups `default_visual_`.
`space.run(site, neighborhood=None)` returns the sites around a site, in the order of a neighbourhood or of the
space's own. Compiling a system fits its spaces, so a space needs `prepare()` only when its facts are read before.

```python
from skcomplex.core.spaces import GeneratorNeighborhood, LatticeSpace, MooreNeighborhood, TriangularNeighborhood, VonNeumannNeighborhood

space = LatticeSpace((4, 3), neighborhood='von_neumann').prepare()
print(space.size_, space.shape_, space.periodic_, space.run(0).tolist())
```

**Local rules.** `LocalRuleLaw(targets, over, rule)` computes a cell's next value from the ordered values of its
neighbourhood. `Neighborhood.moore`, `.von_neumann`, `.radius`, or `Neighborhood(offsets)` give the offsets. A rule is
a `Table` of patterns, a `Totalistic` or `OuterTotalistic` rule, a rule code whose `table()` returns a table, such
as `Wolfram` of the automata models, or a function of NumPy arrays,
one per neighbour, that also receives the parameters it names and, for stochastic rules, uniform random numbers.
Tables and totalistic rules run as lookups, fused into one compiled loop on the Numba backend.

```python
import numpy as np
from skcomplex.core.base import NumericDomain, state
from skcomplex.core.components import Element
from skcomplex.engine.runtime import Simulation
from skcomplex.core.system import System

from skcomplex.core.laws import LocalRuleLaw
from skcomplex.core.laws import OuterTotalistic
from skcomplex.core.spaces import GeneratorNeighborhood, LatticeNeighborhood, LatticeSpace, MooreNeighborhood, TriangularNeighborhood, VonNeumannNeighborhood
from skcomplex.core.elements import LatticeElement

class LifeElement(Element):
    """Life element."""

    alive = state(0, domain=NumericDomain.BOOLEAN)

    def laws(self, s):
        """Return the laws of the life element."""
        return [LocalRuleLaw(s.alive, MooreNeighborhood(), OuterTotalistic({(0, 3): 1, (1, 2): 1, (1, 3): 1}))]
glider = np.zeros((20, 20))
glider[1, 2] = glider[2, 3] = glider[3, 1] = glider[3, 2] = glider[3, 3] = 1
cells = System(elements=[LatticeElement('cells', element=LifeElement, space=LatticeSpace((20, 20)), values={'alive': glider})])
trajectory = Simulation(cells, dt=1).run(until=4).trajectory_
print(np.array_equal(trajectory.lattice('cells.alive')[-1], np.roll(glider, (1, 1), axis=(0, 1))))
```

- Rules update synchronously by default, every cell from the previous configuration. `schedule='random-sequential'`,
  `'random-site'`, or `'checkerboard'` update cells one at a time or one sublattice at a time.
- One unit of time is one application of every rule, so `dt` is an integer, and `trajectory.lattice(name)` returns
  the states in lattice shape.
- `DriveLaw(targets, function)` changes a whole lattice at once, such as dropping a grain on a random site.

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

**Monte Carlo.** A `SampleLaw(energy, move, sampler)` samples states. `Energy(expression, over)` is the local energy of a site, every term of the total energy
that involves it, with neighbour values read through `neighbors`. `MoveLaw.flip`, `MoveLaw.random_value`, and `MoveLaw.rotate`
propose changes, and `SamplerLaw('metropolis' | 'glauber' | 'heat-bath', temperature)` accepts them. The energy change of
a move is derived symbolically, sweeps update the two sublattices in turn, and the temperature is a parameter that
changes live.

```python
from skcomplex.core.base import NumericDomain, state
from skcomplex.core.components import Element
from skcomplex.engine.runtime import Simulation
from skcomplex.core.spaces import GeneratorNeighborhood, LatticeNeighborhood, LatticeSpace, MooreNeighborhood, TriangularNeighborhood, VonNeumannNeighborhood
from skcomplex.core.system import System

from skcomplex.core.specs import EnergySpec, MoveSpec, SamplerSpec
from skcomplex.core.laws import SampleLaw
from skcomplex.core.laws import neighbors
from skcomplex.core.declarations import parameter
from skcomplex.core.elements import LatticeElement

nearest = VonNeumannNeighborhood()
class SpinElement(Element):
    """Spin element."""

    spin = state(1.0, domain=NumericDomain.INTEGER)
    J = parameter(1.0)
    temperature = parameter(1.5)

    def laws(self, s):
        """Return the laws of the spin element."""
        return [
        SampleLaw(
            Energy(-s.J * s.spin * sum(neighbors(s.spin, nearest, reduce='each')), nearest),
            MoveLaw.flip(s.spin),
            SamplerLaw('metropolis', s.temperature),
        ),
    ]
spins = System(elements=[LatticeElement('spins', element=SpinElement, space=LatticeSpace((32, 32)))])
print(abs(Simulation(spins, dt=50, random_state=0).run(until=200).state_.mean()) > 0.9)
```

**Triangular lattices.** `LatticeSpace(shape, geometry='triangular')` keeps integer coordinates on a skewed grid, so
every boundary works as on a square lattice, and places site `(i, j)` at `(j + i / 2, i sqrt(3) / 2)` in the plane,
given by the `positions` property. `TriangularNeighborhood()` holds its six neighbours counterclockwise from 0
degrees, so direction `k + 3` modulo 6 is opposite to direction `k`, as lattice gases need. The cells of the sites
are the hexagons of a hexagonal tiling.

```python
import numpy as np
from skcomplex.core.components import Element
from skcomplex.core.laws import LocalRuleLaw
from skcomplex.engine.runtime import Simulation
from skcomplex.core.spaces import GeneratorNeighborhood, LatticeNeighborhood, LatticeSpace, MooreNeighborhood, TriangularNeighborhood, VonNeumannNeighborhood
from skcomplex.core.system import System
from skcomplex.core.declarations import state
from skcomplex.core.elements import LatticeElement

class BurnElement(Element):
    """Burn element."""

    fire = state(0.0)

    def laws(self, s):
        """Return the laws of the burn element."""
        return [LocalRuleLaw(s.fire, TriangularNeighborhood(center=True), lambda *values: np.max(values, axis=0))]
fire = np.zeros((40, 40))
fire[20, 20] = 1
forest = System(elements=[LatticeElement('forest', element=BurnElement, space=LatticeSpace((40, 40), geometry='triangular'), values={'fire': fire})])
burning = Simulation(forest, dt=1).run(until=3).trajectory_.lattice('forest.fire')
print([int(item.sum()) for item in burning])
```

**Lattices on graphs.** `LatticeElement` also takes a `NetworkSpace`, with one site per node in the order of
`graph.nodes`, and initial values as arrays in node order or mappings from node to value. `Adjacency(center=False)` is
the neighbourhood of a node, its neighbours along its edges, padded to the largest degree with the network's `fill`
value. Local rules, drives, relaxations, Monte Carlo laws, and extremal laws run on graphs as on lattices. Lookup rules
on graphs take the totalistic forms, and a function rule that names the keyword `degree` receives the number of real
neighbours of every site. The padded table grows with the largest degree, so graphs with hubs of very high degree,
beyond scale-free graphs of about 10^5 nodes, use much memory.

**Neighbour sums and colored schedules.** `neighbors(expression, reduce='sum')` sums an expression of a site and one
neighbour, read through `s.other`, over the real neighbours of the site, so local energies are exact on sites of any
degree and outside fixed edges. `schedule='colored'` on local rules and samplers updates the classes of a proper coloring of
the neighbour graph in turn, which NetworkX computes once per law. The checkerboard schedule stays the two-color case
of square lattices.

```python
from skcomplex.core.base import NumericDomain, state
from skcomplex.core.components import Element
from skcomplex.core.specs import EnergySpec, MoveSpec, SamplerSpec
from skcomplex.core.laws import SampleLaw
from skcomplex.engine.runtime import Simulation
from skcomplex.core.system import System

import networkx as nx

from skcomplex.core.laws import neighbors
from skcomplex.core.spaces import AdjacencyNeighborhood, GeneratorNeighborhood, MooreNeighborhood, NetworkSpace, TriangularNeighborhood, VonNeumannNeighborhood
from skcomplex.core.declarations import parameter
from skcomplex.core.elements import LatticeElement

class GraphSpinElement(Element):
    """Graph spin element."""

    spin = state(1.0, domain=NumericDomain.INTEGER)
    J = parameter(1.0)
    temperature = parameter(1.0)

    def laws(self, s):
        """Return the laws of the graph spin element."""
        return [
        SampleLaw(
            Energy(-s.J * s.spin * neighbors(s.other.spin, reduce='sum'), AdjacencyNeighborhood()),
            MoveLaw.flip(s.spin),
            SamplerLaw('metropolis', s.temperature, schedule='colored'),
        ),
    ]
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
import numpy as np
from skcomplex.core.components import Element
from skcomplex.engine.runtime import Simulation
from skcomplex.core.spaces import GeneratorNeighborhood, LatticeNeighborhood, LatticeSpace, MooreNeighborhood, TriangularNeighborhood, VonNeumannNeighborhood
from skcomplex.core.system import System

from skcomplex.core.laws import ExtremalLaw
from skcomplex.core.declarations import state
from skcomplex.core.elements import LatticeElement


def renew(fitness, random):
    return random


class SpeciesElement(Element):
    """Species element."""

    fitness = state(0.5)

    def laws(self, s):
        """Return the laws of the species element."""
        return [ExtremalLaw(s.fitness, LatticeNeighborhood(((-1,), (1,))), renew, repeat=100)]
ring = System(elements=[LatticeElement('ring', element=SpeciesElement, space=LatticeSpace(100), values={'fitness': np.random.default_rng(0).random(100)})])
trajectory = Simulation(ring, dt=10, random_state=0).run(until=100).trajectory_
print(len(trajectory.events), trajectory.events[0].label)
```

Square lattice groups stream to Rerun as images colored by value, and triangular lattices and graph lattices stream as
sites with `SiteVisual`, registered as `'sites'`: hexagonal cells at the site positions, or nodes colored by state with
their edges logged once. The library has `elementary_automaton`, `game_of_life`, `majority_rule`, `langtons_ant`,
`nagel_schreckenberg`, `hodgepodge`, `sandpile`, `forest_fire`, `hexagonal_forest_fire`, `earthquakes`, `ising`,
`graph_ising`, `potts`, `xy`, `fhp_lattice_gas`, and `bak_sneppen`, each in the subfield of its science, such as
`skcomplex.models.computer_science.automata` for `game_of_life` and `skcomplex.models.physics.statistical_mechanics`
for `ising`.

## Networks

A `NetworkSpace` holds a NetworkX graph whose nodes carry the members of groups on its sites and whose edges carry the
members of groups on its edges. The states and the structure of a network can change together: interactions read and
change edge states, laws reduce over the neighbours of a node, and topology requests add and remove nodes and edges.

### Generators and layers

The class methods of `NetworkSpace` wrap the generators of NetworkX: `erdos_renyi`, `watts_strogatz`,
`barabasi_albert`, `configuration_model`, `stochastic_block_model`, `grid`, `complete`, and the spatially embedded
`waxman` and `random_geometric`, whose node positions are the fitted `positions_`. A random generator takes a seed or
a NumPy generator, so a network drawn from the random stream of an experiment is reproducible. A `LayerSpec` names a second
relation over the same nodes, such as work next to contacts. Fitting a network lowers every layer to arrays by its
label: `graphs_` holds its graph, `edges_` the source and target of every edge, and `adjacency_` the neighbours of
every node in compressed sparse row form.

```python
import networkx as nx

from skcomplex.core.spaces import GeneratorNeighborhood, MooreNeighborhood, NetworkSpace, TriangularNeighborhood, VonNeumannNeighborhood

from skcomplex.core.specs import LayerSpec

first = NetworkSpace(nx.watts_strogatz_graph(100, 6, 0.1, seed=7))
second = NetworkSpace(nx.watts_strogatz_graph(100, 6, 0.1, seed=7))
print(sorted(first.graph.edges) == sorted(second.graph.edges))

space = NetworkSpace(nx.path_graph(4), layers=[LayerSpec('work', nx.complete_graph(4))]).prepare()
print(list(space.graphs_), [item.tolist() for item in space.edges_['work']])
```

### Interactions over edges

A group on the edges of a network, `GroupElement(..., on='edges')`, holds one member per edge of its layer. An interaction
with `over = 'neighbors'` binds a group on the nodes and, optionally, a group on the edges of the same network. Its
laws run once per edge and read `s.source`, `s.target`, and `s.edge`: the members at both ends and the member of the
edge. Rate, noise, and jump laws may target any of them, and update laws the edge member. The keyword `layer` chooses
the layer of an interaction without an edge group. Hebbian weights grow with the product of the activities at the ends:

```python
from typing import ClassVar

import networkx as nx
import numpy as np

from skcomplex.core.base import parameter, role
from skcomplex.core.components import Element, GroupElement
from skcomplex.core.base import BaseInteraction
from skcomplex.core.laws import RateLaw
from skcomplex.engine.runtime import Simulation
from skcomplex.engine.solvers import RK4Solver
from skcomplex.core.spaces import GeneratorNeighborhood, MooreNeighborhood, NetworkSpace, TriangularNeighborhood, VonNeumannNeighborhood
from skcomplex.core.system import System
from skcomplex.core.declarations import state
from skcomplex.core.elements import LatticeElement

class NeuronElement(Element):
    """Neuron element."""

    a = state(0.0)
class SynapseElement(Element):
    """Synapse element."""

    w = state(0.1)


class HebbianInteraction(BaseInteraction):
    over: ClassVar[str] = 'neighbors'
    neurons: str = role()
    synapses: str = role()
    eta: float = parameter(1.0)

    def laws(self, s):
        return [RateLaw(s.edge.w, s.eta * s.source.a * s.target.a)]


space = NetworkSpace(nx.cycle_graph(6))
system = System(
    elements=[
        LatticeElement('neurons', element=NeuronElement, space=space, values={'a': [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]}),
        GroupElement('synapses', element=SynapseElement, space=space, on='edges'),
    ],
    interactions=[HebbianInteraction('hebb', neurons='neurons', synapses='synapses')],
)
trajectory = Simulation(system, solver=RK4Solver(dt=0.01), dt=0.5).run(until=1.0).trajectory_
print(np.round(trajectory.values('synapses.w')[-1], 3).tolist())
```

### Reductions over neighbourhoods

`neighbors(expression, over=..., reduce=..., weight=...)` reduces an expression written for one neighbour, through
`s.other`, over the neighbours of every member: the neighbours of a network by default, a layer with
`Adjacency(layer=label)` or the layer itself, the neighbourhood of a lattice, or in a continuous space `Radius(r)`,
`Nearest(k)`, and `Cone(r, angle)`. The reductions are `'sum'`, `'mean'`, `'count'`, `'max'`, `'min'`, and
`'normalized-mean'`. A weight may read the state of the edge group of the layer through a role of an interaction, as
the weight of the edge that joins the member to its neighbour. The Kuramoto coupling on a network is
`RateLaw(s.theta, s.omega + s.K * neighbors(sin(s.other.theta - s.theta), reduce='mean'))`.

```python
import networkx as nx

from skcomplex.engine.compile import compile_system
from skcomplex.core.components import Element, GroupElement
from skcomplex.core.base import BaseInteraction
from skcomplex.core.laws import RateLaw, neighbors
from skcomplex.core.spaces import GeneratorNeighborhood, MooreNeighborhood, NetworkSpace, TriangularNeighborhood, VonNeumannNeighborhood
from skcomplex.core.system import System
from skcomplex.core.declarations import role, state
from skcomplex.core.elements import LatticeElement

class NodeElement(Element):
    """Node element."""

    u = state(0.0)
class LinkElement(Element):
    """Link element."""

    w = state(1.0)
class DiffusionInteraction(BaseInteraction):
    """Diffusion interaction."""

    nodes = role()
    links = role()

    def laws(self, s):
        """Return the laws of the diffusion interaction."""
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

Validation reports a reduction whose neighbourhood the space does not offer, such as `Radius(1.0)` on a network, with
the code `law-space`, and a layer the network lacks with the code `binding`.

### Topology requests

A variable group on the sites or edges of a network, `GroupElement(..., on='sites', variable=True, capacity=n)`, holds rows
for up to `capacity` members, of which the members present are marked active. The NumPy and Numba backends run such
groups with fixed array shapes: laws leave the free rows unchanged, reductions read the members present, and
neighbourhoods and interactions over neighbours read the edges present. The topology requests `add_node`,
`remove_node`, `add_edge`, `remove_edge`, and `rewire` are the targets of update and jump laws:
`UpdateLaw(request, probability)` fires the request with a probability at every step for every member or edge its law runs
for, and `JumpLaw(rate, {request: 1})` fires it at a rate. The iteration and Gillespie steppers apply the requests between
steps. A request that joins nodes draws the new ends with weights from `toward`, a function of the namespace of a
candidate node, such as `lambda node: neighbors(1)` for its degree. A request on a group that is not variable gets the
validation issue `group-fixed`, and a full capacity raises `CapacityError`.

Trajectories hold the values of absent members as NaN, and `trajectory.network(frame)` returns the nodes and edges
present at a frame. Rerun draws the nodes present and the edges at every frame where they change. Barabási-Albert growth
adds one node with two edges at every step:

```python
import numpy as np

from skcomplex.core.components import Element, GroupElement
from skcomplex.core.laws import UpdateLaw, neighbors
from skcomplex.engine.runtime import Simulation
from skcomplex.core.spaces import DiscreteTime, GeneratorNeighborhood, MooreNeighborhood, NetworkSpace, TriangularNeighborhood, VonNeumannNeighborhood
from skcomplex.core.system import System
from skcomplex.core.declarations import state

class NodeElement(Element):
    """Node element."""

    age = state(0.0)

    def laws(self, s):
        """Return the laws of the node element."""
        return [UpdateLaw(s.age, s.age + 1)]
class LinkElement(Element):
    """Link element."""
class GrowthElement(Element):
    """Growth element."""

    def laws(self, s):
        """Return the laws of the growth element."""
        return [UpdateLaw(add_node(edges=2, toward=lambda node: neighbors(1)), 1)]
space = NetworkSpace(nx.complete_graph(3))
system = System(
    elements=[
        GroupElement('nodes', element=NodeElement, space=space, on='sites', variable=True, capacity=203),
        GroupElement('links', element=LinkElement, space=space, on='edges', variable=True, capacity=403),
        GrowthElement('growth'),
    ],
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
import numpy as np

from skcomplex.core.base import NumericDomain, parameter, role, state
from skcomplex.core.components import Element, GroupElement
from skcomplex.core.spaces import ContinuousTime, GeneratorNeighborhood, LatticeSpace, MooreNeighborhood, TriangularNeighborhood, VonNeumannNeighborhood


class WealthDensityElement(Element):
    f: float = state(default=0.0, domain=NumericDomain.NONNEGATIVE)


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
- The three laws are of the law kind `encounter` of the continuous class. They need continuous time, a lattice space,
  and real states, and validation reports a law whose system lacks them, with the fix to declare them.
- Validation reports the encounters whose probabilities do not sum to one over the outcomes, with the candidate and
  field activities and the sum, before compilation.
- A system with encounters has no exact Jacobian, so implicit solvers estimate it by finite differences.

```python continuation
import numpy as np
from skcomplex.core.base import parameter, role
from skcomplex.engine.runtime import Simulation
from skcomplex.core.spaces import ContinuousTime, GeneratorNeighborhood, LatticeSpace, MooreNeighborhood, TriangularNeighborhood, VonNeumannNeighborhood
from skcomplex.core.system import System

from skcomplex.engine.compile import validate
from skcomplex.core.base import BaseInteraction
from skcomplex.core.laws import EncounterLaw


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


class ExchangeInteraction(BaseInteraction):
    population: str = role()
    eta: float = parameter(default=1.0)

    def laws(self, s):
        return [EncounterLaw(s.population.f, s.population.f, s.eta, exchange_table(0.2, 0.3))]


economy = System(
    elements=[wealth],
    interactions=[ExchangeInteraction('exchange', population='wealth')],
    space=LatticeSpace(9, boundary='fixed'),
    time=ContinuousTime(),
)
densities = Simulation(economy, dt=10.0).run(until=100).trajectory_.values('wealth.f')
print(densities.sum(axis=1)[[0, -1]].round(12), densities[-1].round(3))
print(validate(economy).ok)
```

### External actions

External actions need no primitive of their own.

- A micro-scale action, such as a welfare policy that acts on the transitions of the wealth classes, is an encounter
  whose field role binds a group with fixed states, a group no law changes.
- A macro-scale action, such as a recruitment of security forces proportional to the mean criminality, is a `RateLaw`
  that reads the moments of a distribution.

```python continuation
import numpy as np
from skcomplex.core.base import parameter, role
from skcomplex.core.laws import EncounterLaw
from skcomplex.core.base import BaseInteraction
from skcomplex.engine.runtime import Simulation
from skcomplex.core.spaces import ContinuousTime, GeneratorNeighborhood, LatticeSpace, MooreNeighborhood, TriangularNeighborhood, VonNeumannNeighborhood
from skcomplex.core.system import System

class PolicyInteraction(BaseInteraction):
    population: str = role()
    government: str = role()
    tax: float = parameter(default=0.2)

    def laws(self, s):
        rates, table = np.zeros((9, 1)), np.zeros((9, 1, 9))
        for candidate in range(9):
            outcome = candidate + 1 if candidate < 3 else candidate - 1 if candidate >= 6 else candidate
            rates[candidate, 0] = float(outcome != candidate)
            table[candidate, 0, outcome] = 1.0
        return [EncounterLaw(s.population.f, s.government.f, rates * s.tax, table)]


welfare = System(
    elements=[wealth, GroupElement('government', element=WealthDensityElement, size=1, values={'f': 1.0})],
    interactions=[ExchangeInteraction('exchange', population='wealth'), PolicyInteraction('policy', population='wealth', government='government')],
    space=LatticeSpace(9, boundary='fixed'),
    time=ContinuousTime(),
)
print(Simulation(welfare, dt=10.0).run(until=100).trajectory_.values('wealth.f')[-1].round(3))
```

### Moments

`moment(expression, order, coordinates)` is the reduction `sum_j c_j^r f_j` over the members of a group, with the
coordinate `c_j` of every member, such as its activity. The order-0 moment is the density. `normalized=True` divides by
the density, so the normalized order-1 moment is the mean activity, and `central=True` takes the moment about the mean,
so the central order-2 moment is the variance. Moments are expressions of `total` with a shared row of weights, a
vector with one weight per reduced member, so laws read them and observables record them. An observable of a group
receives the namespace of one member, and its function must be defined at module level so that it pickles.

```python continuation
from skcomplex.engine.runtime import Simulation
from skcomplex.core.specs import ObservableSpec

from skcomplex.core.laws.utils import moment


def mean_wealth(s):
    return moment(s.f, 1, classes, normalized=True)


def wealth_variance(s):
    return moment(s.f, 2, classes, central=True)


observables = [
    ObservableSpec('mean_wealth', mean_wealth, group='wealth'),
    ObservableSpec('wealth_variance', wealth_variance, group='wealth'),
]
trajectory = Simulation(economy, dt=10.0, observables=observables).run(until=100).trajectory_
print(trajectory.values('mean_wealth')[-1].round(3), trajectory.values('wealth_variance')[[0, -1]].round(3))
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
import numpy as np
from skcomplex.engine.compile import validate
from skcomplex.engine.runtime import Simulation

from skcomplex.engine.solvers import KineticSolver

sampled = Simulation(economy, dt=10.0, solver=KineticSolver(n=20_000), random_state=0).run(until=100)
print(np.abs(sampled.trajectory_.values('wealth.f')[-1] - densities[-1]).sum().round(2))
print(validate(welfare, solver=KineticSolver()).ok)
```

### Levels

`distribution_from(group, state, activities, element=E)` bins the values of a state of a group's members, from their
initial values or from a frame of a trajectory, into a distribution site group, every member counting at the nearest
activity, so the density is 1. `members_from(distribution, element, n, activities=...)` draws the members of a
variable group from a distribution. A model can then run one level in a partition of a co-simulation and feed another
level its moments.

```python continuation
from skcomplex.core.components import Element

from skcomplex.engine.compile import distribution_from, members_from

class TraderElement(Element):
    w = state(0.0)

traders = members_from(densities[-1], TraderElement, 1000, activities=classes, random_state=0)
binned = distribution_from(traders, 'w', classes, element=WealthDensityElement, label='wealth')
print(binned.values['f'].sum(), round(float(binned.values['f'] @ classes), 3), round(traders.values['w'].mean(), 3))
```

The library has `kinetic_wealth_welfare` in `skcomplex.models.economics.econophysics`, `immune_competition` in
`skcomplex.models.biology.oncology`, `kinetic_opinion` in `skcomplex.models.social.opinion_dynamics`, and `evacuation`
and `criminality` in `skcomplex.models.social.crowds`. `evacuation` holds a crowd on a network of rooms as a site group
on a two-dimensional lattice indexed by room and speed class, whose walking encounters read the doors of the building.
### Automata on groups

A cellular automaton needs only a group: its cells are the elements, and a cell at `g` reads the cells at `g s` for
the elements `s` of a finite memory set (Ceccherini-Silberstein and Coornaert 2023). `CayleySpace(group, ball=r)` is the
Cayley graph of a finitely generated group on the ball of radius `r` in its generators and their inverses, and
`CayleySpace(group, quotient=q)` on a finite quotient, where configurations are periodic. A group space is a space of the
lattice kind, so `LocalRuleLaw`, `SampleLaw`, and the other lattice laws run on it with the NumPy and Numba backends, and a
run reads only its neighbour table. Fitting the space enumerates its sites, numbered by word length: `words_` names
their shortest words, `ball_sizes_` gives the growth of the balls, and cells outside a ball read the space's `fill`. The memory set is a
`Memory` of words such as `'a'`, `'a^-1'`, or `'a*b'`, with `'1'` for the cell itself, and `GeneratorNeighborhood(group.generators)`
gives the identity followed by every generator and its inverse.

The groups formalism, `skcomplex.formalisms.groups`, holds group spaces and the general constructors that give
groups by any presentation whose word problem a library solves:

- `finitely_presented_group(generators, relators)` takes words such as `'t*a*t^-1*a^-2'`. A free group, without
  relators, reduces its words directly, and otherwise KBMAG in GAP looks for a confluent rewriting system by
  Knuth-Bendix completion. A presentation it cannot complete raises `WordProblemError`, which asks for a quotient.
- `automaton_group(recursion)` takes a Mealy automaton as the wreath recursion of its states, mapped to their sections
  and their permutation of the alphabet, such as `{'t': ((None, 't'), (1, 0))}` for the adding machine. Elements are
  told apart by their action on a level of the tree and, when two act alike there, by AutomGrp in GAP.
  `quotient=n` takes the action on level `n`.
- `permutation_group(generators)` and `matrix_group(generators, ring)` take images of points and integer matrices
  over `'ZZ'` or `'GF(p)'`, whose elements are compared as arrays.

GAP comes with the `passagemath-gap` dependency and starts on the first ball that needs it. A ball whose size bound
exceeds `max_sites`, 100,000 by default, raises `SpaceTooLargeError` before GAP is called, and `validate` reports it as
`space-too-large` with the largest radius that fits. A group space lacks the regular geometry of a square lattice and
an image layout, so validation reports a checkerboard schedule as `schedule-unsupported` and an image visual as
`visual-unsupported`. Groups on group spaces stream to Rerun with `CayleyVisual`, registered as `'cayley'`: nodes on
rings by word length, colored by state, with the edges of the generators logged once.

```python
import numpy as np
from skcomplex.core.base import NumericDomain, state
from skcomplex.core.components import Element
from skcomplex.core.laws import LocalRuleLaw
from skcomplex.engine.runtime import Simulation
from skcomplex.core.system import System

from skcomplex.core.spaces import CayleySpace
from skcomplex.core.laws import Totalistic
from skcomplex.core.spaces import GeneratorNeighborhood, MemoryNeighborhood, MooreNeighborhood, TriangularNeighborhood, VonNeumannNeighborhood
from skcomplex.core.elements import LatticeElement

free = finitely_presented_group(['a', 'b'])
tree = CayleySpace(free, ball=4).prepare()
memory = GeneratorNeighborhood(free.generators)
class MajorityElement(Element):
    """Majority element."""

    alive = state(0, domain=NumericDomain.BOOLEAN)

    def laws(self, s):
        """Return the laws of the majority element."""
        return [LocalRuleLaw(s.alive, memory, Totalistic({3: 1, 4: 1, 5: 1}))]
cells = (np.random.default_rng(0).random(tree.size_) < 0.5).astype(float)
majority = System(elements=[LatticeElement('cells', element=MajorityElement, space=tree, values={'alive': cells})])
print(tree.ball_sizes_, Simulation(majority, dt=1).run(until=3).trajectory_.values('cells.alive').shape)
print(CayleySpace(permutation_group([[1, 0, 2], [1, 2, 0]])).prepare().words_)
```

Named groups are realizations of the automata models: `free_group`, `heisenberg`, `lamplighter`, `baumslag_solitar`,
and `grigorchuk_group` of `skcomplex.models.computer_science.automata`. The Grigorchuk group, which acts on the binary
tree by `a = (1, 1)(0 1)`, `b = (a, c)`, `c = (a, d)`, `d = (1, b)`, is not finitely presented, and its balls grow
faster than any polynomial and slower than any exponential.

```python
from skcomplex.core.spaces import CayleySpace

from skcomplex.models.computer_science.automata import grigorchuk_group

ball, quotient = CayleySpace(grigorchuk_group(), ball=6).prepare(), CayleySpace(grigorchuk_group(), quotient=3).prepare()
print(ball.ball_sizes_, quotient.size_)
```

**Theory of automata.** `skcomplex.analysis` decides the properties the theory studies on finite universes, where
they are decidable: periodic configurations on a torus or a finite quotient, and patterns on windows. The checks do
not decide them on infinite groups. `Automaton(space, neighborhood, rule, states)` is the map on configurations of a
local rule, with `step` and `run`.

- `is_injective` and `is_surjective`, which agree on a finite universe, enumerate the configurations of small
  universes and solve a SAT encoding with PySAT on larger ones. `is_pre_injective(automaton, window)` asks whether two
  configurations that differ only inside a window can share their image.
- `garden_of_eden(automaton, window)` returns a pattern without a preimage, by enumeration or by SAT on random
  patterns.
- `inverse(automaton, max_radius)` finds the rule of the inverse of a reversible automaton with the smallest memory,
  and `compose(first, second)` gives the automaton of their composition.

```python
from skcomplex.core.laws import OuterTotalistic
from skcomplex.core.spaces import GeneratorNeighborhood, LatticeNeighborhood, LatticeSpace, MooreNeighborhood, TriangularNeighborhood, VonNeumannNeighborhood

from skcomplex.analysis import Automaton, compose, garden_of_eden, inverse, is_surjective
from skcomplex.core.laws import Table

life = Automaton(LatticeSpace((6, 6)), MooreNeighborhood(), OuterTotalistic({(0, 3): 1, (1, 2): 1, (1, 3): 1}))
print(is_surjective(life), garden_of_eden(life, random_state=0) is not None)
shift = Automaton(LatticeSpace(8), LatticeNeighborhood(((1,),)), Table({(1,): 1}))
print(compose(shift, inverse(shift)).step([1, 0, 1, 1, 0, 0, 0, 1]).tolist())
```

**Linear automata and subshifts.** `LinearAutomaton(space, neighborhood, coefficients, field)` is the element
`sum_s a_s s` of the group ring over `'GF(p)'` or `'QQ'`. On a finite universe it is a matrix whose rank decides
injectivity, surjectivity, and pre-injectivity together, and on `Z^d`, with `infinite=True`, its Laurent polynomial
decides them: it is invertible when the polynomial is a monomial and surjective when it is not zero.
`Subshift(neighborhood, forbidden)` holds the configurations without forbidden patterns: it checks configurations,
counts the admissible ones of a window, and samples them by a Markov chain of single-cell moves.
`configuration_entropy` gives `log |X_F| / |F|` along windows `F`, `spatial_entropy_rates` the entropy rate of every
frame of a run, which never increases under an automaton, and `difference_pattern` the cells where two runs differ
after one cell changes.

```python
from skcomplex.core.spaces import GeneratorNeighborhood, LatticeNeighborhood, LatticeSpace, MooreNeighborhood, TriangularNeighborhood, VonNeumannNeighborhood

from skcomplex.analysis import LinearAutomaton, Subshift, configuration_entropy

laplacian = LinearAutomaton(LatticeSpace(5), LatticeNeighborhood(((-1,), (0,), (1,))), (1, -2, 1), 'GF(2)')
golden = Subshift(LatticeNeighborhood(((0,), (1,))), [(1, 1)])
print(laplacian.is_invertible(), laplacian.is_surjective(infinite=True))
print([golden.count(LatticeSpace(length, boundary='fixed')) for length in range(1, 8)])
print(configuration_entropy(golden, [LatticeSpace(24, boundary='fixed')]).round(3))
```

The library has `free_group_automaton`, `lamplighter_automaton`, `heisenberg_automaton`, `baumslag_solitar_automaton`,
`grigorchuk_automaton`, the Laplacian `linear_ring_automaton`, and the golden mean shift `subshifts` in
`skcomplex.models.computer_science.automata`.

## Fields

A field holds one value per site of a lattice or per node of a network. `Field(label, element, space, boundary=...,
values=...)` places a member of an element type on every site, and the laws of the type, written for one site, read
the neighbouring sites through differential operators. The method of lines turns a field into ordinary differential
equations, so every ODE solver and backend runs it.

```python
import numpy as np
from skcomplex.core.components import Element
from skcomplex.core.laws import RateLaw
from skcomplex.engine.runtime import Simulation
from skcomplex.core.spaces import GeneratorNeighborhood, LatticeSpace, MooreNeighborhood, TriangularNeighborhood, VonNeumannNeighborhood
from skcomplex.core.system import System

from skcomplex.core.elements import FieldElement
from skcomplex.core.laws.utils import laplacian
from skcomplex.core.declarations import parameter, state

class DiffusionInteractionElement(Element):
    """Diffusion interaction element."""

    u = state(0.0)
    D = parameter(1.0)

    def laws(self, s):
        """Return the laws of the diffusion interaction element."""
        return [RateLaw(s.u, s.D * laplacian(s.u))]
initial = np.zeros(400)
initial[200] = 20.0
rod = FieldElement('rod', DiffusionInteractionElement, LatticeSpace(400, spacing=0.05), boundary={'u': 'zero-flux'}, values={'u': initial})
trajectory = Simulation(System(elements=[rod]), dt=0.1).run(until=0.5).trajectory_
print(trajectory.values('rod.u')[-1].max(), trajectory.values('rod.u').sum(axis=1)[[0, -1]])
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

The compiler splits the rate of every field into a linear part with constant coefficients, such as
`D * laplacian(u)` or `-laplacian(laplacian(u))`, and a nonlinear remainder. `IMEXEulerSolver`, `IMEX2Solver`, and
`CrankNicolsonSolver` treat the linear part implicitly, with sparse solves for finite differences and finite elements and
solves per Fourier mode for spectral fields. `ETDRK4Solver` integrates it exactly per mode, coupled states included, so stiff
equations take steps far beyond the explicit limit.

```python
import numpy as np
from skcomplex.core.components import Element
from skcomplex.core.elements import FieldElement
from skcomplex.core.laws.utils import laplacian
from skcomplex.core.laws import RateLaw
from skcomplex.engine.runtime import Simulation
from skcomplex.core.spaces import GeneratorNeighborhood, LatticeSpace, MooreNeighborhood, TriangularNeighborhood, VonNeumannNeighborhood
from skcomplex.core.system import System

from skcomplex.engine.solvers import ETDRK4Solver
from skcomplex.core.laws.utils import divergence
from skcomplex.core.declarations import state

class ChaosElement(Element):
    """Chaos element."""

    u = state(0.0)

    def laws(self, s):
        """Return the laws of the chaos element."""
        return [RateLaw(s.u, -divergence([s.u**2 / 2]) - laplacian(s.u) - laplacian(laplacian(s.u)))]
positions = np.arange(1, 129) * 32 * np.pi / 128
line = FieldElement(
    'line',
    ChaosElement,
    LatticeSpace(128, spacing=32 * np.pi / 128),
    values={'u': np.cos(positions / 16) * (1 + np.sin(positions / 16))},
    discretization='spectral',
)
print(np.abs(Simulation(System(elements=[line]), dt=10, solver=ETDRK4Solver(dt=0.25)).run(until=100).state_).max())
```

The implicit methods of `ScipyODESolver` are the stiff path for finite-difference and finite element fields, with adaptive
error control that the fixed-step split steppers lack. The compiler derives the sparsity pattern of the Jacobian from
the laws: member and shared reads, the neighbour tables of stencils, the sparse matrices of finite elements, the
reconstruction stencils of finite volumes, and dense blocks where couplings are known only at run time, such as pairs,
spectral operators, and callables. `ScipyODESolver(method='BDF')` and `ScipyODESolver(method='Radau')` receive the pattern, so
SciPy estimates the Jacobian with one drift evaluation per group of independent columns and factors it with sparse LU,
and `LSODA` receives the half bandwidths of a pattern narrower than a tenth of the size. The option `jacobian` selects
the path: `'auto'` by default, `'exact'` for the sparse Jacobian of the laws differentiated by Sage, which laws without
a symbolic form cannot give, and `'dense'` for the dense estimate. Explicit methods ignore it, and `RK45` warns when one
call spans more than 1000 stable steps of the diffusion of a field.

```python
from skcomplex.engine.runtime import Simulation

from skcomplex.engine.solvers import ScipyODESolver
from skcomplex.models.biology.ecology import fisher_kpp

stiff = fisher_kpp(400, D=1.0, spacing=0.05)
simulation = Simulation(stiff, dt=1.0, solver=ScipyODESolver(method='BDF')).run(until=5)
print(simulation.stepper_.jacobian_pattern_.nnz, simulation.state_.max().round(3))
```

On Fisher-KPP on 4000 points, whose largest eigenvalue is about 1600, run to `t = 10`, BDF with the dense estimate
took 35 to 50 s, against about 0.2 s with the pattern, 0.3 s with the exact Jacobian, 0.7 s for Radau with the pattern,
and 0.8 s for `CrankNicolsonSolver(dt=0.01)`. Spectral fields keep `ETDRK4Solver` and the IMEX steppers, which win there because
their linear part is diagonal in Fourier space while its pattern is dense.

`skcomplex.models.physics.fluids.incompressible_flow(shape, size, nu=..., forcing=..., buoyancy=...)` builds two-dimensional
incompressible flow in vorticity-streamfunction form, with Boussinesq convection when `buoyancy` gives `Ra` and `Pr`.
The library has `kuramoto_sivashinsky`, `complex_ginzburg_landau`, `swift_hohenberg`, and `heat_l_shape` in the
subfields `fluids`, `pattern_formation`, and `heat` of `skcomplex.models.physics`, and `burgers`, `shallow_water`,
`decaying_turbulence`, and `rayleigh_benard` in `skcomplex.models.physics.fluids`.

## Backends and speed

Compiling a system produces numeric functions with a backend. `'numpy'` evaluates every law with NumPy arrays, and
`'numba'` generates one compiled function per form, with compiled cell lists for pair interactions and compiled loops
for `RK4Solver`, `EulerMaruyamaSolver`, `GillespieSolver`, and `IterationSolver`. The default `'auto'` picks Numba when the groups of a system
hold at least 1000 members in total, because compiling takes a second or two that small systems do not win back.

```python
from skcomplex.engine.runtime import Simulation

from skcomplex.engine.backends import NumbaBackend
from skcomplex.models.physics.statistical_mechanics import ideal_gas
from skcomplex.engine.solvers import RK4Solver

simulation = Simulation(ideal_gas(50_000, random_state=0), solver=RK4Solver(dt=0.01), backend='numba', dt=0.01)
print(simulation.run(until=0.5).problem_.backend)
```

Compiled kernels are cached on disk, under `~/.cache/skcomplex` or the `cache_dir` of `NumbaBackend(cache_dir=...)`,
so running the same system again skips compilation.

A law value can be a function called once per member with `Call(function, arguments)`. A Numba-compiled function
runs inside the compiled kernels, and the NumPy backend calls it from Python.

```python
from skcomplex.core.base import parameter, role
from skcomplex.core.base import BaseInteraction
from skcomplex.engine.runtime import Simulation
from skcomplex.core.system import System

import numba

from skcomplex.core.laws import Call, RateLaw
from skcomplex.models.biology.ecology import PopulationElement
from skcomplex.core.elements import GroupElement


@numba.njit
def saturating(t, n, r):
    return r * n / (1 + n)


class SaturationInteraction(BaseInteraction):
    target: str = role()
    r: float = parameter(1.0)

    def laws(self, s):
        return [RateLaw(s.target.n, Call(saturating, (s.target.n, s.r)))]


herds = System(
    elements=[GroupElement('herds', element=PopulationElement, size=2000, values={'n': 1.0})],
    interactions=[SaturationInteraction('growth', target='herds')],
)
print(Simulation(herds, dt=0.1).run(until=1).trajectory_.values('herds.n')[-1, 0])
```

A plain Python callable of the values, as in `RateLaw(s.target.n, lambda t, values: ...)`, also works with the Numba
backend, but runs in Python and turns off the compiled stepper loops for that kind of law.

## External backends

An external backend runs a whole system in a third-party solver library, and the simulation keeps its frames,
trajectories, live parameters, and Rerun visualization. `LammpsSolver` runs particle systems in LAMMPS, which
integrates them with velocity Verlet.

```python
import numpy as np
from skcomplex.core.elements import ParticleElement
from skcomplex.engine.runtime import Simulation
from skcomplex.core.spaces import ContinuousSpace, GeneratorNeighborhood, MooreNeighborhood, TriangularNeighborhood, VonNeumannNeighborhood
from skcomplex.core.system import System

from skcomplex.engine.backends import LammpsSolver
from skcomplex.models.physics.realizations.particles import LangevinThermostatInteraction, LennardJonesInteraction
from skcomplex.core.elements import GroupElement
side = np.arange(10) * 1.2

atoms = System(
    elements=[GroupElement('atoms', element=ParticleElement, size=100, values={'x': np.stack(np.meshgrid(side, side), axis=-1).reshape(-1, 2)})],
    interactions=[
        LennardJonesInteraction('lj', first='atoms', second='atoms', cutoff=2.5),
        LangevinThermostatInteraction('bath', target='atoms', temperature=0.5, gamma=1.0),
    ],
    space=ContinuousSpace(n_dim=2),
)
trajectory = Simulation(atoms, solver=LammpsSolver(dt=0.005), dt=0.05, random_state=0).run(until=1).trajectory_
print(trajectory.values('atoms.x').shape)
```

Before the first frame the backend checks the system and raises one error that lists every element, interaction, and
law it cannot run. The translation is the following.

| Framework | LAMMPS |
| --- | --- |
| `ParticleElement` and particle groups | one atom type per element or group, with one mass per group |
| two or three dimensions | `dimension 2` with `fix enforce2d`, or `dimension 3` |
| `ReflectingBox` | `fix wall/reflect` at `0` and `size` |
| `PeriodicBox` | periodic boundaries of side `size`, shared by every group |
| `LennardJonesInteraction` | `pair_style lj/cut` |
| other `PairPotentialInteraction` | `pair_style lepton` with the pair energy printed as a Lepton expression |
| `ConstantForceInteraction` | `fix addforce` |
| `LinearDragInteraction` | `fix viscous` |
| `LangevinThermostatInteraction` | `fix langevin`, seeded from the simulation's random state |
| any other interaction that only changes velocities | `fix external`, evaluated by the framework every step |

Importing `skcomplex.formalisms.particles` also registers the solver name `'lammps'`.

## Agent-based models

Agents are ordinary elements whose laws are rules written as code. An `Element` declares the states every agent holds
and the parameters its group shares, and returns `RuleLaw`s whose functions take the agent and the model. An element
or a group whose laws include rules is an agent, and so is every member of a variable group. In a system with rules, a
group whose element type has no laws is a group of passive agents that rules read and change, unless an interaction
whose laws are not rules binds it. A system with rules is
agent-based, runs in discrete time, and runs on the Mesa backend, one Mesa step per unit of time, with every agent
activated in random order. Mesa declares what it runs like any solver, so `validate(system, solver='mesa')` reports
the laws, spaces, and times it does not support.

```python
from skcomplex.engine.runtime import Simulation
from skcomplex.core.system import System

from skcomplex.core.base import NumericDomain, VisualSpec, parameter, state
from skcomplex.core.components import Element
from skcomplex.core.laws import RuleLaw
from skcomplex.core.spaces import GeneratorNeighborhood, LatticeSpace, MooreNeighborhood, TriangularNeighborhood, VonNeumannNeighborhood
from skcomplex.core.elements import GroupElement


class ForagerElement(Element):
    visual = VisualSpec('agents', {'color_by': 'food'})
    food: float = state(5.0)
    hunger: float = parameter(1.0, domain=NumericDomain.NONNEGATIVE)

    def laws(self, s):
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
print(trajectory.values('foragers.count')[-1], len(trajectory.agents('foragers', 20).ids))
```

- An agent reads and sets its states as attributes and reads its group's parameters, which change live like any
  parameter. `neighbors`, `neighbor_cells`, `agents_here`, `move_to`, `create`, and `remove` act on its place and its
  population. Only the members of a group declared `variable=True` are created and removed during a run.
- `GroupElement(label, element=..., size=..., location=...)` places the agents at cell coordinates, network nodes, or positions, one
  per agent or one shared, and agents without a location take random places.
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
- `LatticeSpace` gives agents Moore or von Neumann neighbourhoods and an optional cell capacity, `NetworkSpace` places agents on
  the nodes of a NetworkX graph, and `ContinuousSpace` finds neighbours within a radius.
- Frames and trajectories hold every group's agents with their ids, states, and locations, through
  `trajectory.agents(group, frame)`, and the count of every group as the reporter `<group>.count`.
- `skcomplex.models.social` has Schelling segregation and the voter model, `skcomplex.models.economics.econophysics`
  has Boltzmann wealth exchange, and `skcomplex.models.biology.ecology` has the wolf and sheep model.

```python
from skcomplex.core.base import parameter, role
from skcomplex.core.components import Element
from skcomplex.core.laws import RuleLaw
from skcomplex.core.base import BaseInteraction
from skcomplex.engine.runtime import Simulation
from skcomplex.core.system import System

from skcomplex.core.spaces import ContinuousSpace, GeneratorNeighborhood, MooreNeighborhood, TriangularNeighborhood, VonNeumannNeighborhood
from skcomplex.core.elements import GroupElement


class WalkerElement(Element):
    def laws(self, s):
        return [RuleLaw(self.step)]

    def step(self, agent, model):
        width, height = model.dimensions
        x, y = agent.location
        agent.move_to(((x + model.rng.uniform(0, 2)) % width, y))


class SpreadInteraction(BaseInteraction):
    walkers: str = role()
    reach: float = parameter(10.0)

    def laws(self, s):
        return [RuleLaw(self.measure)]

    def measure(self, model):
        walkers = model.agents('walkers')
        close = [a for a in walkers for b in walkers if a != b and a.distance_to(b) < model.parameter('reach')]
        counts.append(len(close) // 2)


counts = []
road = System(
    elements=[
        System(
            'road',
            elements=[GroupElement('walkers', element=WalkerElement, size=5, location=[(10.0 * index, 0.5) for index in range(5)])],
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

- `DiscreteTime(schedule=PhasesScheduleSpec([...]))` lists the phases of a step. A `PhaseSpec(name, activation, repeat)` runs its agents
  `'random'`ly, `'sequential'`ly in order of creation, `'simultaneous'`ly, or `'by-type'`, for `repeat` rounds such as
  intraday rounds. In a simultaneous phase every rule reads the states of the start of the phase, and the writes apply
  at its end. `RuleLaw(function, phase=name)` puts a rule in a phase. Rules without a phase run in the phase `'default'`,
  which runs first unless the system lists it, so systems without phases keep one random activation per step.
- `MarketInteraction(label, buyers=..., sellers=..., protocol=...)` matches two groups. Buyers state their limit `bid` and
  `demand`, and sellers their `ask` and `supply`. The protocols are `BilateralSearch`, `SampleAndSwitch` (partner
  choice with switching), `RationedQueue`, `Walrasian` (tâtonnement), `OrderBook` (price-time priority), `CallAuction`
  (one uniform price), and `DeferredAcceptance` (stable matching by the `matching` library). Every trade moves
  `money` to the seller and `goods` to the buyer at once, within the buyer's budget. The market reports
  `<market>.price`, `<market>.volume`, and `<market>.unmatched`.
- An `InstitutionInteraction` acts once per round of its phase. It reads `model.observable(name)`, reporters through
  `model.history(name, window)`, and agents, and it sets parameters of any component with
  `model.set_parameter(name, value)`. Rules see the change at once, and the trajectory records it.
- Components declare the values they publish in the class variable `reports` and publish them with
  `model.report(name, value)`. `model.storage` keeps private values of a component across the steps of a run.
- Learning blocks: `memory(agent, state, window)` returns the past values of the states that the element type of an
  agent lists in its class variable `remembers`. `heuristics` has adaptive, trend-following, anchoring, and fundamentalist forecasts.
  `logit_choice` picks a strategy by past performance, `imitate` copies the best neighbour, and `evolve` breeds the
  next generation of strategies with the model's random stream.

```python
import numpy as np
from skcomplex.core.base import parameter, state
from skcomplex.core.components import Element
from skcomplex.core.laws import RuleLaw
from skcomplex.engine.runtime import Simulation
from skcomplex.core.system import System

from skcomplex.core.specs import PhaseSpec, PhasesScheduleSpec
from skcomplex.core.spaces import DiscreteTime, GeneratorNeighborhood, MooreNeighborhood, TriangularNeighborhood, VonNeumannNeighborhood
from skcomplex.models.economics.realizations import CallAuction, InstitutionInteraction, MarketInteraction
from skcomplex.core.elements import GroupElement


class TraderElement(Element):
    value: float = state(0.0)
    bid: float = state(0.0)
    ask: float = state(0.0)
    demand: float = state(0.0)
    supply: float = state(0.0)
    money: float = state(0.0)
    goods: float = state(0.0)

    def laws(self, s):
        return [RuleLaw(self.quote, phase='quote')]

    def quote(self, agent, model):
        agent.bid = model.rng.uniform(0, agent.value) if agent.demand else 0.0
        agent.ask = model.rng.uniform(agent.value, 2.0) if agent.supply else 2.0


class ClerkInteraction(InstitutionInteraction):
    reports = ('traded',)
    fee: float = parameter(0.0)

    def act(self, model):
        traded = sum(agent.goods for agent in model.agents('buyers'))
        model.report('traded', traded)
        model.set_parameter('fee', 0.01 * traded)


rng = np.random.default_rng(0)
auction = System(
    elements=[
        GroupElement('buyers', element=TraderElement, size=20, values={'value': rng.uniform(0.5, 2, 20), 'demand': 1.0, 'money': 2.0}),
        GroupElement('sellers', element=TraderElement, size=20, values={'value': rng.uniform(0, 1.5, 20), 'supply': 1.0}),
    ],
    interactions=[
        MarketInteraction('auction', buyers='buyers', sellers='sellers', protocol=CallAuction(), phase='trade'),
        ClerkInteraction('clerk', phase='trade'),
    ],
    time=DiscreteTime(schedule=PhasesScheduleSpec([PhaseSpec('quote'), PhaseSpec('trade', repeat=3)])),
)
trajectory = Simulation(auction, random_state=0).run(until=5).trajectory_
print(trajectory.values('auction.price')[1:], trajectory.values('clerk.traded')[-1], trajectory.values('clerk.fee')[-1])
```

### Porting Mesa models

A Mesa model is ported to a system: its agent class becomes an element whose step is a `RuleLaw`, its agent attributes
become states, and its model attributes become parameters, which then change live, are validated, and are recorded
like those of any system. Mesa still runs the steps, as the backend of agent systems.

```python
from skcomplex.core.base import parameter, state
from skcomplex.core.components import Element
from skcomplex.core.laws import RuleLaw
from skcomplex.engine.runtime import Simulation
from skcomplex.core.system import System

from skcomplex.core.spaces import DiscreteTime, GeneratorNeighborhood, MooreNeighborhood, TriangularNeighborhood, VonNeumannNeighborhood
from skcomplex.core.elements import GroupElement


class CoinElement(Element):
    heads: float = state(0.0)
    bias: float = parameter(0.5)

    def laws(self, s):
        return [RuleLaw(self.flip)]

    def flip(self, agent, model):
        agent.heads += float(model.rng.random() < agent.bias)


coins = System('coins', components=[GroupElement('coins', element=CoinElement, size=10)], time=DiscreteTime())
print(Simulation(coins, random_state=0).run(until=10).trajectory_.agents('coins', 10).values['heads'].sum())
```

## Rewriting systems

A rewriting system grows by replacing its parts in parallel: a `WordElement` by string productions in the style of
L-systems, a `PatchElement` of tiles by affine substitutions, and a `HostGraphElement` by double-pushout graph productions. It
compiles to the rewriting form, every step is one generation, and `run(until=n)` records the generations `0` to `n`.
The structure lives next to the state vector, whose states are the count of every symbol, prototile, or node label
that is a word, and the size `<element>.size`. `Trajectory.structure(path, frame)` returns the recorded structure,
which later generations do not change.

### Words and productions

A word, a patch, or a host graph rewrites itself by the rewriting laws it returns from `laws`. A `RewriteLaw` holds
fixed rules, and every generation applies the rules of all the laws of a component at once. Strings are words of
one-character symbols, and `Module` gives a symbol parameters.

```python
from skcomplex.engine.runtime import Simulation
from skcomplex.core.system import System

from skcomplex.core.specs import ModuleSpec
from skcomplex.core.laws import ProductionLaw
from skcomplex.core.elements import WordElement


class AlgaeElement(WordElement):
    def laws(self, s):
        return [RewriteLaw([ProductionLaw('A', 'AB'), ProductionLaw('B', 'A')])]


algae = System(elements=[AlgaeElement('algae', axiom='A')])
trajectory = Simulation(algae).run(until=5).trajectory_
print(trajectory.values('algae.size').tolist(), trajectory.structure('algae', 5).text)
```

A production can be stochastic, with probabilities that sum to one per predecessor and context and draws from the
simulation's random stream. It can depend on a left or a right context, which skips the word's `ignore` symbols and
treats bracketed branches as Prusinkiewicz and Lindenmayer (1990) do, and it then takes precedence over context-free
productions. A parametric production binds the predecessor's parameters to symbols, with a condition and successor
parameters written as expressions of them and of the parameters of the rewritten component, which `schedule` and
`update_parameters` change between generations.

```python
from skcomplex.core.specs import ModuleSpec
from skcomplex.core.laws import ProductionLaw
from skcomplex.core.elements import WordElement
from skcomplex.engine.runtime import Simulation
from skcomplex.core.system import System

from skcomplex.core.base import parameter, symbol

x = symbol('x')


class ShrinkingElement(WordElement):
    ratio: float = parameter(0.5)

    def laws(self, s):
        shrink = ProductionLaw(Module('A', (x,)), [Module('F', (x,)), Module('A', (s.ratio * x,))], condition=x > 0.1)
        return [RewriteLaw([shrink])]


plant = System(elements=[ShrinkingElement('plant', axiom=[Module('A', (1.0,))], alphabet={'A': 1, 'F': 1})])
print(Simulation(plant).run(until=5).trajectory_.structure('plant', 5).text)
```

A generation that would make a word longer than `max_size` raises `RewritingError` before allocating it, and the
frames before it stay recorded.

### Word geometries, patches, and measures

A `WordGeometry`, from `skcomplex.formalisms.rewriting`, draws a word as line segments in two or three dimensions,
as the turtle interpretation of L-systems does: `F` draws forward, `f` moves, `+` and `-` turn, `[` and `]` push and pop
a branch, and `contraction` scales the step at every generation. A `PatchElement` holds tiles, each a `Prototile` segment or
polygon under an affine placement, and a `SubstitutionLaw` replaces every tile of a prototile by pieces placed by `Affine`
maps relative to it. Structures with a geometry report `<element>.length` and `<element>.area` at every frame, and give
`bounds()` and `raster(resolution)`, a boolean grid that `box_counting_dimension` reads.

```python
from skcomplex.core.laws import ProductionLaw
from skcomplex.core.elements import WordElement
from skcomplex.engine.runtime import Simulation
from skcomplex.core.system import System

from skcomplex.analysis import box_counting_dimension
from skcomplex.core.specs import AffineSpec, PrototileSpec
from skcomplex.core.elements import PatchElement
from skcomplex.core.laws import SubstitutionLaw
from skcomplex.engine.runtime import WordGeometry


class KochElement(WordElement):
    def laws(self, s):
        return [RewriteLaw([ProductionLaw('F', 'F+F--F+F')])]


koch = System(elements=[KochElement('koch', axiom='F', geometry=WordGeometry(angle=60.0, contraction=1 / 3))])
curve = Simulation(koch).run(until=4).trajectory_.structure('koch', 4)
print(len(curve.segments()), round(curve.length(), 6))

interval = Prototile('interval', [(0, 0), (1, 0)])
thirds = [('interval', Affine(scale=1 / 3)), ('interval', Affine(scale=1 / 3, translation=(2 / 3, 0)))]


class CantorElement(PatchElement):
    def laws(self, s):
        return [RewriteLaw([SubstitutionLaw('interval', thirds)])]


cantor = System(elements=[CantorElement('cantor', prototiles=[interval])])
trajectory = Simulation(cantor).run(until=5).trajectory_
print(trajectory.values('cantor.size')[-1], round(trajectory.values('cantor.length')[-1], 6))
grid = trajectory.structure('cantor', 5).raster(729)
print(round(box_counting_dimension(grid, [3, 9, 27]), 2))
```

The `geometry` visual logs segments and outlines as line strips, a patch with `filled=True` as an image of its raster,
and a host graph as graph nodes and edges, in the bounds the element declares or that its first tiles give.

### Graph rewriting

A `GraphProductionLaw` gives a left graph, an interface, and a right graph that share nodes by name, and matches by their
`label` attributes. Every generation applies it at a maximal set of parallel independent matches, chosen in an order
drawn from the random stream, and skips matches whose deleted nodes would leave dangling edges.

```python
from skcomplex.engine.runtime import Simulation
from skcomplex.core.system import System

import networkx as nx

from skcomplex.core.laws import GraphProductionLaw
from skcomplex.core.elements import HostGraphElement

interface = nx.Graph()
interface.add_nodes_from('ab')
subdivide = GraphProductionLaw(nx.Graph([('a', 'b')]), interface, nx.Graph([('a', 'c'), ('c', 'b')]))


class RingElement(HostGraphElement):
    def laws(self, s):
        return [RewriteLaw([subdivide])]


ring = System(elements=[RingElement('ring', graph=nx.cycle_graph(4))])
print(Simulation(ring, random_state=0).run(until=3).trajectory_.values('ring.size').tolist())
```

## Quantum systems

A quantum system is a set of quantum subsystems whose state is one vector of their joint Hilbert space. `SpinElement` has the
components `x`, `y`, `z` and the ladder operators `plus` and `minus` of spin `j`, `QubitElement` has the Pauli matrices,
`ModeElement` has `annihilate`, `create`, and `number` truncated at `quanta`, and `SiteElement` is a particle on the sites of a
lattice or a network with its `laplacian`, its `adjacency`, and the functions `projector` and `diagonal`. A group of
subsystems, such as a `LatticeElement` of spins, is the tensor product of its members' spaces in member order, and the initial
state of every subsystem is a basis label, an index, or a vector.

### Operators and laws

Laws receive the local operators of their subsystems, which combine by sums, products, numbers, and symbolic
coefficients of parameters and time into an `Operator`. `GeneratorLaw(operator)` adds to the Hamiltonian, with
`ħ = 1`, and `DissipatorLaw(operator, rate)` adds a Lindblad dissipator, which makes the system open. An interaction
with the binding `over = 'neighbors'` is written once for a bond, with `s.first` and `s.second`, and acts on every pair
of neighbours of the space of the group its one role binds. Compiling checks that the Hamiltonian is Hermitian and that the dimension stays within the
`max_dimension` of the quantum solver. A whole-system initial state, such as an entangled vector or `'ground'`, is
given to the simulation as `initial_state={'<system label>.state': value}`.

```python
from operator import attrgetter

import numpy as np

from typing import ClassVar

from skcomplex.core.base import parameter, role
from skcomplex.core.laws import GeneratorLaw
from skcomplex.core.base import BaseInteraction
from skcomplex.core.spaces import GeneratorNeighborhood, LatticeSpace, MooreNeighborhood, TriangularNeighborhood, VonNeumannNeighborhood
from skcomplex.core.system import System
from skcomplex.engine.runtime import Simulation
from skcomplex.models.physics.realizations.quantum import SpinElement
from skcomplex.core.elements import LatticeElement
class CouplingInteraction(BaseInteraction):
    over: ClassVar[str] = 'neighbors'
    target: str = role()
    J: float = parameter(1.0)

    def laws(self, s):
        a, b = s.first, s.second
        return [GeneratorLaw(s.J * (a.x * b.x + a.y * b.y + a.z * b.z))]


chain = System(
    elements=[LatticeElement('chain', element=SpinElement, space=LatticeSpace(6), values={'initial': ['up', 'down'] * 3})],
    interactions=[CouplingInteraction('coupling', target='chain')],
)
observables = [expectation('sz', attrgetter('z'), group='chain')]
trajectory = Simulation(chain, dt=0.5, observables=observables).run(until=2.0).trajectory_
print(trajectory.values('sz').shape, round(float(np.ptp(trajectory.values('energy'))), 8))
```

### Solvers, observables, and spectra

A quantum system compiles to the quantum form, whose state is stored as complex pairs under `<system>.state`, a pure
state for a closed system and a density matrix for an open one. Its solvers hand every frame to QuTiP: `unitary` for
the Schrödinger equation, `lindblad`, the default, for the master equation, and `quantum_trajectories` for quantum
jumps drawn from the run's random stream. `expectation` makes observables of operators, one value per member for a
group, and every quantum run also records its `energy`, and its `purity` when it is open. Groups on a lattice stream
the space-time image of their local expectations to Rerun. The quantum formalism also holds its analyses: `spectrum`
and `ground_state` diagonalise the Hamiltonian exactly, optionally in the sector of a conserved diagonal operator,
`purity` and `entanglement_entropy` read trajectories, and `hamiltonian_of` prints the Hamiltonian in operator
notation.

```python
from skcomplex.analysis import spectrum
from skcomplex.models import create

print(spectrum(create('quantum_heisenberg', n_sites=4), k=2).energies.round(6).tolist())
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
- The `CoSimulationSolver` advances every partition with its own solver, chosen per partition through
  `solvers={'land': ScipyODESolver()}`, and exchanges inputs every `sync_dt`, which defaults to the frame duration and must
  divide it, and must be a whole number of steps for discrete and agent partitions.
- The Gauss-Seidel scheme, the default, advances partitions in system order, each reading the values already advanced
  by those before it. `scheme='jacobi'` advances all of them from the values at the start of the interval.
- Splitting adds an error of first order in `sync_dt`, so halving `sync_dt` roughly halves it at the cost of more
  exchanges. `CoSimulationSolver(split=True)` splits a system that fits one form, which shows the error against the joint
  solution.

## Visualizing with Rerun

`RerunObserver` streams frames to a spawned viewer, a running viewer, or a `.rrd` file. Entity paths follow the system
tree, each element type chooses its visual, and parameters appear under `parameters/`.

```python
from skcomplex.engine.runtime import Simulation
from skcomplex.core.system import System

import tempfile
from pathlib import Path

from skcomplex.core.spaces import ContinuousSpace, GeneratorNeighborhood, MooreNeighborhood, TriangularNeighborhood, VonNeumannNeighborhood
from skcomplex.core.elements import ParticleElement
from skcomplex.models.physics.realizations.particles import SpringInteraction
from skcomplex.engine.viz import RerunObserver

molecule = System(
    elements=[ParticleElement('a', x=[0.0, 0.0, 0.0]), ParticleElement('b', x=[1.5, 0.0, 0.0], v=[0.0, 0.5, 0.0])],
    interactions=[SpringInteraction('bond', first='a', second='b', k=4.0, length=1.0)],
    space=ContinuousSpace(n_dim=3),
)
path = Path(tempfile.mkdtemp()) / 'molecule.rrd'
Simulation(molecule, dt=0.02, observers=[RerunObserver('molecule', save_path=path)]).run(until=5)
```

Particles appear as 3D points, populations as time series, agents on grids and in continuous spaces as points
colored by a state, agents on networks as graph nodes, and reporters under `reporters/`. A group is logged as one batch per state, and
`PointsVisual(position='x', color_by='v')` colors its points by speed. Points and agents in a continuous space without
a given `radius` take a share of the extent of the view, and a group of at most ten points draws the trail of its recent
positions, which `trail=0` turns off. A one-dimensional lattice also logs a space-time image, one row per logged frame
with the oldest at the top, holding at most `ImageVisual(history=...)` rows. A complex state of a lattice or a mesh
appears as two images, its modulus under `<path>/<state>/modulus` colored over the range and its phase under
`<path>/<state>/phase` colored by a cyclic palette over `(-π, π]`, and a complex scalar as the series
`<path>/<state>/real` and `<path>/<state>/imag`. A group can set its own visual with
`GroupElement(..., element=ParticleElement, visual=VisualSpec('points', {...}))`. A new visual subclasses `BaseVisual`, implements the hook `_log`
for elements or `_log_agents` for agents, and is listed under its kind in `VISUALS`, and an element selects it with
its `visual` argument. `visual.prepare(element, problem)` prepares a visual for one element and sets its views
`views_`, and `visual.run(frame, recording)` logs the element at a frame.

### Layouts and blueprints

Every visual declares the Rerun views that show what it logs, as its fitted attribute `views_`. Scalars go in time series
views, points in 2D or 3D spatial views by their dimension, lattices and fields in one 2D view per state, after the
2D views of their space-time images in one dimension, meshes in 3D views, and agents in a 2D view or, on a network, a
graph view. Agents without a space get no spatial view. `default_blueprint` arranges these views into a Rerun
blueprint:
- spatial views come first and side by side, with time series views below them;
- scalar elements of one type share a time series view with at most six series;
- parameters are left out unless `parameters=True`;
- the side panels are collapsed, the top panel is hidden, and the time panel follows the latest frame.

`primary_view` returns the first view, the one that shows the system's signature behaviour. `RerunObserver` sends a
blueprint with `blueprint='default'` or a blueprint of your own, and writes it next to a saved recording as a `.rbl` file.
`save_blueprint` writes any blueprint to a file.

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

Complex systems are studied through many runs. `Experiment` runs a model, a registered name, a factory, or a system,
over a design of factory arguments and dotted paths, scenarios, and replicates. Every design row and replicate gets
its own random stream spawned from `random_state` in a fixed order, so results are the same serially, on threads, or on
processes, and every scenario of a row and replicate shares its stream. Parameter and initial-state overrides reuse one
compiled problem, so sweeps of thousands of runs compile once.

```python
from skcomplex.experiments import Experiment, GridDesign, LatinHypercubeDesign, Scenario, calibrate, sensitivity

ensemble = Experiment('sir', GridDesign(stochastic=[True], gamma=[0.1, 0.2]), replicates=10, until=60, random_state=0)
print(ensemble.run().summary()[0]['I.n.mean'])

policies = [Scenario('baseline'), Scenario('quota', schedule=[(50.0, 'harvest.effort', 0.025)])]
stocks = Experiment('harvesting', scenarios=policies, until=80, outputs=['land/stock.n']).run()
print([round(float(row['land/stock.n'])) for row in stocks.rows])
```

- Designs are `GridDesign`, `RandomDesign`, `LatinHypercubeDesign`, `SobolDesign`, and `RowsDesign`. `prepare` sets their
  rows in `rows_` and `run` returns them. They cross with `product` and concatenate with `concat`, and their bounds are
  pairs or SciPy distributions.
- `outputs` keeps states and observables as series, after `warmup`, and functions of a trajectory as scalars. Results
  give one row per run, `series`, `summary` with means, standard deviations, and quantiles across replicates, and
  `to_frame` for pandas.
- `sensitivity` ranks parameters one at a time, by Morris screening, or by Sobol indices with SALib.
- `calibrate` fits dotted paths to observed series by least squares or a Gaussian likelihood, and reports the
  covariance of the estimates with R², RMSE, MAE, and MAPE.

```python
from skcomplex.experiments import calibrate

import numpy as np

times = np.arange(0.0, 21.0)
observed = 500 / (1 + 49 * np.exp(-0.8 * times))
fit = calibrate('logistic_growth', {'population.n': (times, observed)}, {'growth.r': (0.1, 2.0), 'growth.K': (100, 1000)})
print(round(fit.estimates['growth.r'], 2), round(fit.r2, 4))
```

## Analysing symbolically

`equations_of` returns the equations of a system as Sage relations, and `solve_symbolically` solves an ODE system
exactly where Maxima can, with `desolve` for one equation and `desolve_system` for linear systems with constant
coefficients.

```python
from skcomplex.core.elements import ParticleElement
from skcomplex.core.spaces import ContinuousSpace, GeneratorNeighborhood, MooreNeighborhood, TriangularNeighborhood, VonNeumannNeighborhood
from skcomplex.core.system import System

from skcomplex.analysis import equations_of, solve_symbolically
from skcomplex.models.physics.realizations.particles import HarmonicPotentialInteraction

oscillator = System(
    elements=[ParticleElement('p', x=1.0)],
    interactions=[HarmonicPotentialInteraction('trap', target='p', k=4.0)],
    space=ContinuousSpace(n_dim=1),
)
print(equations_of(oscillator)['p.v[0]'])
print(solve_symbolically(oscillator, use_initial_state=True)['p.x[0]'])
```


## Analysing dynamics and data

`skcomplex.analysis` measures complex dynamics in the same way for every field. Every measure takes a system, or plain
arrays from trajectories or measured data.

**Stability and chaos.** `fixed_points`, `jacobian`, `stability`, `nullclines`, and `divergence` work on the symbolic
vector field of a flow or a map. `lyapunov_spectrum` integrates its tangent dynamics compiled with Numba, and
`orbit_diagram` and `basin_stability` run experiments over parameters or initial conditions.

```python
from skcomplex.analysis import fixed_points, lyapunov_spectrum, orbit_diagram, stability
from skcomplex.models.biology.ecology import lotka_volterra
from skcomplex.models.mathematics.dynamical_systems import logistic_map, lorenz

predation = lotka_volterra()
print([stability(predation, point).classification for point in fixed_points(predation)])
print(lyapunov_spectrum(lorenz(), until=200).round(2))
diagram = orbit_diagram(logistic_map, 'map.r', [2.9, 3.2, 3.5], 'map.x', transient=500, keep=32, dt=1)
print(diagram.periods())
```

**Sections.** `poincare_section` returns the states where a state crosses a value in a direction, located like an
event, and `stroboscopic` samples a periodically forced system once per period.

```python
import numpy as np

import math

from skcomplex.analysis import stroboscopic
from skcomplex.models.physics.nonlinear_oscillators import duffing

section = stroboscopic(duffing(), 2 * math.pi / 1.2, n_points=200)
print(len(np.unique(np.round(section.states, 4), axis=0)))
```

**Distributions and events.** `histogram` with logarithmic bins, `ccdf`, `fit_power_law`, `hill`, and `gini` describe
heavy tails, and `detect_events` turns series into avalanches or bursts with sizes, durations, and waiting times.
`fit_power_law` adapts the `powerlaw` library: it chooses the lower cutoff by Clauset, Shalizi, and Newman's
Kolmogorov-Smirnov criterion, accepts an upper cutoff, uses the exact discrete likelihood for integers, gives a
bootstrap p-value, and compares the power law with exponential, log-normal, stretched exponential, and truncated
power-law tails.

```python
import numpy as np

from skcomplex.analysis import detect_events, fit_power_law, gini

samples = (1 - np.random.default_rng(0).random(5000)) ** (-1 / 1.5)
fit = fit_power_law(samples, bootstrap=20, random_state=0)
print(round(fit.alpha, 2), fit.p_value, round(gini(samples), 2))
print(detect_events([0, 2, 3, 0, 0, 1, 0]).sizes)
```

**Time series and information.** `autocorrelation`, `autocorrelation_time`, `power_spectrum`,
`mean_squared_displacement`, `detrended_fluctuation`, `hurst_exponent`, and `early_warning` describe memory and
diffusion. The mean squared displacement comes from `tidynamics` over all lags of unwrapped positions, with its exponent
and diffusion coefficient fitted over a chosen range of lags. The fluctuation analyses come from `nolds`: detrended
fluctuation of any order, which gives `H + 1` for an integrated series such as a random walk, or the rescaled range.
The early warnings add rolling skewness and spectral reddening to variance and lag-1 autocorrelation. `symbolize`,
`ordinal_patterns`, `mutual_information`, and `transfer_entropy` measure structure and information flow, as in
Lindgren's information theory of complex systems, and `EntropyAnalysis` gives the Shannon, block, rate, excess,
permutation, and Lempel-Ziv entropies by its `kind`. Ordinal patterns are coded by the lexicographic rank of their
permutation with `ordpy`, with a delay and of any length, and the Lempel-Ziv parsing comes from `antropy`.

```python
import numpy as np

from skcomplex.analysis import EntropyAnalysis, hurst_exponent, transfer_entropy

noise = np.random.default_rng(1).standard_normal(20_000)
print(round(hurst_exponent(noise), 2), round(hurst_exponent(np.cumsum(noise)), 2))
for kind in ('rate', 'excess'):
    print(EntropyAnalysis(kind, dimension=4).check().prepare([0, 0, 1, 1] * 200).run().entropy_.round(3))
driver = np.random.default_rng(2).integers(0, 2, 5000)
print(round(transfer_entropy(driver, np.roll(driver, 1)), 2))
```

**Space and networks.** `label_clusters` labels clusters of a grid, with periodic edges, `correlation_function` gives
radially averaged correlations, and `box_counting_dimension` measures fractals. `degrees`, `summarize_network`, and
`centrality` measure NetworkX graphs and network spaces.

```python
import numpy as np

import networkx as nx

from skcomplex.analysis import box_counting_dimension, label_clusters, summarize_network

clusters = label_clusters(np.random.default_rng(3).random((100, 100)) < 0.6)
print(clusters.spans(0), clusters.sizes.max())
rows, columns = np.indices((512, 512))
print(round(box_counting_dimension((rows & columns) == columns), 2))
print(summarize_network(nx.watts_strogatz_graph(200, 6, 0.1, seed=0)).diameter)
```

### Kinds of data and signatures

Every analysis consumes one kind of data, never a model or a field of science: a series, a jump path, a field, agent
snapshots, a network, or an ensemble of runs. The same `structure_factor` serves an Ising lattice and a reaction-diffusion
field, and the same `avalanches` serves a sandpile and an earthquake model. Every analysis takes plain arrays or graphs,
so it works on measured data as well.

The adapters `series_of`, `path_of`, `fields_of`, `snapshots_of`, `networks_of`, and `ensemble_of` extract each kind from
a trajectory or an experiment result. `snapshots_of` unwraps positions across periodic sides, so displacements never
jump across the box, and `path_of` keeps only the frames where a discrete state changed. `data_kinds` names the kinds an
object holds, and `analyses_for` lists the registered analyses that apply to a trajectory, a system, an experiment
result, or an array, with the adapter that feeds each and how it reads the object. The `analysis` decorator registers a
new analysis under its kind.

```python
import numpy as np

from skcomplex.analysis import analyses_for, data_kinds, fields_of, series_of, structure_factor
from skcomplex.models import run

spins = run('ising', size=32, settings={'until': 20}).trajectory_
print([kind.value for kind in data_kinds(spins)])
print(structure_factor(fields_of(spins)[-1]).wavenumbers[:3].round(3))
predation = run('lotka_volterra', settings={'until': 20.0}).trajectory_
print(series_of(predation).labels)
print(sorted({item.name for item in analyses_for(np.zeros((16, 16)))})[:4])
```

`signature_analysis` computes one default analysis per kind of data, with its curves, axis labels, title, and measured
values: the power spectrum and the autocorrelation time of series, or the return map of a map, the waiting times and
the Fano factor of jump paths, the structure factor of the last frame of fields with its correlation length or peak
wavelength, the polar and milling orders of moving agents or their nearest-neighbour distances, the degree distribution
of networks, and the mean with a 90 percent band of ensembles. The gallery draws it for every example that names no
analysis of its own.

```python continuation
from skcomplex.analysis import signature_analysis

for signature in signature_analysis(predation):
    print(signature.kind.value, signature.title, signature.xscale, list(signature.values))
```

### Complexity and information

`EntropyAnalysis` measures every entropy by its `kind`. The kinds `'renyi'` and `'tsallis'` generalize the Shannon
entropy of a distribution. The kinds `'sample'`, `'approximate'`, `'multiscale'`, and `'spectral'` come from `antropy`
and measure the regularity of a series. The kind `'permutation'` reads ordinal patterns of series with a delay, or of
two-dimensional fields in a window, in its Shannon, Rényi, or Tsallis `measure`. Discrete states such as spins have
many equal values, which `ties='random'` ranks at random. `ensemble_entropy` follows the entropy of an ensemble and its
divergence from a reference law frame by frame, as in the H-theorem, and `complexity_entropy` places data on Rosso's
complexity-entropy plane between its bounding curves with `ordpy`. `active_information_storage` measures the information the past of a series holds
about its next value, and `transfer_entropy` estimates the flow between series by plug-in counts of symbols or by the
nearest-neighbour estimator of `ennemi` on continuous values.

```python
import numpy as np

import math

from skcomplex.analysis import EntropyAnalysis, complexity_entropy, transfer_entropy

rng = np.random.default_rng(0)
renyi = EntropyAnalysis('renyi', order=2, probabilities=True).check().prepare(np.full(8, 1 / 8)).run()
print(renyi.entropy_.round(3))
sample = EntropyAnalysis('sample', dimension=2).check().prepare(rng.standard_normal(2000)).run()
print(sample.entropy_.round(1))
plane = complexity_entropy(rng.standard_normal(20_000), 4)
print(round(plane.entropy, 2), round(plane.complexity, 2))
x = rng.standard_normal(5000)
y = np.concatenate([[0.0], 0.8 * x[:-1]]) + 0.6 * rng.standard_normal(5000)
print(transfer_entropy(x, y, estimator='knn', base=math.e) > transfer_entropy(y, x, estimator='knn', base=math.e))
```

### Higher-order information and emergence

The measures of many variables take one row per sample and one column per variable. `total_correlation` counts the
information shared at any order, `dual_total_correlation` the information shared by two or more variables, and
`o_information`, their difference, is positive when the variables are redundant and negative when they are synergistic.
`o_information_gradients` gives the share of every variable. They take the `'plugin'` estimator of `dit` for symbols,
`'gaussian'` for the closed form of a fitted Gaussian, or `'knn'` for chains of `ennemi` estimates. `partial_information`
splits what two sources of symbols tell about a target into redundant, unique, and synergistic parts with a redundancy
measure of `dit`. `local_entropy`, `local_mutual_information`, and `local_transfer_entropy` return the value at every
sample, and their means equal the global measures. `surrogate_test` compares any of these measures with its values on
copies of the data whose variables are shuffled independently.

```python continuation
import numpy as np

from skcomplex.analysis import o_information, partial_information, surrogate_test

bits = rng.integers(0, 2, (5000, 2))
xor = np.column_stack([bits, bits[:, 0] ^ bits[:, 1]])
print(round(o_information(xor), 2))
parts = partial_information(bits[:, 0], bits[:, 1], bits[:, 0] & bits[:, 1])
print(round(parts.redundancy, 2), round(parts.synergy, 2))
print(surrogate_test(o_information, xor, n=19, alternative='less', random_state=0).p_value)
```

`emergence_criteria` computes the criteria of Rosas et al. (2020) for a macro feature of micro variables: Ψ is positive
when the feature predicts its own future beyond what each micro variable predicts, Δ is positive when it predicts a micro
variable beyond the micro variables, and Γ is zero for a feature decoupled from every micro variable. `closure` takes a
partition of micro states, as a label per state or a function of the micro series, and compares how well the macro past
and the micro past predict the macro future. A gap of zero means the coarse-graining is closed. `epsilon_machine`
reconstructs the causal states of a discrete series by causal-state splitting, with their transitions, statistical
complexity, and entropy rate.

```python continuation
import numpy as np

from skcomplex.analysis import closure, emergence_criteria, epsilon_machine

parity = rng.integers(0, 2, (5000, 4))
parity[:, -1] = (parity[:, :-1].sum(axis=1) + np.arange(5000) // 100) % 2
print(emergence_criteria(parity, parity.sum(axis=1) % 2).psi > 0.5)
print(round(closure(rng.integers(0, 4, 5000), [0, 0, 1, 1]).gap, 2))
golden = [1]
for _ in range(20_000):
    golden.append(1 if golden[-1] == 0 else int(rng.random() < 0.5))
machine = epsilon_machine(golden, 4)
print(len(machine.histories), round(machine.statistical_complexity, 2))
```

### Invariants of dynamics

`embed` builds delay coordinates of a series, with the delay at the first minimum of `auto_mutual_information` and the
dimension where `false_nearest_neighbors` fall below a threshold. `correlation_dimension` counts close pairs of any point
set with a k-d tree, so it serves embedded series and agent positions alike. `kaplan_yorke_dimension` and
`kolmogorov_sinai_entropy` follow from a Lyapunov spectrum, `largest_lyapunov_exponent` estimates the largest exponent
from one series with Rosenstein's method of `nolds`, and `ErgodicityAnalysis` of kind `'invariant-density'` or `'ulam'`
estimates the invariant density of a map. `recurrence_matrix` and `recurrence_quantification` give the recurrence rate, determinism,
laminarity, line lengths, trapping time, and line entropy of embedded series.

```python
import numpy as np

from skcomplex.analysis import (
    ErgodicityAnalysis,
    correlation_dimension,
    delay_embedding,
    kaplan_yorke_dimension,
    recurrence_quantification,
)

points = np.random.default_rng(1).random((3000, 2))
print(round(correlation_dimension(points).dimension, 1))
print(round(kaplan_yorke_dimension([0.906, 0.0, -14.572]), 2))
chain = ErgodicityAnalysis('ulam', edges=np.linspace(0, 1, 21), samples=200)
print(chain.check().prepare(lambda x: 4 * x * (1 - x)).run().density_.round(1)[:3])
sine = delay_embedding(np.sin(np.arange(1000) * 0.1), 2, 15)
print(round(recurrence_quantification(sine).determinism, 2))
```

### Surrogates fitted to data

`DMDSurrogate`, `SINDySurrogate`, and `ReservoirSurrogate` are estimators: they take their settings when built,
`prepare` them to one or more trajectories or series sampled at a uniform step, and `run(steps)` forecasts the variables
over that many samples after the last sample. `prepare` sets `system_`, a `System` of one surrogate element, which
`validate` checks and `Simulation` runs like a hand-written system, and `training_error_` and `validation_error_`, the
relative errors of the fit on the training samples and on the end of every series held out by `validation`. A forecast
simulates `system_` from the last sample, and the DMD and reservoir surrogates compute it directly.

- `DMDSurrogate` fits dynamic mode decomposition with PyDMD and sets the continuous eigenvalues `ln(mu) / dt` in
  `eigenvalues_`, the modes, their amplitudes, and a surrogate under continuous time with the matrix rate law
  `dg/dt = L g`. A scikit-learn transformer passed as `dictionary`, such as `PolynomialFeatures(2)`, lifts the
  variables to observables first, which is extended DMD.
- `SINDySurrogate` fits sparse identification of nonlinear dynamics with PySINDy and sets the monomials of the
  variables in `features_`, their coefficients in `coefficients_`, and a surrogate under continuous time with one
  symbolic rate law per variable.
- `ReservoirSurrogate` fits a next-generation reservoir or an echo state network from ReservoirPy with a ridge readout
  of the increments over one sample. Its surrogate runs under discrete time, one step per sample, holds the memory of
  the reservoir as states, and starts at the last sample, so a run of `system_` is the autonomous forecast.

```python
import numpy as np
from skcomplex.engine.runtime import Simulation

from skcomplex.analysis import DMDSurrogate, ReservoirSurrogate, Series, SINDySurrogate
from skcomplex.engine.compile import validate

times = np.arange(0.0, 10.0, 0.01)
spiral = Series(times, np.exp(-0.1 * times)[:, None] * np.column_stack([np.cos(times), np.sin(times)]), ['x', 'y'])
decomposition = DMDSurrogate().prepare(spiral)
print(np.sort(decomposition.eigenvalues_.imag).round(3), validate(decomposition.system_).ok)
sparse = SINDySurrogate(degree=1, threshold=0.05).prepare(spiral)
print(sparse.features_, sparse.coefficients_.round(2).tolist(), sparse.run(50).shape)
reservoir = ReservoirSurrogate().prepare(spiral)
forecast = Simulation(reservoir.system_).run(until=50).trajectory_
print(type(reservoir.system_.time).__name__, np.allclose(forecast.values('surrogate.x')[1:], reservoir.run(50)[:, 0]))
```

### Stochastic processes

`PassageAnalysis` measures first passages, absorption, and returns of paths, and the committor and mean first-passage
time of a transition matrix, by its `kind`, and `laminar_durations` measures laminar phases. `MarkovAnalysis` estimates
transition matrices from paths, reads transition and rate matrices with `matrix`, and gives their stationary laws,
spectral gaps, and relaxation times, and Markov state models and implied timescales with `deeptime`. `fano_factor`,
`event_counts`, and `relative_fluctuations` describe fluctuations, and `KramersMoyalAnalysis` estimates the drift and
diffusion of a series with `kramersmoyal`. `IrreversibilityAnalysis` measures entropy production from rates, jump paths,
Langevin paths, or transition counts, tests detailed balance and the fluctuation theorem, and checks the thermodynamic
uncertainty relation: the squared relative fluctuation of a current times the entropy produced is at least 2. It also
gives the response to a field and the fluctuation-dissipation ratio. `LargeDeviationAnalysis` gives the scaled cumulant
generating function from a tilted generator, from blocks of one run, or by cloning a population of trajectories of a
model through the simulation's steppers, for systems whose tilted generators are too large to write down, and its
Legendre transform at `values`. It also gives rate functions from histograms at several sizes and reweights samples.
`two_time_correlation`, `aging_collapse`, and `collapse_distributions` measure aging, and `fit_extreme_values`,
`record_counts`, `fit_stable`, and `compare_with_law` describe extremes and limit laws.

```python
import numpy as np

from skcomplex.analysis import IrreversibilityAnalysis, LargeDeviationAnalysis, MarkovAnalysis

ring = 2 * np.roll(np.eye(5), 1, axis=1) + np.roll(np.eye(5), -1, axis=1)
stationary = MarkovAnalysis('stationary', matrix='rate').check().prepare(ring).run().stationary_
production = IrreversibilityAnalysis().check().prepare(ring).run().entropy_production_
print(stationary.round(2), round(production, 4))
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
import numpy as np
from skcomplex.core.components import Element
from skcomplex.core.system import System

from skcomplex.analysis import IrreversibilityAnalysis, LargeDeviationAnalysis
from skcomplex.core.laws import JumpLaw
from skcomplex.core.declarations import state

class CounterElement(Element):
    """Counter element."""

    n = state(0.0)

    def laws(self, s):
        """Return the laws of the counter element."""
        return [JumpLaw(1.0, {s.n: 1})]
counter = System(elements=[CounterElement('counter')])
cloning = LargeDeviationAnalysis(
    'cloning', observable='counter.n', tilts=0.5, population=200, until=10.0, random_state=0
)
print(cloning.check().prepare(counter).run().scgf_.round(2), round(float(np.exp(0.5) - 1), 2))
currents = np.random.default_rng(0).normal(10.0, np.sqrt(30.0), 5000)
relation = IrreversibilityAnalysis('uncertainty-relation', entropy=10.0, random_state=0)
print(relation.check().prepare(currents).run().holds_)
```

### Emergence and ergodicity

`partition_coarse_grain` maps micro states to macro states by a partition, an integer label per state or a function of
the state. It maps every frame of a discrete trajectory, or sums the transitions of a transition matrix within every
block with `matrix=True`, weighting the states of a block uniformly or by the stationary distribution.
`effective_information` measures how much a transition matrix constrains the next state: it is the mutual information
in bits between a uniform intervention on the current state and the next state, and it equals the determinism of the
matrix minus its degeneracy. Rows with no observed transition are left out with an `UnobservedStatesWarning` that names
their states. `causal_emergence` reports the gain in effective information of a partition over the micro states, and
without a partition it searches for one, greedily by merging the pair of blocks with the largest gain, or exhaustively
for at most 10 states. In the chain of Hoel, Albantakis, and Tononi, seven noisy states merge into one deterministic
macro state that carries more effective information than the micro states.

```python
import numpy as np

from skcomplex.analysis import causal_emergence, effective_information, partition_coarse_grain

noisy = np.zeros((8, 8))
noisy[:7, :7], noisy[7, 7] = 1 / 7, 1.0
print(round(effective_information(noisy).information, 3))
found = causal_emergence(noisy)
print(found.partition.tolist(), round(found.macro.information, 3), round(found.gain, 3))
print(partition_coarse_grain(noisy, found.partition, matrix=True).tolist())
```

`ErgodicityAnalysis` of kind `'breaking'`, `'metric'`, or `'distance'` diagnoses whether the runs of an ensemble, one
row per run or an `Ensemble` from `ensemble_of`, explore the same states. The ergodicity-breaking parameter is the
variance of the time averages across runs over the variance of all values, which tends to 0 for ergodic runs and stays
near 1 for runs frozen in separate states. The fluctuation metric of Thirumalai and Mountain is the variance of the
running time averages, which decays as `1 / t` for ergodic runs and levels off for frozen ones. The ergodicity distance
is the median total-variation distance between the values of one run over time and the values of the whole ensemble.

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

`structure_factor` averages the power of a field radially on a periodic grid of any dimension, and `correlation_length`
fits its Ornstein-Zernike form or the decay of the correlation function. `domain_size` and `domain_growth` measure
coarsening, `pattern_wavelength` the wavelength and orientation of patterns, and `number_fluctuations` the giant number
fluctuations of density fields. `cluster_sizes`, `percolation`, and `spanning_probability` describe clusters of grids
and components of graphs, and `binder_cumulant`, `binder_crossings`, and `scaling_collapse` locate transitions by
finite-size scaling. `dispersion_relation` predicts patterns from a field model's laws before running it: it
linearises them around a uniform state, turns every Laplacian into its eigenvalue on each Fourier mode of the lattice,
and returns the growth rate and frequency of every mode, exact for the model's discretization. A band of growing modes
is a Turing instability, and a growing mode with a nonzero frequency travels.

```python
import numpy as np

from skcomplex.analysis import binder_cumulant, dispersion_relation, pattern_wavelength, percolation
from skcomplex.models.chemistry.pattern_formation import brusselator

relation = dispersion_relation(brusselator((128,), b=1.9))
print(relation.band.min().round(2), relation.band.max().round(2))

stripes = np.cos(2 * np.pi * np.arange(64) / 8)[:, None] * np.ones(64)
print(pattern_wavelength(stripes).wavelength)
print(percolation(np.random.default_rng(2).random((128, 128)) < 0.7).spanning)
print(round(binder_cumulant(np.random.default_rng(3).standard_normal(10_000)), 1))
```

### Agents

`polar_order`, `nematic_order`, `milling_order`, and `phase_order` measure collective order. `kinetic_temperature` and
`maxwell_boltzmann_test` compare velocities with the Maxwell-Boltzmann law of their dimension.
`nearest_neighbor_distances`, `clark_evans_ratio`, and `radial_distribution` describe point patterns in open or periodic
boxes, and `coarse_grain` turns agents into density or polarization fields for the field analyses.
`MixingAnalysis` measures segregation by the share of like neighbours on grids and networks, or by the dissimilarity
index of two groups across units, and `interface_density` gives the share of unlike neighbouring pairs.

```python
import numpy as np
from skcomplex.models import run

from skcomplex.analysis import clark_evans_ratio, coarse_grain, polar_order, snapshots_of

flock = snapshots_of(run('boids', settings={'until': 5}).trajectory_)
print(polar_order(flock.velocities, vectors=True).shape)
positions = np.random.default_rng(4).random((1000, 2))
print(round(clark_evans_ratio(positions, [1.0, 1.0]), 1), coarse_grain(positions, [1.0, 1.0], [8, 8]).sum())
```

### Criticality

`avalanches` extracts avalanches from activity series or relaxation records, `fit_avalanches` fits their size and
duration laws, and `crackling_noise` checks the scaling relation between the exponents and collapses mean shapes.
`branching_ratio` estimates the branching ratio by the multistep regression of `mrestimator`, which stays unbiased under
subsampling. `order_parameter_sweep` gives the order parameter, the susceptibility, and the Binder cumulant over a
sweep with the peak of the susceptibility, `fit_order_exponent` fits the order-parameter exponent, and
`replica_overlap` compares replicas.

```python
import numpy as np

from skcomplex.analysis import avalanches, branching_ratio

found = avalanches([0, 1, 3, 2, 0, 0, 4, 1, 0])
print(found.sizes, found.durations)
rng = np.random.default_rng(5)
activity = np.zeros(5000)
activity[0] = 50
for t in range(1, activity.size):
    activity[t] = rng.poisson(0.8 * activity[t - 1] + 10)
print(round(branching_ratio(activity), 1))
```

### Networks

`degree_moments`, `assortativity`, `neighbor_degree`, and `communities` measure degrees, mixing, and communities, the
last by Louvain modularity maximization or label propagation with its agreement with a reference partition.
`motif_counts` and `motif_zscores` count motifs of three and four nodes and score them against degree-preserving edge
switches or the configuration model, and `network_frames` measures every frame of an evolving network. The library
model `network_structure_analysis` records a network with these measures.

```python
import networkx as nx

from skcomplex.analysis import communities, motif_counts

club = nx.karate_club_graph()
print(round(communities(club, random_state=0).modularity, 2), motif_counts(club)['triangle'])
```
