# User guide

## Concepts

`scikit-complexity` separates the description of a model from the way it is solved.

- An **element** is an entity that carries state, such as a particle, a population, a compartment, or a market.
- An **interaction** couples elements through **roles** and describes its effect with **laws**.
- A **law** says how a state changes. It is one of four kinds.
    - `Rate` adds a term to the time derivative of a state. Rates on one state add up.
    - `Noise` drives a state with an independent Wiener process.
    - `Jump` fires a random event that changes states by fixed increments.
    - `Update` gives the next value of a state after one discrete step.
- A **space** gives vector states, such as positions, their dimension.
- A **system** holds elements, interactions, child systems, an optional space, and **observables**, named quantities
  computed from its states at every frame.

These concepts carry no field-specific meaning. A population, a particle, a firm, and a cell are all elements, so a
model from any science is a configuration of the same interface, and every ready-made model in `skcomplex.models` is a
factory that builds such a configuration.

Compiling a system infers its mathematical form from its law kinds and builds fast numeric functions.

| Law kinds | Form | Default solver |
| --- | --- | --- |
| `Rate` | ordinary differential equations | `ScipyODE` with `LSODA` |
| `Rate` and `Noise` | stochastic differential equations | `EulerMaruyama` |
| `Jump` | jump process | `Gillespie` |
| `Update` | discrete map | `Iteration` |

## Defining elements and interactions

Elements and interactions declare their fields with `state`, `parameter`, and `role`. Each field becomes a constructor
argument, stored unmodified under its own name.

```python
from skcomplex.core.base import Domain, parameter, role, state
from skcomplex.core.components import Element
from skcomplex.core.interactions import Interaction
from skcomplex.core.laws import Jump, Rate


class Species(Element):
    n: float = state(10.0, domain=Domain.NONNEGATIVE)


class Birth(Interaction):
    target: str = role()
    rate: float = parameter(0.5, domain=Domain.NONNEGATIVE)
    stochastic: bool = False

    def laws(self, s):
        if self.stochastic:
            return [Jump(s.rate * s.target.n, {s.target.n: 1})]
        return [Rate(s.target.n, s.rate * s.target.n)]
```

Inside `laws`, `s.rate` is the symbol of the parameter and `s.target.n` is the symbol of the state of the element bound
to the role `target`. Because parameters stay symbols, a running simulation can change them without compiling again.

A law can also be a Python function of time and current values, which covers rules that have no closed form.

```python
class Threshold(Interaction):
    target: str = role()

    def laws(self, s):
        name = f'{self.target}.n'
        return [Rate(s.target.n, lambda t, values: -1.0 if values[name] > 50 else 0.0)]
```

### Symbolic expressions

Laws are Sage expressions. The symbols in `s` are Sage symbols that carry the domains of their variables, vector states
are Sage vectors, and mathematical functions come from Sage. Sage has no condition-based piecewise expression or
symbolic remainder, so `skcomplex.core.laws` adds `where`, `either`, `both`, and `mod`, which every backend lowers, LAMMPS
included.

```python
from sage.all__sagemath_symbolics import exp, sqrt

from skcomplex.core.laws import where


class Crowding(Interaction):
    target: str = role()
    capacity: float = parameter(100.0, domain=Domain.POSITIVE)

    def laws(self, s):
        n = s.target.n
        return [Rate(n, where(n > s.capacity, -sqrt(n - s.capacity), 0.1 * n * exp(-n / s.capacity)))]
```

Models written with SymPy move to Sage as follows. A SymPy expression in a law raises a `ValidationError`, and
`expression._sage_()` converts one that is already built.

| SymPy | Sage and `skcomplex` |
| --- | --- |
| `sympy.sqrt`, `sympy.exp`, `sympy.sin`, and other functions | `sqrt`, `exp`, `sin`, and others from `sage.all__sagemath_symbolics` |
| `sympy.Symbol('x', positive=True)` | `skcomplex.core.base.symbol('x', Domain.POSITIVE)` |
| `sympy.Matrix([...])` and `a.dot(b)` | `skcomplex.core.base.vector([...])` and `a.dot_product(b)` |
| `sympy.Piecewise((a, condition), (b, True))` | `where(condition, a, b)` |
| `condition1 \| condition2` and `condition1 & condition2` | `either(condition1, condition2)` and `both(condition1, condition2)` |
| `sympy.Mod(x, m)` | `mod(x, m)` |
| `sympy.diff(f, x)` and `f.subs(x, y)` | `f.diff(x)` and `f.subs({x: y})` |
| `sympy.Eq(a, b)`, `.lhs`, and `.rhs` | `a == b`, `.lhs()`, and `.rhs()` |

### Defining without classes

`Element.define` and `Interaction.define` build the same types from declarations and a laws function, which suits
one-off models. The types behave like classes in groups, cloning, and parameter introspection, and pickle when they are
assigned to a module-level name equal to their name.

```python
from skcomplex.core.components import Element

SpeciesDefined = Element.define('SpeciesDefined', states={'n': state(10.0, domain=Domain.NONNEGATIVE)})
BirthDefined = Interaction.define(
    'BirthDefined',
    roles=['target'],
    parameters={'rate': parameter(0.5, domain=Domain.NONNEGATIVE)},
    laws=lambda s: [Rate(s.target.n, s.rate * s.target.n)],
)
print(BirthDefined('birth', target='deer', rate=0.2))
```

### State domains

Besides real, positive, nonnegative, and integer values, a state takes booleans, angles, or the labels of a finite
alphabet. Values are stored as floats: booleans as 0 and 1, labels as their index, and angles wrapped into one turn
after every frame. Trajectories return booleans and labels, and `Simulation(..., checked=True)` raises an error naming
any state that leaves its domain.

```python
health = Domain.categorical('susceptible', 'infected', 'recovered')
Person = Element.define('Person', states={'health': state('susceptible', domain=health)})
Rotor = Element.define('Rotor', states={'theta': state(0.0, domain=Domain.ANGLE)})
```

### Complex states

A state or parameter of `Domain.COMPLEX` takes complex values, and its laws use the imaginary unit `I`, arithmetic,
the elementary functions over the complex plane, and `real_part`, `imag_part`, `conjugate`, `abs`, and `arg`. The
compiler writes every complex law as the laws of its real and imaginary parts, which every solver, stepper, and backend
runs as a real system. A real state takes only real values, so `abs(z)` or `real_part(z)` turns a complex value into
one, and a condition compares the modulus or a part, since complex values have no order. Trajectories and frames return
complex arrays, whose `.real`, `.imag`, `np.abs`, and `np.angle` are the real views that the analysis functions take.

```python
import numpy as np
from sage.all__sagemath_symbolics import I

from skcomplex.core.base import Domain, parameter, state
from skcomplex.core.components import Element
from skcomplex.core.laws import Rate
from skcomplex.core.system import System
from skcomplex.core.runtime import Simulation

Phase = Element.define(
    'Phase',
    states={'z': state(1.0, domain=Domain.COMPLEX)},
    parameters={'omega': parameter(2.0)},
    laws=lambda s: [Rate(s.z, I * s.omega * s.z)],
)
z = Simulation(System(components=[Phase('osc')]), dt=0.5).run(until=10).trajectory_['osc.z']
assert z.dtype == np.complex128 and np.allclose(np.abs(z), 1.0, atol=1e-4)
```

Agent states stay real, and the LAMMPS backend raises an error naming a complex state.

### Equations of any order

Laws give the highest time derivative of a state, and a state declares how high that is with `order`. Newton's second
law for a pendulum, `m l θ'' = -m g sin θ`, is written as it reads. The framework adds the angular velocity as the
companion state `theta_d1`, sets its initial value from the constructor, and reduces the equation to first order for
the solvers.

```python
from sage.all__sagemath_symbolics import sin

from skcomplex.core.laws import Equation, derivative

Pendulum = Element.define(
    'Pendulum',
    states={'theta': state(default=0.5, domain=Domain.ANGLE, order=2)},
    parameters={'m': 1.0, 'g': 9.81, 'l': 1.0},
    laws=lambda s: [Equation(s.m * s.l * derivative(s.theta, 2) == -s.m * s.g * sin(s.theta))],
)
```

- On a state of order `k`, rate and noise laws contribute to the `k`-th derivative and add up, so separate interactions
  supply separate forces, voltages, or growth terms: `Rate(s.target.x, -s.k * s.target.x / s.target.m)` from a spring
  and `Rate(s.target.x, -s.c * derivative(s.target.x, 1) / s.target.m)` from a damper superpose.
- `derivative(x, j)` reads the companion `x_d<j>` below the state's order. An `Equation` may contain the derivative of
  the state's own order, linearly, and the compiler solves all equations of a system jointly for those derivatives,
  which couples, for example, the two angles of a double pendulum.
- Any order works the same way, such as Sprott's chaotic jerk equation `x''' = -A x'' - x' + |x| - 1` on a state of
  order 3.
- Difference equations keep past values with `lags`. `lag(x, j)` reads the value `j` steps before the current one, and
  an update law gives the next value, so `Update(s.x, s.x + lag(s.x, 1))` with one lag is the Fibonacci recurrence.

`equations_of` returns these equations in the order they were written, and `solve_symbolically` takes the initial
values of every derivative and returns a solution only when it has a closed form.

## Composing systems

Systems nest. An interaction refers to an element of a child system by a slash-separated path.

```python
from skcomplex.core.system import System
from skcomplex.models.biology.ecology import Population, Predation

landscape = System(
    components=[
        System('forest', components=[Population('foxes', n=5)]),
        System('meadow', components=[Population('rabbits', n=50)]),
    ],
    interactions=[Predation('hunt', predator='forest/foxes', prey='meadow/rabbits', a=0.02, b=0.01)],
)
```

States and parameters are named by their path, such as `meadow/rabbits.n` and `hunt.a`.

## Simulating

`Simulation` compiles the system, picks the solver of its form, and advances it in frames of `dt`. The default ODE
solver is `ScipyODE` with SciPy's `LSODA`, which starts with Adams methods and switches to BDF methods while the system
is stiff, so stiff systems also run with the default solver.

```python
from skcomplex.core.runtime import RealTime, Simulation

trajectory = Simulation(landscape, dt=0.1).run(until=10).trajectory_
print(trajectory['meadow/rabbits.n'][-1])

simulation = Simulation(landscape, dt=0.1).run(until=10)
simulation.update_parameters({'hunt.a': 0.0})
print(len(simulation.extend(until=20).trajectory_))
```

- `dt` defaults to 1 for discrete maps and agents. For other forms it defaults to 0.01, raised to the smallest of 0.02,
  0.05, 0.1, and so on, in steps of 1, 2, and 5 times a power of ten, that keeps the run at most 1000 frames long, so
  `run(until=1000)` takes frames of 1.
- `clock=RealTime(speed)` paces frames so that simulated time advances `speed` times faster than wall time.
- `start(until)` runs in a background thread, and `pause`, `resume`, `stop`, and `wait` control it.
- `update_parameters({'hunt.a': 0.05})` changes parameters before the next frame. Values set between runs stay in
  force for the later runs of the same simulation, and `clone(simulation).run(until)` starts afresh.
- `random_state` makes stochastic runs reproducible.
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
import numpy as np

from skcomplex.core.laws import Event, Noise, Rate, delayed, step

MackeyGlass = Element.define(
    'MackeyGlass',
    states={'x': state(0.5, history=lambda t: 0.5 + 0.1 * np.sin(t))},
    parameters={'beta': 0.2, 'gamma': 0.1, 'n': 10.0, 'tau': 17.0},
    laws=lambda s: [Rate(s.x, s.beta * delayed(s.x, s.tau) / (1 + delayed(s.x, s.tau) ** s.n) - s.gamma * s.x)],
)
cells = System(components=[MackeyGlass('cells')])
print(Simulation(cells, dt=0.5).run(until=300).trajectory_['cells.x'][-1])
```

**Events.** `Event(guard, reset, direction, terminal, label)` fires when the guard crosses zero, upward for
`direction=1`, downward for `-1`, and either way for `0`. The solver locates the crossing on its interpolant to the
precision of the root finder, applies the reset to every named state at once, and restarts from there. A terminal
event ends the run at the crossing. Every frame carries the events since the previous one, and the trajectory keeps
them all with their times and the states before and after the reset. A guard that fires again right after its reset
raises an error that names the chattering event.

```python
Ball = Element.define(
    'Ball',
    states={'h': state(default=1.0, order=2)},
    parameters={'g': 9.81, 'e': 0.8},
    laws=lambda s: [
        Rate(s.h, -s.g),
        Event(s.h, {derivative(s.h, 1): -s.e * derivative(s.h, 1)}, direction=-1, label='impact'),
    ],
)
trajectory = Simulation(System(components=[Ball('ball')]), dt=0.01).run(until=3).trajectory_
print([round(event.time, 6) for event in trajectory.events])
```

**Drivers.** `step`, `ramp`, `pulse`, `pulse_train`, and `series` with `'linear'` or `'previous'` interpolation are
inputs of time `s.t`. The compiler finds the times where they jump or bend, including the periodic edges of pulse
trains, and the solvers stop at every one and evaluate the inputs on the correct side of it, so a step response is
exact to the solver tolerance.

```python
Filter = Element.define(
    'Filter',
    states={'y': 0.0},
    parameters={'k': 1.0},
    laws=lambda s: [Rate(s.y, -s.k * s.y + step(s.t, 5.0))],
)
print(Simulation(System(components=[Filter('filter')]), dt=0.5).run(until=10).trajectory_['filter.y'][-1])
```

**Noise conventions.** `Noise(..., convention='stratonovich')` or `'anti-ito'` declares how a noise law is read. The
compiler adds the drift correction `α g ∂g/∂x`, with `α = 1/2` and `1`, so every SDE solver integrates the equivalent
Itô equation. `EulerMaruyama` has strong order 1/2. `Milstein`, in Platen's derivative-free form, has strong order 1 for
diagonal noise, and `Heun` averages the drift over every step.

```python
from skcomplex.core.solvers import Milstein

Growth = Element.define(
    'Growth',
    states={'x': 1.0},
    parameters={'mu': 0.5, 'sigma': 0.8},
    laws=lambda s: [Rate(s.x, s.mu * s.x), Noise(s.x, s.sigma * s.x, convention='stratonovich')],
)
growth = System(components=[Growth('asset')])
print(Simulation(growth, solver=Milstein(dt=0.01), random_state=0).run(until=1).state_)
```

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
from skcomplex.core.laws import Update, bernoulli, normal, where

Persistent = Element.define(
    'Persistent',
    states={'d': 1.0, 'x': 0.0},
    parameters={'q': 0.8},
    laws=lambda s: [
        Update(s.d, where(bernoulli('turn', 1 - s.q) > 0, -s.d, s.d)),
        Update(s.x, s.x + where(bernoulli('turn', 1 - s.q) > 0, -s.d, s.d) + normal('jitter', 0, 0.1)),
    ],
)
walkers = System(components=[Persistent.group('walkers', 500)])
final = Simulation(walkers, dt=1.0, random_state=0).run(until=200).trajectory_['walkers.x'][-1]
print(round(float(final.var() / 200), 1))
```

Only update laws and renewal laws take draws. A draw in a rate, noise, jump, boundary, equation, or event law raises
`IncompatibleLawsError`, since noise laws and jump laws give randomness in continuous time. Draws are not states, so
the trajectory does not record them, and a model that wants to keep a draw assigns it to a state.

### Markov chains and hidden Markov sources

`Transition(target, matrix)` is the step of a discrete-time Markov chain. The target is a state over a finite alphabet,
a categorical domain or the integers from 0 to `n - 1`, and its current value selects the row of the transition matrix
that the next value is drawn from. `Emission(target, source, matrix)` draws an observed state from the row of an
emission matrix that the hidden state `source` selects, reading the hidden state after the same step, so the two
sequences align frame by frame. Both are update laws, so they combine with other update laws.

Matrix entries are numbers or expressions of parameters. Compiling checks that the entries are nonnegative and that
every row sums to 1, and `update_parameters` checks again every matrix a changed parameter enters, rejecting the change
with a `ParameterError` before it applies.

```python
from skcomplex.core.laws import Emission, Transition

Source = Element.define(
    'Source',
    states={
        'hidden': state(default='calm', domain=Domain.categorical('calm', 'storm')),
        'symbol': state(default='a', domain=Domain.categorical('a', 'b')),
    },
    parameters={'stay': 0.9},
    laws=lambda s: [
        Transition(s.hidden, [[s.stay, 1 - s.stay], [0.4, 0.6]]),
        Emission(s.symbol, s.hidden, [[0.7, 0.3], [0.2, 0.8]]),
    ],
)
simulation = Simulation(System(components=[Source('source')]), dt=1.0, random_state=0).run(until=10000)
print(round(float((simulation.trajectory_['source.hidden'] == 'calm').mean()), 1))
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
from skcomplex.core.laws import draw

Flier = Element.define(
    'Flier',
    states={'x': 0.0},
    laws=lambda s: [Update(s.x, s.x + draw('jump', 'levy_stable', alpha=1.5, beta=0.0))],
)
flight = Simulation(System(components=[Flier.group('fliers', 100)]), dt=1.0, random_state=0).run(until=20)
print(flight.trajectory_['fliers.x'].shape)
```

### Jumps on clocks of waiting times

`Jump(waiting=..., reset=..., label=...)` fires after a waiting time, applies its reset to the states all at once, and draws the
next waiting time from the states after the reset. Every member of a group renews on its own clock. Draws in the
waiting time and the reset take a fresh value at every renewal of every member, and every use of one name in a
renewal reads the same value. A waiting time that is a single draw must come from a distribution with nonnegative
support, and compiling rejects one that can be negative, such as `normal`.

A system or partition whose only laws are such jumps compiles to the renewal form, which `RenewalStepper` runs
exactly, from renewal to renewal with no time grid. A frame holds the states the last renewal at or before its time
set. These jumps also combine with the rate and noise laws of the same component, which evolve the states between
renewals. `RK4` and `EulerMaruyama` stop every member at its own renewals when the members do not interact, and split
the step at every renewal time for all members when a law sums over the group. Other steppers reject these jumps.

```python
from skcomplex.core.laws import Jump

Walker = Element.define(
    'Walker',
    states={'x': 0.0},
    parameters={'alpha': 0.5},
    laws=lambda s: [Jump(waiting=draw('wait', 'pareto', b=s.alpha), reset={s.x: s.x + normal('jump', 0, 1)}, label='jumps')],
)
walk = Simulation(System(components=[Walker.group('walkers', 1000)]), dt=100.0, random_state=0).run(until=1000)
print(walk.problem_.form.value, walk.trajectory_['jumps.renewals'][-1] > 1000)
```

The trajectory records every renewal pass as a `RenewalRecord` among its events, with the time and the member of every
renewal and the member's states before and after it. The observable `<label>.renewals` counts the renewals up to every
frame, and the Rerun observer streams it as a time series and the renewals as text entries under `renewals/<label>`.
The compiler keeps the count of every member in a hidden state, `<group>.<label>~renewals`.

### Fractional noise

`FractionalNoise(target, value, hurst)` drives a state with fractional Brownian motion of Hurst index `0 < H < 1`,
whose increments anticorrelate for `H < 1/2` and correlate for `H > 1/2`. A Hurst index of 1/2 compiles to an ordinary
noise law. For other indices the amplitude must not depend on any state. `FractionalEulerMaruyama(dt, horizon)` is the
stepper of systems with fractional noise. It draws the increments of the whole run on its grid when it first advances,
with the circulant embedding of Davies and Harte over `scipy.fft`, so frames must fall on multiples of `dt`. A
simulation passes the end of its run as the horizon, advancing beyond it raises `HorizonError`, and other SDE steppers
reject fractional noise with an error that names `FractionalEulerMaruyama`. `fractional_gaussian_noise` returns the
noise paths themselves.

```python
from skcomplex.core.laws import FractionalNoise
from skcomplex.core.solvers import FractionalEulerMaruyama

Rough = Element.define('Rough', states={'x': 0.0}, laws=lambda s: [FractionalNoise(s.x, 1, hurst=0.3)])
rough = Simulation(
    System(components=[Rough.group('walkers', 500)]),
    solver=FractionalEulerMaruyama(dt=0.01),
    dt=0.1,
    random_state=0,
).run(until=1)
print(round(float(rough.trajectory_['walkers.x'][-1].var()), 1))
```

The library has these models ready: `continuous_time_random_walk`, `levy_walk`, `annealed_transient_time_motion`,
`fractional_brownian_motion`, and `levy_flight` in soft matter, and `trap_model` in statistical mechanics.

## Observables

Observables are named quantities of a system, computed at every frame, recorded in trajectories under the system's
path, and logged to Rerun. An observable is either an expression of the system namespace, where earlier observables
appear by label, or a reduction over a group's members of a variable or an expression of one member: `count`, `sum`,
`mean`, `min`, `max`, `var`, or a function. Expressions and reductions given as functions must be module-level
functions, so that systems pickle, and a lambda raises an error.

```python
import numpy as np
from sage.all__sagemath_symbolics import I, exp

from skcomplex.core.system import Observable

Oscillator = Element.define(
    'Oscillator',
    states={'theta': state(0.0, domain=Domain.ANGLE)},
    parameters={'omega': 1.0},
    laws=lambda s: [Rate(s.theta, s.omega)],
)


def phase_vector(member):
    return exp(I * member.theta)


def order_parameter(s):
    return abs(s.z)


oscillators = System(
    components=[Oscillator.group('phases', 100, theta=np.linspace(0, 1, 100), omega=np.linspace(0.9, 1.1, 100))],
)
observables = [Observable('z', phase_vector, group='phases'), Observable('order', order_parameter)]
order = Simulation(oscillators, dt=1.0, observables=observables).run(until=50).trajectory_['order']
print(round(float(order[0]), 3), round(float(order[-1]), 3))
```

A law reads an observable through a role bound to the observable's path, as `s.<role>.value`. A symbolic observable,
an expression or a reduction with a string `reduce` over an element group, is read at the current state by laws of its
own partition, and the compiled problem evaluates it before the law, so the value the law reads equals the recorded
value at every frame. A law of another partition reads the observable as held at the last synchronisation point, so
agents and equations exchange macroscopic quantities in a co-simulation. An observable with a Python function as
`reduce`, or one over agents, is read only from another partition, and reading it in its own partition raises an error.

```python
from sage.all__sagemath_symbolics import cos, sin

MeanField = Interaction.define(
    'MeanField',
    roles=['target', 'X', 'Y'],
    parameters={'K': 1.0},
    laws=lambda s: [Rate(s.target.theta, s.K * (s.Y.value * cos(s.target.theta) - s.X.value * sin(s.target.theta)))],
)


def cosine(member):
    return cos(member.theta)


def sine(member):
    return sin(member.theta)


coupled = System(
    components=[Oscillator.group('phases', 100, theta=np.linspace(0, 6, 100), omega=np.linspace(0.9, 1.1, 100))],
    interactions=[MeanField('coupling', target='phases', X='X', Y='Y', K=2.0)],
)
observables = [Observable('X', cosine, group='phases'), Observable('Y', sine, group='phases')]
print(Simulation(coupled, dt=1.0, observables=observables).run(until=20).trajectory_['X'][-1].round(2))
```

## Stocks and flows

A state tagged with a quantity, `state(..., quantity='water')`, is a stock that changes only through flows. A
`Transfer(source, sink, amount, label=...)` moves an amount of the quantity from one stock to another, or exchanges it with
the environment through `Source()` and `Sink()`. In a system of rate laws the amount is a rate, in a system of update
laws it is moved once per step, and with `rate=...` it is the size of a jump with that propensity. Trajectories record
every flow under `<component path>.<label>`. An `Auxiliary(label, expression)` names an intermediate expression that
later laws use through `auxiliary.symbol`, and trajectories record it too. `lookup(x, points)` is a table function,
piecewise linear between its points and flat outside them.

```python
from skcomplex.core.laws import Auxiliary, Sink, Source, Transfer, lookup


def population_laws(s):
    births = Auxiliary('birth_rate', s.b * s.n * lookup(s.n, [(0, 1.0), (100, 1.0), (200, 0.0)]))
    return [births, Transfer(Source(), s.n, births.symbol, label='births'), Transfer(s.n, Sink(), s.d * s.n, label='deaths')]


People = Element.define(
    'People',
    states={'n': state(10.0, quantity='people')},
    parameters={'b': 0.03, 'd': 0.01},
    laws=population_laws,
)
crowd = System(components=[People('crowd')])
trajectory = Simulation(crowd, dt=1.0).run(until=10).trajectory_
print(round(float(trajectory['crowd.births'][-1]), 3), round(float(trajectory['crowd.deaths'][-1]), 3))
```

A rate or update law outside flows that changes a tagged state raises `ConservationError` when the system compiles.
`conservation(system)` returns the net inflow of every quantity from the environment, after checking that the internal
flows cancel. The compiler accumulates that inflow in the hidden state `__boundary.<quantity>`, and the observer
`ConservationMonitor` checks at every frame that each total changed only by it. `flow_matrix(trajectory, frame)` is
the transaction-flow matrix of a frame, one row per flow and one column per holder, whose rows sum to zero.

```python
from skcomplex.core.analysis import conservation, flow_matrix
from skcomplex.core.runtime import ConservationMonitor

print(conservation(crowd))
monitored = Simulation(crowd, dt=1.0, observers=[ConservationMonitor()]).run(until=10).trajectory_
print(flow_matrix(monitored, -1).values.sum(axis=1))
```

`distributed_delay(value, time, order, conserved=True)` is a material delay of any order, hidden stocks in series whose
outflow it returns, and with `conserved=False` an information smoothing. It adds hidden stocks
`<component path>.<kind>_delay<order>_<index>` to rate laws of single elements and interactions, starting in
equilibrium with their input. The system-dynamics names `delay1`, `delay3`, `smooth`, and `smooth3` are its
realizations in `skcomplex.models.engineering.processes`.

`loops(system)` derives the signed dependency graph of states, auxiliaries, and flows from the laws, and lists its
feedback loops with their polarity, and `loop_dominance(trajectory)` ranks the loops by their gains at every frame.

```python
from skcomplex.core.analysis import loop_dominance, loops

for loop in loops(crowd).loops:
    print(loop.polarity, ' -> '.join(loop.nodes))
dominance = loop_dominance(trajectory)
print(dominance.loops[int(dominance.dominant[-1])].polarity)
```

Agents move conserved quantities between each other inside rules. `transfer(source, sink, amount, state)` moves an
amount between two agents at once, so a negative holding is an amount owed and total holdings stay constant. An agent
returns an `Invariant(label, function)` among its laws for every expression of its states that must keep its first
value, and the stepper checks it for every agent after every step. Removing an agent that still holds a tagged state
raises `ConservationError` until its holdings are settled with transfers.

```python
from skcomplex.core.components import Element
from skcomplex.core.laws import Invariant, Rule, transfer


def net_worth(agent):
    return agent.cash + agent.claims - agent.equity


class Trader(Element):
    cash: float = state(10.0, quantity='money')
    claims: float = state(0.0, quantity='claims')
    equity: float = state(10.0)

    def laws(self, s):
        return [Rule(self.lend), Invariant('cash + claims == equity', net_worth)]

    def lend(self, agent, model):
        borrower = model.agents('traders')[0]
        if borrower != agent:
            transfer(agent, borrower, 1.0, 'cash')
            transfer(borrower, agent, 1.0, 'claims')


traders = Simulation(System(components=[Trader.group('traders', 3)])).run(until=3).trajectory_
print(traders.agents('traders', 3).values['cash'].sum())
```

The library models built on flows are `bathtub`, `rice_milling`, `solid_waste`, `food_relief`, and `irrigation` in
engineering processes, `car_dealer`, `commodity_cycle`, and `beer_game` in markets, `bass_diffusion` in innovation,
`stock_flow_consistent_economy` in macroeconomics, and `success_to_the_successful`, `shifting_the_burden`, and
`escalation` in organizations.

## The model library

The package holds two parts. `skcomplex.core` is the domain-agnostic framework: `base` for the shared leaves such as
domains, registries, and errors, `model` for components, laws, and spaces, `compile`, `solvers`, `runtime`, `viz`,
`experiments`, `analysis`, and `backends`. `skcomplex.models` is the library of ready-made models, organised by
scientific field and subfield as `skcomplex.models.<field>.<subfield>`.

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
print(run('sir').trajectory_['I.n'].max().round())
```

### Models as patterns

The models are examples of the core interface, not a separate one, so each is also the pattern for a model of your
own. The page of every field, under Models in the documentation, shows each model's factory, its component types with
their states, parameters, and laws as written, and, for differential equations and maps, its equations as
`equations_of` returns them. A model of your own follows the same steps: declare the component types, write a factory
that returns the system, and register it with `register_model`.

```python
from skcomplex.core.base import Domain, state
from skcomplex.core.components import Element
from skcomplex.core.laws import Rate
from skcomplex.core.system import System
from skcomplex.core.runtime import create, register_model

Tank = Element.define(
    'Tank',
    states={'level': state(default=1.0, domain=Domain.NONNEGATIVE)},
    parameters={'k': 0.5},
    laws=lambda s: [Rate(s.level, -s.k * s.level)],
)


@register_model(field='examples/processes', references=('Torricelli 1643',), defaults={'until': 5})
def draining_tank(level: float = 2.0) -> System:
    """Return a tank that drains at a rate proportional to its level."""
    return System(components=[Tank('tank', level=level)])


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

### Moved paths

Earlier versions placed the framework at the top level and grouped the models by technique. The framework packages
`skcomplex.model`, `compile`, `solvers`, `runtime`, `viz`, `experiments`, `analysis`, and `backends` moved under
`skcomplex.core`, and the shared leaves moved from `skcomplex.core` to `skcomplex.core.base`. The redesign then split
`skcomplex.core.model` into `core.system`, `core.components`, `core.interactions`, `core.space`, and `core.laws`, and
moved named realizations, such as `LennardJones` and `LangevinThermostat`, to `skcomplex.models.physics.realizations`.
The models moved to their subfields:

| Old package | New subfields |
| --- | --- |
| `ecology`, `epidemiology`, `neuroscience` | `biology.ecology`, `biology.epidemiology`, `biology.neuroscience` |
| `economics`, `finance` | `economics.macroeconomics`, `economics.markets`, `economics.finance` |
| `social` | `social.segregation`, `social.opinion_dynamics`, `economics.econophysics`, `economics.resources` |
| `fluids` | `physics.fluids` |
| `fields` | `physics.heat`, `physics.pattern_formation`, `physics.fluids`, `chemistry.pattern_formation`, `biology.ecology`, `biology.physiology`, `biology.collective_behaviour` |
| `lattices` | `physics.statistical_mechanics`, `physics.criticality`, `earth.seismology`, `computer_science.automata`, `social.opinion_dynamics`, `chemistry.kinetics`, `engineering.traffic` |
| `nonlinear` | `mathematics.dynamical_systems`, `physics.nonlinear_oscillators`, `physics.synchronization`, `physics.soft_matter`, `biology.physiology` |
| `physics` | `physics.mechanics`, `physics.celestial_mechanics`, `physics.statistical_mechanics`, `physics.soft_matter`, `engineering.circuits` |

Every model keeps its name, so `create` and `run` find it wherever it moved.

## Groups of many elements

A group holds N identical elements of one type as arrays with one row per member. It compiles once, whatever its size,
and every law written for one member runs over all members with array operations.

```python
import numpy as np

from skcomplex.core.components import Particle
from skcomplex.core.space import ContinuousSpace

rng = np.random.default_rng(0)
gas = System(
    components=[Particle.group('gas', 10_000, x=rng.uniform(0, 1, (10_000, 2)), v=rng.normal(0, 1, (10_000, 2)))],
    space=ContinuousSpace(1.0, 1.0, boundary='reflecting'),
)
trajectory = Simulation(gas, dt=0.01).run(until=1).trajectory_
print(trajectory['gas.x'].shape)
```

- The label and the size of a group are given by position, `Particle.group('gas', 10_000, ...)`, so a state named
  `size` takes per-member values by keyword. States and parameters cannot be named `label`, `visual`, or `remembers`,
  and those of agents cannot take the names of the agent view, such as `location` or `alive`.
- A value is an array whose first dimension is the size of the group, or one value shared by every member.
- An interaction whose role binds to a group applies to every member. A single element bound next to the group
  receives the sum over the members, unless its law reads the group only through reductions.
- Group variables are named by the group path, such as `gas.x` with shape `(10000, 2)`, and a group parameter takes
  one value for every member or one value per member in `update_parameters`.
- `ideal_gas(n_molecules, kT=..., size=...)` builds a gas of non-interacting molecules in a reflecting box.

### Reductions over a group

`total`, `mean`, `maximum`, `minimum`, `variance`, and `count` reduce an expression over the members of a group inside
any law. Inside the reduction, the symbols of a member stand for the reduced member, and outside it for the reading
member, so `Rate(s.x, -s.x + tanh(total(s.x)))` reads the group total in every member's own law. A reduction is
evaluated once per evaluation, before the member laws, at a cost linear in the size of the group, on NumPy and Numba.

- `where` selects the members with a condition on the reduced member, such as `count(where=s.x > 0.5)`. The reductions
  of no selected member are 0.
- `weights` gives `total` and `mean` a constant matrix, a NumPy array or a SciPy sparse matrix, with one row per
  reading member and one column per reduced member, so member `i` reads `sum_j W_ij a_j`, such as a payoff matrix.
- `over` names the group by its namespace, such as `over=s.neurons`, when the expression reads no symbol of it.
- A reduction reads one group, and an argument that reads two groups raises an error. Reductions nest, so the mean
  fitness `total(s.x * total(s.x, weights=A))` of replicator dynamics evaluates the inner reduction first.
- An interaction reads groups member by member outside reductions, which needs groups of one size, and reads other
  groups through reductions. A law that sets a single element from reductions alone runs once, not once per member.
- Discrete maps read the reductions at step `n`, jump propensities after every event, and events at the state they
  receive.

```python
from sage.all__sagemath_symbolics import tanh

from skcomplex.core.laws import total

Unit = Element.define('Unit', states={'x': 0.0}, laws=lambda s: [Rate(s.x, -s.x + tanh(total(s.x)))])
units = System(components=[Unit.group('units', 50, x=np.linspace(-1, 1, 50))])
print(Simulation(units, dt=0.5).run(until=2).trajectory_['units.x'][-1, :3].round(3))

payoff = np.array([[-0.5, 2.0], [0.0, 1.0]])
Strategy = Element.define('Strategy', states={'x': 0.5})
Replication = Interaction.define(
    'Replication',
    roles=['population'],
    laws=lambda s: [
        Rate(
            s.population.x,
            s.population.x
            * (total(s.population.x, weights=payoff) - total(s.population.x * total(s.population.x, weights=payoff))),
        ),
    ],
)
game = System(
    components=[Strategy.group('population', 2, x=[0.1, 0.9])],
    interactions=[Replication('replication', population='population')],
)
print(Simulation(game, dt=1.0).run(until=40).trajectory_['population.x'][-1].round(3))
```

A role also binds to an element set, a list of single elements of one type. The interaction's laws apply to each
element, the elements keep their own state names, and reductions over the role reduce over the set.

```python
Inhibition = Interaction.define(
    'Inhibition',
    roles=['units'],
    parameters={'beta': 2.0},
    laws=lambda s: [Rate(s.units.x, s.units.x * (1 - s.units.x) - s.beta * s.units.x * (total(s.units.x) - s.units.x))],
)
Activity = Element.define('Activity', states={'x': 0.1})
network = System(
    components=[Activity(f'unit_{index}', x=0.1 * (index + 1)) for index in range(3)],
    interactions=[Inhibition('inhibition', units=['unit_0', 'unit_1', 'unit_2'])],
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
from skcomplex.core.interactions import Interaction
from skcomplex.core.laws import Update
from skcomplex.core.system import System
from skcomplex.core.runtime import Simulation

Node = Element.define('Node', states={'x': 0.0})


class WeightedSum(Interaction):
    target: str = role()
    inputs: list[str] = role(variadic=True)
    weights: list[float] = parameter(1.0, per='inputs')

    def laws(self, s):
        total_input = sum(weight * item.x for weight, item in zip(s.weights, s.inputs, strict=True))
        return [Update(s.target.x, total_input)]


summed = System(
    components=[Node('a', x=1.0), Node('b', x=2.0), Node('out'), Node.group('nodes', 5, x=[0.0, 1.0, 2.0, 3.0, 4.0])],
    interactions=[WeightedSum('sum', target='out', inputs=['a', 'b', 'nodes[3]'], weights=[1.0, 1.0, 10.0])],
)
simulation = Simulation(summed, dt=1).run(until=1)
simulation.update_parameters({'sum.weights[2]': 0.0})
print(simulation.trajectory_['out.x'], simulation.parameters_)
```

An element set and a variadic role both bind a list. An element set applies the laws once per element, each element
read like a single element, while a variadic role gives one law the whole sequence. `random_boolean_network` and
`yeast_cell_cycle` read the inputs of every node through one variadic role.

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
import numpy as np
from sage.all__sagemath_symbolics import tanh

from skcomplex.core.laws import Rate, mod, shift_member

Car = Element.define('Car', states={'x': 0.0, 'v': 1.0}, laws=lambda s: [Rate(s.x, s.v)])
Following = Interaction.define(
    'Following',
    roles=['cars'],
    parameters={'length': 40.0},
    laws=lambda s: [
        Rate(s.cars.v, tanh(mod(shift_member(s.cars.x, 1) - s.cars.x, s.length) - 2.0) + tanh(2.0) - s.cars.v),
    ],
)
ring = System(
    components=[Car.group('cars', 20, x=np.arange(20) * 2.0 + np.eye(20)[0] * 0.1)],
    interactions=[Following('following', cars='cars', length=40.0)],
)
print(Simulation(ring, dt=1.0).run(until=50).trajectory_['cars.v'][-1].std() > 0.1)
```

### Jacobians of reductions

A reduction makes every reading member depend on every reduced member, so the Jacobian of a drift with reductions has
the rank-one block `dF_i/dR dR/dx_j` of every reduction `R`. `problem.jacobian` of an ODE problem evaluates the exact
Jacobian by the chain rule through every reduction, as a SciPy sparse matrix, and `problem.jacobian.pattern()` states
its structure, which the implicit methods of `ScipyODE` receive (see
[Discretizations and stiff solvers](#discretizations-and-stiff-solvers)). When the dense blocks exceed 4 million
entries, `pattern()` returns `None` and an implicit solver estimates a dense Jacobian. Mean-field groups are rarely stiff, so explicit steppers such as `RK4` and
`ScipyODE(method='RK45')` suit large groups with reductions.

```python
from skcomplex.core.compile import compile_system

problem = compile_system(units)
jacobian = problem.jacobian(0.0, problem.initial_state, problem.initial_parameters)
print(jacobian.shape, problem.jacobian.pattern().nnz)
```

`dynamics_of`, `fixed_points`, `stability`, and `jacobian` of `skcomplex.core.analysis` expand a group of at most 64
members into one equation per member, with every reduction written as an explicit sum, so the fixed points of
replicator dynamics are derived symbolically. A larger group raises an error that states the limit.

### Pair interactions

A pair interaction is written once for a pair of members and summed over every pair closer than its cutoff, inside one
group or between two groups. `PairPotential` derives the forces from a pair energy.

```python
from skcomplex.core.base import Domain, parameter
from skcomplex.core.interactions import PairPotential


class LennardJones(PairPotential):
    epsilon: float = parameter(1.0, domain=Domain.POSITIVE)
    sigma: float = parameter(1.0, domain=Domain.POSITIVE)

    def pair_energy(self, s, r):
        return 4 * s.epsilon * ((s.sigma / r) ** 12 - (s.sigma / r) ** 6)


side = np.arange(10) * 1.2
liquid = System(
    components=[Particle.group('atoms', 100, x=np.stack(np.meshgrid(side, side), axis=-1).reshape(-1, 2))],
    interactions=[LennardJones('lj', first='atoms', second='atoms', cutoff=2.5)],
    space=ContinuousSpace(n_dim=2),
)
Simulation(liquid, dt=0.05).run(until=1)
```

`period` gives the side of a periodic box, so pair distances use the nearest periodic image.

### Boundaries

A `Boundary` law sets the value of a state after every internal solver step, however long the frame, so a frame never
reports a state that a boundary would change. Adaptive solvers find the first time in a step where a boundary acts and
restart there. Boundaries combine with every form, so walls and periodic boxes work for differential equations,
stochastic systems, jump processes, and discrete maps.
`ReflectingBox` mirrors positions and velocities at the walls, and `PeriodicBox` wraps positions around the box.

## Lattice models

Cellular automata, sandpiles, and spin models live on lattices. `LatticeSpace(shape, boundary=...)` is a box of `Z^d`
with `'periodic'`, `'fixed'`, or `'reflecting'` edges, and `Element.lattice(label, space, **values)` puts one member
of an element type on every site, with values given as arrays of the lattice shape. Laws are written for one cell.

**Local rules.** `LocalRule(target, neighborhood, rule)` computes a cell's next value from the ordered values of its
neighbourhood. `Neighborhood.moore`, `.von_neumann`, `.radius`, or `Neighborhood(offsets)` give the offsets. A rule is
a `Table` of patterns, a `Totalistic` or `OuterTotalistic` rule, a rule code whose `table()` returns a table, such
as `Wolfram` of the automata models, or a function of NumPy arrays,
one per neighbour, that also receives the parameters it names and, for stochastic rules, uniform random numbers.
Tables and totalistic rules run as lookups, fused into one compiled loop on the Numba backend.

```python
from skcomplex.core.laws import LocalRule, OuterTotalistic
from skcomplex.core.space import LatticeSpace, Neighborhood

Life = Element.define(
    'Life',
    states={'alive': state(0, domain=Domain.BOOLEAN)},
    laws=lambda s: [LocalRule(s.alive, Neighborhood.moore(), OuterTotalistic({(0, 3): 1, (1, 2): 1, (1, 3): 1}))],
)
glider = np.zeros((20, 20))
glider[1, 2] = glider[2, 3] = glider[3, 1] = glider[3, 2] = glider[3, 3] = 1
cells = System(components=[Life.lattice('cells', LatticeSpace((20, 20)), alive=glider)])
trajectory = Simulation(cells, dt=1).run(until=4).trajectory_
print(np.array_equal(trajectory.lattice('cells.alive')[-1], np.roll(glider, (1, 1), axis=(0, 1))))
```

- Rules update synchronously by default, every cell from the previous configuration. `schedule='random-sequential'`,
  `'random-site'`, or `'checkerboard'` update cells one at a time or one sublattice at a time.
- One unit of time is one application of every rule, so `dt` is an integer, and `trajectory.lattice(name)` returns
  the states in lattice shape.
- `Drive(target, function)` changes a whole lattice at once, such as dropping a grain on a random site.

**Relaxation.** `Relaxation(drive, relax, stable)` separates a slow drive from a fast relaxation, as in
self-organised criticality. Every step applies the drive laws once and the relax laws until `stable` holds, and each
relaxation is recorded among the trajectory's events with its size, duration, and area. `activity` marks the sites
that count as updates, such as the toppling sites of a sandpile.

```python
from skcomplex.models.physics.criticality import sandpile

trajectory = Simulation(sandpile((20, 20)), dt=100, random_state=0).run(until=2000).trajectory_
print(max(event.size for event in trajectory.events))
```

**Monte Carlo.** A `Sample(energy, move, sampler)` law samples states. `Energy(expression, neighborhood)` is the local energy of a site, every term of the total energy
that involves it, with neighbour values read through `neighbors`. `Move.flip`, `Move.random_value`, and `Move.rotate`
propose changes, and `Sampler('metropolis' | 'glauber' | 'heat-bath', temperature)` accepts them. The energy change of
a move is derived symbolically, sweeps update the two sublattices in turn, and the temperature is a parameter that
changes live.

```python
from skcomplex.core.laws import Energy, Move, Sample, Sampler, neighbors

nearest = Neighborhood.von_neumann()
Spin = Element.define(
    'Spin',
    states={'spin': state(1.0, domain=Domain.INTEGER)},
    parameters={'J': 1.0, 'temperature': 1.5},
    laws=lambda s: [
        Sample(
            Energy(-s.J * s.spin * sum(neighbors(s.spin, nearest)), nearest),
            Move.flip(s.spin),
            Sampler('metropolis', s.temperature),
        ),
    ],
)
spins = System(components=[Spin.lattice('spins', LatticeSpace((32, 32)))])
print(abs(Simulation(spins, dt=50, random_state=0).run(until=200).state_.mean()) > 0.9)
```

**Triangular lattices.** `LatticeSpace(shape, geometry='triangular')` keeps integer coordinates on a skewed grid, so
every boundary works as on a square lattice, and places site `(i, j)` at `(j + i / 2, i sqrt(3) / 2)` in the plane,
given by the `positions` property. `Neighborhood.triangular()` holds its six neighbours counterclockwise from 0
degrees, so direction `k + 3` modulo 6 is opposite to direction `k`, as lattice gases need. The cells of the sites
are the hexagons of a hexagonal tiling.

```python
Burn = Element.define(
    'Burn',
    states={'fire': 0.0},
    laws=lambda s: [LocalRule(s.fire, Neighborhood.triangular(center=True), lambda *values: np.max(values, axis=0))],
)
fire = np.zeros((40, 40))
fire[20, 20] = 1
forest = System(components=[Burn.lattice('forest', LatticeSpace((40, 40), geometry='triangular'), fire=fire)])
burning = Simulation(forest, dt=1).run(until=3).trajectory_.lattice('forest.fire')
print([int(item.sum()) for item in burning])
```

**Lattices on graphs.** `Element.lattice` also takes a `NetworkSpace`, with one site per node in the order of
`graph.nodes`, and initial values as arrays in node order or mappings from node to value. `Adjacency(center=False)` is
the neighbourhood of a node, its neighbours along its edges, padded to the largest degree with the network's `fill`
value. Local rules, drives, relaxations, Monte Carlo laws, and extremal laws run on graphs as on lattices. Lookup rules
on graphs take the totalistic forms, and a function rule that names the keyword `degree` receives the number of real
neighbours of every site. The padded table grows with the largest degree, so graphs with hubs of very high degree,
beyond scale-free graphs of about 10^5 nodes, use much memory.

**Neighbour sums and colored schedules.** `neighbor_sum(expression)` sums an expression of a site and one neighbour,
read with `adjacent(state)`, over the real neighbours of the site, so local energies are exact on sites of any degree
and outside fixed edges. `schedule='colored'` on local rules and samplers updates the classes of a proper coloring of
the neighbour graph in turn, which NetworkX computes once per law. The checkerboard schedule stays the two-color case
of square lattices.

```python
import networkx as nx

from skcomplex.core.laws import adjacent, neighbor_sum
from skcomplex.core.space import Adjacency, NetworkSpace

GraphSpin = Element.define(
    'GraphSpin',
    states={'spin': state(1.0, domain=Domain.INTEGER)},
    parameters={'J': 1.0, 'temperature': 1.0},
    laws=lambda s: [
        Sample(
            Energy(-s.J * s.spin * neighbor_sum(adjacent(s.spin)), Adjacency()),
            Move.flip(s.spin),
            Sampler('metropolis', s.temperature, schedule='colored'),
        ),
    ],
)
graph = nx.random_regular_graph(4, 200, seed=0)
spins = System(components=[GraphSpin.lattice('spins', NetworkSpace(graph))])
print(abs(Simulation(spins, dt=50, random_state=0).run(until=200).state_.mean()) > 0.9)
```

**Extremal updates.** `Extremal(target, neighborhood, rule, key=None, extremum='min', repeat=1)` finds the site with
the smallest or largest value of a key state, breaking ties with the simulation's random stream, and applies
`rule(*values, random, **parameters)` to the site and its real neighbours, `repeat` times per step. Every application
is recorded among the trajectory's events as an `ExtremalRecord` with its time, site, and extremal value. On the Numba
backend the applications of a step run in one compiled loop when Numba compiles the rule, which then takes its
arguments by position or keyword, with no keyword-only arguments, and both backends give the same events.

```python
from skcomplex.core.laws import Extremal


def renew(fitness, random):
    return random


Species = Element.define(
    'Species',
    states={'fitness': 0.5},
    laws=lambda s: [Extremal(s.fitness, Neighborhood(((-1,), (1,))), renew, repeat=100)],
)
ring = System(components=[Species.lattice('ring', LatticeSpace(100), fitness=np.random.default_rng(0).random(100))])
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

## Fields

A field holds one value per site of a lattice or per node of a network. `Field(label, element, space, boundary=...,
values=...)` places a member of an element type on every site, and the laws of the type, written for one site, read
the neighbouring sites through differential operators. The method of lines turns a field into ordinary differential
equations, so every ODE solver and backend runs it.

```python
from skcomplex.core.components import Field
from skcomplex.core.laws import laplacian

Diffusion = Element.define(
    'Diffusion',
    states={'u': 0.0},
    parameters={'D': 1.0},
    laws=lambda s: [Rate(s.u, s.D * laplacian(s.u))],
)
initial = np.zeros(400)
initial[200] = 20.0
rod = Field('rod', Diffusion, LatticeSpace(400, spacing=0.05), boundary={'u': 'zero-flux'}, values={'u': initial})
trajectory = Simulation(System(components=[rod]), dt=0.1).run(until=0.5).trajectory_
print(trajectory['rod.u'][-1].max(), trajectory['rod.u'].sum(axis=1)[[0, -1]])
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

The same laws of a field run with another discretization by changing `Field(..., discretization=...)`.

- `'spectral'` fields on periodic lattices multiply Fourier modes, with the two-thirds rule removing aliasing from
  every operator, so `laplacian(u)` is exact for smooth fields. `inverse_laplacian(u)` solves Poisson equations, and
  `symmetry={'u': (None, 'odd')}` keeps a state odd or even along an axis, which represents walls on a doubled
  lattice. `LatticeSpace(spacing=(dx, dy))` sets a spacing per axis.
- `'finite-volume'` fields take `Flux(state, flux)` laws, `du/dt + divergence(flux) = 0`, updated with MUSCL
  reconstruction, minmod or van Leer limiters, and Rusanov or exact Godunov fluxes. `SSPRK` steps them by the CFL
  condition and by the stability limit of their Laplacian rates at the current parameters, and conserves their totals.
- `'finite-element'` fields live on a `Mesh`, such as `Mesh.l_shape`, `Mesh.rectangle`, `Mesh.disk`, or
  `Mesh.load(path)`, with stiffness and mass matrices from scikit-fem and boundary values that can be functions of
  the coordinates. They stream to Rerun as colored meshes.

The compiler splits the rate of every field into a linear part with constant coefficients, such as
`D * laplacian(u)` or `-laplacian(laplacian(u))`, and a nonlinear remainder. `IMEXEuler`, `IMEX2`, and
`CrankNicolson` treat the linear part implicitly, with sparse solves for finite differences and finite elements and
solves per Fourier mode for spectral fields. `ETDRK4` integrates it exactly per mode, coupled states included, so stiff
equations take steps far beyond the explicit limit.

```python
from skcomplex.core.laws import divergence
from skcomplex.core.solvers import ETDRK4

Chaos = Element.define(
    'Chaos',
    states={'u': 0.0},
    laws=lambda s: [Rate(s.u, -divergence([s.u**2 / 2]) - laplacian(s.u) - laplacian(laplacian(s.u)))],
)
positions = np.arange(1, 129) * 32 * np.pi / 128
line = Field(
    'line',
    Chaos,
    LatticeSpace(128, spacing=32 * np.pi / 128),
    values={'u': np.cos(positions / 16) * (1 + np.sin(positions / 16))},
    discretization='spectral',
)
print(np.abs(Simulation(System(components=[line]), dt=10, solver=ETDRK4(dt=0.25)).run(until=100).state_).max())
```

The implicit methods of `ScipyODE` are the stiff path for finite-difference and finite element fields, with adaptive
error control that the fixed-step split steppers lack. The compiler derives the sparsity pattern of the Jacobian from
the laws: member and shared reads, the neighbour tables of stencils, the sparse matrices of finite elements, the
reconstruction stencils of finite volumes, and dense blocks where couplings are known only at run time, such as pairs,
spectral operators, and callables. `ScipyODE(method='BDF')` and `ScipyODE(method='Radau')` receive the pattern, so
SciPy estimates the Jacobian with one drift evaluation per group of independent columns and factors it with sparse LU,
and `LSODA` receives the half bandwidths of a pattern narrower than a tenth of the size. The option `jacobian` selects
the path: `'auto'` by default, `'exact'` for the sparse Jacobian of the laws differentiated by Sage, which laws without
a symbolic form cannot give, and `'dense'` for the dense estimate. Explicit methods ignore it, and `RK45` warns when one
call spans more than 1000 stable steps of the diffusion of a field.

```python
from skcomplex.core.solvers import ScipyODE
from skcomplex.models.biology.ecology import fisher_kpp

stiff = fisher_kpp(400, D=1.0, spacing=0.05)
simulation = Simulation(stiff, dt=1.0, solver=ScipyODE(method='BDF')).run(until=5)
print(simulation.stepper_.jacobian_pattern_.nnz, simulation.state_.max().round(3))
```

On Fisher-KPP on 4000 points, whose largest eigenvalue is about 1600, run to `t = 10`, BDF with the dense estimate
took 35 to 50 s, against about 0.2 s with the pattern, 0.3 s with the exact Jacobian, 0.7 s for Radau with the pattern,
and 0.8 s for `CrankNicolson(dt=0.01)`. Spectral fields keep `ETDRK4` and the IMEX steppers, which win there because
their linear part is diagonal in Fourier space while its pattern is dense.

`skcomplex.models.physics.fluids.incompressible_flow(shape, size, nu=..., forcing=..., buoyancy=...)` builds two-dimensional
incompressible flow in vorticity-streamfunction form, with Boussinesq convection when `buoyancy` gives `Ra` and `Pr`.
The library has `kuramoto_sivashinsky`, `complex_ginzburg_landau`, `swift_hohenberg`, and `heat_l_shape` in the
subfields `fluids`, `pattern_formation`, and `heat` of `skcomplex.models.physics`, and `burgers`, `shallow_water`,
`decaying_turbulence`, and `rayleigh_benard` in `skcomplex.models.physics.fluids`.

## Backends and speed

Compiling a system produces numeric functions with a backend. `'numpy'` evaluates every law with NumPy arrays, and
`'numba'` generates one compiled function per form, with compiled cell lists for pair interactions and compiled loops
for `RK4`, `EulerMaruyama`, `Gillespie`, and `Iteration`. The default `'auto'` picks Numba when the groups of a system
hold at least 1000 members in total, because compiling takes a second or two that small systems do not win back.

```python
from skcomplex.core.backends import NumbaBackend
from skcomplex.models.physics.statistical_mechanics import ideal_gas
from skcomplex.core.solvers import RK4

simulation = Simulation(ideal_gas(50_000, random_state=0), solver=RK4(dt=0.01), backend='numba', dt=0.01)
print(simulation.run(until=0.5).problem_.backend)
```

Compiled kernels are cached on disk, under `~/.cache/skcomplex` or the `cache_dir` of `NumbaBackend(cache_dir=...)`,
so running the same system again skips compilation.

A law value can be a function called once per member with `Call(function, *arguments)`. A Numba-compiled function
runs inside the compiled kernels, and the NumPy backend calls it from Python.

```python
import numba

from skcomplex.core.laws import Call, Rate
from skcomplex.models.biology.ecology import Population


@numba.njit
def saturating(t, n, r):
    return r * n / (1 + n)


class Saturation(Interaction):
    target: str = role()
    r: float = parameter(1.0)

    def laws(self, s):
        return [Rate(s.target.n, Call(saturating, s.target.n, s.r))]


herds = System(
    components=[Population.group('herds', 2000, n=1.0)],
    interactions=[Saturation('growth', target='herds')],
)
print(Simulation(herds, dt=0.1).run(until=1).trajectory_['herds.n'][-1, 0])
```

A plain Python callable of the values, as in `Rate(s.target.n, lambda t, values: ...)`, also works with the Numba
backend, but runs in Python and turns off the compiled stepper loops for that kind of law.

## External backends

An external backend runs a whole system in a third-party solver library, and the simulation keeps its frames,
trajectories, live parameters, and Rerun visualization. `LammpsBackend` runs particle systems in LAMMPS, which
integrates them with velocity Verlet.

```python
from skcomplex.core.backends import LammpsBackend
from skcomplex.models.physics.realizations import LangevinThermostat, LennardJones

atoms = System(
    components=[Particle.group('atoms', 100, x=np.stack(np.meshgrid(side, side), axis=-1).reshape(-1, 2))],
    interactions=[
        LennardJones('lj', first='atoms', second='atoms', cutoff=2.5),
        LangevinThermostat('bath', target='atoms', temperature=0.5, gamma=1.0),
    ],
    space=ContinuousSpace(n_dim=2),
)
trajectory = Simulation(atoms, solver=LammpsBackend(dt=0.005), dt=0.05, random_state=0).run(until=1).trajectory_
print(trajectory['atoms.x'].shape)
```

Before the first frame the backend checks the system and raises one error that lists every element, interaction, and
law it cannot run. The translation is the following.

| Framework | LAMMPS |
| --- | --- |
| `Particle` and particle groups | one atom type per element or group, with one mass per group |
| two or three dimensions | `dimension 2` with `fix enforce2d`, or `dimension 3` |
| `ReflectingBox` | `fix wall/reflect` at `0` and `size` |
| `PeriodicBox` | periodic boundaries of side `size`, shared by every group |
| `LennardJones` | `pair_style lj/cut` |
| other `PairPotential` | `pair_style lepton` with the pair energy printed as a Lepton expression |
| `ConstantForce` | `fix addforce` |
| `LinearDrag` | `fix viscous` |
| `LangevinThermostat` | `fix langevin`, seeded from the simulation's random state |
| any other interaction that only changes velocities | `fix external`, evaluated by the framework every step |

Importing `skcomplex.core.backends` also registers the solver name `'lammps'`.

## Agent-based models

Agents are ordinary elements whose laws are rules written as code. An `Element` declares the states every agent holds
and the parameters its group shares, and returns `Rule` laws whose functions take the agent and the model. An element
or a group whose laws include rules is an agent, and so is every member of a variable group. In a system with rules, a
group whose element type has no laws is a group of passive agents that rules read and change, unless an interaction
whose laws are not rules binds it. A system with rules is
agent-based, runs in discrete time, and runs on the Mesa backend, one Mesa step per unit of time, with every agent
activated in random order. Mesa declares what it runs like any solver, so `validate(system, solver='mesa')` reports
the laws, spaces, and times it does not support.

```python
from skcomplex.core.base import Domain, VisualSpec, parameter, state
from skcomplex.core.components import Element
from skcomplex.core.laws import Rule
from skcomplex.core.space import LatticeSpace


class Forager(Element):
    visual = VisualSpec('agents', {'color_by': 'food'})
    food: float = state(5.0)
    hunger: float = parameter(1.0, domain=Domain.NONNEGATIVE)

    def laws(self, s):
        return [Rule(self.forage)]

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
    components=[Forager.group('foragers', 50, variable=True)],
    space=LatticeSpace((30, 30), boundary='periodic'),
)
trajectory = Simulation(foragers, random_state=0).run(until=20).trajectory_
print(trajectory['foragers.count'][-1], len(trajectory.agents('foragers', 20)))
```

- An agent reads and sets its states as attributes and reads its group's parameters, which change live like any
  parameter. `neighbors`, `neighbor_cells`, `agents_here`, `move_to`, `create`, and `remove` act on its place and its
  population. Only the members of a group declared `variable=True` are created and removed during a run.
- `Element.group(label, size, location=...)` places the agents at cell coordinates, network nodes, or positions, one
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
from skcomplex.core.space import ContinuousSpace


class Walker(Element):
    def laws(self, s):
        return [Rule(self.step)]

    def step(self, agent, model):
        width, height = model.dimensions
        x, y = agent.location
        agent.move_to(((x + model.rng.uniform(0, 2)) % width, y))


class Spread(Interaction):
    walkers: str = role()
    reach: float = parameter(10.0)

    def laws(self, s):
        return [Rule(self.measure)]

    def measure(self, model):
        walkers = model.agents('walkers')
        close = [a for a in walkers for b in walkers if a != b and a.distance_to(b) < model.parameter('reach')]
        counts.append(len(close) // 2)


counts = []
road = System(
    components=[
        System(
            'road',
            components=[Walker.group('walkers', 5, location=[(10.0 * index, 0.5) for index in range(5)])],
            interactions=[Spread('spread', walkers='walkers')],
        ),
    ],
    space=ContinuousSpace(50.0, 1.0, boundary='periodic'),
)
Simulation(road, random_state=0).run(until=5)
print(counts)
```

### Phases, markets, institutions, and learning

Economic and social models order each step into phases, match agents in markets, let institutions set policy, and let
agents learn from their past.

- `DiscreteTime(schedule=Phases([...]))` lists the phases of a step. A `Phase(name, activation, repeat)` runs its agents
  `'random'`ly, `'sequential'`ly in order of creation, `'simultaneous'`ly, or `'by-type'`, for `repeat` rounds such as
  intraday rounds. In a simultaneous phase every rule reads the states of the start of the phase, and the writes apply
  at its end. `Rule(function, phase=name)` puts a rule in a phase. Rules without a phase run in the phase `'default'`,
  which runs first unless the system lists it, so systems without phases keep one random activation per step.
- `Market(label, buyers=..., sellers=..., protocol=...)` matches two groups. Buyers state their limit `bid` and
  `demand`, and sellers their `ask` and `supply`. The protocols are `BilateralSearch`, `SampleAndSwitch` (partner
  choice with switching), `RationedQueue`, `Walrasian` (tâtonnement), `OrderBook` (price-time priority), `CallAuction`
  (one uniform price), and `DeferredAcceptance` (stable matching by the `matching` library). Every trade moves
  `money` to the seller and `goods` to the buyer at once, within the buyer's budget. The market reports
  `<market>.price`, `<market>.volume`, and `<market>.unmatched`.
- An `Institution` acts once per round of its phase. It reads `model.observable(name)`, reporters through
  `model.history(name, window)`, and agents, and it sets parameters of any component with
  `model.set_parameter(name, value)`. Rules see the change at once, and the trajectory records it.
- Components declare the values they publish in the class variable `reports` and publish them with
  `model.report(name, value)`. `model.storage` keeps private values of a component across the steps of a run.
- Learning blocks: `memory(agent, state, window)` returns the past values of the states that the element type of an
  agent lists in its class variable `remembers`. `heuristics` has adaptive, trend-following, anchoring, and fundamentalist forecasts.
  `logit_choice` picks a strategy by past performance, `imitate` copies the best neighbour, and `evolve` breeds the
  next generation of strategies with the model's random stream.

```python
from skcomplex.core.space import DiscreteTime, Phase, Phases
from skcomplex.models.economics.realizations import CallAuction, Institution, Market


class Trader(Element):
    value: float = state(0.0)
    bid: float = state(0.0)
    ask: float = state(0.0)
    demand: float = state(0.0)
    supply: float = state(0.0)
    money: float = state(0.0)
    goods: float = state(0.0)

    def laws(self, s):
        return [Rule(self.quote, phase='quote')]

    def quote(self, agent, model):
        agent.bid = model.rng.uniform(0, agent.value) if agent.demand else 0.0
        agent.ask = model.rng.uniform(agent.value, 2.0) if agent.supply else 2.0


class Clerk(Institution):
    reports = ('traded',)
    fee: float = parameter(0.0)

    def act(self, model):
        traded = sum(agent.goods for agent in model.agents('buyers'))
        model.report('traded', traded)
        model.set_parameter('fee', 0.01 * traded)


rng = np.random.default_rng(0)
auction = System(
    components=[
        Trader.group('buyers', 20, value=rng.uniform(0.5, 2, 20), demand=1.0, money=2.0),
        Trader.group('sellers', 20, value=rng.uniform(0, 1.5, 20), supply=1.0),
    ],
    interactions=[
        Market('auction', buyers='buyers', sellers='sellers', protocol=CallAuction(), phase='trade'),
        Clerk('clerk', phase='trade'),
    ],
    time=DiscreteTime(schedule=Phases([Phase('quote'), Phase('trade', repeat=3)])),
)
trajectory = Simulation(auction, random_state=0).run(until=5).trajectory_
print(trajectory['auction.price'][1:], trajectory['clerk.traded'][-1], trajectory['clerk.fee'][-1])
```

### Porting Mesa models

A Mesa model is ported to a system: its agent class becomes an element whose step is a `Rule`, its agent attributes
become states, and its model attributes become parameters, which then change live, are validated, and are recorded
like those of any system. Mesa still runs the steps, as the backend of agent systems.

```python
from skcomplex.core.space import DiscreteTime


class Coin(Element):
    heads: float = state(0.0)
    bias: float = parameter(0.5)

    def laws(self, s):
        return [Rule(self.flip)]

    def flip(self, agent, model):
        agent.heads += float(model.rng.random() < agent.bias)


coins = System('coins', components=[Coin.group('coins', 10)], time=DiscreteTime())
print(Simulation(coins, random_state=0).run(until=10).trajectory_.agents('coins', 10).values['heads'].sum())
```

## Rewriting systems

A rewriting system grows by replacing its parts in parallel: a `Word` by string productions in the style of
L-systems, a `Patch` of tiles by affine substitutions, and a `HostGraph` by double-pushout graph productions. It
compiles to the rewriting form, every step is one generation, and `run(until=n)` records the generations `0` to `n`.
The structure lives next to the state vector, whose states are the count of every symbol, prototile, or node label
that is a word, and the size `<element>.size`. `Trajectory.structure(path, frame)` returns the recorded structure,
which later generations do not change.

### Words and productions

A `Rewrite` interaction rewrites its target with fixed rules. Strings are words of one-character symbols, and
`Module` gives a symbol parameters.

```python
from skcomplex.core.components import Word
from skcomplex.core.interactions import Rewrite
from skcomplex.core.laws import Module, Production

algae = System(
    components=[Word('algae', axiom='A')],
    interactions=[Rewrite('rules', target='algae', rules=[Production('A', 'AB'), Production('B', 'A')])],
)
trajectory = Simulation(algae).run(until=5).trajectory_
print(trajectory['algae.size'].tolist(), str(trajectory.structure('algae', 5)))
```

A production can be stochastic, with probabilities that sum to one per predecessor and context and draws from the
simulation's random stream. It can depend on a left or a right context, which skips the word's `ignore` symbols and
treats bracketed branches as Prusinkiewicz and Lindenmayer (1990) do, and it then takes precedence over context-free
productions. A parametric production binds the predecessor's parameters to symbols, with a condition and successor
parameters written as expressions of them and of the parameters of a `Rewriting` interaction, which `schedule` and
`update_parameters` change between generations.

```python
from skcomplex.core.base import parameter, symbol
from skcomplex.core.interactions import Rewriting

x = symbol('x')


class Shrinking(Rewriting):
    ratio: float = parameter(0.5)

    def productions(self, s):
        return [Production(Module('A', x), [Module('F', x), Module('A', s.ratio * x)], condition=x > 0.1)]


plant = System(
    components=[Word('plant', axiom=[Module('A', 1.0)], alphabet={'A': 1, 'F': 1})],
    interactions=[Shrinking('growth', target='plant')],
)
print(Simulation(plant).run(until=5).trajectory_.structure('plant', 5))
```

A generation that would make a word longer than `max_size` raises `RewritingError` before allocating it, and the
frames before it stay recorded.

### Word geometries, patches, and measures

A `WordGeometry`, from `skcomplex.core.viz`, draws a word as line segments in two or three dimensions, as the turtle
interpretation of L-systems does: `F` draws forward, `f` moves, `+` and `-` turn, `[` and `]` push and pop a branch,
and `contraction` scales the step at every generation. A `Patch` holds tiles, each a `Prototile` segment or polygon
under an affine placement, and a `Substitution` replaces every tile of a prototile by pieces placed by `Affine` maps
relative to it. Structures with a geometry report `<element>.length` and
`<element>.area` at every frame, and give `bounds()` and `raster(resolution)`, a boolean grid that
`box_counting_dimension` reads.

```python
from skcomplex.core.analysis import box_counting_dimension
from skcomplex.core.components import Patch
from skcomplex.core.laws import Affine, Prototile, Substitution
from skcomplex.core.viz import WordGeometry

koch = System(
    components=[Word('koch', axiom='F', geometry=WordGeometry(angle=60.0, contraction=1 / 3))],
    interactions=[Rewrite('rules', target='koch', rules=[Production('F', 'F+F--F+F')])],
)
curve = Simulation(koch).run(until=4).trajectory_.structure('koch', 4)
print(len(curve.segments()), round(curve.length(), 6))

interval = Prototile('interval', [(0, 0), (1, 0)])
thirds = [('interval', Affine(scale=1 / 3)), ('interval', Affine(scale=1 / 3, translation=(2 / 3, 0)))]
cantor = System(
    components=[Patch('cantor', prototiles=[interval])],
    interactions=[Rewrite('rules', target='cantor', rules=[Substitution('interval', thirds)])],
)
trajectory = Simulation(cantor).run(until=5).trajectory_
print(trajectory['cantor.size'][-1], round(trajectory['cantor.length'][-1], 6))
grid = trajectory.structure('cantor', 5).raster(729)
print(round(box_counting_dimension(grid, [3, 9, 27]), 2))
```

The `geometry` visual logs segments and outlines as line strips, a patch with `filled=True` as an image of its raster,
and a host graph as graph nodes and edges, in the bounds the element declares or that its first tiles give.

### Graph rewriting

A `GraphProduction` gives a left graph, an interface, and a right graph that share nodes by name, and matches by their
`label` attributes. Every generation applies it at a maximal set of parallel independent matches, chosen in an order
drawn from the random stream, and skips matches whose deleted nodes would leave dangling edges.

```python
import networkx as nx

from skcomplex.core.components import HostGraph
from skcomplex.core.laws import GraphProduction

interface = nx.Graph()
interface.add_nodes_from('ab')
subdivide = GraphProduction(nx.Graph([('a', 'b')]), interface, nx.Graph([('a', 'c'), ('c', 'b')]))
ring = System(
    components=[HostGraph('ring', graph=nx.cycle_graph(4))],
    interactions=[Rewrite('rules', target='ring', rules=[subdivide])],
)
print(Simulation(ring, random_state=0).run(until=3).trajectory_['ring.size'].tolist())
```

## Quantum systems

A quantum system is a set of quantum subsystems whose state is one vector of their joint Hilbert space. `Spin` has the
components `x`, `y`, `z` and the ladder operators `plus` and `minus` of spin `j`, `Qubit` has the Pauli matrices,
`Mode` has `annihilate`, `create`, and `number` truncated at `quanta`, and `Site` is a particle on the sites of a
lattice or a network with its `laplacian`, its `adjacency`, and the functions `projector` and `diagonal`. A group of
subsystems, such as `Spin.lattice`, is the tensor product of its members' spaces in member order, and the initial
state of every subsystem is a basis label, an index, or a vector.

### Operators and laws

Laws receive the local operators of their subsystems, which combine by sums, products, numbers, and symbolic
coefficients of parameters and time into an `Operator`. `Generator(operator)` adds to the Hamiltonian, with
`ħ = 1`, and `Dissipator(operator, rate)` adds a Lindblad dissipator, which makes the system open. An interaction
with the binding `over = 'neighbours'` is written once for a bond, with `s.first` and `s.second`, and acts on every pair
of neighbours of the space of the group its one role binds. Compiling checks that the Hamiltonian is Hermitian and that the dimension stays within the
`max_dimension` of the quantum solver. A whole-system initial state, such as an entangled vector or `'ground'`, is
given to the simulation as `initial_state={'<system label>.state': value}`.

```python
from operator import attrgetter

import numpy as np

from typing import ClassVar

from skcomplex.core.base import parameter, role
from skcomplex.core.interactions import Interaction
from skcomplex.core.laws import Generator
from skcomplex.core.space import LatticeSpace
from skcomplex.core.system import System, expectation
from skcomplex.core.runtime import Simulation
from skcomplex.models.physics.realizations import Spin


class Coupling(Interaction):
    over: ClassVar[str] = 'neighbours'
    target: str = role()
    J: float = parameter(1.0)

    def laws(self, s):
        a, b = s.first, s.second
        return [Generator(s.J * (a.x * b.x + a.y * b.y + a.z * b.z))]


chain = System(
    components=[Spin.lattice('chain', LatticeSpace(6), initial=['up', 'down'] * 3)],
    interactions=[Coupling('coupling', target='chain')],
)
observables = [expectation('sz', attrgetter('z'), group='chain')]
trajectory = Simulation(chain, dt=0.5, observables=observables).run(until=2.0).trajectory_
print(trajectory['sz'].shape, round(float(np.ptp(trajectory['energy'])), 8))
```

### Solvers, observables, and spectra

A quantum system compiles to the quantum form, whose state is stored as complex pairs under `<system>.state`, a pure
state for a closed system and a density matrix for an open one. Its solvers hand every frame to QuTiP: `unitary` for
the Schrödinger equation, `lindblad`, the default, for the master equation, and `quantum_trajectories` for quantum
jumps drawn from the run's random stream. `expectation` makes observables of operators, one value per member for a
group, and every quantum run also records its `energy`, and its `purity` when it is open. Groups on a lattice stream
the space-time image of their local expectations to Rerun. `spectrum` and `ground_state` diagonalise the Hamiltonian
exactly, optionally in the sector of a conserved diagonal operator, `purity` and `entanglement_entropy` read
trajectories, and `hamiltonian_of` prints the Hamiltonian in operator notation.

```python
from skcomplex.core.analysis import spectrum
from skcomplex.models import create

print(spectrum(create('quantum_heisenberg', n_sites=4), k=2).energies.round(6).tolist())
```

## Co-simulation

A system whose laws do not fit one form, such as agents next to differential equations, is split into partitions at
subsystem boundaries. Every subsystem whose laws fit one form becomes a partition, and a law joins the partition that
owns its target, so an interaction of the parent system may contribute to several partitions. A subsystem whose own
elements mix law kinds raises `IncompatibleLawsError` with its path and the suggestion to split it.

```python
from skcomplex.core.compile import compile_system
from skcomplex.models.economics.resources import harvesting

farming = harvesting(10, effort=0.05)
print([(partition.name, partition.problem.form.value) for partition in compile_system(farming).partitions])
trajectory = Simulation(farming, random_state=0).run(until=50).trajectory_
print(round(trajectory['land/stock.n'][-1]), trajectory.agents('village/farmers', 50).values['wealth'].size)
```

- Every partition holds the states of the other partitions as inputs, which keep their values from the last
  synchronisation point while it advances.
- `Aggregate(label, group=..., reduce='count' | 'sum' | 'mean' | callable, variable=None)` turns an agent group or an
  element group into the single state `value`, which other partitions read by its dotted name, such as
  `village/size.value`.
- Agent rules read the states of other partitions through `model.inputs[name]`, such as
  `model.inputs['land/stock.n']`.
- The `CoSimulation` solver advances every partition with its own solver, chosen per partition through
  `solvers={'land': ScipyODE()}`, and exchanges inputs every `sync_dt`, which defaults to the frame duration and must
  divide it, and must be a whole number of steps for discrete and agent partitions.
- The Gauss-Seidel scheme, the default, advances partitions in system order, each reading the values already advanced
  by those before it. `scheme='jacobi'` advances all of them from the values at the start of the interval.
- Splitting adds an error of first order in `sync_dt`, so halving `sync_dt` roughly halves it at the cost of more
  exchanges. `CoSimulation(split=True)` splits a system that fits one form, which shows the error against the joint
  solution.

## Visualizing with Rerun

`RerunObserver` streams frames to a spawned viewer, a running viewer, or a `.rrd` file. Entity paths follow the system
tree, each element type chooses its visual, and parameters appear under `parameters/`.

```python
import tempfile
from pathlib import Path

from skcomplex.core.space import ContinuousSpace
from skcomplex.core.components import Particle
from skcomplex.models.physics.realizations import Spring
from skcomplex.core.viz import RerunObserver

molecule = System(
    components=[Particle('a', x=[0.0, 0.0, 0.0]), Particle('b', x=[1.5, 0.0, 0.0], v=[0.0, 0.5, 0.0])],
    interactions=[Spring('bond', first='a', second='b', k=4.0, length=1.0)],
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
`Particle.group(..., visual=VisualSpec('points', {...}))`. A new visual subclasses `Visual` and is registered with
`register_visual`, and an element type selects it with its `visual` class variable.

### Layouts and blueprints

Every visual declares the Rerun views that show what it logs, through its `views` method. Scalars go in time series
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

```python
import rerun.blueprint as rrb

from skcomplex.core.viz import default_blueprint, primary_view, save_blueprint
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
from skcomplex.core.experiments import Experiment, Grid, LatinHypercube, Scenario, calibrate, sensitivity

ensemble = Experiment('sir', Grid(stochastic=[True], gamma=[0.1, 0.2]), replicates=10, until=60, random_state=0)
print(ensemble.run().summary()[0]['I.n.mean'])

policies = [Scenario('baseline'), Scenario('quota', schedule=[(50.0, 'harvest.effort', 0.025)])]
stocks = Experiment('harvesting', scenarios=policies, until=80, outputs=['land/stock.n']).run()
print([round(float(row['land/stock.n'])) for row in stocks.rows])
```

- Designs are `Grid`, `RandomDesign`, `LatinHypercube`, `SobolDesign`, and `Rows`. They cross with `*` and
  concatenate with `+`, and their bounds are pairs or SciPy distributions.
- `outputs` keeps states and observables as series, after `warmup`, and functions of a trajectory as scalars. Results
  give one row per run, `series`, `summary` with means, standard deviations, and quantiles across replicates, and
  `to_frame` for pandas.
- `sensitivity` ranks parameters one at a time, by Morris screening, or by Sobol indices with SALib.
- `calibrate` fits dotted paths to observed series by least squares or a Gaussian likelihood, and reports the
  covariance of the estimates with R², RMSE, MAE, and MAPE.

```python
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
from skcomplex.core.analysis import equations_of, solve_symbolically
from skcomplex.models.physics.realizations import HarmonicPotential

oscillator = System(
    components=[Particle('p', x=1.0)],
    interactions=[HarmonicPotential('trap', target='p', k=4.0)],
    space=ContinuousSpace(n_dim=1),
)
print(equations_of(oscillator)['p.v[0]'])
print(solve_symbolically(oscillator, use_initial_state=True)['p.x[0]'])
```


## Analysing dynamics and data

`skcomplex.core.analysis` measures complex dynamics in the same way for every field. Every measure takes a system, or plain
arrays from trajectories or measured data.

**Stability and chaos.** `fixed_points`, `jacobian`, `stability`, `nullclines`, and `divergence` work on the symbolic
vector field of a flow or a map. `lyapunov_spectrum` integrates its tangent dynamics compiled with Numba, and
`orbit_diagram` and `basin_stability` run experiments over parameters or initial conditions.

```python
from skcomplex.core.analysis import fixed_points, lyapunov_spectrum, orbit_diagram, stability
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
import math

from skcomplex.core.analysis import stroboscopic
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
from skcomplex.core.analysis import detect_events, fit_power_law, gini

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
`ordinal_patterns`, `shannon_entropy`, `block_entropy`, `entropy_rate`, `excess_entropy`, `mutual_information`,
`transfer_entropy`, `permutation_entropy`, and `lempel_ziv_complexity` measure structure and information flow, as in
Lindgren's information theory of complex systems. Ordinal patterns are coded by the lexicographic rank of their
permutation with `ordpy`, with a delay and of any length, and the Lempel-Ziv parsing comes from `antropy`.

```python
from skcomplex.core.analysis import entropy_rate, excess_entropy, hurst_exponent, transfer_entropy

noise = np.random.default_rng(1).standard_normal(20_000)
print(round(hurst_exponent(noise), 2), round(hurst_exponent(np.cumsum(noise)), 2))
print(round(entropy_rate([0, 0, 1, 1] * 200, 4), 3), round(excess_entropy([0, 0, 1, 1] * 200, 4), 3))
driver = np.random.default_rng(2).integers(0, 2, 5000)
print(round(transfer_entropy(driver, np.roll(driver, 1)), 2))
```

**Space and networks.** `label_clusters` labels clusters of a grid, with periodic edges, `correlation_function` gives
radially averaged correlations, and `box_counting_dimension` measures fractals. `degrees`, `summarize_network`, and
`centrality` measure NetworkX graphs and network spaces.

```python
import networkx as nx

from skcomplex.core.analysis import box_counting_dimension, label_clusters, summarize_network

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
from skcomplex.core.analysis import analyses_for, data_kinds, fields_of, series_of, structure_factor
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

```python
from skcomplex.core.analysis import signature_analysis

for signature in signature_analysis(predation):
    print(signature.kind.value, signature.title, signature.xscale, list(signature.values))
```

### Complexity and information

`renyi_entropy` and `tsallis_entropy` generalize the Shannon entropy of a distribution, and `ensemble_entropy` follows
the entropy of an ensemble and its divergence from a reference law frame by frame, as in the H-theorem.
`sample_entropy`, `approximate_entropy`, `multiscale_entropy`, and `spectral_entropy` come from `antropy` and measure the
regularity of a series. `permutation_entropy` reads ordinal patterns of series with a delay, or of two-dimensional
fields in a window, in its Shannon, Rényi, or Tsallis form, and `complexity_entropy` places data on Rosso's
complexity-entropy plane between its bounding curves with `ordpy`. Discrete states such as spins have many equal values,
which `break_ties` ranks at random. `active_information_storage` measures the information the past of a series holds
about its next value, and `transfer_entropy` estimates the flow between series by plug-in counts of symbols or by the
nearest-neighbour estimator of `ennemi` on continuous values.

```python
import math

from skcomplex.core.analysis import complexity_entropy, renyi_entropy, sample_entropy, transfer_entropy

rng = np.random.default_rng(0)
print(round(renyi_entropy(np.full(8, 1 / 8), 2, probabilities=True), 3))
print(round(sample_entropy(rng.standard_normal(2000)), 1))
plane = complexity_entropy(rng.standard_normal(20_000), 4)
print(round(plane.entropy, 2), round(plane.complexity, 2))
x = rng.standard_normal(5000)
y = np.concatenate([[0.0], 0.8 * x[:-1]]) + 0.6 * rng.standard_normal(5000)
print(transfer_entropy(x, y, estimator='knn', base=math.e) > transfer_entropy(y, x, estimator='knn', base=math.e))
```

### Invariants of dynamics

`embed` builds delay coordinates of a series, with the delay at the first minimum of `auto_mutual_information` and the
dimension where `false_nearest_neighbors` fall below a threshold. `correlation_dimension` counts close pairs of any point
set with a k-d tree, so it serves embedded series and agent positions alike. `kaplan_yorke_dimension` and
`kolmogorov_sinai_entropy` follow from a Lyapunov spectrum, `largest_lyapunov_exponent` estimates the largest exponent
from one series with Rosenstein's method of `nolds`, and `invariant_density` and `ulam_chain` estimate the invariant
density of a map. `recurrence_matrix` and `recurrence_quantification` give the recurrence rate, determinism,
laminarity, line lengths, trapping time, and line entropy of embedded series.

```python
from skcomplex.core.analysis import (
    correlation_dimension,
    delay_embedding,
    kaplan_yorke_dimension,
    recurrence_quantification,
    ulam_chain,
)

points = np.random.default_rng(1).random((3000, 2))
print(round(correlation_dimension(points).dimension, 1))
print(round(kaplan_yorke_dimension([0.906, 0.0, -14.572]), 2))
chain = ulam_chain(lambda x: 4 * x * (1 - x), np.linspace(0, 1, 21), 200)
print(chain.density.round(1)[:3])
sine = delay_embedding(np.sin(np.arange(1000) * 0.1), 2, 15)
print(round(recurrence_quantification(sine).determinism, 2))
```

### Stochastic processes

`first_passage_times`, `hitting_probabilities`, `return_times`, and `laminar_durations` measure passages and returns of
paths. `transition_matrix`, `stationary_distribution`, `spectral_gap`, `relaxation_times`, and `detailed_balance` read
transition and rate matrices or estimate them from paths. `fano_factor`, `event_counts`, `relative_fluctuations`, and
`kramers_moyal` describe fluctuations, the last with `kramersmoyal`. `entropy_production`, `jump_entropy_production`,
`langevin_entropy_production`, `count_entropy_production`, and `fluctuation_theorem` measure irreversibility.
`tilted_generating_function`, `block_generating_function`, `legendre_transform`, `histogram_rate_function`, and
`reweight` give large deviations. `markov_state_model`, `implied_timescales`, `committor`, and
`mean_first_passage_time` estimate Markov state models with `deeptime`. `two_time_correlation`, `aging_collapse`,
`collapse_distributions`, `response`, and `fluctuation_dissipation_ratio` measure aging, and `fit_extreme_values`,
`record_counts`, `fit_stable`, and `compare_with_law` describe extremes and limit laws.

```python
from skcomplex.core.analysis import (
    entropy_production,
    legendre_transform,
    spectral_gap,
    stationary_distribution,
    tilted_generating_function,
)

ring = 2 * np.roll(np.eye(5), 1, axis=1) + np.roll(np.eye(5), -1, axis=1)
print(stationary_distribution(ring, rates=True).round(2), round(entropy_production(ring), 4))
print(round(spectral_gap([[0.9, 0.1], [0.2, 0.8]]), 2))
tilts = np.linspace(-3, 3, 601)
print(round(float(legendre_transform(tilts, tilted_generating_function([[2.0]], [[1.0]], tilts), 3.0)[0]), 3))
```

### Fields and lattices

`structure_factor` averages the power of a field radially on a periodic grid of any dimension, and `correlation_length`
fits its Ornstein-Zernike form or the decay of the correlation function. `domain_size` and `domain_growth` measure
coarsening, `pattern_wavelength` the wavelength and orientation of patterns, and `number_fluctuations` the giant number
fluctuations of density fields. `cluster_sizes`, `percolation`, and `spanning_probability` describe clusters of grids
and components of graphs, and `binder_cumulant`, `binder_crossings`, and `scaling_collapse` locate transitions by
finite-size scaling.

```python
from skcomplex.core.analysis import binder_cumulant, pattern_wavelength, percolation

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
`like_neighbor_share`, `interface_density`, and `dissimilarity_index` measure segregation on grids and networks.

```python
from skcomplex.core.analysis import clark_evans_ratio, coarse_grain, polar_order, snapshots_of

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
from skcomplex.core.analysis import avalanches, branching_ratio

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
from skcomplex.core.analysis import communities, motif_counts

club = nx.karate_club_graph()
print(round(communities(club, random_state=0).modularity, 2), motif_counts(club)['triangle'])
```
