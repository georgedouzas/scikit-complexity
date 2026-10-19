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
| `Rate` | ordinary differential equations | `ScipyODE` |
| `Rate` and `Noise` | stochastic differential equations | `EulerMaruyama` |
| `Jump` | jump process | `Gillespie` |
| `Update` | discrete map | `Iteration` |

## Defining elements and interactions

Elements and interactions declare their fields with `state`, `parameter`, and `role`. Each field becomes a constructor
argument, stored unmodified under its own name.

```python
from skcomplex.core.base import Domain, parameter, role, state
from skcomplex.core.model import Element, Interaction, Jump, Rate


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
symbolic remainder, so `skcomplex.core.model` adds `where`, `either`, `both`, and `mod`, which every backend lowers, LAMMPS
included.

```python
from sage.all__sagemath_symbolics import exp, sqrt

from skcomplex.core.model import where


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
from skcomplex.core.model import Element

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

### Equations of any order

Laws give the highest time derivative of a state, and a state declares how high that is with `order`. Newton's second
law for a pendulum, `m l θ'' = -m g sin θ`, is written as it reads. The framework adds the angular velocity as the
companion state `theta_d1`, sets its initial value from the constructor, and reduces the equation to first order for
the solvers.

```python
from sage.all__sagemath_symbolics import sin

from skcomplex.core.model import Equation, derivative

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
from skcomplex.core.model import System
from skcomplex.models.biology.ecology import Population, Predation

landscape = System(
    systems=[
        System('forest', elements=[Population('foxes', n=5)]),
        System('meadow', elements=[Population('rabbits', n=50)]),
    ],
    interactions=[Predation('hunt', predator='forest/foxes', prey='meadow/rabbits', a=0.02, b=0.01)],
)
```

States and parameters are named by their path, such as `meadow/rabbits.n` and `hunt.a`.

## Simulating

`Simulation` compiles the system, picks the solver of its form, and advances it in frames of `dt`.

```python
from skcomplex.core.runtime import RealTime, Simulation

trajectory = Simulation(landscape, dt=0.1).run(until=10).trajectory_
print(trajectory['meadow/rabbits.n'][-1])
```

- `clock=RealTime(speed)` paces frames so that simulated time advances `speed` times faster than wall time.
- `start(until)` runs in a background thread, and `pause`, `resume`, `stop`, and `wait` control it.
- `update_parameters({'hunt.a': 0.05})` changes parameters before the next frame.
- `random_state` makes stochastic runs reproducible.
- `schedule={'hunt.a': lambda t: 0.02 + 0.001 * t}` sets parameters from functions of time at every frame, such as a
  slow ramp of a control parameter.

## Delays, events, drivers, and noise

Laws can read the past, change states at discrete events, follow inputs that switch in time, and take noise in the
convention of the field.

**Delays.** `delayed(x, tau)` is the state `x` at `t - tau`, for a constant or a state-dependent delay. The solvers keep
a history of dense output and step no further than the smallest delay, by the method of steps. Before the start the
history is the initial value, or what `history={path: value or function of time}` on the system gives.

```python
import numpy as np

from skcomplex.core.model import Event, Noise, Rate, delayed, step

MackeyGlass = Element.define(
    'MackeyGlass',
    states={'x': 0.5},
    parameters={'beta': 0.2, 'gamma': 0.1, 'n': 10.0, 'tau': 17.0},
    laws=lambda s: [Rate(s.x, s.beta * delayed(s.x, s.tau) / (1 + delayed(s.x, s.tau) ** s.n) - s.gamma * s.x)],
)
cells = System(elements=[MackeyGlass('cells')], history={'cells.x': lambda t: 0.5 + 0.1 * np.sin(t)})
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
trajectory = Simulation(System(elements=[Ball('ball')]), dt=0.01).run(until=3).trajectory_
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
print(Simulation(System(elements=[Filter('filter')]), dt=0.5).run(until=10).trajectory_['filter.y'][-1])
```

**Noise conventions.** `System(..., noise='stratonovich')` or `'anti-ito'` declares how noise laws are read. The
compiler adds the drift correction `α g ∂g/∂x`, with `α = 1/2` and `1`, so every SDE solver integrates the equivalent
Itô equation. `EulerMaruyama` has strong order 1/2. `Milstein`, in Platen's derivative-free form, has strong order 1 for
diagonal noise, and `Heun` averages the drift over every step.

```python
from skcomplex.core.solvers import Milstein

Growth = Element.define(
    'Growth',
    states={'x': 1.0},
    parameters={'mu': 0.5, 'sigma': 0.8},
    laws=lambda s: [Rate(s.x, s.mu * s.x), Noise(s.x, s.sigma * s.x)],
)
growth = System(elements=[Growth('asset')], noise='stratonovich')
print(Simulation(growth, solver=Milstein(dt=0.01), random_state=0).run(until=1).state_)
```

SciPy, RK4, Euler-Maruyama, Milstein, and Heun handle delays, events, and drivers. Velocity Verlet, Gillespie, and map
iteration reject delays and events with an error that names the solvers to use. The library has these models ready:
`mackey_glass`, `delayed_kuramoto`, and `stochastic_resonance` in nonlinear dynamics, `hutchinson` in ecology,
`bouncing_ball`, `newtons_cradle`, `circular_billiard`, and `brownian_near_wall` in physics, and
`integrate_and_fire` and `izhikevich` in neuroscience.

## Observables

Observables are named quantities of a system, computed at every frame, recorded in trajectories under the system's
path, and logged to Rerun. An observable is either an expression of the system namespace, where earlier observables
appear by label, or a reduction over a group's members of a variable or an expression of one member: `count`, `sum`,
`mean`, `min`, `max`, `var`, or a function.

```python
import numpy as np
from sage.all__sagemath_symbolics import I, exp

from skcomplex.core.model import Observable

Oscillator = Element.define(
    'Oscillator',
    states={'theta': state(0.0, domain=Domain.ANGLE)},
    parameters={'omega': 1.0},
    laws=lambda s: [Rate(s.theta, s.omega)],
)
oscillators = System(
    elements=[Oscillator.group('phases', 100, theta=np.linspace(0, 1, 100), omega=np.linspace(0.9, 1.1, 100))],
    observables=[
        Observable('z', lambda member: exp(I * member.theta), group='phases'),
        Observable('order', lambda s: abs(s.z)),
    ],
)
order = Simulation(oscillators, dt=1.0).run(until=50).trajectory_['order']
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
coupled = System(
    elements=[Oscillator.group('phases', 100, theta=np.linspace(0, 6, 100), omega=np.linspace(0.9, 1.1, 100))],
    interactions=[MeanField('coupling', target='phases', X='X', Y='Y', K=2.0)],
    observables=[
        Observable('X', lambda member: cos(member.theta), group='phases'),
        Observable('Y', lambda member: sin(member.theta), group='phases'),
    ],
)
print(Simulation(coupled, dt=1.0).run(until=20).trajectory_['X'][-1].round(2))
```

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
from skcomplex.core.model import Element, Rate, System
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
    return System(elements=[Tank('tank', level=level)])


print(create('draining_tank').elements[0].level)
```

Every registered factory needs defaults for all of its arguments, so the catalog can create, compile, simulate,
visualize, and clone every model through the same interface.

### Coverage

The model inventory records every model surveyed from the reference books and articles, with its sources and status:
implemented, pending a planned change, blocked by a missing feature of the core, or out of scope. The coverage page,
under Models in the documentation, renders it by field and subfield. A model the core cannot express is recorded as
blocked on the OpenSpec change that specifies the missing general feature, never worked around inside the model
package, and a test checks the inventory against the catalog and the changes.

### Moved paths

Earlier versions placed the framework at the top level and grouped the models by technique. The framework packages
`skcomplex.model`, `compile`, `solvers`, `runtime`, `viz`, `experiments`, `analysis`, and `backends` moved under
`skcomplex.core`, the shared leaves moved from `skcomplex.core` to `skcomplex.core.base`, and the domain-agnostic
particle components, such as `Particle`, `LennardJones`, `PeriodicBox`, and `LangevinThermostat`, moved from
`skcomplex.models.physics` to `skcomplex.core.model`. The models moved to their subfields:

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

from skcomplex.core.model import EuclideanSpace, MesaSystem
from skcomplex.core.model import Particle, ReflectingBox

rng = np.random.default_rng(0)
gas = System(
    elements=[Particle.group('gas', 10_000, x=rng.uniform(0, 1, (10_000, 2)), v=rng.normal(0, 1, (10_000, 2)))],
    interactions=[ReflectingBox('box', target='gas', size=1.0)],
    space=EuclideanSpace(2),
)
trajectory = Simulation(gas, dt=0.01).run(until=1).trajectory_
print(trajectory['gas.x'].shape)
```

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
- An interaction acts on one group and reads others through reductions. A law that sets a single element from
  reductions alone runs once, not once per member.
- Discrete maps read the reductions at step `n`, jump propensities after every event, and events at the state they
  receive.

```python
from sage.all__sagemath_symbolics import tanh

from skcomplex.core.model import total

Unit = Element.define('Unit', states={'x': 0.0}, laws=lambda s: [Rate(s.x, -s.x + tanh(total(s.x)))])
units = System(elements=[Unit.group('units', 50, x=np.linspace(-1, 1, 50))])
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
    elements=[Strategy.group('population', 2, x=[0.1, 0.9])],
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
    elements=[Activity(f'unit_{index}', x=0.1 * (index + 1)) for index in range(3)],
    interactions=[Inhibition('inhibition', units=['unit_0', 'unit_1', 'unit_2'])],
)
print(Simulation(network, dt=1.0).run(until=30).state_.round(3))
```

### Jacobians of reductions

A reduction makes every reading member depend on every reduced member, so the Jacobian of a drift with reductions has
the rank-one block `dF_i/dR dR/dx_j` of every reduction `R`. `problem.jacobian` of an ODE problem evaluates the exact
Jacobian by the chain rule through every reduction, as a SciPy sparse matrix, and `problem.jacobian.pattern()` states
its structure. When the dense blocks of the reductions exceed 4 million entries, `pattern()` returns `None` and an
implicit solver estimates a dense Jacobian. Mean-field groups are rarely stiff, so explicit steppers such as `RK4` and
`ScipyODE` with `RK45` suit large groups with reductions.

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
from skcomplex.core.model import PairPotential


class LennardJones(PairPotential):
    epsilon: float = parameter(1.0, domain=Domain.POSITIVE)
    sigma: float = parameter(1.0, domain=Domain.POSITIVE)

    def pair_energy(self, s, r):
        return 4 * s.epsilon * ((s.sigma / r) ** 12 - (s.sigma / r) ** 6)


side = np.arange(10) * 1.2
liquid = System(
    elements=[Particle.group('atoms', 100, x=np.stack(np.meshgrid(side, side), axis=-1).reshape(-1, 2))],
    interactions=[LennardJones('lj', first='atoms', second='atoms', cutoff=2.5)],
    space=EuclideanSpace(2),
)
Simulation(liquid, dt=0.05).run(until=1)
```

`period` gives the side of a periodic box, so pair distances use the nearest periodic image.

### Boundaries

A `Boundary` law sets the value of a state after every solver step. Boundaries combine with every form, so walls and
periodic boxes work for differential equations, stochastic systems, jump processes, and discrete maps.
`ReflectingBox` mirrors positions and velocities at the walls, and `PeriodicBox` wraps positions around the box.

## Lattice models

Cellular automata, sandpiles, and spin models live on lattices. `LatticeSpace(shape, boundary=...)` is a box of `Z^d`
with `'periodic'`, `'fixed'`, or `'reflecting'` edges, and `Element.lattice(label, space, **values)` puts one member
of an element type on every site, with values given as arrays of the lattice shape. Laws are written for one cell.

**Local rules.** `LocalRule(target, neighborhood, rule)` computes a cell's next value from the ordered values of its
neighbourhood. `Neighborhood.moore`, `.von_neumann`, `.radius`, or `Neighborhood(offsets)` give the offsets. A rule is
a `Wolfram` number, a `Table` of patterns, a `Totalistic` or `OuterTotalistic` rule, or a function of NumPy arrays,
one per neighbour, that also receives the parameters it names and, for stochastic rules, uniform random numbers.
Tables and totalistic rules run as lookups, fused into one compiled loop on the Numba backend.

```python
from skcomplex.core.model import LatticeSpace, LocalRule, Neighborhood, OuterTotalistic

Life = Element.define(
    'Life',
    states={'alive': state(0, domain=Domain.BOOLEAN)},
    laws=lambda s: [LocalRule(s.alive, Neighborhood.moore(), OuterTotalistic.life(born=(3,), survive=(2, 3)))],
)
glider = np.zeros((20, 20))
glider[1, 2] = glider[2, 3] = glider[3, 1] = glider[3, 2] = glider[3, 3] = 1
cells = System(elements=[Life.lattice('cells', LatticeSpace((20, 20)), alive=glider)])
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

**Monte Carlo.** `Energy(expression, neighborhood)` is the local energy of a site, every term of the total energy
that involves it, with neighbour values read through `neighbors`. `Move.flip`, `Move.random_value`, and `Move.rotate`
propose changes, and `Sampler('metropolis' | 'glauber' | 'heat-bath', temperature)` accepts them. The energy change of
a move is derived symbolically, sweeps update the two sublattices in turn, and the temperature is a parameter that
changes live.

```python
from skcomplex.core.model import Energy, Move, Sampler, neighbors

nearest = Neighborhood.von_neumann()
Spin = Element.define(
    'Spin',
    states={'spin': state(1.0, domain=Domain.INTEGER)},
    parameters={'J': 1.0, 'temperature': 1.5},
    laws=lambda s: [
        Energy(-s.J * s.spin * sum(neighbors(s.spin, nearest)), nearest),
        Move.flip(s.spin),
        Sampler('metropolis', s.temperature),
    ],
)
spins = System(elements=[Spin.lattice('spins', LatticeSpace((32, 32)))])
print(abs(Simulation(spins, dt=50, random_state=0).run(until=200).state_.mean()) > 0.9)
```

Lattice groups stream to Rerun as images colored by value. The library has `elementary_automaton`, `game_of_life`,
`majority_rule`, `langtons_ant`, `nagel_schreckenberg`, `hodgepodge`, `sandpile`, `forest_fire`, `earthquakes`,
`ising`, `potts`, and `xy`, each in the subfield of its science, such as `skcomplex.models.computer_science.automata`
for `game_of_life` and `skcomplex.models.physics.statistical_mechanics` for `ising`.

## Fields

A field holds one value per site of a lattice or per node of a network. `Field(label, element, space, boundary=...,
values=...)` places a member of an element type on every site, and the laws of the type, written for one site, read
the neighbouring sites through differential operators. The method of lines turns a field into ordinary differential
equations, so every ODE solver and backend runs it.

```python
from skcomplex.core.model import Field, laplacian

Diffusion = Element.define(
    'Diffusion',
    states={'u': 0.0},
    parameters={'D': 1.0},
    laws=lambda s: [Rate(s.u, s.D * laplacian(s.u))],
)
initial = np.zeros(400)
initial[200] = 20.0
rod = Field('rod', Diffusion, LatticeSpace(400, spacing=0.05), boundary={'u': 'zero-flux'}, values={'u': initial})
trajectory = Simulation(System(elements=[rod]), dt=0.1).run(until=0.5).trajectory_
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
  condition and conserves their totals.
- `'finite-element'` fields live on a `Mesh`, such as `Mesh.l_shape`, `Mesh.rectangle`, `Mesh.disk`, or
  `Mesh.load(path)`, with stiffness and mass matrices from scikit-fem and boundary values that can be functions of
  the coordinates. They stream to Rerun as colored meshes.

The compiler splits the rate of every field into a linear part with constant coefficients, such as
`D * laplacian(u)` or `-laplacian(laplacian(u))`, and a nonlinear remainder. `IMEXEuler`, `IMEX2`, and
`CrankNicolson` treat the linear part implicitly, with sparse solves for finite differences and finite elements and
solves per Fourier mode for spectral fields. `ETDRK4` integrates it exactly per mode, coupled states included, so stiff
equations take steps far beyond the explicit limit.

```python
from skcomplex.core.model import divergence
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
print(np.abs(Simulation(System(elements=[line]), dt=10, solver=ETDRK4(dt=0.25)).run(until=100).state_).max())
```

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
from skcomplex.core.compile import NumbaBackend
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

from skcomplex.core.model import Call, Rate
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
    elements=[Population.group('herds', 2000, n=1.0)],
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
from skcomplex.core.model import LangevinThermostat, LennardJones

atoms = System(
    elements=[Particle.group('atoms', 100, x=np.stack(np.meshgrid(side, side), axis=-1).reshape(-1, 2))],
    interactions=[
        LennardJones('lj', first='atoms', second='atoms', cutoff=2.5),
        LangevinThermostat('bath', target='atoms', temperature=0.5, gamma=1.0),
    ],
    space=EuclideanSpace(2),
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

Agents follow rules written as code. An `AgentType` declares the states every agent holds and the parameters its group
shares, and returns `Rule` laws whose functions take the agent and the model. A system with rules is agent-based and
runs on Mesa, one Mesa step per unit of time, with every agent activated in random order.

```python
from skcomplex.core.base import Domain, VisualSpec, parameter, state
from skcomplex.core.model import AgentType, GridSpace, Rule


class Forager(AgentType):
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


foragers = System(elements=[Forager.group('foragers', 50)], space=GridSpace(30, 30, torus=True))
trajectory = Simulation(foragers, random_state=0).run(until=20).trajectory_
print(trajectory['foragers.count'][-1], len(trajectory.agents('foragers', 20)))
```

- An agent reads and sets its states as attributes and reads its group's parameters, which change live like any
  parameter. `neighbors`, `neighbor_cells`, `agents_here`, `move_to`, `create`, and `remove` act on its place and its
  population.
- The model gives the random generator `model.rng`, the other agents with `model.agents(group)`, and empty cells.
- `GridSpace` has Moore or von Neumann neighbourhoods and an optional cell capacity, `NetworkSpace` places agents on
  the nodes of a NetworkX graph, and `ContinuousSpace` finds neighbours within a radius.
- Frames and trajectories hold every group's agents with their ids, states, and locations, through
  `trajectory.agents(group, frame)`, and the count of every group as the reporter `<group>.count`.
- `skcomplex.models.social` has Schelling segregation and the voter model, `skcomplex.models.economics.econophysics`
  has Boltzmann wealth exchange, and `skcomplex.models.biology.ecology` has the wolf and sheep model.

### Existing Mesa models

`MesaSystem` runs any Mesa model with the framework's runtime, Rerun visualization, and live parameters, without
changing its code. Numeric constructor arguments become parameters, model reporters become reporters, and agent
reporters become agent states.

```python
import mesa


class Coin(mesa.Agent):
    def __init__(self, model):
        super().__init__(model)
        self.heads = 0

    def step(self):
        self.heads += int(self.model.random.random() < self.model.bias)


class Coins(mesa.Model):
    def __init__(self, n=10, bias=0.5, rng=None):
        super().__init__(rng=rng)
        self.bias = bias
        Coin.create_agents(self, n)

    def step(self):
        self.agents.do('step')


coins = MesaSystem(
    'coins',
    model=Coins,
    parameters={'n': 10, 'bias': 0.5},
    model_reporters={'heads': lambda model: sum(agent.heads for agent in model.agents)},
    agent_reporters={'heads': 'heads'},
)
print(Simulation(coins, random_state=0).run(until=10).trajectory_['coins.heads'][-1])
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

from skcomplex.core.model import EuclideanSpace
from skcomplex.core.model import Particle, Spring
from skcomplex.core.viz import RerunObserver

molecule = System(
    elements=[Particle('a', x=[0.0, 0.0, 0.0]), Particle('b', x=[1.5, 0.0, 0.0], v=[0.0, 0.5, 0.0])],
    interactions=[Spring('bond', first='a', second='b', k=4.0, length=1.0)],
    space=EuclideanSpace(3),
)
path = Path(tempfile.mkdtemp()) / 'molecule.rrd'
Simulation(molecule, dt=0.02, observers=[RerunObserver('molecule', save_path=path)]).run(until=5)
```

Particles appear as 3D points, populations as time series, agents on grids and in continuous spaces as points
colored by a state, agents on networks as graph nodes, and reporters under `reporters/`. A group is logged as one batch per state, and
`PointsVisual(position='x', color_by='v')` colors its points by speed. A group can set its own visual with
`Particle.group(..., visual=VisualSpec('points', {...}))`. A new visual subclasses `Visual` and is registered with
`register_visual`, and an element type selects it with its `visual` class variable.

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
from skcomplex.core.model import HarmonicPotential

oscillator = System(
    elements=[Particle('p', x=1.0)],
    interactions=[HarmonicPotential('trap', target='p', k=4.0)],
    space=EuclideanSpace(1),
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

**Distributions and events.** `histogram` with logarithmic bins, `ccdf`, `fit_power_law` with Clauset, Shalizi, and
Newman's cutoff, bootstrap p-value, and likelihood-ratio comparisons, `hill`, and `gini` describe heavy tails, and
`detect_events` turns series into avalanches or bursts with sizes, durations, and waiting times.

```python
from skcomplex.core.analysis import detect_events, fit_power_law, gini

samples = (1 - np.random.default_rng(0).random(5000)) ** (-1 / 1.5)
fit = fit_power_law(samples, bootstrap=20, random_state=0)
print(round(fit.alpha, 2), fit.p_value, round(gini(samples), 2))
print(detect_events([0, 2, 3, 0, 0, 1, 0]).sizes)
```

**Time series and information.** `autocorrelation`, `power_spectrum`, `mean_squared_displacement`, `hurst_exponent` by
detrended fluctuation analysis, and `early_warning` for critical slowing down describe memory and diffusion.
`symbolize`, `shannon_entropy`, `block_entropy`, `entropy_rate`, `excess_entropy`, `mutual_information`,
`transfer_entropy`, `permutation_entropy`, and `lempel_ziv_complexity` measure structure and information flow, as in
Lindgren's information theory of complex systems.

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
