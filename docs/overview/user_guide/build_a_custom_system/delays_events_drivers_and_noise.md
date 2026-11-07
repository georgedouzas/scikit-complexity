# Delays, events, drivers, and noise

Make laws read the past, change states at discrete events, follow inputs that switch in time, and take noise in the
convention of your field.

## Delays

Read a state as it was some time ago, as in a feedback that acts with a lag. `delayed(x, tau)` from
`skcomplex.core.components.laws.utils` is the state `x` at `t - tau`.

```python
from typing import Annotated

import numpy as np

from skcomplex.core.components.elements import Element
from skcomplex.core.components.laws import RateLaw
from skcomplex.core.components.laws.utils import delayed
from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.specs import ParameterSpec, StateSpec


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
round(float(Simulation(cells, dt=0.5).prepare().run(until=300).trajectory_.values('cells.x')[-1]), 3)
```

- The delay is a constant or depends on the state.
- The solvers keep a history of dense output and step no further than the smallest delay, by the method of steps.
- Before the start, the history is the initial value, or what the state declares with `StateSpec(history=...)`, a
  number or a function of time.

## Events

Change states at the moment a condition is met, such as a ball that bounces off the floor.
`CrossingLaw(guard, reset, direction, terminal, label='')` fires when the guard crosses zero.

```python
from typing import Annotated

from skcomplex.core.components.elements import Element
from skcomplex.core.components.laws import CrossingLaw, RateLaw
from skcomplex.core.components.laws.utils import derivative
from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.specs import ParameterSpec, StateSpec


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


trajectory = Simulation(System(elements=[BallElement('ball')]), dt=0.01).prepare().run(until=3).trajectory_
[round(event.time, 6) for event in trajectory.events]
```

- The guard fires upward for `direction=1`, downward for `-1`, and either way for `0`.
- The solver locates the crossing on its interpolant to the precision of the root finder. It applies the reset to every
  named state at once, and restarts from there.
- A terminal event ends the run at the crossing.
- Every frame carries the events since the previous one. The trajectory keeps them all, with their times and the states
  before and after the reset.
- A guard that fires again right after its reset raises an error that names the chattering event.

## Drivers

Feed a system an input that switches or bends in time, such as a step, a ramp, or measured data. `step`, `ramp`,
`pulse`, `pulse_train`, and `series` from `skcomplex.core.components.laws.utils` are inputs of the time `s.t`.

```python
from typing import Annotated

from skcomplex.core.components.elements import Element
from skcomplex.core.components.laws import RateLaw
from skcomplex.core.components.laws.utils import step
from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.specs import ParameterSpec, StateSpec


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


filtered = Simulation(System(elements=[FilterElement('filter')]), dt=0.5).prepare().run(until=10).trajectory_
round(float(filtered.values('filter.y')[-1]), 4)
```

`series` interpolates data with `'linear'` or `'previous'` interpolation. The compiler finds the times where the inputs
jump or bend, including the periodic edges of pulse trains. The solvers stop at every one and evaluate the inputs on the
correct side of it, so a step response is exact to the solver tolerance.

## Noise conventions

Read noise in the convention of your field, Itô, Stratonovich, or anti-Itô.
`NoiseLaw(..., convention='stratonovich')` or `'anti-ito'` declares how a noise law is read.

```python
from typing import Annotated

from skcomplex.core.components.elements import Element
from skcomplex.core.components.laws import NoiseLaw, RateLaw
from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.engine.solvers import MilsteinSolver
from skcomplex.core.specs import ParameterSpec, StateSpec


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
round(float(Simulation(growth, solver=MilsteinSolver(dt=0.01), random_state=0).prepare().run(until=1).state_[0]), 3)
```

The compiler adds the drift correction `α g ∂g/∂x`, with `α = 1/2` for Stratonovich and `1` for anti-Itô. So every SDE
solver integrates the equivalent Itô equation.

- `EulerMaruyamaSolver` has strong order 1/2.
- `MilsteinSolver`, in Platen's derivative-free form, has strong order 1 for diagonal noise.
- `HeunSolver` averages the drift over every step.

## Shared and correlated noise

Drive several states with one source of noise, or with sources that move together, such as the prices of two related
assets. `NoiseLaw(..., process=...)` names the Wiener process a law draws.

```python
from typing import Annotated

import numpy as np

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import NoiseLaw
from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.engine.solvers import EulerMaruyamaSolver
from skcomplex.core.specs import RoleSpec, StateSpec, WienerProcessSpec


class StockElement(Element):
    def __init__(self, label, *, n: Annotated[float, StateSpec()] = 0.0):
        super().__init__(label)
        self.n = n


class CorrelatedNoiseInteraction(Interaction):
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
    interactions=[CorrelatedNoiseInteraction('drive', first='a', second='b', noise=processes)],
)
solver = EulerMaruyamaSolver(dt=1.0)
trajectory = Simulation(market, solver=solver, dt=1.0, random_state=0).prepare().run(until=2000).trajectory_
round(float(np.corrcoef(np.diff(trajectory.values('a.n')), np.diff(trajectory.values('b.n')))[0, 1]), 1)
```

The measured correlation of the increments is the 0.8 of the covariance.

- Without a process, every element and every group member takes its own independent process.
- Laws of one element, or of one member, that name the same label share one increment. So a reaction can drive every
  species it changes with one process.
- A `WienerProcessSpec(label, dimension, covariance)` from `skcomplex.core.specs` declares processes whose increments
  over a step `h` have the covariance `h C`.
- A `WienerProcessSpec` is no element. An interaction binds it through a role annotated with `RoleSpec(noise=True)`, and
  its noise laws name process `k` with `s.<role>.dW[k]`, which every element, and every member of a group, can share.
- Interactions of one system that bind processes of one label share them and their correlations.
- The covariance is fixed when the system compiles. NumPy factors it once, so the steppers scale standard normal
  increments drawn from the run's random generator.

The Stratonovich and anti-Itô corrections of each law use the full diffusion matrix and the covariance. So a state that
several laws drive, and laws that share a process, are corrected exactly. Laws that share a process declare one
convention. Validation reports:

- laws of different conventions on one process,
- a Stratonovich or anti-Itô amplitude that reads a state another element drives through its process,
- a covariance that is not symmetric positive semidefinite,
- a handle `dW` used anywhere but as the process of a noise law.

## Conserved noise

Add thermal noise to a field without changing its total, as in System B and active field theories.
`NoiseLaw(target, value, conserved=True)` on a state of a field on a lattice adds the divergence of a random flux,
`div(value eta)`.

```python
from typing import Annotated

from sage.all__sagemath_symbolics import sqrt

from skcomplex.core.components.elements import Element, FieldElement
from skcomplex.core.components.laws import NoiseLaw, RateLaw
from skcomplex.core.components.laws.utils import laplacian
from skcomplex.core.components.spaces import LatticeSpace
from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.engine.solvers import EulerMaruyamaSolver
from skcomplex.core.specs import ParameterSpec, StateSpec


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
density = Simulation(gas, solver=solver, random_state=0).prepare().run(until=5).trajectory_.values('gas.rho')
round(float(density[-1].sum()), 6), bool(density[-1].std() > 0)
```

The density fluctuates, and its total stays at 256, one per site.

- Every link between neighbouring sites draws one Wiener process. Site `i` gains `value / h` times the increment of the
  link from its previous neighbour and loses the same amount on the link to its next one, along every dimension of
  spacing `h`.
- The total of the field then changes only by its rate laws, exactly to rounding.
- The links that leave a lattice through a zero-flux or fixed edge carry no process.
- The value is a number or an expression of parameters, the same on every link, and the field uses finite differences
  on a square lattice.
- Validation reports a conserved noise law on any other state with the code `noise-unsupported`.

## Which solvers support what

Solvers and backends declare the noise structures they draw.

- `'diagonal'` is one process per state.
- `'shared'` is a process that drives several states.
- `'correlated'` is the processes of a `WienerProcessSpec` with a covariance.
- `'conserved'` is the link processes of conserved noise.

`EulerMaruyamaSolver`, `HeunSolver`, and the NumPy and Numba backends support all four. `MilsteinSolver`, whose
derivative-free correction holds for diagonal noise only, and `LammpsSolver` support diagonal noise. Validation
reports a system with shared, correlated, or conserved noise for them before compiling it. The [solvers page](../../../parts/solvers.md)
lists their supported tags.

SciPy, RK4, Euler-Maruyama, Milstein, and Heun handle delays, events, and drivers. Velocity Verlet, Gillespie, and map
iteration reject delays and events, with an error that names the solvers to use.

## Ready-made systems with delays and events

The catalog has these systems ready.

- Delays, events, and drivers: `mackey_glass` in physiology, `hutchinson` in ecology, `integrate_and_fire` and
  `izhikevich` in neuroscience, `delayed_kuramoto` in synchronization, `stochastic_resonance` and `brownian_near_wall`
  in soft matter, and `bouncing_ball`, `newtons_cradle`, and `circular_billiard` in mechanics.
- Noise: `BindingInteraction` and `DissociationInteraction` take `langevin=True` for chemical Langevin equations, as in
  `chemical_langevin_network`. `correlated_geometric_brownian_motion` drives two assets with correlated processes, and
  `active_system_b` and `cahn_hilliard` with a `temperature` take conserved noise.
