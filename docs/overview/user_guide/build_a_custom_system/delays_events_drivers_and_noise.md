# Delays, events, drivers, and noise

This page makes laws read the past, change states at discrete events, follow inputs that switch in time, and take
noise in the convention of your field. Each section holds the values in an
[`Element`](../core_components/elements.md#element) and writes the laws in an
[`Interaction`](../core_components/interactions.md#interaction), whose laws read every state as a function of the time
`t`.

## Delays

Here a law reads a state as it was some time ago, as in a feedback that acts with a lag. `x(t=t - tau)`, written in
Sage, is the state `x` at `t - tau`. The Mackey and Glass model of blood cell
production feeds back the cell count of 17 days before through a [`RateLaw`](../core_components/laws.md#ratelaw).

```python
from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import RateLaw
from skcomplex.core.specs import Role
from skcomplex.core.systems import System


class MackeyGlass(Interaction):
    def __init__(self, label, *, target: Role, beta=0.2, gamma=0.1, n=10.0, tau=17.0):
        super().__init__(label)
        self.target = target
        self.beta = beta
        self.gamma = gamma
        self.n = n
        self.tau = tau

    def _prepare(self, s):
        super()._prepare(s)
        x = s.target.x
        past = x(t=s.t - s.tau)
        self.laws_ = [RateLaw(x, s.beta * past / (1 + past**s.n) - s.gamma * x)]


cells = System(elements=[Element('cells', values={'x': 0.5})], interactions=[MackeyGlass('feedback', target='cells')])
round(float(cells.prepare().simulate(until=300, dt=0.5).get_values('cells.x')[-1]), 3)
```

The count is 1.004 at the day 300. With this delay the model is chaotic, so the value depends on the history before the
start.

- The delay is a constant or depends on the state.
- The solvers keep a history of dense output and step no further than the smallest delay, by the method of steps.
- Before the start, the history is the initial value.

## Events

Here a law changes states at the moment a condition is met, such as a ball that bounces off the floor. A
[`EventLaw`](../core_components/laws.md#eventlaw), `EventLaw(guard, reset, direction, terminal, label='')`, fires when the
guard crosses zero. The ball falls from the height 1 and keeps 0.8 of its speed `h'` at each impact.

```python
from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import EventLaw, RateLaw
from skcomplex.core.systems import System


def bouncing(s):
    h, g, e = s.target.h, 9.81, 0.8
    speed = h.diff(s.t)
    return [RateLaw(h.diff(s.t, 2), -g), EventLaw(h, {speed: -e * speed}, direction=-1, label='impact')]


fall = Interaction('fall', roles={'target': 'ball'}, laws=bouncing)
ball = System(elements=[Element('ball', values={'h': 1.0})], interactions=[fall])
trajectory = ball.prepare().simulate(until=3, dt=0.01)
[round(event.time, 6) for event in trajectory.events]
```

The first impact comes at 0.451524, the time `sqrt(2 / 9.81)` of a fall from the height 1. The later impacts come at
1.173961, 1.751912, 2.214272, 2.58416, and 2.880071, closer together as the bounces shrink.

- The guard fires upward for `direction=1`, downward for `-1`, and either way for `0`.
- The solver locates the crossing on its interpolant to the precision of the root finder. It applies the reset to every
  named state at once, and restarts from there.
- A terminal event ends the run at the crossing.
- Every frame carries the events since the previous one. The trajectory keeps them all, with their times and the states
  before and after the reset.
- A guard that fires again right after its reset raises an error that names the chattering event.

## Drivers

Here a system follows an input that switches or bends in time, such as a step, a ramp, or measured data. `step`,
`ramp`, `pulse`, `pulse_train`, and `series` from `skcomplex.core.components.laws.utils` are inputs of the time `s.t`.
A first-order filter receives a unit step at the time 5.

```python
from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import RateLaw
from skcomplex.core.components.laws.utils import step
from skcomplex.core.systems import System


def respond(s):
    y = s.target.y
    return [RateLaw(y, -s.target.k * y + step(s.t, 5.0))]


lowpass = Element('filter', values={'y': 0.0, 'k': 1.0})
system = System(elements=[lowpass], interactions=[Interaction('response', roles={'target': 'filter'}, laws=respond)])
filtered = system.prepare().simulate(until=10, dt=0.5)
round(float(filtered.get_values('filter.y')[-1]), 4)
```

The output is 0.9933 at the time 10, which is `1 - exp(-5)`, the response five time units after the step.

`series` interpolates data with `'linear'` or `'previous'` interpolation. The compiler finds the times where the inputs
jump or bend, including the periodic edges of pulse trains. The solvers stop at every one and evaluate the inputs on the
correct side of it, so a step response is exact to the solver tolerance.

## Noise conventions

Here a noise law is read in the convention of your field. A convention fixes where in a step the amplitude of the
noise is evaluated: at the start for Itô, at the midpoint for Stratonovich, and at the end for anti-Itô. A
[`NoiseLaw`](../core_components/laws.md#noiselaw) with `convention='stratonovich'` or `'anti-ito'` declares how it is read. The
example grows an asset at the rate 0.5 with a noise of amplitude `0.8 x`.

```python
from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import NoiseLaw, RateLaw
from skcomplex.core.engine.solvers import MilsteinSolver
from skcomplex.core.systems import System


def grow(s):
    x = s.target.x
    return [RateLaw(x, s.target.mu * x), NoiseLaw(x, s.target.sigma * x, convention='stratonovich')]


asset = Element('asset', values={'x': 1.0, 'mu': 0.5, 'sigma': 0.8})
growing = Interaction('growth', roles={'target': 'asset'}, laws=grow)
growth = System(elements=[asset], interactions=[growing], solver=MilsteinSolver(dt=0.01))
round(float(growth.prepare().simulate(until=1, random_state=0).states[-1][0]), 3)
```

The asset is worth 4.235 after one time unit on this draw. The compiler adds the drift correction `α g ∂g/∂x`, with
`α = 1/2` for Stratonovich and `1` for anti-Itô. So every SDE [solver](../glossary.md#solver) integrates the equivalent
Itô equation.

- `EulerMaruyamaSolver` has strong order 1/2.
- `MilsteinSolver`, in Platen's derivative-free form, has strong order 1 for diagonal noise.
- `HeunSolver` averages the drift over every step.

## Shared and correlated noise

Here several states share one source of noise, or sources that move together, such as the prices of two related
assets. `NoiseLaw(..., process=...)` names the Wiener process a law draws. The interaction below takes two processes
with the correlation 0.8 as an argument.

```python
import numpy as np

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction, WienerProcessSpec
from skcomplex.core.components.laws import NoiseLaw
from skcomplex.core.engine.solvers import EulerMaruyamaSolver
from skcomplex.core.specs import Role
from skcomplex.core.systems import System


class CorrelatedNoise(Interaction):
    def __init__(self, label, *, first: Role, second: Role, noise: WienerProcessSpec):
        super().__init__(label)
        self.first = first
        self.second = second
        self.noise = noise

    def _prepare(self, s):
        super()._prepare(s)
        first, second = s.noise.dW[0], s.noise.dW[1]
        self.laws_ = [NoiseLaw(s.first.n, 1.0, process=first), NoiseLaw(s.second.n, 1.0, process=second)]


processes = WienerProcessSpec('market', dimension=2, covariance=[[1.0, 0.8], [0.8, 1.0]])
market = System(
    elements=[Element('a', values={'n': 0.0}), Element('b', values={'n': 0.0})],
    interactions=[CorrelatedNoise('drive', first='a', second='b', noise=processes)],
    solver=EulerMaruyamaSolver(dt=1.0),
)
trajectory = market.prepare().simulate(until=2000, dt=1.0, random_state=0)
round(float(np.corrcoef(np.diff(trajectory.get_values('a.n')), np.diff(trajectory.get_values('b.n')))[0, 1]), 1)
```

The measured correlation of the increments is the 0.8 of the covariance.

- Without a process, every element and every group member takes its own independent process.
- Laws of one element, or of one member, that name the same label share one increment. So a reaction can drive every
  species it changes with one process.
- A `WienerProcessSpec(label, dimension, covariance)` from `skcomplex.core.components.interactions` declares
  processes whose increments over a step `h` have the covariance `h C`.
- A `WienerProcessSpec` is no element. An interaction takes it as an argument, and its noise laws name process `k`
  with `s.<argument>.dW[k]`, which every element, and every member of a group, can share.
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

Here thermal noise enters a field without changing its total, as in System B and active field theories.
`NoiseLaw(target, value, conserved=True)` on a state of a field on a lattice adds the divergence of a random flux,
`div(value eta)`. The field is an [`Element`](../core_components/elements.md#element) on a
[`LatticeSpace`](../core_components/spaces.md#latticespace) of 16 by 16 sites.

```python
from sage.all__sagemath_symbolics import sqrt

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import NoiseLaw, RateLaw
from skcomplex.core.components.laws.utils import laplacian
from skcomplex.core.components.spaces import LatticeSpace
from skcomplex.core.engine.solvers import EulerMaruyamaSolver
from skcomplex.core.systems import System


def diffuse(s):
    rho, D, T = s.target.rho, 1.0, 0.1
    return [RateLaw(rho, D * laplacian(rho)), NoiseLaw(rho, sqrt(2 * T), conserved=True)]


gas = Element('gas', values={'rho': 1.0}, space=LatticeSpace((16, 16)))
diffusion = Interaction('diffusion', roles={'target': 'gas'}, laws=diffuse)
system = System(elements=[gas], interactions=[diffusion], solver=EulerMaruyamaSolver(dt=0.01))
density = system.prepare().simulate(until=5, random_state=0).get_values('gas.rho')
round(float(density[-1].sum()), 6), bool(density[-1].std() > 0)
```

The block returns `(256.0, True)`. The density fluctuates, and its total stays at 256, one per site.

- Every link between neighbouring sites draws one Wiener process. Site `i` gains `value / h` times the increment of the
  link from its previous neighbour and loses the same amount on the link to its next one, along every dimension of
  spacing `h`.
- The total of the field then changes only by its rate laws, exactly to rounding.
- The links that leave a lattice through a zero-flux or fixed edge carry no process.
- The value is a number or an expression of parameters, the same on every link, and the field uses finite differences
  on a square lattice.
- Validation reports a conserved noise law on any other state with the code `noise-unsupported`.

## Which solvers support what

This section matches a solver to the noise of a system. Solvers and backends declare the noise structures they draw.

- `'diagonal'` is one process per state.
- `'shared'` is a process that drives several states.
- `'correlated'` is the processes of a `WienerProcessSpec` with a covariance.
- `'conserved'` is the link processes of conserved noise.

`EulerMaruyamaSolver`, `HeunSolver`, and the NumPy and Numba backends support all four. `MilsteinSolver`, whose
derivative-free correction holds for diagonal noise only, and `LammpsSolver` support diagonal noise. Validation
reports a system with shared, correlated, or conserved noise for them before compiling it. The [solvers page](../../../components/solvers.md)
lists their supported tags.

The solvers of SciPy, RK4, Euler-Maruyama, Milstein, and Heun integrate delays, events, and drivers. Velocity Verlet,
Gillespie, and map iteration reject delays and events, with an error that names the solvers to use.

## Ready-made systems with delays and events

This section lists ready-made systems to start from. Each uses delays, events, drivers, or noise, and each is a
class of the subfield of `skcomplex.systems` in brackets.

- Delays, events, and drivers: `MackeyGlassSystem` (`biology.physiology`), `HutchinsonSystem` (`biology.ecology`),
  `IntegrateAndFireSystem` and `IzhikevichSystem` (`biology.neuroscience`), `DelayedKuramotoSystem`
  (`physics.synchronization`), `StochasticResonanceSystem` and `BrownianNearWallSystem` (`physics.soft_matter`), and
  `BouncingBallSystem`, `NewtonsCradleSystem`, and `CircularBilliardSystem` (`physics.mechanics`).
- Noise: `BindingInteraction` and `DissociationInteraction` take `langevin=True` for chemical Langevin equations, as in
  `ChemicalLangevinNetworkSystem` (`chemistry.kinetics`). `CorrelatedGeometricBrownianMotionSystem`
  (`economics.finance`) drives two assets with correlated processes, and `ActiveSystemBSystem`
  (`physics.active_matter`) and `CahnHilliardSystem` (`physics.pattern_formation`) with a `temperature` take conserved
  noise.

```python
from skcomplex.systems.physics.mechanics import BouncingBallSystem

len(BouncingBallSystem().prepare().simulate().events) > 0
```

The bouncing ball of the catalog records its impacts as events, so the block returns `True`.
