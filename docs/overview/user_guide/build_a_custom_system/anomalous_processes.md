# Anomalous processes

Write processes with memory: heavy-tailed jumps, events that fire after waiting times from any distribution, noise
whose increments are correlated over the whole run, and derivatives of fractional order.

## Draws from SciPy distributions

Draw from any distribution of `scipy.stats`, such as a Lévy stable law for long flights.
`draw(name, distribution, **parameters)` takes the name of the distribution, such as `'pareto'`, `'lomax'`, `'expon'`,
`'beta'`, or `'levy_stable'`, with its shape parameters, `loc`, and `scale` as keywords.

```python
from typing import Annotated

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.laws import UpdateLaw
from skcomplex.core.components.laws.utils import draw
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.specs import StateSpec


class FlierElement(Element):
    def __init__(self, label, *, x: Annotated[float, StateSpec()] = 0.0):
        super().__init__(label)
        self.x = x

    def _get_laws(self, s):
        return [UpdateLaw(s.x, s.x + draw('jump', 'levy_stable', alpha=1.5, beta=0.0))]


fliers = CompositeSystem(elements=[GroupElement('fliers', element=FlierElement, size=100)])
flight = simulate(fliers, until=20, dt=1.0, random_state=0)
flight.get_values('fliers.x').shape
```

- The keywords can be expressions of states and parameters.
- A SciPy draw follows the rules of the other named draws, and it samples with the distribution's `rvs` and the
  random generator of the run.
- Compiling rejects a name `scipy.stats` does not provide, listing the closest names, and keywords the distribution does
  not take.
- In update laws, a SciPy draw is sampled for every member before each step, and the map then steps in Python with the
  backend's compiled update.

## Jumps on clocks of waiting times

Fire an event after a random waiting time, such as a trapped particle that escapes after a heavy-tailed wait.
`JumpLaw(waiting=..., reset=..., label=...)` fires after a waiting time, applies its reset to the states all at once,
and draws the next waiting time from the states after the reset.

```python
from typing import Annotated

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.laws import JumpLaw
from skcomplex.core.components.laws.utils import draw, normal
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.specs import ParameterSpec, StateSpec


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


walkers = CompositeSystem(elements=[GroupElement('walkers', element=WalkerElement, size=1000)])
walk = simulate(walkers, until=1000, dt=100.0, random_state=0)
walk.problem.form.value, bool(walk.get_values('jumps.renewals')[-1] > 1000)
```

- Every member of a group renews on its own clock.
- Draws in the waiting time and the reset take a fresh value at every renewal of every member, and every use of one
  name in a renewal reads the same value.
- A waiting time that is a single draw must come from a distribution with nonnegative support. Compiling rejects one
  that can be negative, such as `normal`.

A system or partition whose only laws are such jumps compiles to the renewal form. `RenewalSolver` runs it exactly, from
renewal to renewal with no time grid, and a frame holds the states the last renewal at or before its time set.

These jumps also combine with the rate and noise laws of the same element, which evolve the states between renewals.
`RK4Solver` and `EulerMaruyamaSolver` stop every member at its own renewals when the members do not interact. When a
law sums over the group, they split the step at every renewal time for all members. Other steppers reject these jumps.

The trajectory records every renewal pass as a `RenewalRecord` among its events, with the time and the member of every
renewal and the member's states before and after it. The observable `<label>.renewals` counts the renewals up to every
frame. The Rerun observer streams it as a time series, and the renewals as text entries under `renewals/<label>`. The
compiler keeps the count of every member in a hidden state, `<group>.<label>~renewals`.

## Fractional noise

Drive a state with noise that remembers its past, as in rough or persistent paths.
`NoiseLaw(targets, value, hurst=hurst)` drives a state with fractional Brownian motion of Hurst index
`0 < H < 1`.

```python
from typing import Annotated

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.laws import NoiseLaw
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.solvers import FractionalEulerMaruyamaSolver
from skcomplex.core.specs import StateSpec


class RoughElement(Element):
    def __init__(self, label, *, x: Annotated[float, StateSpec()] = 0.0):
        super().__init__(label)
        self.x = x

    def _get_laws(self, s):
        return [NoiseLaw(s.x, 1, hurst=0.3)]


walkers = CompositeSystem(elements=[GroupElement('walkers', element=RoughElement, size=500)])
rough = simulate(walkers, until=1, dt=0.1, solver=FractionalEulerMaruyamaSolver(dt=0.01), random_state=0)
round(float(rough.get_values('walkers.x')[-1].var()), 1)
```

- The increments anticorrelate for `H < 1/2` and correlate for `H > 1/2`.
- A Hurst index of 1/2 compiles to a noise law without one. For other indices the amplitude must not depend on any state.
- `FractionalEulerMaruyamaSolver(dt, horizon)` is the stepper of systems with fractional noise. When it first advances,
  it draws the increments of the whole run on its grid, with the circulant embedding of Davies and Harte over
  `scipy.fft`. So frames must fall on multiples of `dt`.
- `simulate` passes the end of the run as the horizon, and advancing beyond it raises `HorizonError`.
- Other SDE steppers reject fractional noise with an error that names `FractionalEulerMaruyamaSolver`.
- `fractional_gaussian_noise` returns the noise paths themselves.

These ready-made systems have memory: `ContinuousTimeRandomWalkSystem`, `LevyWalkSystem`,
`AnnealedTransientTimeMotionSystem`, `FractionalBrownianMotionSystem`, and `LevyFlightSystem` of
`skcomplex.systems.physics.soft_matter`, and `TrapSystem` of `skcomplex.systems.physics.statistical_mechanics`.

## Fractional derivatives

Give a state a derivative of fractional order, for relaxation that is slower than any exponential. A state declared
with a fractional order, `StateSpec(order=0.6)`, has a Caputo derivative of that order.

```python
from typing import Annotated

from skcomplex.core.analysis import symbolic
from skcomplex.core.components.elements import Element
from skcomplex.core.components.laws import RateLaw
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.solvers import L1Solver
from skcomplex.core.specs import StateSpec


class RelaxingElement(Element):
    def __init__(self, label, *, x: Annotated[float, StateSpec(order=0.6)] = 1.0):
        super().__init__(label)
        self.x = x

    def _get_laws(self, s):
        return [RateLaw(s.x, -s.x)]


system = CompositeSystem(elements=[RelaxingElement('a')])
relaxing = simulate(system, until=20, dt=1.0, solver=L1Solver(dt=0.01, memory=10.0))
solution = symbolic(system, kind='solve', use_initial_state=True).solutions['a.x'].rhs()
exact = solution.subs({solution.variables()[0]: 20.0})
relaxing.problem.form.value, abs(float(relaxing.get_values('a.x')[-1]) - float(exact)) < 1e-3
```

The simulated relaxation matches the exact Mittag-Leffler solution to `1e-3` at time 20.

The rate laws on a fractional state give its derivative, `D^alpha x = f(t, x)`, and superpose as for any state, so no
other law kind is needed. Order 1 is the ordinary rate law. The derivative of order `alpha` at a time depends on the
whole past of the state, so the solution relaxes like a Mittag-Leffler function, slower than any exponential.

A system with a state of a fractional order compiles to the fractional form, with the order of every state component
in `problem.orders`. The states of order 1 of the same system keep their ordinary equations. Three steppers integrate
the fractional form on a fixed grid from the start of the run, so frames must fall on multiples of their `dt`.

- `GrunwaldLetnikovSolver(dt, memory=None)` weighs the past states with the binomial weights
  `(-1)^j binomial(alpha, j)`.
- `L1Solver(dt, memory=None)` interpolates the state linearly between grid points inside the Caputo integral.
- `FractionalABMSolver(dt)` is the predictor-corrector of Diethelm, Ford, and Freed, of order `1 + alpha` for smooth
  solutions, and the default.

The two explicit schemes are of order 1. All three keep the whole history, so a run of `n` steps costs `n^2`
operations. `memory` bounds that cost. The history of the last `memory` time units enters with its exact weights, and
the older history enters through about 40 exponential modes fitted to the weights, so every step costs the same. For
fractional relaxation of order 0.6, a window of 10 time units stays within `1e-8` of the whole history at time 20. A
change of a fractional state between frames, such as one a `schedule` makes, starts a new history from
the changed state.

`simulate` reports the misuses of fractional orders before compiling, each with the path of the state and a fix.

- `fractional-order`: an order that is neither an integer nor in `(0, 1)`.
- `fractional-time`: a fractional state in a system whose declared time is not continuous.
- `order-unsupported`: a chosen solver or backend that does not integrate fractional orders.
- `solver-unsupported`: a law on a fractional state that no fractional solver runs, such as a noise law.

Solvers and backends declare the kinds of order they integrate in the `orders` field of their `SupportSpec`,
`'integer'` or `'fractional'`. The NumPy and Numba backends run the fractional form.

`symbolic(system, kind='solve')` solves one linear fractional relaxation `D^alpha x = c x` as
`x(0) E_alpha(c t^alpha)`. It writes the solution with the symbolic function `mittag_leffler`, which evaluates
numerically. It sums the power series in arbitrary precision with Sage for moderate arguments, and takes the asymptotic
expansion for large ones.
`symbolic(system)` writes a Caputo derivative as `caputo(x(t), t, alpha)`.

The order belongs to the element type, so a sweep over orders builds one system per order. These ready-made systems of
`skcomplex.systems.mathematics.fractional_dynamics` take the order as a model parameter: `FractionalRelaxationSystem`,
`FractionalOscillatorSystem`, `FractionalVanDerPolSystem`, and `FractionalLorenzSystem`.
