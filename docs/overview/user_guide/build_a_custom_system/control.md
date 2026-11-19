# Control

This page steers a system with a control input. It drives the input by a schedule, computes the optimal control for an
objective, or closes the loop with a feedback controller. A control input is a value of an element that no law
changes but that a schedule, a solver, or a controller sets during a run. Its bounds are its allowed values, written in
`Annotated`, such as `power: Annotated[float, RealSet.closed(0, 1)]`.

## Control inputs

This section declares a control input and drives it by a schedule. A heater warms a room at the rate `power - T`, and
its power is a control input between 0 and 1. The interaction `warming` sets a
[`RateLaw`](../core_components/laws.md#ratelaw) that reads the power like any parameter. `prepare` raises a
`ValidationError` for an initial value outside the bounds.

```python
from typing import Annotated

from sage.sets.real_set import RealSet

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import RateLaw
from skcomplex.core.errors import ValidationError
from skcomplex.core.systems import System


class Heater(Element):
    def __init__(self, label, T=0.0, power: Annotated[float, RealSet.closed(0, 1)] = 0.0):
        super().__init__(label)
        self.T = T
        self.power = power


def warm(s):
    return [RateLaw(s.target.T, s.target.power - s.target.T)]


warming = Interaction('warming', roles={'target': 'heater'}, laws=warm)
try:
    System(elements=[Heater('heater', power=2.0)], interactions=[warming]).prepare()
except ValidationError as error:
    message = str(error)
message
```

The message reads "heater.power = 2.0 is not in its domain [0, 1]", and the fix is an initial value inside the
bounds. A schedule supplies the value of a control input at every time. Here it switches
the heater on at time 1.

```python continuation
room = System(elements=[Heater('heater')], interactions=[warming]).prepare()
schedule = {'heater.power': lambda t: 1.0 if t >= 1.0 else 0.0}
heated = room.simulate(until=3, dt=0.5, schedule=schedule)
heated.get_values('heater.power').tolist()
```

The recorded power is 0.0 at the times 0 and 0.5 and 1.0 from time 1 on. Laws read a control input like a parameter,
and trajectories record it like a state. A schedule, an optimal control solver, or a feedback controller supplies its
value. Without one it keeps its initial value.

## Objectives and optimal control

This section finds the control that minimises a cost over a run. The cost is an objective law that names the values it
steers, `ObjectiveLaw(cost, controls=[s.plant.u])`, and the bounds of a control are the allowed values of its element.
The plant `x' = u` starts at `x = 1`, and the cost integrates `(x^2 + u^2) / 2` over one time unit, so the control must
pull the state to zero without spending too much effort. `PontryaginSolver` solves the problem by the maximum
principle of Pontryagin.

```python
from typing import Annotated

from sage.sets.real_set import RealSet

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import ObjectiveLaw, RateLaw
from skcomplex.core.engine.solvers import CollocationSolver, PontryaginSolver
from skcomplex.core.systems import System


class PlantElement(Element):
    def __init__(self, label, x=1.0, u: Annotated[float, RealSet.closed(-10, 10)] = 0.0):
        super().__init__(label)
        self.x = x
        self.u = u


def regulate(s):
    plant = s.target
    return [RateLaw(plant.x.diff(s.t), plant.u), ObjectiveLaw((plant.x**2 + plant.u**2) / 2, controls=[plant.u])]


regulator = Interaction('regulator', roles={'target': 'plant'}, laws=regulate)
plant = System(elements=[PlantElement('plant')], interactions=[regulator], solver=PontryaginSolver())
optimal = plant.prepare().simulate(until=1.0, dt=0.1)
solution = PontryaginSolver(until=1.0).prepare(optimal.problem).solution_
round(float(optimal.get_values('plant.u')[0]), 3), round(float(solution.cost), 4)
```

The optimal control starts at -0.762, and the cost of the run is 0.3808. Both match the exact solution of this problem,
`u(0) = -tanh(1)` and a cost of `tanh(1) / 2`. `CollocationSolver`, the solver a system with controls takes by
default, solves the same problem by direct transcription.

```python continuation
transcribed = plant.set_params(solver=CollocationSolver(nodes=21)).prepare().simulate(until=1.0, dt=0.1)
round(float(transcribed.get_values('plant.u')[0]), 3)
```

Collocation on 21 nodes gives the initial control -0.761, within 0.001 of the sweep. The objective law has two parts.

- Its value is the running cost, integrated over the run.
- `terminal` is the terminal cost at the horizon, the time `until` of the run, such as the value of the capital an
  economy leaves, `ObjectiveLaw(-utility, controls=[s.target.c], terminal=-value)`.

The run minimises the integral of the running cost plus the terminal cost. A cost to maximise, such as a profit, is
minimised with its sign changed.

Two solvers optimise the objective over the controls of the system.

- `PontryaginSolver()` builds the Hamiltonian `H = L + λ · f` of the running cost `L` and the rates `f`. It derives
  the costate equations `λ' = -∂H/∂x` and the stationarity condition `∂H/∂u = 0` symbolically, and projects the
  controls onto their bounds. Where `H` is linear in a control, it takes a bound. It solves the two-point problem by a
  forward-backward sweep.
- `CollocationSolver(nodes=..., scheme=...)` transcribes the problem with the `'hermite-simpson'` or `'trapezoidal'`
  scheme, and solves it with the `minimize` of SciPy, with the Jacobians derived symbolically. Collocation solves
  bang-bang controls, singular arcs, and saddle paths on which the sweep diverges.

Both solve the problem over the whole run when it starts. A solver prepared on the `problem` of a trajectory, with the
same `until`, holds in its `solution_` the solution on its grid, its cost, and, for `PontryaginSolver`, the costates.

Every solver declares what it supports. The optimal control solvers declare continuous time, no space, rate,
constraint, and objective laws, real and angle states, and elements. `PontryaginSolver` also applies boundary laws
after every step of its sweep, and `CollocationSolver` keeps the states of nonnegative domains nonnegative instead.

| Solver | Laws | Method |
| --- | --- | --- |
| `PontryaginSolver` | rate, constraint, objective, boundary | symbolic costates and a forward-backward sweep |
| `CollocationSolver` | rate, constraint, objective | direct transcription solved by SLSQP or trust-constr |

`simulate` raises a `SolverError` for each of these cases.

- A system with noise, jumps, or updates runs with an optimal control solver.
- A system has no objective law with controls.
- A system runs under discrete time, since an objective needs static or continuous time.

```python continuation
from skcomplex.core.errors import SolverError

drift = Interaction('drift', roles={'target': 'plant'}, laws=lambda s: [RateLaw(s.target.x.diff(s.t), s.target.u)])
try:
    System(elements=[PlantElement('plant')], interactions=[drift], solver=PontryaginSolver()).prepare().simulate(until=1.0)
except SolverError as error:
    message = str(error)
message
```

The message reads "PontryaginSolver has no objective", and the fix is to state the cost with an objective law that
names its controls, such as `ObjectiveLaw(s.plant.x**2 + s.plant.u**2, controls=[s.plant.u])`.

## Feedback controllers

This section closes the loop with a controller of your own. The core holds no feedback controller, because a
controller is an [`Interaction`](../core_components/interactions.md#interaction) that you write.

- Its control input is a plain value of the plant, such as the `force` of
  `Element('cart', values={'x': 0.0, 'force': 0.0})`, and it sets that value with an
  [`UpdateLaw`](../core_components/laws.md#updatelaw) beside the rate laws, such as `UpdateLaw(s.plant.force, -K * x)`.
  A value that such an update sets after each solver step is a state that holds within the step, so the plant declares
  nothing.
- A proportional-integral-derivative controller sets it to `kp e + ki ∫ e dt - kd y'`, from the error
  `e = setpoint - y` of a measured state. It adds a `RateLaw` for the integral of the error on an element of its own.
- A state feedback controller sets it to `-K x`.

The update sets the control after every solver step. So a fixed-step solver with small steps makes the controller
close to a continuous one. `PidInvertedPendulumSystem` and `LqrOscillatorSystem` of the catalog are built this way.

## Linear analysis

This section linearises a system near an operating point, checks that it can be controlled and observed, and designs a
linear-quadratic regulator. A system is controllable when its inputs can steer every state, and observable when its
outputs reveal every state. The analysis `linearization(system, controls=..., outputs=..., at=...)` finds the
state-space matrices `A`, `B`, `C`, and `D` of a system under continuous time near an operating point, from its
symbolic Jacobians. The controls are named by their dotted names. They are parameters that no law of the plant changes,
such as the force `cart.force` below, or adaptive parameters. A cart on a rail, pushed by a force, serves as the system,
and its position is the output.

```python
from types import SimpleNamespace

from skcomplex.core.analysis import linearization
from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import RateLaw
from skcomplex.core.types import Role
from skcomplex.core.systems import System


class Push(Interaction):
    def __init__(self, label, *, target: Role):
        super().__init__(label)
        self.target = target

    def _prepare(self, context=None):
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            cart, t = context.target, context.t
            self.laws_ = [RateLaw(cart.x.diff(t, 2), cart.force)]


cart = System(elements=[Element('cart', values={'x': 0.0, 'force': 0.0})], interactions=[Push('push', target='cart')])
linear = linearization(
    cart,
    controls=['cart.force'],
    outputs=['cart.x'],
    state_weights=[[1, 0], [0, 1]],
    control_weights=[[1]],
)
linear.states, linear.controllability, linear.observability
```

The controllability and the observability are both 2, the ranks of the Kalman matrices, so the force steers both
states and the position reveals both. With the weights `state_weights` and `control_weights`, the result holds the
gain of the linear-quadratic regulator, from the solution of the Riccati equation by SciPy.

```python continuation
linear.gain.round(3).tolist()
```

The gain is `[[1.0, 1.732]]`, which is `[1, sqrt(3)]`, so the regulator pushes with the force `-x - 1.732 x'`. The
states are `cart.x` and `cart.x'`, named with a prime for the derivative as in trajectories.

The systems of the catalog with optimal controls are `OptimalHarvestingSystem` in resources, `OptimalConsumptionSystem`
in macroeconomics, `SIRVaccinationSystem` in epidemiology, and `OptimalChemotherapySystem` and
`OptimalImmunotherapySystem` in oncology.
