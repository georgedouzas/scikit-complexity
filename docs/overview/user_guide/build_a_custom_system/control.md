# Control

Steer a system with a control input: drive it by a schedule, compute the optimal control for an objective, or close the
loop with a feedback controller. A control input is an adaptive parameter with bounds, annotated with
`ParameterSpec(adaptive=True, bounds=(lower, upper))`.

```python
from typing import Annotated

from skcomplex.core.components.elements import Element
from skcomplex.core.components.laws import RateLaw
from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.errors import ParameterError
from skcomplex.core.specs import ParameterSpec, StateSpec


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


try:
    simulate(System(elements=[HeaterElement('heater', power=2.0)]), until=3)
except ParameterError as error:
    message = str(error)
message
room = System(elements=[HeaterElement('heater')])
schedule = {'heater.power': lambda t: 1.0 if t >= 1.0 else 0.0}
heated = simulate(room, until=3, dt=0.5, schedule=schedule)
heated.values('heater.power').tolist()
```

Laws read a control input like a parameter, and trajectories record it like a state. A schedule, an optimal control
solver, or a feedback controller supplies its value. Without one it keeps its initial value. `simulate` raises a
`ParameterError` for an initial value outside the bounds, at the parameter, and the fix in the message names the bounds. Above, a
power of 2 is outside the bounds `(0, 1)`.

## Objectives and optimal control

Find the control that minimises or maximises a cost over a run. An objective defines a control problem, not a part of a
system, so the optimal control solvers take it as arguments.

```python
from typing import Annotated

from skcomplex.core.components.elements import Element
from skcomplex.core.components.laws import RateLaw
from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.solvers import CollocationSolver, PontryaginSolver
from skcomplex.core.specs import ParameterSpec, StateSpec


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


optimal = simulate(plant, until=1.0, dt=0.1, solver=PontryaginSolver(cost))
solution = PontryaginSolver(cost, until=1.0).prepare(optimal.problem).solution_
round(float(optimal.values('plant.u')[0]), 3), round(float(solution.cost), 4)
transcribed = simulate(plant, until=1.0, dt=0.1, solver=CollocationSolver(cost, nodes=21))
round(float(transcribed.values('plant.u')[0]), 3)
```

The objective has two parts:

- `objective` is the running cost.
- `terminal` is the terminal cost at the horizon, the time `until` of the run.

Each is a number, or a function of the namespace `s` of the states, parameters, and time of the system by dotted name,
such as `lambda s: s.plant.x**2`. The value of a run is the integral of the running cost over the run plus the terminal
cost. `sense` is `'min'` to minimise it or `'max'` to maximise it.

Two solvers optimise the objective over the bounded adaptive parameters of the system:

- `PontryaginSolver(objective)` builds the Hamiltonian `H = L + λ · f` of the running cost `L` and the rates `f`. It
  derives the costate equations `λ' = -∂H/∂x` and the stationarity condition `∂H/∂u = 0` symbolically, and projects
  the controls onto their bounds. Where `H` is linear in a control, it takes a bound. It solves the two-point problem by
  a forward-backward sweep.
- `CollocationSolver(objective, nodes=..., scheme=...)` transcribes the problem with the `'hermite-simpson'` or
  `'trapezoidal'` scheme, and solves it with SciPy's `minimize`, with the Jacobians derived symbolically. Collocation
  handles bang-bang controls, singular arcs, and saddle paths on which the sweep diverges.

Both solve the problem over the whole run when it starts. A solver prepared on the `problem` of a trajectory, with the
same `until`, holds in its `solution_` the solution on its grid, its cost, and, for `PontryaginSolver`, the costates.

Every solver declares what it supports. The optimal control solvers declare continuous time, no space, rate and
equation laws, real and angle states, and elements. `PontryaginSolver` also applies boundary laws after every step of
its sweep, and `CollocationSolver` keeps the states of nonnegative domains nonnegative instead.

| Solver | Laws | Method |
| --- | --- | --- |
| `PontryaginSolver` | rate, equation, boundary | symbolic costates and a forward-backward sweep |
| `CollocationSolver` | rate, equation | direct transcription solved by SLSQP or trust-constr |

`simulate` raises a `SolverError` for each of these:

- a system with noise, jumps, or updates run with an optimal control solver,
- a solver without an objective,
- a system under discrete time, since an objective needs static or continuous time.

```python continuation
from skcomplex.core.errors import SolverError

try:
    simulate(plant, until=1.0, solver=PontryaginSolver())
except SolverError as error:
    message = str(error)
message
```

## Feedback controllers

Close the loop with a controller of your own. The core holds no feedback controller, because a controller is an
`Interaction` you write:

- Its control input is an adaptive parameter of the plant, and it sets that parameter with a `BoundaryLaw`.
- A proportional-integral-derivative controller sets it to `kp e + ki ∫ e dt - kd y'`, from the error
  `e = setpoint - y` of a measured state. It adds a `RateLaw` for the integral of the error on an element of its own.
- A state feedback controller sets it to `-K x`.

A boundary law sets the control after every solver step. So a fixed-step solver with small steps makes the controller
close to a continuous one. `PidInvertedPendulumSystem` and `LqrOscillatorSystem` are built this way.

## Linear analysis

Linearise a system near an operating point to check that it can be controlled and observed, and to design a
linear-quadratic regulator.

```python
from typing import Annotated

from skcomplex.core.analysis import linearization
from skcomplex.core.components.elements import Element
from skcomplex.core.components.laws import RateLaw
from skcomplex.core.components.system import System
from skcomplex.core.specs import ParameterSpec, StateSpec


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


cart = System(elements=[CartElement('cart')])
linear = linearization(cart, outputs=['cart.x'], state_weights=[[1, 0], [0, 1]], control_weights=[[1]])
linear.controllability, linear.observability
linear.gain.round(3).tolist()
```

`linearization(system, controls=, outputs=, at=)` finds the state-space matrices `A`, `B`, `C`, and `D` of a system
under continuous time near an operating point, from its symbolic Jacobians. The controls are named by their adaptive
parameters. Its `controllability` and `observability` are the ranks of the Kalman matrices. With the weights
`state_weights` and `control_weights`, its `gain` is the gain of the linear-quadratic regulator, from SciPy's solution
of the Riccati equation.

The catalog systems with optimal controls are `OptimalHarvestingSystem` in resources, `OptimalConsumptionSystem` in
macroeconomics, `SIRVaccinationSystem` in epidemiology, and `OptimalChemotherapySystem` and `OptimalImmunotherapySystem`
in oncology.
