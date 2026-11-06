# Control

Steer a system with a control input: drive it by a schedule, compute the optimal control for an objective, or close the
loop with a feedback controller. A control input is an adaptive parameter with bounds, annotated with
`ParameterSpec(adaptive=True, bounds=(lower, upper))`.

```python
from typing import Annotated

from skcomplex.core.components.elements import Element
from skcomplex.core.components.laws import RateLaw
from skcomplex.core.components.system import System
from skcomplex.core.engine.compile import PassCompiler
from skcomplex.core.engine.runtime import Simulation
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


room = System(elements=[HeaterElement('heater')])
report = PassCompiler().prepare(System(elements=[HeaterElement('heater', power=2.0)])).report_
[issue.path for issue in report.errors]
schedule = {'heater.power': lambda t: 1.0 if t >= 1.0 else 0.0}
heated = Simulation(room, dt=0.5, schedule=schedule).prepare().run(until=3).trajectory_
heated.values('heater.power').tolist()
```

Laws read a control input like a parameter, and trajectories record it like a state. A schedule, an optimal control
solver, or a feedback controller supplies its value. Without one it keeps its initial value. The `report_` of a
compiler holds an initial value outside the bounds as an error at the parameter, and the fix of the error names the
bounds. Above, a power of 2 is outside the bounds `(0, 1)`.

## Objectives and optimal control

Find the control that minimises or maximises a cost over a run. An objective defines a control problem, not a part of a
system, so the optimal control solvers take it as arguments.

```python
from typing import Annotated

from skcomplex.core.components.elements import Element
from skcomplex.core.components.laws import RateLaw
from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import Simulation
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


optimal = Simulation(plant, solver=PontryaginSolver(cost), dt=0.1).prepare().run(until=1.0)
round(float(optimal.trajectory_.values('plant.u')[0]), 3), round(float(optimal.stepper_.solution_.cost), 4)
transcribed = Simulation(plant, solver=CollocationSolver(cost, nodes=21), dt=0.1).prepare().run(until=1.0)
round(float(transcribed.trajectory_.values('plant.u')[0]), 3)
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

Both solve the problem over the whole run when it starts. `simulation.stepper_.solution_` holds the solution on its
grid, its cost, and, for `PontryaginSolver`, the costates.

Every solver declares what it supports. The optimal control solvers declare continuous time, no space, rate and
equation laws, real and angle states, and elements. `PontryaginSolver` also applies boundary laws after every step of
its sweep, and `CollocationSolver` keeps the states of nonnegative domains nonnegative instead.

| Solver | Laws | Method |
| --- | --- | --- |
| `PontryaginSolver` | rate, equation, boundary | symbolic costates and a forward-backward sweep |
| `CollocationSolver` | rate, equation | direct transcription solved by SLSQP or trust-constr |

The `report_` of the compiler holds each of these as a `SolverError`:

- a system with noise, jumps, or updates run with an optimal control solver,
- a solver without an objective,
- a system under discrete time, since an objective needs static or continuous time.

```python continuation
from skcomplex.core.engine.compile import PassCompiler

[issue.code for issue in PassCompiler(solver=PontryaginSolver()).prepare(plant).report_.errors]
```

## Feedback controllers

Close the loop with a controller of your own. The core holds no feedback controller, because a controller is an
`Interaction` you write:

- Its control input is an adaptive parameter of the plant, and it sets that parameter with a `BoundaryLaw`.
- A proportional-integral-derivative controller sets it to `kp e + ki ∫ e dt - kd y'`, from the error
  `e = setpoint - y` of a measured state. It adds a `RateLaw` for the integral of the error on an element of its own.
- A state feedback controller sets it to `-K x`.

A boundary law sets the control after every solver step. So a fixed-step solver with small steps makes the controller
close to a continuous one. The systems `pid_inverted_pendulum` and `lqr_oscillator` are built this way.

## Linear analysis

Linearise a system near an operating point to check that it can be controlled and observed, and to design a
linear-quadratic regulator.

```python
from typing import Annotated

from skcomplex.core.analysis import LinearizationAnalysis
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


linear = LinearizationAnalysis(outputs=['cart.x'], state_weights=[[1, 0], [0, 1]], control_weights=[[1]])
linear.prepare(System(elements=[CartElement('cart')])).run()
linear.controllability_, linear.observability_
linear.gain_.round(3).tolist()
```

`LinearizationAnalysis(controls, outputs, at)` finds the state-space matrices `A_`, `B_`, `C_`, and `D_` of a system
under continuous time near an operating point, from its symbolic Jacobians. The controls are named by their adaptive
parameters. Its `controllability_` and `observability_` are the ranks of the Kalman matrices. With the weights
`state_weights` and `control_weights`, its `gain_` is the gain of the linear-quadratic regulator, from SciPy's solution
of the Riccati equation.

The catalog systems with optimal controls are `optimal_harvesting` in resources, `optimal_consumption` in
macroeconomics, `sir_vaccination` in epidemiology, and `optimal_chemotherapy` and `optimal_immunotherapy` in oncology.
