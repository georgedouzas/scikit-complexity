# Design

## Context

The current package has about 800 lines in `base`, `component`, and `system`. It builds every unset parameter as a
Sage variable, solves only through Sage `desolve`, and has no numeric runtime. The constitution requires a `core`
subpackage for shared leaves, imports that run downward, one errors module, private implementation modules, and the
scikit-learn conventions for parameters and derived state. See proposal.md for the motivation and the specs for the
required behavior.

## Goals / Non-Goals

**Goals:**

- One description layer that covers the ODE, SDE, jump process, and discrete map forms without the user choosing the
  form.
- Numeric functions fast enough for interactive real-time runs of systems with tens to hundreds of state variables.
- Extension points for new law kinds, solvers, spaces, and visuals that need no change to existing modules.

**Non-Goals:**

- Element groups stored as arrays for thousands of agents or particles. This change compiles each element on its
  own, and groups come in a later change.
- Ports, connections, DAE, PDE, equilibrium, and optimization forms, and co-simulation of subsystems with different
  forms.
- Calibration (`fit`, `predict`, `score`), stability, bifurcation, and sensitivity analysis.
- Replaying a recorded trajectory into Rerun in one batch.

## Decisions

### Package layout

```
skcomplex/
  core/       errors, domains, variable declarations, registry
  model/      Element, Interaction, System, Space, law kinds, Symbols
  compile/    compile_system, Problem, Form, StateLayout
  solvers/    Stepper protocol, ODE, SDE, jump, and discrete steppers
  runtime/    Simulation, clocks, Observer, Frame, Recorder, Trajectory
  viz/        Visual protocol, ScalarsVisual, PointsVisual, RerunObserver
  analysis/   equations_of, solve_symbolically
  models/     physics, ecology, epidemiology, economics, finance
```

Imports run downward in this order, so `models` imports only the public surfaces of the layers above it. The
top-level `skcomplex` re-exports the most used names (`System`, `Simulation`, `EuclideanSpace`, law kinds) because
the constitution allows the top-level `__init__` to be the library's front page. Each subpackage re-exports its own
surface once.

### Components are dataclasses with declared variables

`Element` and `Interaction` subclasses declare fields with `state(...)`, `parameter(...)`, and `role()`. The base
class `Component` turns every subclass into a keyword-only dataclass in `__init_subclass__` and is marked with
`typing.dataclass_transform`, so type checkers see the generated `__init__`. The `label` field stays positional.

- The generated `__init__` stores every argument unmodified, which satisfies the scikit-learn contract, and
  `Component` inherits `sklearn.base.BaseEstimator`, so `get_params`, `set_params`, and `clone` work.
- Alternative considered: hand-written `__init__` methods plus a class-level schema. Rejected because every variable
  would be declared twice.
- Alternative considered: Pydantic models. Rejected because it adds a dependency and conflicts with the estimator
  contract.

Each declared variable carries a `Domain` (real, positive, nonnegative, integer, nonnegative integer), which maps to
SymPy assumptions and to a validation check at compile time. A state can be `vector=True`, which takes its dimension
from the nearest enclosing space.

### Laws see symbols through one namespace

`Interaction.laws(self, s)` and `Element.laws(self, s)` receive a `Symbols` namespace and return a list of laws.
`s.a` is the symbol of the interaction's parameter `a`, `s.prey.n` is the symbol of the state `n` of the element bound
to the role `prey`, `s.prey.m` is an element parameter, and `s.t` is time. A vector state gives a SymPy column
`Matrix`.

- Interaction parameters are symbols, not numbers, inside `laws`, so the compiled functions take them as arguments
  and live parameter changes need no recompilation.
- Alternative considered: passing roles as keyword arguments to `laws`. Rejected because the parameters would still
  need a second channel.

### Four law kinds, combined by kind

| Law | Fields | Combines |
| --- | --- | --- |
| `Rate(target, value)` | the time derivative contribution | sums per target |
| `Noise(target, value)` | the amplitude of one independent Wiener process | one column per law |
| `Jump(rate, changes)` | a propensity and a mapping from target to increment | one channel per law |
| `Update(target, value)` | the next value of the target | one per target, a second is an error |

`value` and `rate` take a SymPy expression, anything `sympy.sympify` accepts (Sage expressions included), or a
callable `f(t, values)`, where `values` maps dotted state and parameter names to current numbers. The form is inferred
from the set of kinds: `{Rate}` gives ODE, `{Rate, Noise}` or `{Noise}` gives SDE, `{Jump}` gives a jump process, and
`{Update}` gives a discrete map. Any other set raises `IncompatibleLawsError`.

### Compilation produces flat numeric functions

`compile_system(system)` flattens the tree into paths, validates labels, roles, and domains, lays out every state
component in one vector and every parameter in another, collects the laws, and lowers them.

- Symbolic parts are lowered once with `sympy.lambdify` to NumPy functions of `(t, x, p)`. Callable parts are wrapped
  and added to the symbolic part at evaluation time through a `values` mapping built from `x` and `p`.
- Symbols are named `<path>__<variable>` with slashes replaced, so `lambdify` sees valid identifiers. User-facing names
  stay dotted, like `meadow/rabbits.n`.
- `Problem` is a frozen dataclass holding the form, the state and parameter layouts, the initial vectors, the numeric
  functions for its form, and the symbolic equations.

### Steppers share one protocol

```python
class Stepper(Protocol):
    def initialize(self, problem: Problem, t0: float, x0: NDArray, rng: Generator) -> None: ...
    def advance(self, t: float, x: NDArray, p: NDArray, until: float) -> tuple[float, NDArray]: ...
```

`advance` integrates from `(t, x)` to `until` with the given parameters and returns the new time and state. Every call
receives the current parameters, so a parameter change applies at the next frame without special handling.

- ODE: `scipy.integrate.solve_ivp` from `t` to `until` per frame, with a configurable method and tolerances.
  Restarting per frame costs one step size estimate, which is small next to a frame at interactive rates.
- SDE: Euler-Maruyama with a fixed internal step.
- Jump: the Gillespie direct method. A pending event past `until` is discarded and resampled at the next call, which
  is exact because waiting times are memoryless.
- Discrete: applies the map once per integer time step up to `until`.

Steppers are registered per form in `core.registry`, and `Simulation` resolves `solver='auto'`, a registered name, or
an instance.

### The runtime owns time, clocks, and observers

`Simulation(system, solver='auto', dt=None, clock=None, observers=(), record=True, random_state=None)` keeps its
arguments unmodified. `run(until)` compiles, initializes, and loops over frames. `start(until=inf)` runs the same loop
in a daemon thread, and `pause`, `resume`, `stop`, and `wait` control it. `update_parameters(values)` puts a mapping
of dotted names to values on a thread-safe queue that the loop drains before each frame. The name avoids a clash with
`set_params` from the estimator contract.

- `dt` defaults to 1 for the discrete form and to 0.01 otherwise.
- `RealTime(speed)` sleeps until wall time catches up with `t / speed` and records the largest lag in `max_lag_`.
  `AsFastAsPossible` never sleeps.
- `Observer` has `on_start(problem)`, `on_frame(frame)`, and `on_stop()`. `Recorder` builds the `Trajectory`, which
  `Simulation` exposes as `trajectory_` when `record=True`.

### Rerun visualization

`RerunObserver(application_id, spawn=False, save_path=None, connect=False, interval=1)` creates its own
`rerun.RecordingStream`, so several simulations never share global state. On each logged frame it sets the
`sim_time` (duration) and `step` (sequence) timelines, asks each element type's `visual` for archetypes, and logs them
under the element's path. Parameters go under `parameters/<name>` as `Scalars`.

- `ScalarsVisual` is the default and logs each scalar state as `rerun.Scalars` under `<path>/<state>`.
- `PointsVisual(position)` logs a vector state as `Points2D` or `Points3D` under `<path>`, and its scalar states as
  series.
- Visuals are classes registered in `core.registry`, so domain modules add their own.

### Symbolic analysis

`equations_of(system)` returns the SymPy equations kept by the compiled problem. `solve_symbolically(system,
use_initial_state=False)` builds one `Function` of `t` per state component, substitutes them into the ODE equations,
and calls `sympy.dsolve`. A Sage expression reaches SymPy through `sympify`, which calls its `_sympy_` method, so the
`sage` extra needs no adapter code.

### Errors

`core/_errors.py` defines `SkcomplexError` and its subclasses `ValidationError`, `LabelError`, `RoleError`,
`SpaceError`, `IncompatibleLawsError`, `SolverError`, `ParameterError`, and `AnalysisError`. The `core` package
re-exports them.

## Risks / Trade-offs

- [Each element compiles on its own, so a system of N pairwise interacting particles has N² symbolic terms] →
  Acceptable for interactive sizes, and element groups in a later change replace this with vectorized laws.
- [Restarting `solve_ivp` per frame loses the step size history] → The per-frame cost is one step estimate. A
  persistent stepper can replace it behind the same protocol if profiling shows a need.
- [Callable laws cannot be differentiated or solved symbolically] → They are excluded from `equations_of` and
  `solve_symbolically` raises `AnalysisError` when a system contains one.
- [Rerun is a heavy runtime dependency] → The project decided Rerun is part of the core experience. The viewer only
  spawns when a user asks for it, which keeps import side effect free.

## Migration Plan

The package is pre-1.0 and has no public release of the system API, so the old modules are removed in this change.
The classical mechanics tests are rewritten against `skcomplex.models.physics`, keeping the behaviors they check:
mass validation, constant velocity, constant acceleration, and the harmonic oscillator solution.
