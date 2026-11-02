# Design

## Context

Laws read parameters and time, drivers from `add-hybrid-dynamics` give time functions, and the symbolic layer derives
Jacobians. Live parameter updates already let code change inputs during a run.

After `redesign-system-model`, a system is `System(label, components=, interactions=, space=, time=)`. A parameter
declared `parameter(adaptive=True)` may be changed by laws of its component or of interactions bound to it, and it is
recorded in trajectories like a state. The `Objective` law kind belongs to the constraint class and is valid under
static time (optimisation) and continuous time (control over a horizon). Every solver declares `Supports`, every law
declares `Requires`, and `validate()` reports every invalid combination before compilation.

## Goals / Non-Goals

**Goals:**

- One control input declaration, a bounded adaptive parameter, used by open-loop optimal controls, closed-loop
  controllers, and linear analysis.
- Pontryagin's equations derived symbolically, as Dilão (2023 ch. 9) derives them by hand.
- Optimal control expressed in the system's own vocabulary: costs are objective laws, and the methods are solvers that
  declare what they support.

**Non-Goals:**

- Stochastic optimal control and dynamic programming over large state spaces.
- Model predictive control. Collocation over a receding horizon can add it later.

## Decisions

### Control inputs are bounded adaptive parameters

`parameter(initial, adaptive=True, bounds=(lower, upper))` declares an input that laws use like a parameter. A schedule
of values, an optimal solution, or a controller supplies the value. Without one, the control holds its initial value.
`bounds=` is new on `parameter` in `core.components`: optimal control solvers project onto the bounds, and validation
reports a bound that does not contain the initial value. The redesign's adaptive parameters replace the `Control` class
this change first planned.

### Optimal control

The cost is an `Objective(expression, terminal=0, sense='min', label='objective')` law returned by a component or an
interaction, in `core.laws`. Under continuous time, the value of a run is `∫ expression dt + terminal(T)`, with `T` the
horizon `until` of the run, minimised for `sense='min'` and maximised for `sense='max'`. The compiler keeps the
objective next to the recorded values of the stock-flow pass, as `Problem.objectives`, so the trajectory records the
running value under `<component path>.<label>`, and leaves the form of the system to its other laws. State bounds are
`Boundary` laws for `Pontryagin`, and the nonnegative domains of states for `Collocation`.

Two solvers in `core.solvers` minimise the cost over the system's bounded adaptive parameters:
- `Pontryagin(nodes, relaxation, tolerance, max_iterations)` builds `H = L + λ · f` from the running cost `L` and the
  drift `f`, derives the costate equations `λ' = -∂H/∂x`, the terminal costates `∂φ/∂x`, and the stationarity condition
  `∂H/∂u = 0` symbolically with Sage's `solve`, projects onto the bounds, takes the bound that minimises `H` where `H`
  is linear in the controls, and solves the two-point problem by a forward-backward sweep of fourth-order Runge-Kutta
  steps with relaxation. The weight of the update halves whenever the distance to the update stops falling, which
  breaks the two-cycles the plain sweep falls into, and a sweep that diverges or does not converge raises an error that
  suggests `Collocation`.
- `Collocation(nodes, scheme='hermite-simpson' | 'trapezoidal', method='SLSQP' | 'trust-constr')` transcribes the
  problem and solves it with `scipy.optimize.minimize`, with the gradient of the cost and the Jacobian of the defects
  assembled from symbolic Jacobians. Hermite-Simpson is the default for its accuracy on smooth problems, and the
  trapezoidal scheme suits bang-bang controls with singular arcs, on which Hermite-Simpson rings.

Both run through `Simulation(system, solver=..., observables=...).run(until=T)`. They share the base
`OptimalControlSolver`, which reads the problem, solves it over the whole run when the run starts, from the initial
state and the initial parameters, and then follows the solution frame by frame. The trajectory records the controls as
adaptive parameters, the states, and the running value. The costates, the cost, and the iterations are in
`simulation.stepper_.solution_`, a `ControlSolution`, since a stepper adds no recorded values of its own. The
`OptimalControlProblem` object and the `pontryagin` and `collocation` functions this change first planned are replaced
by these solvers.

Declarations:

| Solver | `Supports` | Issue when misused |
|---|---|---|
| `Pontryagin` | `time={CONTINUOUS}`, `space={NONE}`, `laws={RATE, EQUATION, OBJECTIVE, BOUNDARY}`, `domains={REAL, ANGLE}`, `components={ELEMENT, SYSTEM}` | a system without a bounded adaptive parameter or an objective, or with noise, jump, or update laws |
| `Collocation` | the same, without `BOUNDARY` | the same |

Equation laws are supported because the compiler solves them into rates. The `Objective` law declares
`Requires(time={STATIC, CONTINUOUS})`. `validate()` reports an objective under discrete time, and an objective run with
a named solver that does not support objectives, whose fix lists `collocation` and `pontryagin`. `compile_system` and
the catalog validate without a solver, so the default solver of the ODE form refuses a problem with objectives when it
initialises instead, through the capability check every stepper already runs.

### Feedback controllers are library realizations

`PID(plant, integrator, control, measure, kp, ki, kd, setpoint, limits)` and
`StateFeedback(plant, control, states, gain, limits)` live in `skcomplex.models.engineering.realizations`, which the
engineering control models import. An element cannot read the states of another component, and an interaction holds
no state, so each controller is an `Interaction` bound to its plant:
- `PID` holds the integral of the error in an `Integrator` element bound to its role `integrator`, as a state with a
  `Rate` law, and sets the control from the measured state of the plant;
- `StateFeedback` sets the control to `u = -K x` from named states of the plant.

The compiler solves `Equation` laws for highest derivatives only, so it has no algebraic assignment of a state.
Controllers set the control input by a `Boundary` law, the value a state takes after every solver step, which makes
them sampled controllers at the step of the solver, and the control models run with `RK4(dt=0.001)`. Controllers are
part of the compiled system, so solvers and backends need no change and validation checks them like any component. A
model of another field that needs a controller defines its own, as the redesign requires.

### Linear analysis

`linearize(system, controls, outputs, at)` in `core.analysis` returns a `StateSpace` of `A, B, C, D` from symbolic
Jacobians, with the controls named by their adaptive parameters, the bounded ones by default, and the outputs named by
states or controls, every state by default. `controllability(A, B)` and `observability(A, C)` return the ranks of the
Kalman matrices, and `lqr(A, B, Q, R)` returns the gain `K = R^-1 B' P` from SciPy's `solve_continuous_are`.
`linearize` validates the system first and raises a validation error for a system that is not under continuous time.

## Risks / Trade-offs

- [Sweep convergence] → The sweep relaxes the control update, halves the weight when it stops converging, and the
  error suggests `Collocation` when it diverges or fails to converge.
- [Sampled controllers] → Controllers set the control after every solver step, so they act like continuous ones only
  with small fixed steps, and the control models choose `RK4(dt=0.001)`.
- [Nonsmooth bang-bang controls] → Collocation with enough nodes captures switches, and the user guide shows the
  harvesting example.

## Resolved Questions

- The signature of the `Objective` law is shared by three changes: this one (running and terminal costs under
  continuous time), `add-static-models`, and `add-constraint-based-models` (linear lexicographic objectives under static
  time). This change goes first and defines the law, one class for every time kind:
  `Objective(expression, terminal=0, sense='min', label='objective')`, with `Requires(time={STATIC, CONTINUOUS})`.
  - `expression` is the running value under continuous time and the value itself under static time, so the static
    changes reuse it unchanged.
  - `terminal` is meaningful under continuous time only, and the static changes leave it at 0 or reject a nonzero one
    under static time.
  - `sense` is `'min'` or `'max'` and `label` names the recorded value, both shared by every time kind.
  - A lexicographic priority is a later keyword argument of the same class, such as `priority=0`, added by
    `add-constraint-based-models`, so lexicographic objectives are several `Objective` laws ordered by priority rather
    than a second class.
  - The compiler exposes objectives as `Problem.objectives`, `CompiledObjective` records with the running and terminal
    values in the problem's symbols and their references, which a static solver reads the same way.
- The library models are rebuilt on the formalism under the names of their inventory entries. The Ramsey model
  `optimal_consumption` runs with `Collocation`, since the forward-backward sweep diverges on its saddle path, and
  `optimal_harvesting` with the trapezoidal scheme for its singular arc.
