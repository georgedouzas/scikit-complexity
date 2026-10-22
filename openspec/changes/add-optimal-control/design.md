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

The cost is an `Objective(running, terminal=0, sense='min')` law returned by a component or an interaction, in
`core.laws`. Under continuous time, the cost over a run is `∫ running dt + terminal(T)`, with `T` the horizon `until` of
the run. State bounds are `Boundary` laws, as the redesign's constraint class already holds them.

Two solvers in `core.solvers` minimise the cost over the system's bounded adaptive parameters:
- `Pontryagin()` builds `H = L + λ · f` from the running cost `L` and the drift `f`, derives the costate equations
  `λ' = -∂H/∂x` and the stationarity condition `∂H/∂u = 0` symbolically, projects onto the bounds, and solves the
  two-point problem by a forward-backward sweep with relaxation.
- `Collocation(nodes, scheme='trapezoidal' | 'hermite-simpson')` transcribes the problem and solves it with
  `scipy.optimize.minimize` using the SLSQP or trust-constr method, with Jacobians derived symbolically.

Both run through `Simulation(system, solver=..., observables=...).run(until=T)`. The trajectory records the controls as
adaptive parameters, the states, the costates where defined, and the cost. The `OptimalControlProblem` object and the
`pontryagin` and `collocation` functions this change first planned are replaced by these solvers.

Declarations:

| Solver | `Supports` | Issue when misused |
|---|---|---|
| `Pontryagin` | `time={CONTINUOUS}`, `laws={RATE, OBJECTIVE, BOUNDARY}`, `domains={REAL, ANGLE}`, `components={ELEMENT, SYSTEM}` | a system without a bounded adaptive parameter, or with noise, jump, or update laws |
| `Collocation` | the same as `Pontryagin` | the same |

The `Objective` law under continuous time declares `Requires(time={STATIC, CONTINUOUS})`. `validate()` reports an
objective under discrete time, and an objective in a continuous-time system run with a solver that does not support
objectives, whose fix lists `Pontryagin` and `Collocation`.

### Feedback controllers are library realizations

`PID(control, measure, setpoint, kp, ki, kd)` and `StateFeedback(control, gain)` live in
`skcomplex.models.engineering.realizations`, which the engineering control models import. Each is an `Element`:
- `PID` holds the integral of the error as a state with a `Rate` law, and sets the control by an algebraic `Equation`
  on the bound adaptive parameter, from an expression of states as its measure;
- `StateFeedback` sets the control by an `Equation` `u = -K x`.

Controllers are therefore part of the compiled system, so solvers and backends need no change and validation checks
them like any element. A model of another field that needs a controller defines its own, as the redesign requires.

### Linear analysis

`linearize(system, controls, outputs, at)` in `core.analysis` returns `A, B, C, D` from symbolic Jacobians, with the
controls named by their adaptive parameters and the outputs as expressions of states. `controllability` and
`observability` return the ranks of the Kalman matrices, and `lqr(A, B, Q, R)` solves the Riccati equation with SciPy.
`linearize` validates the system first and raises the validation error for a system that is not under continuous time.

## Risks / Trade-offs

- [Sweep convergence] → The sweep relaxes the control update and the report suggests `Collocation` when it fails to
  converge.
- [Nonsmooth bang-bang controls] → Collocation with enough nodes captures switches, and the user guide shows the
  harvesting example.

## Open Questions

- The signature of the `Objective` law is shared by three pending changes: this one (running and terminal costs under
  continuous time), `add-static-models`, and `add-constraint-based-models` (linear lexicographic objectives under static
  time). Which change defines the law, and whether `terminal=` and a lexicographic priority are arguments of one class,
  needs a decision before the first of them is applied.
