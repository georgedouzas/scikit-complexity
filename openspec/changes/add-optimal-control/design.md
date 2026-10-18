# Design

## Context

Laws read parameters and time, drivers from `add-hybrid-dynamics` give time functions, and the symbolic layer derives
Jacobians. Live parameter updates already let code change inputs during a run.

## Goals / Non-Goals

**Goals:**

- One control input declaration used by open-loop optimal controls, closed-loop controllers, and linear analysis.
- Pontryagin's equations derived symbolically, as Dilão (2023 ch. 9) derives them by hand.

**Non-Goals:**

- Stochastic optimal control and dynamic programming over large state spaces.
- Model predictive control. Collocation over a receding horizon can add it later.

## Decisions

### Control inputs

`Control(label, bounds=(lower, upper), initial=0.0)` declares an input that laws use like a parameter. A schedule of
values, an optimal solution, or a controller supplies the value. Without one, the control holds its initial value.

### Optimal control

`OptimalControlProblem(system, controls, running_cost, terminal_cost=0, horizon, state_bounds=None)` holds the problem.
`pontryagin(problem)` builds the Hamiltonian `H = L + λ · f`, derives the costate equations `λ' = -∂H/∂x` and the
stationarity condition `∂H/∂u = 0` symbolically, projects onto the bounds, and solves the two-point problem by a
forward-backward sweep with relaxation. `collocation(problem, nodes)` transcribes the problem with trapezoidal or
Hermite-Simpson collocation and solves it with `scipy.optimize.minimize` using the SLSQP or trust-constr method, with
Jacobians derived symbolically. Both return the control as a time series, the states, the costates where defined, and
the cost.

### Feedback controllers

`PID(control, measure, setpoint, kp, ki, kd)` and `StateFeedback(control, gain)` are components that add the
controller's states, such as the integral of the error, as rate laws. Controllers are therefore part of the compiled
system, so solvers and backends need no change.

### Linear analysis

`linearize(system, controls, outputs, at)` returns `A, B, C, D` from symbolic Jacobians. `controllability` and
`observability` return the ranks of the Kalman matrices, and `lqr(A, B, Q, R)` solves the Riccati equation with SciPy.

## Risks / Trade-offs

- [Sweep convergence] → The sweep relaxes the control update and falls back to collocation when it fails to converge.
- [Nonsmooth bang-bang controls] → Collocation with enough nodes captures switches, and the user guide shows the
  harvesting example.
