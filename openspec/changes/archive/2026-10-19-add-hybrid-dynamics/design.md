# Design

## Context

ODE problems run through SciPy, RK4, and velocity Verlet. SDEs run through Euler-Maruyama, which is Itô. Boundary laws
apply after every step without locating crossings, and time `t` is already a symbol that laws can use.

## Goals / Non-Goals

**Goals:**

- Delay and event models written as their equations, with accuracy controlled by the solver tolerances.
- Correct multiplicative noise without users deriving drift corrections by hand.

**Non-Goals:**

- Neutral delay equations and distributed delays with general kernels. Discrete delays and the linear-chain trick cover
  the references.
- Fractional-order derivatives (Trzaska 2023 §2.3.6) and optimal control (Dilão 2023 ch. 9), which
  `add-fractional-dynamics` and `add-optimal-control` cover.

## Decisions

### Delays

`delayed(state, tau)` is a Sage function that stays symbolic in the equations. The compiler collects delayed states and
their delays, and the stepper keeps a history buffer of dense output per step, using cubic Hermite interpolation over
stored states and derivatives. The history before `t = 0` comes from `history={path: function or value}` on the system.
Constant delays use the method of steps with breakpoints at multiples of τ. State-dependent delays evaluate τ at every
step, and the history must cover the earliest time they reach. Fixed-step steppers interpolate the same way.

### Events

`Event(guard, reset=None, direction=0, terminal=False, label=None)` is a law. The guard is a symbolic function of states
and time, and the reset is a mapping of states to new values, symbolic or callable. SciPy's `events` locate crossings
for the SciPy solver. Fixed-step steppers locate them by bisection on the step's interpolant and restart the step from
the reset state. Trajectories store event times, labels, and states before and after the reset.

### Drivers

`step(t0, height)`, `ramp(t0, t1, slope)`, `pulse(t0, width, height)`, `pulse_train(period, width, height)`, and
`series(times, values, kind='linear' | 'previous')` are Sage functions of `t`. They lower to NumPy and Numba code, and
their discontinuities are passed to solvers as breakpoints. Scheduled parameter ramps reuse the live parameter mechanism
from inside the stepper.

### Noise conventions

`System(..., noise='ito' | 'stratonovich' | 'anti-ito')` declares the convention. The compiler adds the drift
correction `α Σ g ∂g/∂x` with α = 1/2 for Stratonovich and 1 for anti-Itô, derived symbolically from the noise terms,
so every stepper integrates the Itô form. `Milstein()` and `Heun()` steppers improve the strong order where the noise is
diagonal.

### Poincaré sections

`poincare_section(system, guard, direction, until)` runs the system with a non-terminal event and returns the states at
the crossings. `stroboscopic(system, period)` samples at multiples of the period.

## Risks / Trade-offs

- [History memory] → The buffer keeps only the last `max τ` of dense output.
- [Chattering events] → Events have a minimum separation, and a guard that triggers again within it raises a named
  error.

## Implementation Notes

- The compiler replaces every distinct `delayed(x, tau)` by a hidden parameter `__delayed_<k>`, and the hybrid steppers
  fill it from the history buffer before every evaluation of the drift. Equations and kernels therefore stay ordinary
  functions of `(t, x, p)`, and `equations_of` shows the delayed states as `x(t - tau)` again. Stability and Lyapunov
  analysis reject delay systems, whose state space is infinite-dimensional.
- SciPy steps manually with the integrator classes instead of `solve_ivp(events=...)`. Each step adds its dense output
  to the history and is checked for strict sign changes of every guard. The root is found with `brentq` on the
  interpolant and then moved just past the root, so a restart never finds the same crossing again. RK4 interpolates its
  steps with cubic Hermite polynomials. The SDE steppers interpolate linearly.
- The maximum step is capped at the smallest delay, by the method of steps. A delay that depends on states declares its
  smallest value as `minimum`. A fixed step longer than the smallest delay raises an error.
- Breakpoints come from the time relations in `where` conditions, including periodic `mod` patterns in pulse trains.
  Within an interval that ends at a breakpoint, inputs are evaluated at the last floating-point time before it, so a
  step response is exact to the solver tolerance. Steppers take the hybrid path only when a problem has delays or
  events, or a breakpoint falls in the frame including its end, so other problems keep their compiled fast path.
- Event labels default to the owner's path, with an index only when the owner declares several events.
- Noise corrections are symbolic. A component that receives several noise laws under a non-Itô convention is rejected,
  because the correction would need the cross terms. `Milstein` is Platen's derivative-free scheme and `Heun` is the
  Itô-consistent predictor-corrector that averages only the drift.
- `Simulation(schedule=...)` sets parameters from functions of time at every frame, and the frame records the value
  that holds until the next frame.
- `poincare_section` adds a non-terminal event to a copy of the compiled problem. `stroboscopic` samples frames at the
  period, so long sections inherit the per-frame cost of the runtime.
- The Brownian particle near a wall uses the smooth diffusion `D0 z² / (z² + a²)` with laws symmetric about the wall,
  because boundaries apply at frame ends and the noise amplitude must stay defined between them.
- Izhikevich and integrate-and-fire live in the new `neuroscience` field. Newton's cradle leaves small gaps between
  balls, so collisions are separate events rather than simultaneous ones.
