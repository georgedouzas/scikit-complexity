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
