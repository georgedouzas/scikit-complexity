# hybrid-dynamics Specification

## Purpose

Lets a user write models with delays, discrete events, time-dependent drivers, and noise of a chosen convention, and
derive Poincaré sections from them.

## Requirements

### Requirement: Laws read delayed states
Laws SHALL read states at `t - τ` for constant and state-dependent delays, with histories before the start given as
functions or constants. Every ODE solver SHALL integrate delays of any positive length, including delays shorter than
its first step. A delay that is an expression of parameters SHALL take its smallest value from the parameter values.

#### Scenario: Mackey-Glass chaos
- **WHEN** the Mackey-Glass equation runs with τ = 17 and its standard parameters
- **THEN** the largest Lyapunov exponent of the run is positive, and with τ = 6 the run settles on a periodic orbit

#### Scenario: Delay-induced oscillation
- **WHEN** Hutchinson's delayed logistic equation runs with `r * τ` above π/2
- **THEN** the population oscillates around `K` with a growing then stable amplitude

#### Scenario: Very short delay
- **WHEN** `dx/dt = -x(t - τ)` with `x = 1` before the start runs with `ScipyODE` until time 0.2 for τ = 1e-4 and
  τ = 1e-6
- **THEN** both runs finish without error, and the final state matches `exp(-0.2)` within 1e-3

#### Scenario: Delay as an expression of parameters
- **WHEN** a law reads `delayed(x, 2 * tau)` for a parameter `tau = 0.5` and `x = 1` before the start
- **THEN** the system compiles with the smallest delay 1, and the run matches the method of steps for `x' = -x(t - 1)`

### Requirement: Events reset states at located crossings
A law SHALL declare an event whose guard crossing zero in a given direction resets states or ends the run, and the
solver SHALL locate the crossing within its tolerance.

#### Scenario: Bouncing ball
- **WHEN** a ball dropped from height 1 bounces with restitution 0.8
- **THEN** the event times match the analytic impact times within 1e-6, and the trajectory records each impact

#### Scenario: Terminal event
- **WHEN** an event marked terminal fires at time 3.2
- **THEN** the run stops at 3.2 and the trajectory ends with the state at the crossing

### Requirement: Drivers force laws in time
Laws SHALL use step, ramp, pulse, pulse train, and interpolated series inputs of time, and the solver SHALL respect their
discontinuities, also when it keeps one integrator across frames and the run has no end time.

#### Scenario: Step response
- **WHEN** a first-order system receives a step input at time 5
- **THEN** its response before 5 is exactly the unforced solution and after 5 matches the analytic step response within
  1e-6

#### Scenario: Pulse train in a persistent integrator
- **WHEN** a decay receives a pulse train of period 21 and width 1/24 and runs with `ScipyODE` until time 10
- **THEN** the next breakpoint after time 1 is 21, and the final state matches the analytic response within a relative
  tolerance of 1e-5

### Requirement: Noise follows a declared convention
Systems SHALL declare Itô, Stratonovich, or anti-Itô noise, and the framework SHALL add the drift correction for the
convention, with Milstein and Heun steppers available.

#### Scenario: Geometric Brownian motion in Stratonovich form
- **WHEN** `dX = μ X dt + σ X ∘ dW` runs as Stratonovich over an ensemble
- **THEN** the mean of `X` at time 1 matches `exp(μ + σ² / 2)` within 2 standard errors

### Requirement: Poincaré sections come from events
The analysis layer SHALL return the states where a trajectory crosses a section in a given direction, and stroboscopic
samples of forced systems.

#### Scenario: Forced Duffing section
- **WHEN** a user takes the stroboscopic section of the forced Duffing oscillator in its chaotic regime
- **THEN** the section shows a fractal set of thousands of distinct points rather than a finite periodic orbit
