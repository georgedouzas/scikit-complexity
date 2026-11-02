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
Each noise law SHALL declare Itô, Stratonovich, or anti-Itô noise through its convention, and the framework SHALL add
the drift correction for the convention, with Milstein and Heun steppers available. The correction SHALL cover shared
and correlated Wiener processes and states that several noise laws drive. Laws that share one process SHALL declare
one convention, and validation SHALL report laws that share a process with different conventions.

#### Scenario: Geometric Brownian motion in Stratonovich form
- **WHEN** `dX = μ X dt + σ X ∘ dW` runs with a noise law declared `convention='stratonovich'` over an ensemble
- **THEN** the mean of `X` at time 1 matches `exp(μ + σ² / 2)` within 2 standard errors

#### Scenario: Cross terms of a shared process
- **WHEN** `dX = σ Y ∘ dW` and `dY = σ X ∘ dW` share one process in one element, both laws declared Stratonovich, and
  run over an ensemble
- **THEN** the mean of `X + Y` at time 1 matches `(X₀ + Y₀) exp(σ² / 2)` within 2 standard errors

#### Scenario: Several laws on one state
- **WHEN** two Stratonovich noise laws with independent processes drive one state
- **THEN** the system compiles, and its drift correction is the sum of both laws' corrections

#### Scenario: Mixed conventions on one process
- **WHEN** an Itô noise law and a Stratonovich noise law name the same process
- **THEN** validation reports one issue that names both laws and the process, with the fix of one convention for both

#### Scenario: Amplitude outside the law's scope
- **WHEN** a Stratonovich noise law depends on a state that another component drives through a shared process
- **THEN** validation reports an issue that names both laws and suggests the Itô convention or one component

### Requirement: Poincaré sections come from events
The analysis layer SHALL return the states where a trajectory crosses a section in a given direction, and stroboscopic
samples of forced systems.

#### Scenario: Forced Duffing section
- **WHEN** a user takes the stroboscopic section of the forced Duffing oscillator in its chaotic regime
- **THEN** the section shows a fractal set of thousands of distinct points rather than a finite periodic orbit

### Requirement: Series inputs of any length compile
An interpolated series input SHALL compile and run on every backend for any number of data points, with the same
values and breakpoints as for a short series.

#### Scenario: Long held series
- **WHEN** a law reads `series(s.t, times, values, 'previous')` with 1000 points at times 0 to 999
- **THEN** the system compiles on the NumPy and the Numba backends without a syntax or recursion error, and the input
  at time `k + 0.5` equals the value of point `k` for every `k`

#### Scenario: Long linear series
- **WHEN** a law reads a linear series of 1000 points of `sin` at times 0 to 999
- **THEN** the input at every time equals NumPy's linear interpolation of the data within 1e-12, and every data time
  is a breakpoint of the solver

### Requirement: State conditions of rate laws restart integrators
A condition on states inside `where` or `step` in a rate law SHALL compile into a hidden event without a reset, so
integrators with event location restart at the crossing, and hidden events SHALL not appear in the trajectory's events.

#### Scenario: Bathtub drains empty
- **WHEN** `bathtub(water=20.0)` runs until 6 with the default solver
- **THEN** it finishes without a stall warning, the water stays nonnegative within 1e-6, and the trajectory has no
  events

#### Scenario: Escapement of a clock
- **WHEN** `huygens_clocks` is compiled
- **THEN** the escapement `sign(|theta| - theta_c)` of each clock compiles into one hidden event
