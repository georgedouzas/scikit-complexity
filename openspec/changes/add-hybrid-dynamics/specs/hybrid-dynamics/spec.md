# Spec Delta

## Purpose

Lets a user write models with delays, discrete events, time-dependent drivers, and noise of a chosen convention, and
derive Poincaré sections from them.

## ADDED Requirements

### Requirement: Laws read delayed states
Laws SHALL read states at `t - τ` for constant and state-dependent delays, with histories before the start given as
functions or constants.

#### Scenario: Mackey-Glass chaos
- **WHEN** the Mackey-Glass equation runs with τ = 17 and its standard parameters
- **THEN** the largest Lyapunov exponent of the run is positive, and with τ = 6 the run settles on a periodic orbit

#### Scenario: Delay-induced oscillation
- **WHEN** Hutchinson's delayed logistic equation runs with `r * τ` above π/2
- **THEN** the population oscillates around `K` with a growing then stable amplitude

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
discontinuities.

#### Scenario: Step response
- **WHEN** a first-order system receives a step input at time 5
- **THEN** its response before 5 is exactly the unforced solution and after 5 matches the analytic step response within
  1e-6

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
