## MODIFIED Requirements

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
