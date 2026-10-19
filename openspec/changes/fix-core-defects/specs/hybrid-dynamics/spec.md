## MODIFIED Requirements

### Requirement: Laws read delayed states
Laws SHALL read states at `t - τ` for constant and state-dependent delays, with histories before the start given as
functions or constants. Every ODE solver SHALL integrate delays of any positive length, including delays shorter than
its first step.

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
