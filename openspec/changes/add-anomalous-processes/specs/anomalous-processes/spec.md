# Spec Delta

## Purpose

Lets a user build non-Markovian stochastic processes from symbolic laws: renewal processes whose events fire after
waiting times from any declared distribution, and states driven by fractional Gaussian noise with long-range
correlations.

## ADDED Requirements

### Requirement: Draws come from any named distribution
The modelling layer SHALL provide a named draw from any distribution that SciPy names, with arguments that MAY be
expressions of states and parameters, alongside the uniform, normal, Bernoulli, and categorical draws. An unknown
distribution name or an argument the distribution does not accept SHALL be rejected at compile time.

#### Scenario: Draw statistics
- **WHEN** a renewal law resets a state to a named draw from SciPy's normal distribution with mean 2 and scale 3, and
  10⁵ renewals are recorded
- **THEN** the mean of the reset values lies within 0.05 of 2 and their standard deviation within 0.05 of 3

#### Scenario: Unknown distribution
- **WHEN** a law draws from a distribution named `paretto`
- **THEN** compiling raises a named error that names the law and the unknown distribution, and lists the closest
  available names

### Requirement: Renewal laws accept draws
Compiling a system SHALL accept draws in the waiting time and the reset of renewal laws, as well as in update laws. A
draw in a renewal law SHALL take a fresh value at every renewal of every member, and every use of the same draw name in
one renewal SHALL read the same value. Runs with the same random state SHALL be equal.

#### Scenario: Shared draw within a renewal
- **WHEN** a renewal resets `a` to `normal('kick', 0, 1)` and `b` to `-normal('kick', 0, 1)`
- **THEN** `a + b` equals 0 after every renewal

#### Scenario: Reproducible draws
- **WHEN** a model with draws in renewal laws runs twice with random state 0
- **THEN** the two trajectories are equal

### Requirement: Renewal laws fire after drawn waiting times
A renewal law SHALL fire after a waiting time drawn from its declared distribution, apply its reset to the states all at
once, and draw the next waiting time. A waiting-time distribution SHALL have nonnegative support, and one that can give
negative values SHALL be rejected at compile time. Every member of a group SHALL renew on its own clock.

#### Scenario: Exponential waiting times
- **WHEN** a renewal law with exponential waiting times of mean 1 runs to time 1000 over an ensemble of 100 runs
- **THEN** the mean number of renewals lies within 2 standard errors of 1000

#### Scenario: Power-law waiting times
- **WHEN** a renewal law draws waiting times from a Pareto distribution with tail exponent 0.5 for a group of 10⁴
  members
- **THEN** the mean number of renewals up to time `t` grows as `t^0.5`, with a fitted exponent within 0.05 of 0.5

#### Scenario: Independent member clocks
- **WHEN** a group of 1000 members with one renewal law runs to time 10
- **THEN** the trajectory records each member's renewals at that member's own times, and no two members share all
  their renewal times

#### Scenario: Negative waiting time
- **WHEN** a renewal law draws its waiting time from a normal distribution
- **THEN** compiling raises a named error that names the law and asks for a distribution with nonnegative support

### Requirement: Renewals combine with continuous laws
A system SHALL combine renewal laws with rate and noise laws. Between two renewals the states SHALL evolve by the rate
and noise laws, and the solver SHALL stop at each renewal time, apply the reset, and continue from the reset state.

#### Scenario: Ballistic flights
- **WHEN** a state moves at a velocity that a renewal law redraws, and the rate law is `dx/dt = v`
- **THEN** at every renewal the position equals the position at the previous renewal plus the velocity times the
  elapsed waiting time, within 1e-12

### Requirement: Systems of renewals alone run without a time grid
A system or partition whose only laws are renewal laws SHALL compile to a renewal form. Its stepper SHALL jump from
renewal to renewal exactly, and SHALL return the states at any requested time from the last renewal before it.

#### Scenario: Renewal form
- **WHEN** a system contains only renewal laws
- **THEN** the compiled problem has the renewal form, and a renewal-form system and an ODE system in sibling
  subsystems compile to a partitioned problem

#### Scenario: State between renewals
- **WHEN** a renewal-form simulation is asked for frames every 0.1 time units
- **THEN** each frame holds the states set by the last renewal at or before its time

### Requirement: Renewals are recorded and counted
The trajectory SHALL record every renewal with its time, its member, and the states before and after it. The renewal
count of every renewal law SHALL be available as an observable, so it is analysed and streamed to Rerun like other
observables.

#### Scenario: Renewal count observable
- **WHEN** a trajectory of a renewal model is asked for the observable `renewals` of its renewal law
- **THEN** it returns at every frame the number of renewals recorded up to that frame's time

### Requirement: States are driven by fractional Gaussian noise
A fractional noise law SHALL drive a state with the increments of fractional Brownian motion of a declared Hurst index
`0 < H < 1`, scaled by an amplitude. The increments of one law SHALL have the covariance of fractional Gaussian noise
across the whole run, and SHALL be independent of other laws and other group members. A Hurst index of 1/2 SHALL give
the same process as an ordinary noise law.

#### Scenario: Increment correlation
- **WHEN** a fractional noise law with `H = 0.75` drives 1000 group members for 1000 steps
- **THEN** the correlation of consecutive increments lies within 0.02 of `2^(2H-1) - 1`

#### Scenario: Superdiffusion
- **WHEN** a user computes the ensemble mean squared displacement of 1000 members driven by fractional noise with
  `H = 0.75`
- **THEN** the scaling exponent lies within 0.05 of `2H = 1.5`

#### Scenario: Hurst index of one half
- **WHEN** a fractional noise law with `H = 0.5` and the same amplitude as an ordinary noise law run over ensembles
- **THEN** the variances of the state at time 1 agree within 2 standard errors

### Requirement: Fractional noise laws are validated
Compiling SHALL reject a Hurst index outside the open interval from 0 to 1, and an amplitude that depends on a state
when the Hurst index differs from 1/2.

#### Scenario: Hurst index out of range
- **WHEN** a fractional noise law has `H = 1`
- **THEN** compiling raises a named error that names the law and the value, and states the open interval

#### Scenario: State-dependent amplitude
- **WHEN** a fractional noise law with `H = 0.7` has an amplitude proportional to its own state
- **THEN** compiling raises a named error that names the law and the state, and says that only state-independent
  amplitudes are supported for a Hurst index other than 1/2

### Requirement: Fractional noise is integrated on a declared horizon
Systems with fractional noise SHALL run with a stepper that draws the correlated increments of the whole run on a fixed
step up to a horizon, which a simulation run sets from its end time. Advancing beyond the horizon, or choosing a stepper
without correlated increments, SHALL raise a named error.

#### Scenario: Beyond the horizon
- **WHEN** a frame-by-frame simulation with fractional noise and a horizon of 10 is advanced to time 12
- **THEN** it raises a named error that names the horizon and the requested time, and asks for a larger horizon

#### Scenario: Wrong stepper
- **WHEN** a system with a fractional noise law with `H = 0.7` is run with the Euler-Maruyama stepper
- **THEN** it raises a named error that names the fractional Euler-Maruyama stepper as the one to use
