# Spec Delta

## Purpose

Lets a user estimate the hidden states of any system from noisy and partial observations, smooth them over a run,
estimate parameters jointly with the states, and forecast from the latest estimate, with the observation model of the
data as the measurement model and the noise laws of the system as the process noise.

## ADDED Requirements

### Requirement: Assimilation filters the states of a system
`assimilate(system, data, method=...)` SHALL estimate the states of a system at the observation times of one observed
run, with `'kalman'`, `'ekf'`, `'ukf'`, `'enkf'`, or `'particle'`, reading the `ObservationSpec` of the data as the
measurement model, and SHALL return a frozen `AssimilationResult` with the estimated trajectory, the covariances or the
ensemble, the innovations, and the log-likelihood of the run.

#### Scenario: Noisy pendulum
- **WHEN** the noisy pendulum of Särkkä (Example 5.1) is observed through `sin(theta)` with Gaussian noise and filtered
  over 20 seeded runs
- **THEN** the mean root mean squared error of the angle is within 0.02 of 0.12 for `'ekf'`, of 0.11 for `'ukf'`, and
  of 0.12 for `'particle'` with 10000 particles

#### Scenario: Linear tracking
- **WHEN** the Wiener velocity model of Särkkä (Example 4.3) is filtered with `'kalman'` over 20 seeded runs
- **THEN** the mean position error is within 0.05 of 0.43, below the error of the measurements alone

#### Scenario: Ensemble filter on Lorenz-96
- **WHEN** a Lorenz-96 system of 40 variables with forcing 8, observed in every variable every 0.05 with unit Gaussian
  noise, is filtered with `'enkf'` and 40 members over 500 observation times
- **THEN** the root mean squared error of the states after the first 100 times is below 0.5, a bound to be tightened
  to the DAPPER benchmark value once that value is checked

#### Scenario: Same seed
- **WHEN** `'enkf'` or `'particle'` runs twice with the same `random_state`
- **THEN** both results hold equal trajectories and log-likelihoods

### Requirement: Assimilation smooths over the run
`assimilate` with `smoother=True` SHALL return the smoothed means of the states given every observation of the run, by
the Rauch-Tung-Striebel smoother after `'kalman'`, `'ekf'`, and `'ukf'` and by the backward simulation smoother after
`'particle'`, and SHALL refuse a smoother for `'enkf'`.

#### Scenario: Smoothed pendulum
- **WHEN** the noisy pendulum is smoothed over 20 seeded runs
- **THEN** the mean error of the angle is within 0.01 of 0.033 for `'ekf'` and of 0.028 for `'ukf'` and
  `'particle'`, and below the error of the matching filter

#### Scenario: Smoothed tracking
- **WHEN** the Wiener velocity model is smoothed with `'kalman'` over 20 seeded runs
- **THEN** the mean position error is within 0.05 of 0.27

#### Scenario: Ensemble smoother
- **WHEN** `smoother=True` is given with `'enkf'`
- **THEN** it raises `ValidationError` that names the methods with a smoother

### Requirement: Assimilation estimates parameters with the states
`assimilate` SHALL append the parameters named by `augmented` to the state with a random walk of the given variance,
and SHALL report their final means and variances as `estimates`.

#### Scenario: Gravity of the pendulum
- **WHEN** the noisy pendulum is filtered with `'ukf'` and `augmented=('g',)`, started from `g=7`
- **THEN** the estimate of `g` at the last time is within 5% of 9.81, and its variance is below its initial variance

### Requirement: Assimilated systems forecast
The `system` of an `AssimilationResult` SHALL hold the last estimate of the states and the estimates of augmented
parameters, so that `simulate` forecasts from them.

#### Scenario: Forecast from the last estimate
- **WHEN** the system of a result is simulated
- **THEN** its first frame equals the last frame of the result's trajectory within 1e-12

### Requirement: Assimilation checks its models
`assimilate` SHALL raise `ValidationError` before filtering when the method needs linear laws, derivatives, or Gaussian
noise that the system or the observation model does not have, or when the data hold several runs and `run` is not
given, naming the law, tag, noise, or argument.

#### Scenario: Nonlinear system with the Kalman filter
- **WHEN** `'kalman'` is called on the pendulum
- **THEN** it raises `ValidationError` that names the rate of the angular velocity and the methods for nonlinear laws

#### Scenario: Counts with a Gaussian filter
- **WHEN** `'ukf'` is called on data observed with Poisson noise
- **THEN** it raises `ValidationError` that names the noise and the method `'particle'`

#### Scenario: Several runs
- **WHEN** the data hold 3 runs and `run` is not given
- **THEN** it raises `ValidationError` that lists the labels of the runs and names `run`
