# Spec Delta

## ADDED Requirements

### Requirement: Calibration by particle marginal Metropolis-Hastings
`calibrate(method='pmmh')` SHALL sample the posterior of the parameters by a random walk Metropolis-Hastings chain whose
likelihood is that of a bootstrap particle filter over every run of the data, and SHALL accept systems with any
stochastic laws and observation models with any noise.

#### Scenario: Noise variance of the pendulum
- **WHEN** the measurement variance of the noisy pendulum, observed through `sin(theta)`, is calibrated with `'pmmh'`
  and a uniform prior on `[0.02, 0.3]`, as in Särkkä (Example 12.2)
- **THEN** the true variance 0.1 lies inside the 90% posterior interval

#### Scenario: Outbreak with jump laws
- **WHEN** the contact rate of a stochastic SIR system with jump laws is calibrated with `'pmmh'` on simulated counts
  observed with negative binomial noise
- **THEN** the true rate lies inside the 90% posterior interval, and the result reports the acceptance rate

#### Scenario: Same seed
- **WHEN** `'pmmh'` runs twice with the same `random_state` on the serial executor
- **THEN** both results hold equal draws
