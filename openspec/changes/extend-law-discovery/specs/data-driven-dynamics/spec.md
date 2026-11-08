# Spec Delta

## ADDED Requirements

### Requirement: Sparse identification in weak form
`sparse_identification` SHALL take `weak=True`, which fits the laws to integrals of the series and the library against
compactly supported test functions over subdomains of every run, so no derivative of the series is estimated. The
fitted system SHALL be the same kind of system as without the weak form.

#### Scenario: Noisy Lorenz
- **WHEN** `sparse_identification` with `weak=True` and `degree=2` runs on a Lorenz trajectory with Gaussian noise of 5%
  of the standard deviation of every variable
- **THEN** it keeps exactly the seven terms of the Lorenz system, with coefficients within 5% of their true values

#### Scenario: Same system without noise
- **WHEN** it runs with and without the weak form on a noise-free Lorenz trajectory
- **THEN** both results keep the same terms, and their coefficients agree within 1%

### Requirement: Ensemble sparse identification
`sparse_identification` SHALL take `ensemble`, a number of models fitted on resampled samples or on libraries with
terms dropped, and SHALL then set `inclusion`, the share of the models that keep every term, and fit the system with
the median coefficients of the models.

#### Scenario: Inclusion of the true terms
- **WHEN** an ensemble of 50 models runs with a fixed `random_state` on a Lorenz trajectory with noise of 5%
- **THEN** every true term has an inclusion of at least 0.9, every other term at most 0.1, and a second run with the same
  `random_state` gives the same result

#### Scenario: Ensemble size not positive
- **WHEN** `ensemble=0` is given
- **THEN** it raises `AnalysisError` that names the argument and says it is a positive integer or `None`

### Requirement: Parametric sparse identification
`sparse_identification` SHALL take `parameters`, a mapping from parameter names to one value per run, and SHALL fit a
library of products of the terms of the states and of the parameters, so coefficients depend on the parameters. The
fitted system SHALL declare every parameter, which `set_params` and the `schedule` of `simulate` change.

#### Scenario: Lorenz over rho
- **WHEN** it runs on Lorenz runs with `rho` set to 20, 28, and 35, with `parameters={'rho': [20, 28, 35]}`
- **THEN** the rate of `y` holds the term `rho x` with a coefficient within 1% of 1, and its system with `rho=24` set
  through `set_params` matches a Lorenz run at `rho=24` within 1% for one time unit

#### Scenario: Wrong number of values
- **WHEN** a parameter is given two values for three runs
- **THEN** it raises `AnalysisError` that names the parameter, the number of runs, and the number of values

### Requirement: Sparse identification with control inputs
`sparse_identification` SHALL take `inputs`, the names of variables that are control inputs, and SHALL fit the laws of
the other variables over a library of the states and inputs. Its system SHALL take every input as a bounded adaptive
parameter that its rate laws read, as the system of `mode_decomposition` does.

#### Scenario: Forced oscillator
- **WHEN** it runs on runs of a damped oscillator forced by a random piecewise-constant input `u`, with `inputs=['u']`
- **THEN** the rate of the velocity holds `u` with a coefficient within 1% of the true gain, and no rate is given for `u`

#### Scenario: Input as a control
- **WHEN** the `system` of the result is simulated with a schedule of its input parameter
- **THEN** the trajectory records the input at every frame, and the states follow the fitted laws

### Requirement: Sparse identification from experiment results
`sparse_identification` SHALL accept the result of `ensemble` or `sweep` as its source and SHALL fit one run per design
point. With `replicates='mean'` a run is the mean of the replicates of its design point, and with `replicates='each'`
every replicate is a run.

#### Scenario: Mean of stochastic replicates
- **WHEN** the result of `ensemble` over five initial states with 20 replicates each of a stochastic system is given with
  `replicates='mean'`
- **THEN** the fit uses five runs, each the mean over its 20 replicates at every frame

#### Scenario: Unknown replicates option
- **WHEN** `replicates='median'` is given
- **THEN** it raises `AnalysisError` that lists `'mean'` and `'each'`

### Requirement: Koopman eigenfunctions
The result of `mode_decomposition` SHALL hold the Koopman eigenfunctions, as coefficients over the observables of the
left eigenvectors of the one-sample map, and the relative error of every eigenfunction on the held-out pairs of
samples, `|phi(g(t + dt)) - mu phi(g(t))|` over `|phi(g(t))|`.

#### Scenario: Slow manifold
- **WHEN** it runs on runs of `x' = -0.1 x`, `y' = -(y - x^2)` with the observables `x`, `y`, and `x^2`
- **THEN** it gives three eigenfunctions, proportional to `x`, `x^2`, and `y - x^2 / 0.8`, each with an error below 1e-6

#### Scenario: Spurious eigenfunction
- **WHEN** it runs on a Duffing trajectory with polynomial observables of degree 3
- **THEN** every eigenfunction whose error is above 0.1 is marked as not valid in the result

### Requirement: Lifted systems record the original variables
With a dictionary, `mode_decomposition` SHALL fit a linear readout of the variables from the observables, and its
system SHALL declare one observable per variable through that readout, so trajectories of the system give the original
variables.

#### Scenario: Variables of a lifted run
- **WHEN** the system of a fit with `PolynomialFeatures(2)` on runs of the slow manifold is simulated
- **THEN** its trajectory records `x` and `y` as observables, within 1e-6 of a run of the true system

### Requirement: Time-delay observables
`mode_decomposition` SHALL take `delays`, a number of past samples stacked with every sample as observables, and SHALL
declare the delayed copies as states of its system, so a series of fewer variables than modes is decomposed.

#### Scenario: One sine series
- **WHEN** it runs with `delays=1` on the series `sin(t)` sampled every 0.1
- **THEN** its continuous eigenvalues are `i` and `-i` within 1e-6

#### Scenario: Negative delays
- **WHEN** `delays=-1` is given
- **THEN** it raises `AnalysisError` that names the argument

### Requirement: Dictionaries chosen by validation error
`mode_decomposition` SHALL accept a list of candidate dictionaries, fit every candidate on the training samples, keep the
one of smallest validation error, and record the validation error of every candidate.

#### Scenario: Polynomial chosen
- **WHEN** the candidates `None` and `PolynomialFeatures(2)` are given on runs of the slow manifold
- **THEN** the result keeps `PolynomialFeatures(2)`, and its validation errors hold one value per candidate

#### Scenario: No validation samples
- **WHEN** candidates are given with `validation=0`
- **THEN** it raises `AnalysisError` that says the choice needs a positive validation share

### Requirement: Many-body interaction laws by cluster expansion
`cluster_expansion(source, group)` SHALL learn one acceleration law shared by the members of a particle group from
trajectories with positions and velocities. The law SHALL read the velocity of a member and the relative positions and
velocities of its neighbours within `cutoff`, and SHALL be a sum of one term per body order up to `order`.

#### Scenario: Equivariance
- **WHEN** the learned law is evaluated on an environment, and on the same environment with every position and velocity
  rotated by one rotation
- **THEN** the second acceleration is the first rotated, within 1e-10, in two and in three dimensions

#### Scenario: Permutation invariance
- **WHEN** the neighbours of an environment are listed in another order
- **THEN** the acceleration is the same within 1e-12

#### Scenario: Group without velocities
- **WHEN** the group of the trajectory has positions but no velocities
- **THEN** it raises `AnalysisError` that names the group and says the law needs positions and velocities

### Requirement: Body orders are separate
The term of body order `k` SHALL be a sum over the distinct sets of `k - 1` neighbours, so it vanishes on every
environment with fewer than `k - 1` neighbours, and the result SHALL hold the contribution of every body order to the
acceleration of every sample.

#### Scenario: Pairwise collective
- **WHEN** it runs with `order=3` on runs of `bird_swarm`, whose law is pairwise
- **THEN** the relative force error on held-out runs is below 5%, and the 3-body contribution is below 1% of the
  acceleration in norm

#### Scenario: Many-body collective
- **WHEN** it runs with `order=4` on runs of `many_body_alignment_swarm` with its 3-body and 4-body terms on
- **THEN** the contribution of every body order is within 10% of the true term in norm, while a fit with `order=2` has
  a relative force error above 20%

### Requirement: Cluster expansions are fitted by Bayesian ridge
The coefficients SHALL be fitted by ridge regression whose penalty maximises the evidence of the training samples, with
a penalty that grows with the polynomial degree of every basis function, so no samples are held out to choose it. The
result SHALL hold the coefficients, the penalty, and the predictive standard deviation of the acceleration.

#### Scenario: Same data, same law
- **WHEN** it runs twice on the same runs
- **THEN** the coefficients are equal element by element

#### Scenario: Uncertainty away from the data
- **WHEN** the law is evaluated on an environment with neighbours closer than any training sample
- **THEN** the predictive standard deviation is larger than its median over the training samples

### Requirement: Learned interaction laws are group systems
The result of `cluster_expansion` SHALL hold `system`, a system of one particle group under continuous time whose
velocity follows the learned law through one interaction per body order, so `simulate`, `sweep`, and the analyses
accept it like any group system.

#### Scenario: Law and system agree
- **WHEN** the system of the result starts from a frame of a training run
- **THEN** the rate of the velocity of every member equals the acceleration the law predicts within 1e-8

#### Scenario: Free run
- **WHEN** the system learned from `bird_swarm` runs from a held-out initial state until 20 time units
- **THEN** the group stays cohesive like the true system, with the mean distance to the centre within 10% of the true
  one at the end
