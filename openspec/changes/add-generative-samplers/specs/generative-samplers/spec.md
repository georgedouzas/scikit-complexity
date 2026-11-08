# Spec Delta

## Purpose

Lets a user learn a sampler of the Boltzmann density of a Monte Carlo system, weigh its samples into unbiased estimates
with an effective sample size and a free energy, and run the system with the learned sampler as its proposal.

## ADDED Requirements

### Requirement: Learned samplers of Boltzmann densities
`generative_sampler(system)` SHALL learn a sampler of the density `exp(-intensity * E) / Z` of the real-valued state of
the sampling interaction of a system, whose total energy `E` the interaction declares, and SHALL return a frozen result
with the trained sampler, its samples, their log importance weights, and the training losses.

#### Scenario: Free field
- **WHEN** a coupling flow is trained by its reverse divergence on `LatticePhi4System` on an 8 × 8 lattice with `λ = 0`
  and `m² = 1`
- **THEN** the relative effective sample size of 4,096 samples is above 0.5

#### Scenario: Missing energy
- **WHEN** the sampling interaction of the system declares no energy
- **THEN** `AnalysisError` is raised and names the interaction and the setting `energy`

#### Scenario: Discrete states
- **WHEN** the system is `IsingSystem`, whose spins are discrete
- **THEN** `AnalysisError` is raised and names the domain tag of the state

### Requirement: Samplers are flows or flow matching
The setting `kind` SHALL choose an affine coupling flow, a rational-quadratic spline coupling flow, or flow matching,
whose density comes from integrating its learned velocity field and its divergence. The setting `training` SHALL choose
the reverse divergence on the energy or maximum likelihood on samples, given or drawn from the system's own chain.
Flow matching SHALL train on samples only, and a reverse divergence for it SHALL raise `AnalysisError`.

#### Scenario: Flow matching on chain samples
- **WHEN** flow matching is trained on 2,000 configurations of the local chain of the free field on a 4 × 4 lattice
- **THEN** the relative effective sample size of its samples is above 0.3, and the log densities of the same samples
  computed twice agree within 1e-8

#### Scenario: Reproducible training
- **WHEN** the same sampler is trained twice with the same `random_state`
- **THEN** the two results hold equal samples and weights

### Requirement: Importance weights give unbiased estimates
The result SHALL hold the normalised weights, the effective sample size `(Σw)² / Σw²` and its share of the samples, the
estimate of the free energy `-log Z / intensity`, and, for every observable given as a function of a batch of
configurations, its weighted mean and standard error.

#### Scenario: Reweighted field moments
- **WHEN** the free-field sampler estimates the mean of `φ²` over the sites
- **THEN** the estimate lies within 3% of the exact lattice propagator value, and the free energy lies within 1% of the
  exact Gaussian value

### Requirement: The learned system proposes from the sampler
The result SHALL hold `system`, a copy of the system whose sampling interaction proposes whole configurations from the
trained sampler, and that system SHALL run through `simulate` and `sweep` like any Monte Carlo system.

#### Scenario: Flow proposals in a run
- **WHEN** the `system` of a sampler trained on `LatticePhi4System` on a 4 × 4 lattice with `λ = 3` and `m² = -1`
  runs for 2,000 units of time
- **THEN** the mean of `|φ|` over the frames agrees with that of a long local chain within three standard errors, and the
  run records the acceptance rate of the proposals
