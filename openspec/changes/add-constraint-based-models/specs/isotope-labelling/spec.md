# Spec Delta

## Purpose

Lets a user attach atom transitions to the processes of a constraint network, simulate how labelled atoms of a tracer
spread through the network at metabolic and isotopic steady state, and estimate the fluxes from measured labelling
patterns and measured fluxes, with confidence intervals and a goodness-of-fit test.

## ADDED Requirements

### Requirement: Processes declare atom transitions
The modelling layer SHALL let a process of a constraint network declare an atom map, which pairs each reactant and
product occurrence with a string of atom letters, so that every traced atom of a product names the reactant atom it
comes from. A pool SHALL declare its number of traced atoms through its maps and MAY be declared symmetric. Invalid
maps SHALL raise a named error that names the process and the offending pool or letter.

#### Scenario: Valid map
- **WHEN** a process consumes `AcCoA` with letters `ab` and `OAC` with letters `cdef` and produces `Cit` with letters
  `fedbac`
- **THEN** the network accepts it, and records that atom 1 of `Cit` comes from atom 4 of `OAC` and atom 5 of `Cit` from
  atom 1 of `AcCoA`

#### Scenario: Unbalanced letters
- **WHEN** a product names a letter that no reactant carries
- **THEN** construction raises a named validation error that names the process and the letter

#### Scenario: Inconsistent atom count
- **WHEN** one process gives `Glu` five letters and another gives it four
- **THEN** construction raises a named validation error that names the pool and both processes

### Requirement: Labelling is simulated at isotopic steady state
The library SHALL compute the mass isotopomer distributions of chosen fragments of pools, given the fluxes, including the
forward and backward fluxes of reversible processes, and the isotopomer fractions of the tracer pools. It SHALL use the
elementary metabolite unit decomposition, solving one sparse linear system per unit size, and SHALL return
distributions whose entries are non-negative and sum to 1.

#### Scenario: Published TCA example
- **WHEN** the TCA example network of Antoniewicz et al. (2007) runs with `v1 = v2 = 100`, `v3 = v4 = v5 = v8 = v9 = 50`,
  fumarase forward 125 and backward 75, an acetyl-CoA tracer of 50% unlabelled, 25% [2-13C], and 25% [1,2-13C], and
  unlabelled aspartate
- **THEN** the mass isotopomer distribution of glutamate atoms 1–5 is `[0.3464, 0.2695, 0.2708, 0.0807, 0.0286,
  0.0039]` within 5e-5, as published, and matches mfapy within 1e-10

#### Scenario: Unlabelled tracer
- **WHEN** every tracer pool is unlabelled
- **THEN** every simulated distribution is `[1, 0, ..., 0]` within 1e-12

#### Scenario: Symmetric pool
- **WHEN** succinate is declared symmetric and labelled only at atom 1 by its producing process
- **THEN** the fumarate it feeds carries half of that label at atom 1 and half at atom 4

### Requirement: Fluxes are estimated from labelling data
The library SHALL estimate the fluxes of a labelling experiment by weighted least squares over measured mass isotopomer
distributions and measured fluxes, with one or more parallel tracer experiments. It SHALL parametrise the fluxes by the
free net and exchange fluxes, which keeps every balance exact. It SHALL start from several seeded feasible starts and
SHALL report the best fit, the residual sum of squares, and the simulated distributions.

#### Scenario: Noise-free data
- **WHEN** the TCA example is fitted to the noise-free distributions of glutamate from two parallel experiments, 50%
  [1-13C] and 50% [1,2-13C] acetyl-CoA, generated with the true fluxes and `v1` fixed at 100
- **THEN** every net flux is recovered within 1e-4 relative error, and the residual sum of squares is below 1e-10

#### Scenario: Reproducible starts
- **WHEN** the same estimation runs twice with the same seed
- **THEN** both runs return the same estimates and the same residual sum of squares

#### Scenario: Data that cannot be fitted
- **WHEN** a measured flux contradicts the steady-state balances
- **THEN** the fit reports a residual sum of squares above the acceptance range of the goodness-of-fit test, and the
  result says that the fit is rejected

### Requirement: Estimates have confidence intervals and a goodness-of-fit test
An estimation SHALL report, for every net and exchange flux, a 95% confidence interval from the linearised covariance by
default and from the profile of the residual sum of squares on request. It SHALL test the fit against the chi-square
distribution with the number of measurements minus the number of free fluxes as its degrees of freedom.

#### Scenario: Noisy data
- **WHEN** the TCA example is fitted to distributions with Gaussian noise of standard deviation 0.002 at seed 0
- **THEN** every net flux's 95% profile interval contains its true value, and the chi-square test accepts the fit at
  the 5% level

#### Scenario: Unidentifiable exchange
- **WHEN** the fumarase exchange flux has no effect on the measured distributions within the fit tolerance
- **THEN** its profile interval reaches a bound of the process, and the result marks it as not identifiable

### Requirement: Estimation runs as a simulation
A labelling experiment SHALL compile to an estimation problem whose run records one frame per accepted iteration of the
fit, in the order of its starts, and a last frame with the best estimate. Each frame SHALL record the fluxes, the
residual sum of squares, the start, and the simulated distributions. The last frame SHALL also record the intervals
and the test.

#### Scenario: Iterations as frames
- **WHEN** a user runs an estimation with 3 starts through `Simulation`
- **THEN** the trajectory's axis is `iteration`, the residual sum of squares never increases within a start, and the
  last frame holds the estimate that the analysis function returns for the same seed
