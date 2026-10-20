# Spec Delta

## Purpose

Measures signatures of criticality in any activity series, relaxation record, or ensemble over a control parameter:
avalanches, crackling-noise scaling, branching ratios, order-parameter exponents, and susceptibility peaks.

## ADDED Requirements

### Requirement: Avalanches are measured and their laws fitted
The analysis layer SHALL extract avalanches from an activity series or from relaxation records, with their sizes,
durations, and mean shapes, and SHALL fit their size and duration laws with a power-law fit compared against
exponential, log-normal, and truncated power-law alternatives.

#### Scenario: Same avalanche statistics on a sandpile and an earthquake model
- **WHEN** a user measures 10^5 avalanches of a 128 × 128 Bak-Tang-Wiesenfeld sandpile and of a 64 × 64
  Olami-Feder-Christensen model with `alpha = 0.2` and open edges
- **THEN** the sandpile size exponent lies between 1.15 and 1.35, the earthquake size exponent between 1.7 and 2.1, and
  in both the power law is preferred over the exponential with a p-value below 0.05

#### Scenario: Mean-field branching
- **WHEN** a user measures 10^5 avalanches of a critical Galton-Watson process with Poisson offspring of mean 1
- **THEN** the size exponent lies within 0.05 of 3/2 and the duration exponent above duration 10 within 0.1 of 2

### Requirement: Crackling-noise scaling is checked
The analysis layer SHALL fit the exponent of the mean size against duration, compare it with the value
`(tau_T - 1) / (tau_S - 1)` from the fitted laws, and collapse the mean avalanche shapes of several durations.

#### Scenario: Branching process scaling
- **WHEN** a user checks the scaling of the critical Galton-Watson avalanches, fitting the mean size over durations of
  at least 10
- **THEN** the fitted exponent lies within 0.1 of 2, agrees with the exponent from the laws within 0.15, and the
  mean shapes of durations 20, 40, and 80 collapse onto one parabola within 10 percent on average

### Requirement: Branching ratios are estimated
The analysis layer SHALL estimate the branching ratio of an activity series by the multistep regression estimator,
which stays unbiased when only a fraction of the units is observed.

#### Scenario: Subsampled branching network
- **WHEN** a user estimates the branching ratio of a branching network with ratio 0.98 and external drive, from all
  units and from 5 percent of them
- **THEN** both estimates lie within 0.01 of 0.98

#### Scenario: Neuronal and epidemic activity
- **WHEN** a user estimates the branching ratio of the spike counts of a critical stochastic neural network and of the
  daily cases of a subcritical stochastic epidemic with reproduction number 0.9
- **THEN** the network gives within 0.02 of 1 and the epidemic within 0.03 of 0.9

### Requirement: Order parameters and susceptibilities locate transitions
The analysis layer SHALL compute the mean order parameter, the overlap between replicas, the susceptibility from
fluctuations, and the Binder cumulant over a sweep or over disorder samples, locate the susceptibility peak, and fit the
order-parameter exponent near the transition.

#### Scenario: Same susceptibility peak on two transitions
- **WHEN** a user sweeps a 64 × 64 Ising lattice over temperature and 2000 Kuramoto oscillators with Lorentzian
  frequencies of width 1 over coupling
- **THEN** the Ising peak lies within 3 percent of `2 / ln(1 + sqrt 2)`, and the Kuramoto peak within 10 percent of
  `K_c = 2`

#### Scenario: Order-parameter exponent
- **WHEN** a user fits the order-parameter exponent of the mean-field Ising model and of the Kuramoto model with
  Lorentzian frequencies just above their transitions
- **THEN** both exponents lie within 0.05 of 1/2

#### Scenario: Replica overlap
- **WHEN** a user computes the overlap of two replicas of a 32 × 32 Ising lattice run with different seeds,
  at temperature 1.5 and 5
- **THEN** the mean absolute overlap exceeds 0.9 at 1.5 and lies below 0.1 at 5
