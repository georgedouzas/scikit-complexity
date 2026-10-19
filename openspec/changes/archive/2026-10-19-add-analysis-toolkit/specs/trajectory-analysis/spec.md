# Spec Delta

## Purpose

Gives a user field-agnostic measures of complex dynamics on trajectories and data: chaos, distributions, events, time
series, information, spatial structure, and networks.

## ADDED Requirements

### Requirement: Lyapunov exponents measure chaos
The analysis layer SHALL compute the Lyapunov spectrum of continuous systems and maps from their tangent dynamics.

#### Scenario: Lorenz attractor
- **WHEN** a user computes the largest Lyapunov exponent of the Lorenz-63 model with σ = 10, ρ = 28, β = 8/3
- **THEN** the result lies within 0.05 of 0.906

#### Scenario: Logistic map
- **WHEN** a user computes the Lyapunov exponent of the logistic map at r = 4
- **THEN** the result lies within 0.01 of ln 2

### Requirement: Orbit diagrams and basins come from sweeps
The analysis layer SHALL build orbit diagrams over a parameter range and basin stability estimates over initial
conditions, from experiment sweeps.

#### Scenario: Period doubling
- **WHEN** a user builds the orbit diagram of the logistic map for r in [2.8, 4]
- **THEN** the diagram has one point per r below 3, two between 3 and 3.449, and four between 3.449 and 3.544

### Requirement: Heavy tails are fitted and tested
The analysis layer SHALL compute log-binned histograms, complementary cumulative distributions, maximum-likelihood
power-law fits with their lower cutoff and goodness of fit, comparisons with alternative distributions, Hill
estimates, and Gini coefficients.

#### Scenario: Recover an exponent
- **WHEN** a user fits 10,000 samples drawn from a power law with exponent 2.5 above 1
- **THEN** the fitted exponent lies within 0.1 of 2.5 and the goodness-of-fit p-value exceeds 0.1

#### Scenario: Gini of equal wealth
- **WHEN** a user computes the Gini coefficient of equal values
- **THEN** the result is 0

### Requirement: Events are detected in series
The analysis layer SHALL detect events, runs of a series above a threshold or relaxations that happen between drive
steps, and report their sizes, durations, and waiting times.

#### Scenario: Avalanche sizes
- **WHEN** a user detects avalanches in a series of topplings per drive step
- **THEN** each avalanche's size equals the sum of topplings over its consecutive nonzero steps

### Requirement: Time-series measures characterise memory and diffusion
The analysis layer SHALL compute autocorrelation, power spectra, mean squared displacement with its scaling exponent,
Hurst exponents by detrended fluctuation analysis, and rolling variance and autocorrelation as early-warning
indicators.

#### Scenario: Brownian motion
- **WHEN** a user computes the mean squared displacement of 1000 simulated Brownian particles
- **THEN** the scaling exponent lies within 0.05 of 1

#### Scenario: White noise memory
- **WHEN** a user computes the Hurst exponent of white noise
- **THEN** the result lies within 0.05 of 0.5

### Requirement: Information measures quantify structure
The analysis layer SHALL compute Shannon entropy, Kullback-Leibler divergence, block entropies, entropy rate, excess
entropy, mutual information, transfer entropy, permutation entropy, and Lempel-Ziv complexity, from symbols or from
series symbolised by bins, quantiles, or ordinal patterns.

#### Scenario: Periodic sequence
- **WHEN** a user computes the entropy rate and excess entropy of the repeated sequence `0011`
- **THEN** the entropy rate is 0 and the excess entropy is 2 bits

#### Scenario: Direction of information flow
- **WHEN** a user computes transfer entropy between a random binary series `x` and `y` that copies `x` with a lag of one
- **THEN** the transfer entropy from `x` to `y` is close to 1 bit and from `y` to `x` close to 0

### Requirement: Spatial and network structure is measured
The analysis layer SHALL label clusters on grids with their size distribution, compute spatial correlation functions
and box-counting dimensions, and compute degree distributions, clustering, path lengths, components, and centralities
of network snapshots.

#### Scenario: Percolation threshold
- **WHEN** a user labels clusters of site percolation on a 200 × 200 grid at occupation probabilities around 0.5927
- **THEN** the probability of a spanning cluster crosses one half within 0.02 of 0.5927

#### Scenario: Sierpinski dimension
- **WHEN** a user computes the box-counting dimension of a Sierpinski triangle image
- **THEN** the result lies within 0.05 of log 3 / log 2
