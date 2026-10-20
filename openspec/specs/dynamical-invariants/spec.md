# dynamical-invariants Specification

## Purpose
Measures the invariants of deterministic and noisy dynamics from a system or from recorded series: embeddings, fractal
dimensions, Lyapunov exponents, entropy rates of chaos, invariant densities, and recurrences.

## Requirements

### Requirement: Series are embedded in delay coordinates
The analysis layer SHALL build delay embeddings of a series, choose the delay at the first minimum of the auto mutual
information, and choose the dimension where the fraction of false nearest neighbours falls below a threshold.

#### Scenario: Lorenz embedding
- **WHEN** a user embeds 20,000 samples of the Lorenz-63 `x` with step 0.01
- **THEN** the chosen delay lies between 0.10 and 0.20 time units, and the chosen dimension is 3

#### Scenario: Map embedding
- **WHEN** a user embeds 10,000 iterates of the Hénon map `x`
- **THEN** the chosen dimension is 2

### Requirement: Fractal dimensions of attractors and point sets
The analysis layer SHALL compute the correlation dimension of any point set, embedded series or agent positions, with
the correlation sums over a range of radii and the scaling region used for the fit.

#### Scenario: Lorenz attractor
- **WHEN** a user computes the correlation dimension of 20,000 points of the Lorenz-63 attractor
- **THEN** the result lies within 0.1 of 2.05

#### Scenario: Hénon attractor
- **WHEN** a user computes the correlation dimension of 20,000 points of the Hénon attractor
- **THEN** the result lies within 0.05 of 1.21

#### Scenario: Agents in a plane
- **WHEN** a user computes the correlation dimension of 5000 agents placed uniformly in a unit square
- **THEN** the result lies within 0.05 of 2

### Requirement: Invariants follow from the Lyapunov spectrum
The analysis layer SHALL derive the Kaplan-Yorke dimension and the Kolmogorov-Sinai entropy, the sum of the positive
exponents, from a Lyapunov spectrum, and SHALL estimate the largest exponent from a recorded series alone.

#### Scenario: Kolmogorov-Sinai entropy of a map and a flow
- **WHEN** a user derives the Kolmogorov-Sinai entropy of the logistic map at r = 4 and of the Lorenz-63 model
- **THEN** the map gives ln 2 within 0.01 and the flow 0.906 within 0.05

#### Scenario: Kaplan-Yorke dimension
- **WHEN** a user derives the Kaplan-Yorke dimension of the Hénon map and of the Lorenz-63 model
- **THEN** the map gives 1.26 within 0.02 and the flow 2.06 within 0.03

#### Scenario: Largest exponent from data
- **WHEN** a user estimates the largest exponent from 10,000 iterates of the logistic map at r = 4 and from 6000
  samples of the Lorenz-63 `x` every 0.05, leaving the first 0.5 time units of divergence out of the fit
- **THEN** the map gives ln 2 within 0.1 and the flow 0.906 within 0.15

### Requirement: Invariant densities are measured and coarse-grained
The analysis layer SHALL estimate the invariant density of a map from a long run, and SHALL build the coarse-grained
Markov chain of a map on a partition by Ulam's method, whose stationary vector approximates that density.

#### Scenario: Logistic map at r = 4
- **WHEN** a user estimates the invariant density of 10^6 iterates of the logistic map at r = 4 and its Ulam chain
  on 100 cells
- **THEN** both lie within 0.05 in total variation of `1 / (pi sqrt(x (1 - x)))`

### Requirement: Recurrences are quantified
The analysis layer SHALL build recurrence matrices of embedded series with a fixed radius or a fixed recurrence rate,
and SHALL compute recurrence rate, determinism, laminarity, mean and longest diagonal lines, trapping time, and the
entropy of diagonal line lengths.

#### Scenario: Order and noise
- **WHEN** a user quantifies 2000 samples of a sine embedded in 2 dimensions, of the Lorenz-63 `x` embedded in 3
  dimensions, and of white noise, at a recurrence rate of 5 percent
- **THEN** the sine has determinism above 0.95, the noise below 0.3, and the Lorenz series lies between them

#### Scenario: Intermittency on two domains
- **WHEN** a user quantifies the logistic map just below the period-3 window at r = 3.8282 and the Lorenz-63 model
  just above its intermittency threshold at ρ = 166.2, each sampled once per laminar orbit, every third iterate of the
  map and every second maximum of `z` of the flow
- **THEN** each has a laminarity higher than the same model at r = 4 and at ρ = 28
