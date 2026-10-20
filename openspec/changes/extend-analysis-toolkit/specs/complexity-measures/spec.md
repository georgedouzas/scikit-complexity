# Spec Delta

## Purpose

Measures the randomness, regularity, and structure of any series, symbol sequence, distribution, or field with
entropies and complexities, and the storage and transfer of information between series.

## ADDED Requirements

### Requirement: Generalized entropies of distributions
The analysis layer SHALL compute Rényi entropies of any order and Tsallis entropies of any index from samples or
probabilities, equal to the Shannon entropy in the limit of order or index 1, and their time-dependent values over an
ensemble with the Kullback-Leibler divergence to a reference law.

#### Scenario: Uniform distribution
- **WHEN** a user computes the Rényi entropies of orders 0, 0.5, 1, 2, and infinity of a uniform law on 8 outcomes
- **THEN** every order gives 3 bits within 1e-12

#### Scenario: Tsallis limit
- **WHEN** a user computes the Tsallis entropy with index 1 + 1e-9 of the occupation of a lattice gas and of the wealth
  classes of an exchange economy
- **THEN** each equals the Shannon entropy of the same probabilities in nats within 1e-6

#### Scenario: H-theorem of a relaxing ensemble
- **WHEN** a user computes the divergence to the stationary law of the ensemble histogram of a lazy Ehrenfest urn and of a
  birth-death chain started away from stationarity
- **THEN** the divergence never increases between frames by more than its sampling error, and ends below 0.01 bits

### Requirement: Regularity entropies of series
The analysis layer SHALL compute approximate, sample, multiscale, and spectral entropies of a series, with the
embedding dimension, the tolerance as a fraction of the standard deviation, and the scales as parameters.

#### Scenario: White noise
- **WHEN** a user computes the sample entropy with dimension 2 and tolerance 0.2 of 10,000 Gaussian white noise values
- **THEN** the result lies within 0.1 of 2.2 nats

#### Scenario: Periodic orbits on two domains
- **WHEN** a user computes the sample entropy of the Van der Pol limit cycle and of the logistic map at r = 3.2
- **THEN** both results lie below 0.05

#### Scenario: Multiscale signature
- **WHEN** a user computes the multiscale entropy over scales 1 to 10 of white noise and of 1/f noise
- **THEN** the entropy of white noise falls with the scale while that of 1/f noise stays within 15 percent of its
  value at scale 1

#### Scenario: Spectral entropy
- **WHEN** a user computes the normalized spectral entropy of a pure sine of a whole number of periods and of white noise
- **THEN** the sine gives less than 0.1 and the noise more than 0.9

### Requirement: Ordinal patterns span series and fields
The analysis layer SHALL compute permutation entropies of series with an embedding delay and of two-dimensional fields
with ordinal patterns of a given window, their Rényi and Tsallis forms, and the statistical complexity that places
data on the complexity-entropy plane with its bounding curves.

#### Scenario: Complexity-entropy plane
- **WHEN** a user places the logistic map at r = 4, Gaussian white noise, and a pure sine on the plane with patterns of
  length 6
- **THEN** the noise has entropy above 0.99 and complexity below 0.02, the logistic map has complexity above 0.3, and
  every point lies between the minimum and maximum complexity curves

#### Scenario: Fields of two domains
- **WHEN** a user computes the two-dimensional permutation entropy of an Ising lattice at temperature 10 and of the
  Gray-Scott spot pattern, ranking equal values at random
- **THEN** the hot lattice gives above 0.98 and the spot pattern a value lower than the hot lattice by at least 0.1

### Requirement: Information storage and transfer
The analysis layer SHALL compute active information storage of a series, and transfer entropy between series both from
symbols by plug-in counts and from continuous values by nearest-neighbour estimates, with history lengths and a lag.

#### Scenario: Periodic storage
- **WHEN** a user computes the active information storage of the repeated sequence `0011` with history 2
- **THEN** the result is 1 bit, and an independent fair coin gives below 0.01 bits

#### Scenario: Linear Gaussian transfer
- **WHEN** a user computes the continuous transfer entropy from `x` to `y` of 20,000 steps of the coupled
  autoregressive pair `y_t = 0.5 y_(t-1) + 0.4 x_(t-1) + e_t`, with `x` white
- **THEN** the result lies within 0.02 nats of the Granger value `0.5 ln(1 + 0.16)`, and the reverse direction lies
  below 0.01 nats

#### Scenario: Directed coupling of two domains
- **WHEN** a user computes transfer entropies of two logistic maps coupled one way, symbolized, and of a Rössler
  oscillator driving a detuned one, continuous with a target history of 2
- **THEN** in both the transfer from the driver exceeds the reverse transfer by a factor of at least 5
