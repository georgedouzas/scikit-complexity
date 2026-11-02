# complexity-measures Specification

## Purpose
Measures the randomness, regularity, and structure of any series, symbol sequence, distribution, or field with
entropies and complexities, and the storage and transfer of information between series.

## Requirements

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

### Requirement: Higher-order information of many variables
The analysis SHALL compute total correlation, dual total correlation, O-information, and the O-information gradient of each variable, for discrete data and with Gaussian estimators.

#### Scenario: Synergy of XOR
- **WHEN** the O-information of X, Y, and Z = X xor Y with fair independent bits is computed
- **THEN** it equals -1 bit within the estimator error, a synergistic system

#### Scenario: Redundancy of copies
- **WHEN** the O-information of three copies of one fair bit is computed
- **THEN** it equals +1 bit, a redundant system

### Requirement: Partial information decomposition
The analysis SHALL decompose the information two sources give about a target into redundant, unique, and synergistic parts, with the redundancy measure as a keyword, wrapping `dit`.

#### Scenario: AND gate
- **WHEN** the decomposition of Z = X and Y with fair bits uses the minimum mutual information redundancy
- **THEN** redundancy is about 0.311 bits, synergy 0.5 bits, and unique parts are zero

### Requirement: Local information measures
The analysis SHALL return local, pointwise entropy, mutual information, and transfer entropy, whose means equal the global measures.

#### Scenario: Local means
- **WHEN** local mutual information is computed on a long sample of correlated Gaussians
- **THEN** its mean equals the global mutual information within 1e-9

### Requirement: Emergence criteria of a macro feature
The analysis SHALL compute the ΦID criteria Ψ, Δ, and Γ of a macro feature of a micro system, and SHALL test the closure of a coarse-graining by comparing how well macro and micro series predict the macro future.

#### Scenario: Emergent parity
- **WHEN** the parity of many noisy bits that persist as a whole is the macro feature, as in Rosas et al. (2020)
- **THEN** Ψ is positive

#### Scenario: Closed coarse-graining
- **WHEN** the closure of a lumpable chain's block coarse-graining is tested
- **THEN** the macro series is closed within the estimator error

### Requirement: Epsilon-machines of discrete series
The analysis SHALL reconstruct the ε-machine of a discrete series, with its causal states, transitions, statistical complexity, and entropy rate.

#### Scenario: Golden mean process
- **WHEN** the ε-machine of a long sample of the golden mean process is reconstructed
- **THEN** it has two causal states and a statistical complexity within 0.01 of 0.918 bits

### Requirement: Partitions coarse-grain states and transitions
The analysis SHALL coarse-grain discrete micro states into macro states by a partition, given as an integer label per
state or a function of the state, and SHALL apply it to discrete trajectories and to transition matrices. The macro
transition matrix SHALL aggregate micro transitions within each block, weighted by the uniform or the stationary
distribution over the block as a keyword selects, and every macro row SHALL sum to one.

#### Scenario: Lumpable chain
- **WHEN** a four-state chain whose states {0, 1} and {2, 3} move as blocks is coarse-grained by the partition
  `[0, 0, 1, 1]`
- **THEN** the macro transition matrix equals the block transition matrix within 1e-12, with either weighting

#### Scenario: Partition of a trajectory
- **WHEN** a discrete trajectory is coarse-grained by a labelling function
- **THEN** each frame holds the label of its micro state, and the macro matrix estimated from it equals the
  coarse-graining of the micro matrix within the sampling error

### Requirement: Effective information of transitions
The analysis SHALL compute the effective information of a transition matrix, the mutual information in bits between a
uniform intervention on the current state and the next state, together with its determinism and degeneracy, so that the
effective information equals determinism minus degeneracy. States without observed transitions SHALL be dropped with a
named warning that names them.

#### Scenario: Deterministic permutation
- **WHEN** the effective information of an `n`-state permutation matrix is computed
- **THEN** it equals `log2 n` within 1e-12, with determinism `log2 n` and degeneracy 0

#### Scenario: Fully noisy chain
- **WHEN** the effective information of a matrix whose rows are all uniform is computed
- **THEN** it is 0 within 1e-12

### Requirement: Causal emergence of a coarse-graining
The analysis SHALL report the causal emergence of a partition as the effective information of the macro transition
matrix minus that of the micro one, and SHALL search partitions by greedily merging the pair of blocks with the largest
gain until no merge increases it, returning the partition and its gain. Below a documented state count it SHALL offer an
exhaustive search.

#### Scenario: Emergence from noisy micro states
- **WHEN** an eight-state chain whose first seven states move uniformly among themselves and whose eighth state is
  absorbing is analysed, as in Hoel, Albantakis, and Tononi (2013)
- **THEN** the micro effective information is `3 - (7/8) log2 7 ≈ 0.544` bits, the partition that merges the first
  seven states has a macro effective information of 1 bit and so a causal emergence of about 0.456 bits, and the greedy
  search returns that partition

#### Scenario: No emergence in a permutation
- **WHEN** the greedy search runs on a permutation matrix
- **THEN** it returns the identity partition with zero gain
