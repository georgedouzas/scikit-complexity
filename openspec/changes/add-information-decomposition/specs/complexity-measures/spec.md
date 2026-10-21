# Spec Delta

## ADDED Requirements

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
