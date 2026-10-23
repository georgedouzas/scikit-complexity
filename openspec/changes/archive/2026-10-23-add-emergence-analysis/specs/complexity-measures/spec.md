# Spec Delta

## ADDED Requirements

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
