## ADDED Requirements

### Requirement: Bootstrap significance of the causal emergence gain
The causal emergence of a matrix of transition counts SHALL, when resamples are asked for, resample the observed
transitions with a seed, recompute the gain of the same partition on every resample, and return the resampled gains,
their percentile interval at a given confidence, and the p-value, the fraction of resampled gains at or below zero.

#### Scenario: Emergent chain
- **WHEN** a user measures the causal emergence of the eight-state chain of Hoel, Albantakis, and Tononi (2013) from
  counts of 200 transitions per state with 1000 resamples
- **THEN** the interval at 0.95 lies above 0 and the p-value is below 0.05

#### Scenario: No emergence
- **WHEN** a user measures the gain of the partition `[0, 0, 1, 1]` of a four-state chain with uniform rows from counts
  of 200 transitions per state with 1000 resamples
- **THEN** the p-value is above 0.05

#### Scenario: Probabilities instead of counts
- **WHEN** a user asks for resamples of a matrix whose rows are probabilities that are not whole numbers
- **THEN** an `AnalysisError` asks for the transition counts

#### Scenario: Same seed
- **WHEN** a user resamples one matrix of counts twice with the same seed
- **THEN** both results hold the same resampled gains

### Requirement: Causal emergence over a hierarchy of scales
`emergence(kind='scales')` SHALL compute the causal primitives of a transition matrix at every scale of a path of
nested, dynamically consistent partitions from the micro scale, given or searched, and return the value of every
scale, the contribution of every step, the total causal emergence, and the emergent complexity, after Hoel (2025, arXiv
2503.13395).

#### Scenario: Chain of the paper
- **WHEN** a user measures the eight-state chain of Figure 2 of the paper with determinism and specificity
- **THEN** the total causal emergence is 0.33 within 0.01

#### Scenario: Equal gains
- **WHEN** a path of four steps gives four equal positive gains
- **THEN** the emergent complexity is 2 bits within 1e-12

#### Scenario: Mesoscale structure
- **WHEN** a user measures the two chains of Figure 3 of the paper, without and with mesoscale structure
- **THEN** the emergent complexity of the chain with mesoscale structure is the larger

#### Scenario: Inconsistent partition in a given path
- **WHEN** a user gives a path whose second partition is not consistent with the micro chain
- **THEN** it raises `AnalysisError`, whose message names the step of the path

### Requirement: Synergy capacity and coalitions of a group
`emergence(kind='synergy-capacity')` SHALL return the median over pairs of agents of the synergy that their present
states hold about their joint future, and `emergence(kind='coalition')` SHALL return the median over triplets of the
information that a triplet holds about a macro signal minus the largest information of its pairs, after Riedl (2025,
arXiv 2510.05174).

#### Scenario: Exclusive-or dynamics
- **WHEN** a user measures two random binary agents whose next states both equal the exclusive-or of their present
  states
- **THEN** the synergy capacity exceeds 0.9 bits

#### Scenario: Independent copies
- **WHEN** a user measures four binary agents that copy their own state with noise
- **THEN** the synergy capacity lies below 0.05 bits

#### Scenario: Three-way macro signal
- **WHEN** a user measures the coalition gain of three random bits whose macro signal at the next step is their
  exclusive-or
- **THEN** the median gain exceeds 0.9 bits, and it lies below 0.05 bits when the macro signal is the first bit

### Requirement: Surrogate tests keep chosen structure
`surrogate_test` SHALL take `method`: `'shuffle'`, the independent shuffles of every variable and the default, `'shift'`,
a circular shift of every variable by its own random lag, `'swap'`, a permutation of the variables at every time, and
`'block'`, shuffles of blocks of a given length.

#### Scenario: Shifts keep autocorrelation
- **WHEN** a user draws shift surrogates of an autoregressive series of coefficient 0.9
- **THEN** the lag-one autocorrelation of every surrogate equals that of the data within 0.01

#### Scenario: Swaps keep each time
- **WHEN** a user draws swap surrogates of three variables
- **THEN** the sorted values at every time equal those of the data

#### Scenario: Default unchanged
- **WHEN** a user runs a surrogate test without `method` and with a seed
- **THEN** the result equals the result of the same call before this change
