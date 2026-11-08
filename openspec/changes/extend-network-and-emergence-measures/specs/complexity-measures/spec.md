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
