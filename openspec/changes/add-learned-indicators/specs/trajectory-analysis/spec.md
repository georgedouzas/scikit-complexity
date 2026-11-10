# Spec Delta

## ADDED Requirements

### Requirement: Rolling indicators of Dakos et al.
`rolling` SHALL take the kinds `'kurtosis'`, `'return-rate'`, `'standard-deviation'`, and `'coefficient-of-variation'`,
and `bandwidth`, which subtracts a Gaussian kernel smoothing of the whole series of that bandwidth before the windows,
after Dakos et al. (2012).

#### Scenario: Return rate of an autoregression
- **WHEN** the rolling return rate of 20000 steps of `x(t) = 0.8 x(t-1) + e` is computed in windows of 2000
- **THEN** every value is within 5% of 1.25

#### Scenario: Kurtosis of Gaussian noise
- **WHEN** the rolling kurtosis of 20000 seeded Gaussian draws is computed in windows of 2000
- **THEN** every value is within 0.3 of 3

#### Scenario: Slow trend removed
- **WHEN** 20000 steps of seeded unit white noise plus a sine of amplitude 5 and period 5000 steps are analysed for
  lag-one autocorrelation in windows of 500 with `bandwidth=200`
- **THEN** every rolling value lies within 0.1 of 0, while without `bandwidth` the values exceed 0.5

#### Scenario: Defaults unchanged
- **WHEN** `rolling` runs with an existing kind and no `bandwidth`
- **THEN** its result equals the result before this change within 1e-12
