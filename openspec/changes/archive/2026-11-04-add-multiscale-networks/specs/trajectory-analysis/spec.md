# Spec Delta

## ADDED Requirements

### Requirement: Scaling laws across sizes
`ScalingAnalysis(kind='allometric')` SHALL fit the exponent of a quantity against system size across a set of runs,
read as a mapping from size to samples, and SHALL set the exponent with a bootstrap confidence interval.

#### Scenario: Known exponent
- **WHEN** a quantity generated as `N^1.15` times lognormal noise across 50 sizes is fitted
- **THEN** the 95% confidence interval contains 1.15
