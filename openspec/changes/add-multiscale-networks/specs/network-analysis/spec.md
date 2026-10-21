# Spec Delta

## ADDED Requirements

### Requirement: Networks renormalize across scales
The analysis SHALL renormalize a network by its Laplacian or by its geometry, returning the network at each scale with its degree distribution and clustering.

#### Scenario: Self-similar network
- **WHEN** a popularity-similarity network is renormalized geometrically three times
- **THEN** its degree distribution exponent stays within 0.2 of the original at every scale

### Requirement: Scaling laws across sizes
The analysis SHALL fit a quantity against system size across a sweep of runs, returning the exponent with a bootstrap confidence interval.

#### Scenario: Known exponent
- **WHEN** a quantity generated as `N^1.15` times lognormal noise across 50 sizes is fitted
- **THEN** the confidence interval contains 1.15
