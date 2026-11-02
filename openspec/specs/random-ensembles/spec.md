# random-ensembles Specification

## Purpose
Lets a user reduce over the members of a group without overflow in log space and observe the contact clusters of random
points.

## Requirements

### Requirement: Log-sum-exp reduces over members without overflow
The law utilities SHALL give `log_sum_exp`, the logarithm of the sum of the exponentials of an expression over the
members of a group, computed without overflow for values of any size on the NumPy and Numba backends.

#### Scenario: Large values
- **WHEN** `log_sum_exp` reduces values of order 10^4 over 1000 members
- **THEN** the result is finite and equals the largest value plus a correction of at most `ln 1000`

#### Scenario: Random energy model entropy
- **WHEN** the entropy per spin `(ln Z + U/T)/N` of 2^20 Gaussian energies of variance `N/2` is written with
  `log_sum_exp` and averaged over 10 replicates of an `Experiment`
- **THEN** it is within 0.02 of `ln 2 - 1/(4 T^2)` at `T = 1` and `T = 2`, and below 0.1 at `T = 0.3`, below the glass
  temperature `1/(2 sqrt(ln 2))`

### Requirement: Contact clusters of points are analysed
`ClusterAnalysis` SHALL read, with a contact `radius`, the positions of the points of a group, and SHALL give the sizes
of the clusters of points closer than twice the radius, the percolation measures of the largest cluster, and whether a
cluster joins the two opposite faces of the space along an axis.

#### Scenario: Spanning probability
- **WHEN** disks of radius 1 are drawn in a square of side 60 at filling factors 1.0 and 1.25, over 50 replicates each
- **THEN** the share of replicates with a spanning cluster is below 0.1 at 1.0 and above 0.9 at 1.25

#### Scenario: Isolated points
- **WHEN** `ClusterAnalysis(kind='sizes', radius=0.1)` reads points on a grid of spacing 1
- **THEN** every cluster has size 1
