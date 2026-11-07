# Spec Delta

## ADDED Requirements

### Requirement: Networks built from new topology requests
The library SHALL provide in `mathematics/networks` `explosive_percolation` (Achlioptas, D'Souza, and Spencer 2009),
`davidsen_ebel_bornholdt` and `node_merging` (Thurner et al. 4.5.4.6), `duplication_deletion` (Jensen 8.5.1), and
`network_superposition` (Thurner et al. 4.5.2), whose linking probability is drawn once per Erdős-Rényi subgraph, the
reading that gives Eq. 4.34.

#### Scenario: Abrupt giant component
- **WHEN** `explosive_percolation` adds edges by the product rule to 2000 nodes
- **THEN** its largest component jumps from below 10 percent to above half of the nodes between 0.8 and 0.95 edges per
  node, where Erdős-Rényi graphs already hold most nodes at 0.8

#### Scenario: Turnover sets the degree
- **WHEN** `davidsen_ebel_bornholdt` reaches its steady state with turnover `gamma`
- **THEN** its mean degree is within 15 percent of `1 + 1 / gamma` and its clustering is above 0.2

#### Scenario: Merging makes hubs
- **WHEN** `node_merging` runs for 20 steps per node
- **THEN** its largest degree exceeds ten times the mean degree, far above the largest degree of an Erdős-Rényi graph
  of the same size and mean

#### Scenario: Steady connectance
- **WHEN** `duplication_deletion` runs for 40 steps per node
- **THEN** its mean connectance is within 15 percent of Eq. 8.84

#### Scenario: Superposed degrees
- **WHEN** `network_superposition` draws power-law linking probabilities for 1000 subgraphs of 100 nodes
- **THEN** its degree distribution is within a total variation distance of 0.05 of Eq. 4.34

### Requirement: Links regulated by several signed regulators
The library SHALL provide `spatial_triadic_percolation` (Millán et al. 2023, arXiv 2311.14877, Methods, Eqs. 1-2) on a
torus, whose hyperedges list the ends of a link, its negative regulators, and then its positive regulators, whose
numbers the link holds, and whose links are active with probability `p` when no negative regulator is active and a
positive regulator is active, or no positive regulator exists.

#### Scenario: Signed regulators
- **WHEN** `spatial_triadic_percolation` runs one step with `p = 1` on the NumPy and Numba backends
- **THEN** every link is active exactly when none of its negative regulators and one of its positive regulators, if it
  has any, were active

#### Scenario: Blinking patterns
- **WHEN** `spatial_triadic_percolation` runs with `c+ = c- = 0.2`, `d0 = dr = 0.25`, `c = 0.4`, and `p = 0.8`
- **THEN** the giant component keeps changing size, the active nodes of consecutive steps are anti-correlated, and the
  activity of a node correlates with that of the nodes within `d0` of it
