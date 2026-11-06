# Spec Delta

## ADDED Requirements

### Requirement: Multilevel selection with pairwise group conflict
The library SHALL provide the density of group compositions of defectors and altruistic punishers under within-group
replicator dynamics and pairwise between-group conflict with a victory probability that is not additively separable,
the group Fermi rule, the local rule normalized by the payoffs of both groups, or the Tullock contest function, as
`nonlinear_pairwise_group_conflict` under `social/cooperation` (Cooney 2024, arXiv 2405.18419 §3.1 and §6). The
inventory SHALL record it as implemented.

#### Scenario: Strong between-group conflict sustains punishers
- **WHEN** the pairwise conflict with the group Fermi rule runs from a uniform density with `lambda = 2` and with
  `lambda = 0.1`
- **THEN** the mean fraction of punishers ends higher with `lambda = 2` than with `lambda = 0.1`, and the total stays
  one

### Requirement: Trimorphic multilevel selection on the simplex
The library SHALL provide the density on the 2-simplex of cooperators, defectors, and altruistic punishers with the
victory probability of the fraction of non-defectors or of the normalized payoff difference, as
`trimorphic_multilevel_selection` under `social/cooperation` (arXiv 2405.18419 §3.2 and §7). The inventory SHALL record
it as implemented.

#### Scenario: Trimorphic selection lowers defection
- **WHEN** the trimorphic model with the victory probability of non-defectors runs from a uniform density on the
  simplex
- **THEN** the mean fraction of defectors decreases, and the total stays one

### Requirement: Size-structured forest with shading and dispersal
The library SHALL provide the McKendrick-von Foerster density of trees over position and height, with growth slowed by
Beer-Lambert shading from taller trees, mortality from canopy and resource competition, and recruitment at the
sapling height from seeds dispersed over a strip with uncolonized land beyond its edges, as `forest_size_structure`
under `biology/ecology` (Grilletta et al. 2026, arXiv 2608.11918 Eq. 9, 10, and 16). The inventory SHALL record it as
implemented.

#### Scenario: Edges deplete small trees
- **WHEN** the forest runs on a strip towards its stationary state
- **THEN** the density of trees is lower at the edges than in the bulk, and smaller trees are depleted at the edges by a
  larger fraction than taller trees
