## ADDED Requirements

### Requirement: Metropolis sampling accepts independence proposals of whole configurations
`MetropolisSolver` SHALL take a `ProposalSpec` of `skcomplex.core.engine.solvers` as the move of a state in its `moves`,
which samples whole configurations of the state with their log densities `log q`, `hits` times per unit of time. Each
proposal SHALL be accepted with probability `min(1, exp(-(E' - E) / T) q(x) / q(x'))`, where `T` is the temperature of
the solver and `E` is the total energy, the sum over members of the `total` of the objective law. Preparing a system
whose proposal move has no `total` to read, or whose rule is not Metropolis, SHALL raise `ValidationError`.

#### Scenario: Exact proposal
- **WHEN** the free field on a 4 × 4 lattice is sampled with the exact Gaussian density as its proposal
- **THEN** every proposal is accepted, and the run records an acceptance rate of 1

#### Scenario: Detailed balance with a wrong proposal
- **WHEN** the free field is sampled with a Gaussian proposal twice as wide as the exact density for 5,000 units of
  time
- **THEN** the mean of `φ²` over the sites and frames lies within three standard errors of the exact value

#### Scenario: Missing total energy
- **WHEN** a system whose solver has a proposal move states an objective law without `total`
- **THEN** preparing the system raises `ValidationError` naming the field `total`

### Requirement: Total energies are given per member
`ObjectiveLaw` SHALL take `total`, an expression of the namespace of a member whose sum over the members is the total
objective, and a run SHALL record the total energy as `<interaction label>.energy` when it is given. The local
objective SHALL keep scoring local moves.

#### Scenario: Energy of the free field
- **WHEN** the free field is run at temperature 1 with a given `total` for 1,000 units of time
- **THEN** the mean recorded energy per site lies within 2% of 1/2, as equipartition gives for a quadratic action
