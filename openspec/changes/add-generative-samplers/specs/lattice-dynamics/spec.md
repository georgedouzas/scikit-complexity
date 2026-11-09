## ADDED Requirements

### Requirement: Monte Carlo accepts independence proposals of whole configurations
`SamplingInteraction` SHALL take a `ProposalSpec` as its move, which samples whole configurations of the state with
their log densities `log q`, `hits` times per unit of time. Each proposal SHALL be accepted with probability
`min(1, exp(-intensity * (E' - E)) q(x) / q(x'))`, where `E` is the total energy, the sum over sites of the declared
`energy` of a site. A proposal move without an energy, or with a rule other than Metropolis, SHALL raise
`ValidationError`.

#### Scenario: Exact proposal
- **WHEN** the free field on a 4 × 4 lattice is sampled with the exact Gaussian density as its proposal
- **THEN** every proposal is accepted, and the run records an acceptance rate of 1

#### Scenario: Detailed balance with a wrong proposal
- **WHEN** the free field is sampled with a Gaussian proposal twice as wide as the exact density for 5,000 units of
  time
- **THEN** the mean of `φ²` over the sites and frames lies within three standard errors of the exact value

#### Scenario: Missing energy
- **WHEN** a sampling interaction with a proposal move declares no energy
- **THEN** preparing the system raises `ValidationError` naming the setting `energy`

### Requirement: Total energies are declared per site
`SamplingInteraction` SHALL take `energy`, a function of the namespace of a site whose sum over the sites is the total
objective, and a run SHALL record the total energy as `<label>.energy` when it is declared. The local objective SHALL
keep scoring local moves.

#### Scenario: Energy of the free field
- **WHEN** the free field is run at intensity 1 with a declared energy for 1,000 units of time
- **THEN** the mean recorded energy per site lies within 2% of 1/2, as equipartition gives for a quadratic action
