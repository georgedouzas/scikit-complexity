## ADDED Requirements

### Requirement: Stochastic outcome of the Deffuant encounters
`DeffuantSystem` SHALL take `steepness`. With `None` it SHALL keep the deterministic threshold. With a number `rho` an
encounter SHALL move the pair with the probability `1 / (1 + exp(-rho * (confidence - |x_u - x_v|)))`, after Lenti et
al. (2023, arXiv 2310.02766, Eq. 1), drawn by a named draw that runs record.

#### Scenario: Large steepness
- **WHEN** the system runs with steepness 1e4 and with `None` from the same seed for 50 steps
- **THEN** the final opinions agree within 1e-6

#### Scenario: Recorded outcomes
- **WHEN** the system runs with steepness 20 and `record_draws=True`
- **THEN** the trajectory holds one Bernoulli outcome per encounter, with its probability

### Requirement: Off-lattice tumour growth
The library SHALL provide `OffLatticeTumourGrowthSystem` in `biology/oncology`, the biphasic cell-based model of Wang
et al. (2024, arXiv 2409.19675, section 2.1.1): cell centres in a square joined by springs along `DelaunayNeighborhood`,
cancer cells that divide with probability `p0 (1 - d / dmax)` once older than `g_age`, which switches at time `tau`,
and invade healthy neighbours with probability `p_psc`, with the tumour area as an observable.

#### Scenario: Biphasic growth
- **WHEN** the system runs 30 days with `g_age` 20 hours before `tau = 10` days and 5 hours after
- **THEN** the tumour area grows faster per day after day 10 than before it

#### Scenario: Springs keep cells apart
- **WHEN** the system runs 10 days with the default parameters
- **THEN** no two cell centres are closer than half the natural spring length

### Requirement: Cell cycle and invasion
The library SHALL provide `CellCycleInvasionSystem` in `biology/oncology`, the cell invasion model of Carr et al. (2021)
as 2409.19675 (section 2.1.2) uses it: cells on a triangular lattice in the phases G1, eS, and S/G2/M, with transition
rates `Rr`, `Ry`, and `Rg`, moves to a vacant neighbour at rates `Mr`, `My`, and `Mg`, and division into two G1 cells
next to a vacancy, run by Gillespie. It SHALL declare the cells per phase as observables.

#### Scenario: Scratch assay
- **WHEN** the system runs 48 hours from a monolayer with a vacant scratch in its middle
- **THEN** the number of cells inside the scratch grows

#### Scenario: Full lattice
- **WHEN** the system runs from a full lattice with all move rates 0
- **THEN** the number of cells stays constant and no cell leaves G1 before a vacancy appears next to it

### Requirement: Differentiable COVID agents
The library SHALL provide `DifferentiableCovidAbmSystem` in `biology/epidemiology`, after Quera-Bofarull et al. (2023,
arXiv 2305.15340, Eq. 6): agents with discrete states in households, schools, and companies, infected by relaxed
Bernoulli draws of probability `1 - exp(-psi * sum_L beta_L * dt * I_L)`, so it holds the tag `differentiable`, with the
daily infections as an observable.

#### Scenario: Gradient through the run
- **WHEN** the gradient of the total infections over 30 days with respect to the three betas, at fixed uniform values
  of the draws, is compared with central finite differences of step 1e-6
- **THEN** they agree within 1e-5 relative error

#### Scenario: No transmission
- **WHEN** the system runs with every beta set to 0
- **THEN** the daily infections are 0 after the initial cases recover

### Requirement: Calibration probes are recorded in the system inventory
The inventory SHALL record `off_lattice_tumour_growth`, `cell_cycle_invasion`, and `differentiable_covid_abm` as
implemented with their sources and primitives, and the gallery SHALL hold their examples, each showing a calibration.

#### Scenario: Inventory and gallery
- **WHEN** the inventory check and the strict gallery coverage run
- **THEN** they report no problem for the three systems
