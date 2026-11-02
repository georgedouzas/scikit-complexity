# Spec Delta

## ADDED Requirements

### Requirement: Static models
The library SHALL provide `lq_cell_survival` in `biology/oncology`, `random_energy_model` and `continuum_percolation`
in `physics/statistical_mechanics`, `oligopoly_equilibria` in `economics/markets`, `von_thunen` in
`economics/resources`, `gravity_urban_scaling` in `social/organizations`, and `mandelbrot_least_effort` in
`social/language`, as registered factories of systems under `StaticTime` that run through `create` and `run` and sweep
their parameters through `Experiment`.

#### Scenario: Fractionation
- **WHEN** `lq_cell_survival` evaluates 30 fractions of 2 Gy with `alpha/beta = 10`
- **THEN** the biologically effective dose is 72 Gy (Padmanabhan et al. 2021 Eq. 6.8), and the dose-protraction factor
  of a constant dose rate tends to 1 as the delivery time tends to 0

#### Scenario: Glass transition
- **WHEN** an `Experiment` sweeps the temperature of `random_energy_model` with `N = 20` over 10 replicates
- **THEN** the entropy follows `ln 2 - 1/(4 T^2)` above `T_g = 0.6006` and stays below 0.1 at `T = 0.3`

#### Scenario: Mixed price equilibrium
- **WHEN** `oligopoly_equilibria` solves the pricing game of `m` firms with loyal share `alpha` and switchers
  `beta = 1 - m alpha`
- **THEN** the symmetric equilibrium price distribution is `F(P) = 1 - (alpha (1 - P)/(beta P))^(1/(m - 1))` on
  `[alpha/(alpha + beta), 1]`, and each firm's expected profit is `alpha` (Forrest 2018 Lemma 20.1)

#### Scenario: Von Thünen rings
- **WHEN** `von_thunen` assigns the cells of a plain around one market to the crop of highest bid rent
  `Y_i (p_i - c_i) - Y_i f_i d` at distance `d`, after Giordano (2024, arXiv 2407.02181 §7)
- **THEN** the land use forms concentric rings whose boundaries lie where the bid rents of neighbouring crops are equal,
  within one cell, and raising a crop's transport rate `f_i` shrinks its ring

#### Scenario: Percolation threshold
- **WHEN** an `Experiment` sweeps the filling factor of `continuum_percolation` over 0.8 to 1.5 for two sizes
- **THEN** the spanning probabilities that `ClusterAnalysis` gives cross between filling factors 1.0 and 1.25

#### Scenario: Urban scaling exponent
- **WHEN** an `Experiment` runs `gravity_urban_scaling` at five populations from 100 to 1600 with area exponent `alpha`
  and interaction exponent `gamma < 2`, and `ScalingAnalysis(kind='power')` fits the total interaction against the
  population
- **THEN** the fitted exponent is within 0.05 of `2 - alpha gamma / 2`, for one seed up to `gamma = 1` and for totals
  averaged over seeds at `gamma = 1.5`, where close pairs make single cities heavy-tailed

#### Scenario: Least-effort frequencies
- **WHEN** `mandelbrot_least_effort` solves for word costs `c_i = log(i + 1)`
- **THEN** the probabilities follow `(i + 1)^-lambda` with the solved `lambda` within 1e-9, and the ratio of cost to
  entropy at the solved `lambda` is lower than at `lambda ± 0.1`

### Requirement: Dynamic equilibrium, satisfiability, and search models
The library SHALL provide the fictitious play of `oligopoly_equilibria` with `dynamics='fictitious-play'`, built on
`ChoiceInteraction` and `ReinforcementInteraction`, `perceptron_satisfiability` in `computer_science/neural_networks`,
built on `SatisfiabilityInteraction`, and `evolutionary_optimisers` in `computer_science/evolutionary_computation`,
built on `OptimizationInteraction` and on `SamplingInteraction` with `cooling`, all under discrete time.

#### Scenario: Fictitious play of a discretised duopoly
- **WHEN** `oligopoly_equilibria` runs with `dynamics='fictitious-play'` for 2000 rounds
- **THEN** the empirical distribution of prices is within 0.2 of `F(P)` in the Kolmogorov distance

#### Scenario: Satisfiable fraction
- **WHEN** an `Experiment` runs `perceptron_satisfiability` at `N = 15` and `N = 21` over 20 replicates
- **THEN** the share of satisfiable replicates against `p/N` falls from above 0.9 at 0.5 to below 0.1 at 1.2

#### Scenario: Algorithms compared
- **WHEN** `evolutionary_optimisers` runs differential evolution, particle swarm, and annealing on Rastrigin for 100
  generations with the same seed
- **THEN** every run reports a best value that never increases, and differential evolution ends below 1e-3

### Requirement: Static models are recorded in the model inventory
Every model this change adds SHALL register under the field path of its subfield and SHALL be marked implemented in
the model inventory, with a gallery example whose analysis plot comes from an `Experiment` over its swept parameter.

#### Scenario: Inventory models registered
- **WHEN** the change is applied and the inventory check runs
- **THEN** these entries are registered under their field paths and marked implemented:
  - `lq_cell_survival` in `biology/oncology`
  - `random_energy_model` in `physics/statistical_mechanics`
  - `oligopoly_equilibria` in `economics/markets`
  - `von_thunen` in `economics/resources`
  - `perceptron_satisfiability` in `computer_science/neural_networks`
  - `evolutionary_optimisers` in `computer_science/evolutionary_computation`
  - `continuum_percolation` in `physics/statistical_mechanics`
  - `gravity_urban_scaling` in `social/organizations`
  - `mandelbrot_least_effort` in `social/language`
