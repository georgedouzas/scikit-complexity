# Spec Delta

## ADDED Requirements

### Requirement: Static, sampled, and search models
The library SHALL provide models that are solved, sampled, enumerated, or optimised as registered factories. They
SHALL run through `create` and `run` with default sweeps or iteration counts, stream to Rerun, and have a gallery
example with a Rerun thumbnail and an analysis plot.

#### Scenario: Fractionation
- **WHEN** `lq_cell_survival` evaluates 30 fractions of 2 Gy with `alpha/beta = 10`
- **THEN** the biologically effective dose is 72 Gy (Padmanabhan et al. 2021 Eq. 6.8), and the dose-protraction
  factor of a constant dose rate tends to 1 as the delivery time tends to 0

#### Scenario: Mixed price equilibrium
- **WHEN** `oligopoly_equilibria` solves the pricing game of `m` firms with loyal share `alpha` and switchers
  `beta = 1 - m alpha`
- **THEN** the symmetric equilibrium price distribution is `F(P) = 1 - (alpha (1 - P)/(beta P))^(1/(m - 1))` on
  `[alpha/(alpha + beta), 1]`, and each firm's expected profit is `alpha` (Forrest 2018 Lemma 20.1)

#### Scenario: Glass transition
- **WHEN** `random_energy_model` sweeps the temperature with `N = 20` over 10 draws
- **THEN** the entropy follows `ln 2 - 1/(4 T^2)` above `T_g = 0.6006` and stays below 0.1 at `T = 0.3`

### Requirement: Models are recorded in the model inventory
Every model this change adds SHALL register under the field path of its subfield and SHALL have an entry in the model
inventory, and a model the inventory records as pending this change SHALL register under the name of its entry.

#### Scenario: Inventory models registered
- **WHEN** the change is applied and the inventory check runs
- **THEN** these entries are registered under their field paths and marked implemented:
  - `lq_cell_survival` in `biology/oncology` (Pa 6.2)
  - `random_energy_model` in `physics/statistical_mechanics` (Bt 1.5.3)
  - `oligopoly_equilibria` in `economics/markets` (Fo 16, Fo 20, Fo 21)
  - `perceptron_satisfiability` in `computer_science/neural_networks` (Bt 6.3.3, Bt 6.4)
  - `evolutionary_optimisers` in `computer_science/evolutionary_computation` (Rm 6.3.3.3, Rm 6.3.4)
  - `continuum_percolation` in `physics/statistical_mechanics` (Sa 3)

#### Scenario: New model recorded
- **WHEN** the change registers a model with no inventory entry
- **THEN** the inventory check fails until the entry is added with its sources and field path
