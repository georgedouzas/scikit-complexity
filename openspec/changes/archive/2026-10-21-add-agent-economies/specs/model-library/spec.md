# Spec Delta

## ADDED Requirements

### Requirement: Economic and social agent models
The library SHALL provide registered agent models that exercise the phases, market protocols, institutions, and
learning blocks in distinct ways, and SHALL cover spatial cooperation with the existing lattice model of the spatial
prisoner's dilemma rather than registering it again.

#### Scenario: Spatial cooperation
- **WHEN** the spatial prisoner's dilemma runs with imitate-the-best updating and a temptation payoff of 1.6
- **THEN** cooperators survive in clusters for 200 steps instead of dying out

### Requirement: Agent economy models are recorded in the model inventory
Every model this change adds SHALL register under the field path of its subfield and SHALL have an entry in the model
inventory, and a model the inventory records as pending this change SHALL register under the name of its entry. A
model that the library already implements SHALL be rebuilt on the new formalism under the same name, not registered
twice.

#### Scenario: Inventory models registered
- **WHEN** the change is applied and the inventory check runs
- **THEN** these entries are registered under their field paths and marked implemented:
  - `credit_networks` in `economics/macroeconomics`
  - `fractional_reserve_banking` in `economics/macroeconomics`
  - `double_auction` in `economics/markets`
  - `labour_market` in `economics/markets`
  - `fair_food` in `economics/markets`
  - `sugarscape` in `economics/markets`
  - `brock_hommes` in `economics/markets`
  - `kinetic_wealth_exchange` in `economics/econophysics`
  - `minority_game` in `economics/econophysics`
  - `el_farol` in `economics/econophysics`
  - `kolkata_restaurant` in `economics/econophysics`
  - `axelrod` in `social/opinion_dynamics`

#### Scenario: Example entries left pending
- **WHEN** the change is applied
- **THEN** these entries, which repeat patterns the implemented models already exercise, stay pending this change:
  - `goods_traders` in `economics/markets`
  - `shops` in `economics/markets`
  - `barter` in `economics/markets`
  - `order_book` in `economics/finance`
  - `tragedy_of_the_commons` in `economics/resources`
  - `ridesharing` in `engineering/traffic`

#### Scenario: New model recorded
- **WHEN** the change registers a model with no inventory entry
- **THEN** the inventory check fails until the entry is added with its sources and field path
