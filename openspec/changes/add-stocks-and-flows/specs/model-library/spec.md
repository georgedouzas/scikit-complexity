# Spec Delta

## ADDED Requirements

### Requirement: System-dynamics and stock-flow consistent models
The library SHALL provide Meadows' systems zoo (bathtub, coffee cooling, thermostat, inventory with delays, renewable
and nonrenewable resources), the system traps (success to the successful, shifting the burden, escalation), Bass
diffusion, the beer game, and a stock-flow consistent monetary economy as registered factories, with `bathtub` and
`car_dealer` rebuilt on flows under their names. The tragedy of the commons belongs to `add-agent-economies`.

#### Scenario: Inventory oscillation
- **WHEN** the car-dealership inventory model receives a 10% step in demand
- **THEN** the inventory oscillates before settling, and shortening the perception delay reduces the oscillation's
  amplitude

### Requirement: Stock-flow models are recorded in the model inventory
Every model this change adds SHALL register under the field path of its subfield and SHALL have an entry in the model
inventory, and a model the inventory records as pending this change SHALL register under the name of its entry. A
model that the library already implements SHALL be rebuilt on the new formalism under the same name, not registered
twice.

#### Scenario: Inventory models registered
- **WHEN** the change is applied and the inventory check runs
- **THEN** these entries, which probe flows, sources and sinks, auxiliaries, lookups, delays, smoothing, loops, and
  discrete flows in distinct ways, are registered under their field paths and marked implemented:
  - `stock_flow_consistent_economy` in `economics/macroeconomics`
  - `commodity_cycle` in `economics/markets`
  - `beer_game` in `economics/markets`
  - `bass_diffusion` in `economics/innovation`
  - `success_to_the_successful` in `social/organizations`
  - `shifting_the_burden` in `social/organizations`
  - `escalation` in `social/organizations`
  - `rice_milling` in `engineering/processes`
  - `solid_waste` in `engineering/processes`
  - `food_relief` in `engineering/processes`
  - `irrigation` in `engineering/processes`

#### Scenario: Repeated patterns stay pending
- **WHEN** a surveyed model only repeats a pattern of stocks and flows that an implemented model already probes
- **THEN** its entry stays pending this change: `energy_return` in `earth/climate`, and `cocoa`, `hilsa_fishery`,
  `food_security`, `renewable_resource`, and `nonrenewable_resource` in `economics/resources`

#### Scenario: New model recorded
- **WHEN** the change registers a model with no inventory entry
- **THEN** the inventory check fails until the entry is added with its sources and field path
