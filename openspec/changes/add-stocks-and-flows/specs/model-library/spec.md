# Spec Delta

## ADDED Requirements

### Requirement: System-dynamics and stock-flow consistent models
The library SHALL provide Meadows' systems zoo (bathtub, coffee cooling, thermostat, inventory with delays, renewable
and nonrenewable resources), the system traps (tragedy of the commons, success to the successful, shifting the burden,
escalation), Bass diffusion, the beer game, and a stock-flow consistent monetary economy as registered factories.

#### Scenario: Inventory oscillation
- **WHEN** the car-dealership inventory model receives a 10% step in demand
- **THEN** the inventory oscillates before settling, and shortening the perception delay reduces the oscillation's
  amplitude
