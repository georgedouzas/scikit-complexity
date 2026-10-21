# Spec Delta

## ADDED Requirements

### Requirement: State conditions of rate laws restart integrators
A condition on states inside `where` or `step` in a rate law SHALL compile into a hidden event without a reset, so
integrators with event location restart at the crossing, and hidden events SHALL not appear in the trajectory's events.

#### Scenario: Bathtub drains empty
- **WHEN** `bathtub(water=20.0)` runs until 6 with the default solver
- **THEN** it finishes without a stall warning, the water stays nonnegative within 1e-6, and the trajectory has no
  events
