# Spec Delta

## ADDED Requirements

### Requirement: Mechanics models
The library SHALL provide simple, double, and coupled pendulums, the Kepler problem, the restricted three-body problem,
the Chirikov standard map, and Hénon-Heiles as registered factories built from Hamiltonians or Lagrangians.

#### Scenario: Kepler orbit
- **WHEN** the Kepler problem runs for ten periods of a bound orbit with the Yoshida stepper
- **THEN** the orbit closes on itself within 1e-6 relative error and the energy drift stays below 1e-8
