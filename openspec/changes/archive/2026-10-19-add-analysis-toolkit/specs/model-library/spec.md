# Spec Delta

## ADDED Requirements

### Requirement: Nonlinear dynamics models
The library SHALL provide the Lorenz, Rössler, Hénon, logistic map, Duffing, Van der Pol, and Hénon-Heiles models as
registered factories with their standard parameters.

#### Scenario: Strange attractor
- **WHEN** a user runs the Lorenz model with its defaults for 50 time units
- **THEN** the trajectory stays bounded and visits both lobes of the attractor
