# Spec Delta

## ADDED Requirements

### Requirement: Coupled harvesting example
The library SHALL provide a socio-ecological model in which farmer agents harvest a resource whose stock regrows by a
logistic ODE, and the harvest depends on the number of farmers through an aggregate.

#### Scenario: Overharvesting collapses the resource
- **WHEN** the harvesting model runs with a harvest effort high enough that harvest exceeds the largest regrowth
- **THEN** the resource stock falls towards zero over the run

#### Scenario: Sustainable effort
- **WHEN** the harvesting model runs with a harvest effort below the largest regrowth
- **THEN** the resource stock settles at a positive level
