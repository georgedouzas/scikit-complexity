# Spec Delta

## ADDED Requirements

### Requirement: Field models
The library SHALL provide heat diffusion, Fisher-KPP, Gray-Scott, Brusselator, FitzHugh-Nagumo spirals, Schnakenberg
Turing patterns, and pheromone foraging as registered factories.

#### Scenario: Fisher front speed
- **WHEN** the Fisher-KPP model runs with growth rate `r` and diffusion `D` from a step profile
- **THEN** the front moves at a speed within 5% of `2 * sqrt(r * D)`

#### Scenario: Turing spots
- **WHEN** the Schnakenberg model runs in its Turing-unstable regime from small noise
- **THEN** a stationary pattern with a dominant wavelength within 10% of the linear prediction forms
