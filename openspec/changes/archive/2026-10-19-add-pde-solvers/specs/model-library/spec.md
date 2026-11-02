# Spec Delta

## ADDED Requirements

### Requirement: PDE models
The library SHALL provide Kuramoto-Sivashinsky, complex Ginzburg-Landau, Swift-Hohenberg, Burgers' shock, the
shallow-water dam break, decaying two-dimensional turbulence, Rayleigh-Bénard convection, and heat on an L-shaped
domain as registered factories.

#### Scenario: Swift-Hohenberg rolls
- **WHEN** the Swift-Hohenberg model runs above onset from small noise
- **THEN** a roll pattern forms with a wavelength within 5% of 2π
