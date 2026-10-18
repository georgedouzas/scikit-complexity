# Spec Delta

## ADDED Requirements

### Requirement: Acausal component library and engineering models
The library SHALL provide electrical, translational, rotational, thermal, and hydraulic components with ports and
equations, and the RLC circuit, Chua's circuit, a quarter-car suspension, a DC motor with an inertia, and a heated
room with a thermostat as registered factories.

#### Scenario: Chua's double scroll
- **WHEN** Chua's circuit runs with its standard component values
- **THEN** the trajectory visits both scrolls and its largest Lyapunov exponent is positive
