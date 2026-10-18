# Spec Delta

## ADDED Requirements

### Requirement: Control models
The library SHALL provide optimal harvesting, SIR vaccination, Ramsey optimal growth, a PID-stabilised inverted
pendulum, and a linear-quadratic regulated oscillator as registered factories.

#### Scenario: Vaccination lowers the peak
- **WHEN** the SIR vaccination problem is solved with a bounded vaccination rate and a cost on infections
- **THEN** the optimally controlled epidemic peak is lower than the uncontrolled one
