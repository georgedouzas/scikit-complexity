# Spec Delta

## ADDED Requirements

### Requirement: Fractional models
The library SHALL provide fractional relaxation, a fractional damped oscillator, fractional Van der Pol, and fractional
Lorenz as registered factories with the order as a parameter.

#### Scenario: Slower than exponential
- **WHEN** fractional relaxation runs with orders 1 and 0.5 from the same state
- **THEN** the order 0.5 trajectory stays above the exponential one at long times
