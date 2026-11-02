# Spec Delta

## ADDED Requirements

### Requirement: Higher-order models across fields
The library SHALL provide, as registered factories written in their natural order:
- the nonlinear pendulum as Newton's law;
- a driven damped oscillator;
- a series RLC circuit as a second-order charge equation;
- Sprott's jerk equation;
- Samuelson's multiplier-accelerator;
- a Fibonacci-type recurrence.

#### Scenario: Circuit and oscillator agree
- **WHEN** the RLC circuit and the damped oscillator run with matching coefficients `L = m`, `R = c`, and `1 / C = k`
- **THEN** the charge and the displacement trajectories are identical
