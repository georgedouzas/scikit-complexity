# Spec Delta

## ADDED Requirements

### Requirement: Open-loop minimum time control
`MinimumTimeControlSystem` SHALL take `control`, `'feedback'` (the default, the sampled feedback on the switching curve)
or `'open-loop'`, which applies the control `-1` until the time `switch` and `1` after it, or the reverse for a start
below the switching curve.

#### Scenario: Switch of the Pontryagin solution
- **WHEN** the free particle runs from `(3, 2)` with `control='open-loop'` and `switch = 2 + sqrt(5)`
- **THEN** its state at `2 + 2 sqrt(5)` is within 1e-3 of the origin

#### Scenario: Feedback unchanged
- **WHEN** the system runs with its defaults
- **THEN** its run equals the run of the system before this change within 1e-12
