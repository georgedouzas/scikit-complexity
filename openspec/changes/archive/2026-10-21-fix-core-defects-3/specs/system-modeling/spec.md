# Spec Delta

## ADDED Requirements

### Requirement: Angles wrap into one turn centred on zero
A state of the `ANGLE` domain SHALL hold values in `(-pi, pi]` after every step of every form, and encoding a value in
the `ANGLE` domain SHALL wrap it into the same range.

#### Scenario: Small swing around zero
- **WHEN** a rotor with the angle state `phi' = -1` starts at 0.1 and runs until 1
- **THEN** every recorded angle lies in `(-pi, pi]`, and the last one equals -0.9 within 1e-6

#### Scenario: Encoding an angle
- **WHEN** `Domain.ANGLE.encode` receives `[3 pi / 2, -pi, pi]`
- **THEN** it returns `[-pi / 2, pi, pi]`

### Requirement: Systems replicate into vectorised ensembles
`replicate(system, size, values)` SHALL return a system whose elements are groups of `size` members of the same
element types and labels, with per-member values by dotted path, and SHALL raise a named error for a system with
interactions, subsystems, groups, fields, or agents.

#### Scenario: Many initial conditions in one run
- **WHEN** the logistic map at `r = 3.7` is replicated over 100 initial values and run for 50 iterations
- **THEN** member `i` of `map.x` equals the run of the single map from the `i`-th initial value within 1e-12

#### Scenario: A system with interactions
- **WHEN** a system with an interaction is replicated
- **THEN** a named error says that only systems of plain elements replicate
