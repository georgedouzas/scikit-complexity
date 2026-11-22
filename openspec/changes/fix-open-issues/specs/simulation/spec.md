## ADDED Requirements

### Requirement: Derived states are recorded at the first frame
States computed from the structure of a system SHALL be recorded at frame 0 with their values on the initial
structure.

#### Scenario: Initial degrees
- **WHEN** a `CoevolvingVoterSystem(200, mean_degree=4)` simulates
- **THEN** the degrees of frame 0 sum to twice the number of initial edges
