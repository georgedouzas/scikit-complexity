# Spec Delta

## ADDED Requirements

### Requirement: Groups are logged as one batch
The Rerun observer SHALL log a group as one batch per state under the group path, so the number of log calls per frame
does not grow with the group size.

#### Scenario: Gas as one point cloud
- **WHEN** a group of 10000 molecules in two dimensions is logged
- **THEN** one `Points2D` with 10000 points is logged under the group path per frame

### Requirement: Group points can be colored by a state
A group points visual SHALL accept a scalar state or the norm of a vector state that colors each point on a fixed
color scale.

#### Scenario: Molecules colored by speed
- **WHEN** a group visual colors molecules by the norm of their velocity
- **THEN** faster molecules get colors further along the color scale than slower ones
