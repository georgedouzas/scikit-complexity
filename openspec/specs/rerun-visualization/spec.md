# rerun-visualization Specification

## Purpose

Streams a running simulation to the Rerun viewer or to a recording file, laid out by the structure of the system, so
a user sees a model evolve while it runs.

## Requirements

### Requirement: Simulations stream to Rerun
A Rerun observer SHALL log every frame it receives on a simulated time timeline and a step timeline, to a spawned
viewer, a running viewer, or a recording file.

#### Scenario: Save a recording
- **WHEN** a simulation runs with a Rerun observer that saves to a file
- **THEN** the file exists after the run and holds a Rerun recording

### Requirement: Entity paths follow the system tree
The Rerun observer SHALL log each element under an entity path built from its path in the system tree.

#### Scenario: Nested element
- **WHEN** the element `rabbits` of the subsystem `meadow` is logged
- **THEN** its data appears under the entity path `meadow/rabbits`

### Requirement: Element types choose their visuals
Each element type SHALL declare how its state is drawn, scalar states SHALL default to time series, and position
states of Euclidean spaces SHALL be drawable as points.

#### Scenario: Particle as a point
- **WHEN** a particle in a three-dimensional space is logged
- **THEN** its position is logged as a three-dimensional point

#### Scenario: Population as a time series
- **WHEN** a population element is logged
- **THEN** its count is logged as a scalar time series

### Requirement: Parameters are visible
The Rerun observer SHALL log the parameter values with every frame, so live changes appear next to their effects.

#### Scenario: Parameter series
- **WHEN** a parameter changes during a run
- **THEN** its logged series shows the new value from the frame after the change

### Requirement: Logging can be thinned
The Rerun observer SHALL accept a frame interval and log only every n-th frame.

#### Scenario: Every tenth frame
- **WHEN** an observer with interval 10 receives 100 frames
- **THEN** it logs 10 of them

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
