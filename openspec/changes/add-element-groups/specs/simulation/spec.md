# Spec Delta

## ADDED Requirements

### Requirement: Trajectories return group variables as arrays
A trajectory SHALL return a group variable by name as an array with one leading entry per frame followed by the group
shape.

#### Scenario: Positions of a gas over time
- **WHEN** a user asks the trajectory of a 100-frame run for `gas.x` of a group of 500 molecules in two dimensions
- **THEN** it returns an array of shape `(100, 500, 2)`

### Requirement: Group parameters change while running
A running simulation SHALL accept a new value for a group parameter, either one value for every member or one value
per member.

#### Scenario: Heat one species
- **WHEN** a user sets the mass of every member of a group to 2 while the simulation runs
- **THEN** the frames after the change use mass 2 for every member

### Requirement: Groups run large in real time
A simulation SHALL advance a group of 10000 non-interacting members fast enough to keep a real-time clock of
speed 1 with frames every 0.01 time units.

#### Scenario: Ideal gas in real time
- **WHEN** an ideal gas of 10000 molecules runs for 5 seconds with a real-time clock of speed 1 and dt 0.01
- **THEN** the run falls behind by no more than 0.1 seconds
