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

### Requirement: Agents are drawn in their space
The Rerun observer SHALL draw agents on a grid or in a continuous space as one batch of points per agent group, and
agents on a network as graph nodes and edges, with colors from an agent variable.

#### Scenario: Schelling grid
- **WHEN** a Schelling segregation model on a 20 by 20 grid is logged
- **THEN** one `Points2D` batch per agent group is logged per frame, at the agents' cell coordinates

#### Scenario: Voter model network
- **WHEN** a voter model on a network is logged
- **THEN** the network is logged as graph nodes and edges, and node colors follow the agents' opinions

### Requirement: Changing populations are logged consistently
The Rerun observer SHALL log agents with their ids, so an agent keeps its identity across frames and removed agents
disappear.

#### Scenario: Agent removed
- **WHEN** an agent is removed between two frames
- **THEN** it is absent from the batch logged at the second frame

### Requirement: Observables are logged
The Rerun logger SHALL log every observable as a time series under the entity path of the system that declares it.

#### Scenario: Live order parameter
- **WHEN** a simulation of oscillators with the observable `order` streams to Rerun
- **THEN** the viewer shows `order` as a time series under the system's entity path

### Requirement: Rewritten geometry is drawn in a fixed frame
A geometry visual SHALL log the turtle segments and the segment tiles of a rewriting element as one batch of
two-dimensional or three-dimensional line strips per frame, and the polygon tiles as outlines or, when filled, as an
image of their raster. The view SHALL use the bounds the element declares, or the bounding box of a patch's initial tiles, so successive
generations are drawn at the same scale, and colours SHALL follow the prototile or the depth of the branch.

#### Scenario: Koch curve in Rerun
- **WHEN** the Koch curve runs 5 generations with a Rerun observer
- **THEN** one `LineStrips2D` per frame is logged under `koch`, the last one holding 1024 segments, and the view
  shows the whole curve at every frame

#### Scenario: Filled gasket
- **WHEN** the Sierpiński triangle is drawn with the filled option at generation 6
- **THEN** one image of its raster is logged per frame, and the filled pixels of the raster cover `(3/4)^6` of the
  triangle's pixels within 2%

### Requirement: Host graphs are drawn as graphs
A geometry visual SHALL log the host graph of a graph rewriting system as graph nodes, labelled by their `label`
attribute, and graph edges per frame, shown in a graph view.

#### Scenario: Subdivided ring in Rerun
- **WHEN** a ring whose edges are subdivided runs 2 generations with a Rerun observer
- **THEN** one batch of graph nodes and edges is logged per frame under the host graph's path, in a graph view
