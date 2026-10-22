# Spec Delta

## Purpose

Lets a user place agent groups in three-dimensional continuous spaces and in geographic spaces built from GIS layers,
and validate, run, record, and draw them with the same rules, trajectories, and visualization as agents on lattices and
networks.

## ADDED Requirements

### Requirement: Agents run in continuous spaces of two or three dimensions
The Mesa backend SHALL run groups with rule laws in a `ContinuousSpace` of two or three sides, and SHALL declare that
support in its `Supports` declaration. Two-dimensional spaces SHALL behave as before. Validation SHALL report a
continuous space the backend does not support, and an initial location whose length differs from the space's
dimension, each as an issue with the path of the space or group and a fix.

#### Scenario: Three-dimensional box
- **WHEN** a user simulates agents in `ContinuousSpace(10.0, 5.0, 2.0)` on the Mesa backend
- **THEN** the space has 3 dimensions, and agents placed without a location get positions inside the box

#### Scenario: Two-dimensional space unchanged
- **WHEN** an existing model on a two-dimensional continuous space is simulated twice with random state 0
- **THEN** the trajectories equal the ones recorded before this change

#### Scenario: Wrong number of coordinates
- **WHEN** a group in a three-dimensional space is given an initial location with two coordinates
- **THEN** validation reports an issue whose path is the group and whose message names the space dimension and the
  location's length

#### Scenario: Unsupported dimension
- **WHEN** a system with rule laws in a continuous space of four sides is validated with the Mesa backend
- **THEN** the report holds one error that says the backend runs continuous spaces of two or three dimensions

### Requirement: Radius queries and wrapping hold in three dimensions
Rules SHALL find the neighbours of an agent within a radius by the Euclidean distance over all coordinates of the
space, and a space with a periodic boundary SHALL wrap every coordinate, including the third.

#### Scenario: Neighbour across the depth boundary
- **WHEN** two agents in `ContinuousSpace(10.0, 10.0, 10.0, boundary='periodic')` sit at third coordinates 0.2 and 9.9
  with equal other coordinates
- **THEN** each finds the other within radius 0.5

#### Scenario: Depth counts in the distance
- **WHEN** two agents share their first two coordinates and differ by 3 in the third
- **THEN** neither finds the other within radius 2, and each finds the other within radius 4

### Requirement: Geographic spaces hold layers in a coordinate reference system
A geographic space SHALL be a space kind of its own, with a coordinate reference system and vector layers given as
GeoPandas data frames or as paths to vector files. Layers in another reference system SHALL be reprojected into the
space's system. The geographic space SHALL declare its requirements, the Mesa backend SHALL declare support for it, and
validation SHALL report a geographic space with a solver or backend that does not support it.

#### Scenario: Layer reprojected
- **WHEN** a layer in EPSG:4326 is added to a geographic space in EPSG:3857
- **THEN** the space holds the layer in EPSG:3857 with the same number of features

#### Scenario: Unknown reference system
- **WHEN** a layer has no coordinate reference system
- **THEN** building the space raises a named error that names the layer and asks for a reference system

#### Scenario: Unsupported solver
- **WHEN** a system on a geographic space is validated with the solver `'rk4'`
- **THEN** the report holds one error that says the solver does not support geographic spaces and names the Mesa
  backend in its fix

### Requirement: Groups are created from layer features
A group in a geographic space SHALL be created from a layer, with one member per feature or a number of members per
feature read from an attribute. Each member SHALL be placed at a random point inside its feature, drawn from the
simulation's random stream, and SHALL take feature attributes as initial state values. Validation SHALL report a group
created from a layer the space does not hold.

#### Scenario: Agents per census region
- **WHEN** a group is created from three region polygons whose `population` attribute is 100, 50, and 0
- **THEN** the group has 150 members, 100 of them inside the first polygon and 50 inside the second

#### Scenario: Missing layer
- **WHEN** a group is created from the layer `census` in a system whose geographic space holds only `boundary`
- **THEN** validation reports an issue whose path is the group and whose message lists the layers the space holds

#### Scenario: Reproducible placement
- **WHEN** the same geographic model is simulated twice with random state 0
- **THEN** the agents start at the same points in both runs

### Requirement: Rules query and move agents in geographic spaces
Rules SHALL find a geographic agent's neighbours within a distance in the units of the space, list the layer features
that contain the agent, and move the agent to a point or to a random point inside a feature.

#### Scenario: Distance neighbours
- **WHEN** an agent asks for its neighbours within 1000 metres in a space in a metric reference system
- **THEN** it receives exactly the other agents whose points lie at most 1000 metres away

#### Scenario: Move to a favourite place
- **WHEN** a rule moves an agent to a feature of a places layer
- **THEN** the agent's point lies inside that feature, and listing the features that contain the agent includes it

### Requirement: Trajectories report positions in every space
A trajectory SHALL report the location of an agent in a three-dimensional space as three coordinates, and in a
geographic space as the point's coordinates in the space's reference system, with the agent ids at every frame.

#### Scenario: Three coordinates per agent
- **WHEN** a user asks a trajectory of a model in a three-dimensional space for the agent locations at frame 5
- **THEN** it returns an array with one row of 3 coordinates per agent alive at frame 5

### Requirement: Agents are drawn in three-dimensional and geographic spaces
The Rerun observer SHALL draw agents in a three-dimensional space as one batch of 3D points per group, and agents in a
geographic space as 2D points with the space's layers drawn once as static outlines.

#### Scenario: Boids in 3D
- **WHEN** a model of boids in a three-dimensional space is logged
- **THEN** one `Points3D` batch per group is logged per frame at the agents' positions

#### Scenario: Island map
- **WHEN** a model in a geographic space with a boundary layer is logged
- **THEN** the boundary is logged once as static line strips, and the agents are logged per frame as `Points2D`

### Requirement: Geographic spaces need the geo extra
Geographic spaces SHALL work only when the `geo` optional extra is installed, and the rest of the framework SHALL work
without it. Validation SHALL report the missing extra as an unavailable backend before any work.

#### Scenario: Extra missing
- **WHEN** a user validates or simulates a model with a geographic space in an environment without the `geo` extra
- **THEN** the report holds one error whose fix says that the `geo` extra is needed and how to install it
