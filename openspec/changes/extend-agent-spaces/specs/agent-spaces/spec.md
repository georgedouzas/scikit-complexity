# Spec Delta

## Purpose

Lets a user place agents in three-dimensional continuous spaces and in geographic spaces built from GIS layers, and
run, record, and draw them with the same rules, trajectories, and visualization as agents on grids and networks.

## ADDED Requirements

### Requirement: Continuous spaces have two or three dimensions
A continuous agent space SHALL accept an optional depth. With a depth it SHALL have three dimensions. Without one it
SHALL have two dimensions and behave as before.

#### Scenario: Three-dimensional box
- **WHEN** a user creates a continuous space of width 10, height 5, and depth 2
- **THEN** the space has 3 dimensions, and agents placed without a location get positions inside the box

#### Scenario: Two-dimensional space unchanged
- **WHEN** an existing model on a two-dimensional continuous space is simulated twice with random state 0
- **THEN** the trajectories equal the ones recorded before this change

#### Scenario: Wrong number of coordinates
- **WHEN** an agent in a three-dimensional space is given a location with two coordinates
- **THEN** compiling or placing raises a named error that names the agent group, the space dimension, and the
  location's length

### Requirement: Radius queries and wrapping hold in three dimensions
Rules SHALL find the neighbours of an agent within a radius by the Euclidean distance over all coordinates of the
space, and a toroidal space SHALL wrap every coordinate, including depth.

#### Scenario: Neighbour across the depth boundary
- **WHEN** two agents in a toroidal 10 by 10 by 10 space sit at depths 0.2 and 9.9 with equal other coordinates
- **THEN** each finds the other within radius 0.5

#### Scenario: Depth counts in the distance
- **WHEN** two agents share their first two coordinates and differ by 3 in depth
- **THEN** neither finds the other within radius 2, and each finds the other within radius 4

### Requirement: Geographic spaces hold layers in a coordinate reference system
A geographic agent space SHALL have a coordinate reference system and SHALL hold vector layers given as GeoPandas data
frames or as paths to vector files. Layers in another reference system SHALL be reprojected into the space's system.

#### Scenario: Layer reprojected
- **WHEN** a layer in EPSG:4326 is added to a geographic space in EPSG:3857
- **THEN** the space holds the layer in EPSG:3857 with the same number of features

#### Scenario: Unknown reference system
- **WHEN** a layer has no coordinate reference system
- **THEN** creating the space raises a named error that names the layer and asks for a reference system

### Requirement: Agent groups are created from layer features
An agent group in a geographic space SHALL be created from a layer, with one agent per feature or a number of agents
per feature read from an attribute. Each agent SHALL be placed at a random point inside its feature, drawn from the
simulation's random stream, and SHALL take feature attributes as initial state values.

#### Scenario: Agents per census region
- **WHEN** a group is created from three region polygons whose `population` attribute is 100, 50, and 0
- **THEN** the group has 150 agents, 100 of them inside the first polygon and 50 inside the second

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
The Rerun observer SHALL draw agents in a three-dimensional space as one batch of 3D points per agent group, and agents
in a geographic space as 2D points with the space's layers drawn once as static outlines.

#### Scenario: Boids in 3D
- **WHEN** a model of boids in a three-dimensional space is logged
- **THEN** one `Points3D` batch per agent group is logged per frame at the agents' positions

#### Scenario: Island map
- **WHEN** a model in a geographic space with a boundary layer is logged
- **THEN** the boundary is logged once as static line strips, and the agents are logged per frame as `Points2D`

### Requirement: Geographic spaces need the geo extra
Geographic spaces SHALL work only when the `geo` optional extra is installed, and the rest of the framework SHALL work
without it.

#### Scenario: Extra missing
- **WHEN** a user simulates a model with a geographic space in an environment without the `geo` extra
- **THEN** an import error says that the `geo` extra is needed and how to install it
