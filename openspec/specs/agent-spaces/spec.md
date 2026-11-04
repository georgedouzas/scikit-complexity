# agent-spaces Specification

## Purpose

Lets a user place agent groups in three-dimensional continuous spaces and in continuous spaces with region layers,
and check, run, record, and draw them with the same interactions, trajectories, and visualization as agents on
lattices and networks.

## Requirements

### Requirement: Agents run in continuous spaces of two or three dimensions
`MesaSolver` SHALL run agent groups in a `ContinuousSpace` of two or three sides, and two-dimensional spaces SHALL
behave as before. It SHALL reject a continuous space of more sides or without an extent before the first frame, and
the compile report SHALL hold an initial location whose length differs from the dimension of its space.

#### Scenario: Three-dimensional box
- **WHEN** a user simulates agents in `ContinuousSpace(extent=(10.0, 5.0, 2.0))` with `MesaSolver`
- **THEN** the model view reports 3 dimensions, and agents placed without a location get positions inside the box

#### Scenario: Two-dimensional space unchanged
- **WHEN** an existing model on a two-dimensional continuous space is simulated twice with random state 0
- **THEN** the trajectories equal the ones recorded before this change

#### Scenario: Wrong number of coordinates
- **WHEN** a group in a three-dimensional space is given an initial location with two coordinates
- **THEN** the compile report holds a `location-dimension` issue whose path is the group and whose message names the
  space dimension and the location's length

#### Scenario: Unsupported dimension
- **WHEN** a system with agent rules in a continuous space of four sides is prepared with `MesaSolver`
- **THEN** a `BackendError` names the space and says the solver runs continuous spaces of two or three dimensions

### Requirement: Movement and radius queries hold in three dimensions
`MovementInteraction` steps and agent radius queries SHALL use every coordinate of the space, and a space with a
periodic boundary SHALL wrap every coordinate, including the third.

#### Scenario: Neighbour across the depth boundary
- **WHEN** two agents in `ContinuousSpace(extent=(10.0, 10.0, 10.0), boundary='periodic')` sit at third coordinates
  0.2 and 9.9 with equal other coordinates
- **THEN** each finds the other within radius 0.5

#### Scenario: Depth counts in the distance
- **WHEN** two agents share their first two coordinates and differ by 3 in the third
- **THEN** neither finds the other within radius 2, and each finds the other within radius 4

#### Scenario: Gaussian steps in depth
- **WHEN** agents in a three-dimensional periodic box move by a `MovementInteraction` with `step='gaussian'` for 100
  steps
- **THEN** the variance of their third coordinate grows, and every position stays inside the box

### Requirement: Continuous spaces hold region layers
`ContinuousSpace` SHALL take region layers, each a sequence of polygons in the coordinates of a two-dimensional space,
and SHALL reject an invalid region with a named error that names the layer and the region.

#### Scenario: Regions are indexed
- **WHEN** a space is prepared with a layer of three region polygons
- **THEN** its derived index answers which of the three regions contains a given point

#### Scenario: Invalid region
- **WHEN** a layer holds a self-intersecting polygon
- **THEN** preparing the space raises a `SpaceError` that names the layer and the index of the region

### Requirement: Agents are placed and moved in regions
A member location given as a polygon SHALL place the agent at a point drawn uniformly over the polygon's area from
the simulation's generator, in the initial locations of a group, in created agents, and in moves, by choosing a
triangle of the polygon's constrained Delaunay triangulation with probability proportional to its area and a uniform
point in that triangle, with no rejection loop or fallback point.

#### Scenario: Agents per census region
- **WHEN** a group of 150 members is given the first of three region polygons as the location of 100 members and the
  second as the location of 50
- **THEN** 100 agents start inside the first polygon and 50 inside the second

#### Scenario: Reproducible placement
- **WHEN** the same model with region locations is simulated twice with random state 0
- **THEN** the agents start at the same points in both runs

#### Scenario: Uniform placement in a concave polygon
- **WHEN** 20000 agents are placed in an L-shaped polygon with a hole, made of three unit squares around a missing
  fourth, with a hole of area 0.1 in one of them
- **THEN** every agent lies inside the polygon and outside the hole, and the share of agents in each unit square
  matches its area share within 0.02

### Requirement: Rules query the regions of an agent
Rules SHALL list the regions of a layer that contain an agent and read the polygon of a region, so a
`MovementInteraction` step moves an agent to a region and an agent-form interaction acts on where the agent stands.

#### Scenario: Move to a favourite place
- **WHEN** a `MovementInteraction` whose step returns the polygon of a region of a places layer moves an agent
- **THEN** the agent's point lies inside that region, and the regions of that layer that contain the agent include it

### Requirement: Trajectories report positions in three dimensions
A trajectory SHALL report the location of an agent in a three-dimensional space as three coordinates, with the agent
ids at every frame.

#### Scenario: Three coordinates per agent
- **WHEN** a user asks a trajectory of a model in a three-dimensional space for the agent locations at frame 5
- **THEN** it returns an array with one row of 3 coordinates per agent alive at frame 5

### Requirement: Agents are drawn in three dimensions and over regions
`AgentsVisual` SHALL draw agents in a three-dimensional space as one batch of 3D points per group, and the region
layers of a space once as static outlines under the agents' 2D points.

#### Scenario: Boids in 3D
- **WHEN** a model of boids in a three-dimensional space is logged
- **THEN** one `Points3D` batch per group is logged per frame at the agents' positions

#### Scenario: Island map
- **WHEN** a model in a space with a boundary layer is logged
- **THEN** the boundary is logged once as static line strips, and the agents are logged per frame as `Points2D`
