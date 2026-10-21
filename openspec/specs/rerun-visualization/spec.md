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

### Requirement: Time series views show the whole run
Every time series view of a default blueprint SHALL show the whole run, whichever visual declares it.

#### Scenario: One-dimensional points
- **WHEN** the default blueprint of an ideal gas in one dimension is built
- **THEN** the time series view of the positions shows the time range from the first frame to the last

#### Scenario: Other states of a points visual
- **WHEN** a points visual in two dimensions also logs a scalar state of its members
- **THEN** the time series view of that state shows the whole run, like the views of scalar elements

### Requirement: Visuals declare their views
Every visual SHALL declare the Rerun views that show its entities: scalars in time series views, points, agents,
lattices, and fields in 2D or 3D spatial views by their dimension, meshes drawn with their value as height in 3D
spatial views, and networks in graph views. A visual registered by
a user SHALL be able to declare its views the same way.

#### Scenario: Views of the built-in visuals
- **WHEN** the views of the scalars, points in three dimensions, image, and agents on a network visuals are listed
- **THEN** they are a time series view, a 3D spatial view, a 2D spatial view, and a graph view, each with the visual's
  entity path as origin

### Requirement: One-dimensional lattices show their space-time image
A one-dimensional lattice or field SHALL log, besides its current row, a space-time image with one row per logged frame,
the oldest at the top, holding a bounded number of the most recent rows. Its view SHALL come before the view of the row.

#### Scenario: Space-time view
- **WHEN** the primary view of `elementary_automaton` is asked for
- **THEN** it is the 2D view of the space-time image of its cells, whose rows are the generations

### Requirement: Points and agents stay legible at any scale
Points without a given radius SHALL take a share of the extent of the positions logged so far, and a group of at most
ten points in two or three dimensions SHALL log the trail of its recent positions as line strips. Agents in a
continuous space without a given radius SHALL take a share of the side of the space. Agents without a space SHALL get
no spatial view.

#### Scenario: Legible points
- **WHEN** `two_body_orbit` is recorded with its default visual
- **THEN** the bodies have a radius of a fiftieth of the extent of the orbit, and each draws its recent orbit as a line

#### Scenario: Agents without a space
- **WHEN** the default blueprint of an agent model without a space, such as `moran_process`, is built
- **THEN** it holds no spatial view, and its primary view is the time series view of the reporters

### Requirement: Systems get a default blueprint
`default_blueprint(system)` SHALL return a Rerun blueprint built from the views of the system's visuals, ordered with
the primary view first, which `primary_view(system)` returns, unless `primary` names another view. Its scalar views SHALL be grouped by element, its
parameters left out unless asked for, its side panels collapsed, and its time panel following the latest frame.

#### Scenario: Blueprint of an ODE model
- **WHEN** the default blueprint of `lotka_volterra` is built
- **THEN** it holds one time series view with the prey and predator populations, no parameter view, collapsed side
  panels, and a time panel that follows the latest frame

#### Scenario: Blueprint of a field model
- **WHEN** the default blueprint of `gray_scott` is built
- **THEN** it holds one 2D spatial view per field state, `u` and `v`, side by side

#### Scenario: Primary view
- **WHEN** `primary_view` of `gray_scott` and of `lotka_volterra` is asked for
- **THEN** it is the 2D view of a field state and the time series view of the populations, the first view of each
  default blueprint

#### Scenario: Primary view by name
- **WHEN** the default blueprint of `gray_scott` is built with `primary='cells/v'`
- **THEN** its first view is the 2D view of the `v` field

#### Scenario: Parameters on request
- **WHEN** the default blueprint is built with `parameters=True`
- **THEN** it also holds a time series view of the parameters

### Requirement: The observer sends and saves blueprints
`RerunObserver` SHALL accept a `blueprint` argument: `None` for today's behaviour, `'default'` for the default
blueprint of the simulated system, or a Rerun blueprint. It SHALL send the blueprint with the recording and save it
next to a saved recording. `save_blueprint(blueprint, path)` SHALL write a `.rbl` file.

#### Scenario: Saved recording with its blueprint
- **WHEN** a simulation runs with `RerunObserver(save_path='run.rrd', blueprint='default')`
- **THEN** `run.rrd` and `run.rbl` exist, and opening both in the viewer shows the default layout

#### Scenario: No blueprint
- **WHEN** a simulation runs with `RerunObserver()` and no blueprint
- **THEN** the recording is the same as before this change

### Requirement: Angles are colored by a cyclic palette
An image of a real state of the `ANGLE` domain SHALL be colored by the cyclic palette of complex phases, so angles one
turn apart share a color.

#### Scenario: Angles one turn apart
- **WHEN** an image visual logs the angles of the XY model
- **THEN** the colors are those of the cyclic palette at the angles, so a site at `-pi` has the color of a site at `pi`

### Requirement: 2D positions are drawn with the vertical axis up
Visuals that log 2D positions SHALL log them with the second coordinate negated, so Rerun's 2D views, whose vertical
axis points down, draw them with the vertical axis up.

#### Scenario: A point above the origin
- **WHEN** a points visual logs a 2D position `(1, 2)`
- **THEN** the logged point is `(1, -2)`

### Requirement: Bodies, rods, trails, and outlines are drawn by a visual
The `'bodies'` visual SHALL draw bodies at positions computed from the states of an element, rods from a pivot through
the bodies, fading trails of chosen bodies, and static outlines, and SHALL log every state as a time series.

#### Scenario: Double pendulum
- **WHEN** the visual of `double_pendulum` logs 20 frames
- **THEN** the logs hold the two bobs under `pendulum/bodies`, one rod strip of three points from the pivot under
  `pendulum/rods`, the trail of the lower bob under `pendulum/trail` with colors more transparent for older segments,
  and the angles as time series

#### Scenario: Billiard table
- **WHEN** the visual of `circular_billiard` logs its frames
- **THEN** the wall of the table is logged once as a static line strip under `particle/outline`

### Requirement: Series of an element have distinct colors
The scalars visual SHALL give every state of an element a distinct series color from a fixed categorical palette.

#### Scenario: Daisyworld
- **WHEN** the scalars visual logs `daisyworld`
- **THEN** the series of `planet/white` and `planet/black` have static colors that differ

### Requirement: Agent traits without a space are drawn
The default visual of an agent group without a space SHALL log the histogram of a numeric trait and the mean and
standard deviation of every numeric trait.

#### Scenario: Moran process
- **WHEN** the default blueprint of `moran_process` is built and its visual logs 10 frames
- **THEN** the first view of the agents is a bar chart of a trait, and the logs hold the mean of every numeric trait

### Requirement: One-dimensional agents are drawn as a space-time diagram
Agents whose position state has one coordinate SHALL be logged as one series per agent, shown in a time series view
without a legend.

#### Scenario: Traffic on a road
- **WHEN** the visual of `charbonneau_traffic` logs a frame
- **THEN** it logs the positions of the cars as scalars under `cars/x`, and its view is a time series view of
  `cars/x`

### Requirement: Fields on networks are drawn as colored graphs
A field on a network space without a visual SHALL take the colored graph visual of lattice groups on networks.

#### Scenario: Network diffusion
- **WHEN** the default blueprint of `network_diffusion` is built
- **THEN** its view of the nodes is a graph view

### Requirement: Color ranges widen with the values
A visual that colors by a state without a given range SHALL widen its range to every value logged so far.

#### Scenario: Agents that start equal
- **WHEN** an agents visual colors by a state that is 0 for every agent at the first frame and spreads later
- **THEN** the agents of a later frame have different colors

### Requirement: Oscillator rings show phase against position
The `'phase_ring'` visual SHALL log the wrapped phases of a group against the member position with the cyclic palette.

#### Scenario: Chimera
- **WHEN** the visual of a chimera model logs a frame
- **THEN** it logs one point per oscillator at its index and its phase in `(-pi, pi]`

### Requirement: Auxiliaries and flows of stocks and flows are logged
The observer SHALL log the auxiliaries and flows of stocks and flows as series under the path of their element, so a
view of an auxiliary is not blank.

#### Scenario: Hilsa fishery
- **WHEN** a Rerun observer records `hilsa_fishery` for 10 frames
- **THEN** the recording holds the series `hilsa/adult_weight`
