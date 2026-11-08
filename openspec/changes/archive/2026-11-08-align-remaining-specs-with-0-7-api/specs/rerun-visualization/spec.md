## MODIFIED Requirements

### Requirement: One-dimensional lattices show their space-time image
A one-dimensional lattice or field SHALL log, besides its current row, a space-time image with one row per logged frame,
the oldest at the top, holding a bounded number of the most recent rows. Its view SHALL come before the view of the row.

#### Scenario: Space-time view
- **WHEN** the primary view of `ElementaryAutomatonSystem` is asked for
- **THEN** it is the 2D view of the space-time image of its cells, whose rows are the generations

### Requirement: Points and agents stay legible at any scale
Points without a given radius SHALL take a share of the extent of the positions logged so far, and a group of at most
ten points in two or three dimensions SHALL log the trail of its recent positions as line strips. Agents in a
continuous space without a given radius SHALL take a share of the side of the space. Agents without a space SHALL get
no spatial view.

#### Scenario: Legible points
- **WHEN** `TwoBodyOrbitSystem` is recorded with its default visual
- **THEN** the bodies have a radius of a fiftieth of the extent of the orbit, and each draws its recent orbit as a line

#### Scenario: Agents without a space
- **WHEN** the default blueprint of an agent system without a space, such as `MoranProcessSystem`, is built
- **THEN** it holds no spatial view, and its primary view is the time series view of the reporters

### Requirement: Systems get a default blueprint
`default_blueprint(system)` SHALL return a Rerun blueprint built from the views of the system's visuals, ordered with
the primary view first, which `primary_view(system)` returns, unless `primary` names another view. Its scalar views SHALL be grouped by element, its
parameters left out unless asked for, its side panels collapsed, and its time panel following the latest frame.

#### Scenario: Blueprint of an ODE system
- **WHEN** the default blueprint of `LotkaVolterraSystem` is built
- **THEN** it holds one time series view with the prey and predator populations, no parameter view, collapsed side
  panels, and a time panel that follows the latest frame

#### Scenario: Blueprint of a field system
- **WHEN** the default blueprint of `GrayScottSystem` is built
- **THEN** it holds one 2D spatial view per field state, `u` and `v`, side by side

#### Scenario: Primary view
- **WHEN** `primary_view` of `GrayScottSystem` and of `LotkaVolterraSystem` is asked for
- **THEN** it is the 2D view of a field state and the time series view of the populations, the first view of each
  default blueprint

#### Scenario: Primary view by name
- **WHEN** the default blueprint of `GrayScottSystem` is built with `primary='cells/v'`
- **THEN** its first view is the 2D view of the `v` field

#### Scenario: Parameters on request
- **WHEN** the default blueprint is built with `parameters=True`
- **THEN** it also holds a time series view of the parameters

### Requirement: Bodies, rods, trails, and outlines are drawn by a visual
The `'bodies'` visual SHALL draw bodies at positions computed from the states of an element, rods from a pivot through
the bodies, fading trails of chosen bodies, and static outlines, and SHALL log every state as a time series.

#### Scenario: Double pendulum
- **WHEN** the visual of `DoublePendulumSystem` logs 20 frames
- **THEN** the logs hold the two bobs under `pendulum/bodies`, one rod strip of three points from the pivot under
  `pendulum/rods`, the trail of the lower bob under `pendulum/trail` with colors more transparent for older segments,
  and the angles as time series

#### Scenario: Billiard table
- **WHEN** the visual of `CircularBilliardSystem` logs its frames
- **THEN** the wall of the table is logged once as a static line strip under `particle/outline`

### Requirement: Agent traits without a space are drawn
The default visual of an agent group without a space SHALL log the histogram of a numeric trait and the mean and
standard deviation of every numeric trait. The histogram of every trait SHALL bin by the range of its own values,
widened to every value logged so far, and the color range of the visual SHALL only start the range of its colored
trait.

#### Scenario: Moran process
- **WHEN** the default blueprint of `MoranProcessSystem` is built and its visual logs 10 frames
- **THEN** the first view of the agents is a bar chart of a trait, and the logs hold the mean of every numeric trait

#### Scenario: Integer trait outside the color range
- **WHEN** a histogram visual colored by a trait in `(0, 1)` logs an integer trait with negative values
- **THEN** the histogram of that trait counts every agent and its range covers the negative values

### Requirement: One-dimensional agents are drawn as a space-time diagram
Agents whose position state has one coordinate SHALL be logged as one series per agent, shown in a time series view
without a legend.

#### Scenario: Traffic on a road
- **WHEN** the visual of `CharbonneauTrafficSystem` logs a frame
- **THEN** it logs the positions of the cars as scalars under `cars/x`, and its view is a time series view of
  `cars/x`

### Requirement: Fields on networks are drawn as colored graphs
A field on a network space without a visual SHALL take the colored graph visual of lattice groups on networks.

#### Scenario: Network diffusion
- **WHEN** the default blueprint of `NetworkDiffusionSystem` is built
- **THEN** its view of the nodes is a graph view

### Requirement: Flows and observables of stocks and flows are logged
The observer SHALL log the flows of stocks and flows and the observables of the run as series under the path of their
element, so a view of an observed expression is not blank.

#### Scenario: Hilsa fishery
- **WHEN** a Rerun observer records `HilsaFisherySystem` for 10 frames with the observables of its system
- **THEN** the recording holds the series `hilsa/adult_weight`

### Requirement: Matrix-valued states are drawn as heat maps
`MatrixVisual`, chosen by `VisualSpec('matrix')`, SHALL log a matrix-valued state or parameter as an image colored by
value, one image per frame under `<path>/<variable>`, with a fixed color range when one is given and the range of the
frame otherwise.

#### Scenario: Recurrent weights
- **WHEN** the `ElmanNetworkSystem` system with 4 hidden units streams to Rerun
- **THEN** `net/recurrence/A` is a 4 by 4 image at every frame, and its pixels change between the first and the
  last frame

### Requirement: Systems that train stream their loss and parameters
A system holding a gradient-step law SHALL stream the loss and metrics the law records as time series over epochs and
its trained matrices as heat maps. The first view of the default blueprint of `RerunObserver` SHALL be the time series
view of the loss and metrics, with the heat maps of the trained matrices next to it.

#### Scenario: First view of a system that trains
- **WHEN** the default blueprint of `ElmanNetworkSystem` is built
- **THEN** its first view is the time series view of `training/loss` and the metrics, with epochs on the time
  axis

#### Scenario: Default blueprint
- **WHEN** the default blueprint of `ElmanNetworkSystem` is built
- **THEN** it holds the loss view and one heat map view per trained matrix, and no view of the inner system's states
