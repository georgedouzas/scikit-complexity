# Spec Delta

## ADDED Requirements

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
