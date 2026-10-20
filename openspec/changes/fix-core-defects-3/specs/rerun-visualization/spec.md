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
