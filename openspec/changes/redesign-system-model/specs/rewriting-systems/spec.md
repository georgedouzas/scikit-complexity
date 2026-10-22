# Spec Delta

## MODIFIED Requirements

### Requirement: Turtles interpret words as geometry
A word geometry, the turtle interpretation of L-systems, SHALL interpret a word as line segments in two or three
dimensions from declared symbol meanings, moving forward with or without drawing, turning by a parameter angle,
pushing and popping its state with brackets, and scaling its step by a factor per generation. A module's first
parameter SHALL override the step or the angle where the module has one. The word geometry SHALL be the visual
`WordGeometry` of `skcomplex.core.viz`, which a word takes as its `geometry`.

#### Scenario: Koch curve length
- **WHEN** the word of the Koch L-system `F → F+F--F+F` with a 60° angle and a step of `3^-n` is interpreted at
  generation `n = 4`
- **THEN** the geometry has 256 segments, its total length is `(4/3)^4` within 1e-12, and it starts at `(0, 0)` and
  ends at `(1, 0)`

#### Scenario: Branches return
- **WHEN** the word `F[+F]F` is interpreted with a 90° angle and a unit step
- **THEN** the geometry has 3 segments, and the third starts where the second segment's branch began
