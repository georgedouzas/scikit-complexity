# Spec Delta

## MODIFIED Requirements

### Requirement: Lattice models run fast and stream as images
Lattice rules SHALL run vectorised on the NumPy and Numba backends, and lattices SHALL stream to Rerun as images colored
by state, unless the lattice group or its element type declares another visual.

#### Scenario: Large Game of Life
- **WHEN** a 1000 × 1000 Game of Life runs on the Numba backend with real-time pacing
- **THEN** it sustains 30 steps per second, and the viewer shows the lattice as an image

#### Scenario: Visual of the element type
- **WHEN** an element type declares the visual `VisualSpec('heights')` and a lattice group of it is created without a
  visual argument
- **THEN** the group's visual is `heights`, not an image

#### Scenario: Element type without a visual
- **WHEN** an element type declares no visual and a lattice group of it is created without a visual argument
- **THEN** the group's visual is an image
