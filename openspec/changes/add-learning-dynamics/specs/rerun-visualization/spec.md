# Spec Delta

## ADDED Requirements

### Requirement: Matrix-valued states are drawn as heat maps
`MatrixVisual`, chosen by `VisualSpec('matrix')`, SHALL log a matrix-valued state or parameter as an image colored by
value, one image per frame under `<path>/<variable>`, with a fixed color range when one is given and the range of the
frame otherwise.

#### Scenario: Recurrent weights
- **WHEN** the `elman_network` system with 4 hidden units streams to Rerun
- **THEN** `elman/net/recurrence/A` is a 4 by 4 image at every frame, and its pixels change between the first and the
  last frame

### Requirement: Systems that train stream their loss and parameters
A system holding a gradient-step law SHALL stream the loss and metrics the law records as time series over epochs and
its trained matrices as heat maps. The first view of the default blueprint of `RerunObserver` SHALL be the time series
view of the loss and metrics, with the heat maps of the trained matrices next to it.

#### Scenario: First view of a system that trains
- **WHEN** the default blueprint of `elman_network` is built
- **THEN** its first view is the time series view of `elman/training/loss` and the metrics, with epochs on the time
  axis

#### Scenario: Default blueprint
- **WHEN** the default blueprint of `elman_network` is built
- **THEN** it holds the loss view and one heat map view per trained matrix, and no view of the inner model's states
