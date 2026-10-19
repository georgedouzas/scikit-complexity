# Spec Delta

## ADDED Requirements

### Requirement: Matrix-valued states are drawn as heat maps
A matrix-valued state or parameter SHALL be loggable as an image colored by value, one image per frame under
`<path>/<variable>`, with a fixed color range when one is given and the range of the frame otherwise.

#### Scenario: Recurrent weights
- **WHEN** a learning system of an Elman network with 4 hidden units streams to Rerun
- **THEN** `elman/recurrence/A` is a 4 by 4 image at every frame, and its pixels change between the first and the last
  frame

### Requirement: Learning systems stream their loss and parameters
A learning system SHALL stream its loss and metrics as time series over epochs and its trained matrices as heat maps.
Its primary view SHALL be the time series view of the loss and metrics, and its default blueprint SHALL hold the heat
maps of the trained matrices next to it.

#### Scenario: Primary view of a learning system
- **WHEN** the primary view of `elman_network` is asked for
- **THEN** it is the time series view of `elman/loss` and the metrics, with epochs on the time axis

#### Scenario: Default blueprint
- **WHEN** the default blueprint of `elman_network` is built
- **THEN** it holds the loss view and one heat map view per trained matrix, and no view of the inner model's states
