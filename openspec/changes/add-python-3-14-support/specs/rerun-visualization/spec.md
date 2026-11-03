# Spec Delta

## ADDED Requirements

### Requirement: A viewer that cannot run or be reached does not stop a run
A run asked to spawn a Rerun viewer on a Linux machine without a display SHALL warn and run without a viewer. A run
streaming to a viewer SHALL wait a limited time for it when the run ends, then warn and keep its results. A run that only
saves to a file SHALL wait for the whole recording.

#### Scenario: No display
- **WHEN** a user runs a simulation with `RerunObserver(spawn=True)` on a Linux machine without a display
- **THEN** no viewer is spawned, a warning says so, and the run returns its trajectory

#### Scenario: Viewer out of reach
- **WHEN** a run streams to a viewer that cannot be reached
- **THEN** the run warns within the limit and returns its trajectory

#### Scenario: Saving to a file
- **WHEN** a user runs a simulation with `RerunObserver(save_path=...)` and no viewer
- **THEN** the run waits until the whole recording is written
