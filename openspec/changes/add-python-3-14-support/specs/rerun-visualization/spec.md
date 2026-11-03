# Spec Delta

## ADDED Requirements

### Requirement: A viewer that cannot be reached does not stop a run
A run streaming to a Rerun viewer SHALL wait a limited time for the viewer when it ends, then warn and keep its results.
A run that only saves to a file SHALL wait for the whole recording.

#### Scenario: No display
- **WHEN** a user runs a simulation with `RerunObserver(spawn=True)` on a machine without a display
- **THEN** the run returns its trajectory and a warning names the failure, rather than hanging

#### Scenario: Saving to a file
- **WHEN** a user runs a simulation with `RerunObserver(save_path=...)` and no viewer
- **THEN** the run waits until the whole recording is written
