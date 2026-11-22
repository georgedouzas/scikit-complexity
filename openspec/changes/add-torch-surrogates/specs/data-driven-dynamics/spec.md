## ADDED Requirements

### Requirement: Neural-operator surrogates of fields
`neural_operator(source, ...)` SHALL learn a neural operator that maps the fields of a field system at one time to its
fields `lead` samples later, from one or more runs of the system on the same grid, as trajectories, an experiment
result, or observed runs, with whole runs held out for validation, and SHALL return a frozen result with the training
and validation errors and a `forecast` of `horizon` steps.

#### Scenario: Gray-Scott surrogate
- **WHEN** `neural_operator` learns from 16 runs of `GrayScottSystem` on a 64 by 64 grid from random initial spots
- **THEN** the relative error of one step on held-out runs is below 0.05

#### Scenario: Source that is not a field
- **WHEN** a user passes the trajectory of a system without fields
- **THEN** it raises `ValidationError`, whose message asks for runs of a field system

### Requirement: Neural-operator surrogates are systems
The result of `neural_operator` SHALL hold `system`, a ready-made system of one element on the lattice of the source
whose update law applies the learned map at every step. Its model parameter `shape` SHALL accept a grid of another
size, so the learned map runs at a finer resolution than it was trained on.

#### Scenario: Surrogate runs
- **WHEN** the `system` of a surrogate of `KuramotoSivashinskySystem` is prepared and simulated for 50 steps
- **THEN** `prepare` reports no error, `form_` is the discrete map, and the trajectory holds the field at every step

#### Scenario: Finer grid
- **WHEN** a surrogate trained on a 64-point Kuramoto-Sivashinsky grid sets `shape=(128,)` on its system and steps once
  from the 128-point field of a true run
- **THEN** the relative error of the step is below twice the error at 64 points

### Requirement: Neural operators learn across parameters
When the runs of the source differ in the values of model parameters, such as the runs of a `sweep`, `neural_operator`
SHALL take those parameters as inputs of the map, and the surrogate system SHALL take them as its model parameters.

#### Scenario: Surrogate over a sweep
- **WHEN** `neural_operator` learns from a sweep of `GrayScottSystem` over `F` and its system is set to an `F`
  inside the swept range that no run used
- **THEN** the surrogate runs and its one-step error at that `F` is below 0.1
