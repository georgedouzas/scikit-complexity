## MODIFIED Requirements

### Requirement: Gradient-step laws evolve adaptive parameters over epochs
`GradientStepLaw` SHALL be a discrete-time law, returned by a `TrainingInteraction` bound to a nested system through
its role `system`, that trains adaptive parameters of that system named by dotted path. A system holding it under
discrete time SHALL advance one epoch, a pass over its dataset, per unit of time, record the trained parameters with
their inner paths and shapes, and hold the inner system's initial values in frame 0, without any subclass of `CompositeSystem`.

#### Scenario: Epochs as frames
- **WHEN** a system `elman` holding an Elman network `net` with 4 hidden units and a training interaction runs with
  `dt=1` until 200
- **THEN** the trajectory has 201 frames at times 0 to 200, `trajectory.values('net/readout.W')` has shape
  `(201, 1, 4)`, and its first row equals the initial readout weights of the inner system

#### Scenario: Time between frames
- **WHEN** a system holding a gradient-step law runs with `dt=0.5`
- **THEN** the simulation raises a named error that says the time between frames of a system that trains is a positive
  whole number of epochs and names the value

#### Scenario: Untrained parameter path
- **WHEN** a gradient-step law names `readout.V` as trained and the inner system has no such parameter
- **THEN** the compiler's report holds an issue with the code `learning-trained-path` whose path is the law and whose
  message names the path and lists the parameters of the inner system

#### Scenario: Parameter not adaptive
- **WHEN** a gradient-step law names a trained parameter that is not declared `adaptive=True`
- **THEN** the compiler's report holds an issue with the code `learning-not-adaptive` whose fix says to declare the
  parameter `ParameterSpec(adaptive=True)`
