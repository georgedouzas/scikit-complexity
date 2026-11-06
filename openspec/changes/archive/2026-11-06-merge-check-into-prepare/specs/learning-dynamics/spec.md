## MODIFIED Requirements

### Requirement: Optimizers are configured and change while running
`TrainingInteraction` SHALL take an optimizer by name, at least `'sgd'`, `'momentum'`, `'rmsprop'`, and `'adam'`, and
its hyperparameters SHALL be parameters of the interaction, so `Simulation.update_parameters` and a simulation's
`schedule` change them between epochs. The optimizer's internal state SHALL
persist across frames and SHALL reset when the simulation is reset.

#### Scenario: Zero learning rate
- **WHEN** a system holding a gradient-step law runs for 5 epochs with a learning rate of 0
- **THEN** every trained parameter keeps its initial value exactly

#### Scenario: Learning rate schedule
- **WHEN** a simulation has a schedule that halves `training.learning_rate` every 50 epochs
- **THEN** the recorded parameter `training.learning_rate` halves at frames 50, 100, and 150

#### Scenario: Unknown optimizer
- **WHEN** a `TrainingInteraction` names the optimizer `'lbfgs-b'`
- **THEN** `prepare` raises `ValidationError` that names the optimizer and lists the supported ones
