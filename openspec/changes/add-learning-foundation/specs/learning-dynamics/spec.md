# Spec Delta

## MODIFIED Requirements

### Requirement: Datasets feed inner parameters at every step
`TrainingInteraction` SHALL take as `data` an `ObservedRunsSpec`. Every run SHALL start from the inner system's initial
states with its conditions set, SHALL set every input parameter to its value at the current inner time, held since its
last given time, before the laws of the step are evaluated, as a schedule does in a simulation, and SHALL be unrolled on
the union of its observation and input times. Missing values SHALL not enter the loss, and runs with different times
SHALL be padded and masked so a batch compiles once.

#### Scenario: Inputs and targets
- **WHEN** the runs give the inputs `inputs.x` and `outputs.target` for an input group and an output group of one member
  each, at 12 steps for each of 996 runs
- **THEN** at inner step `k` of run `i` the input reads the value of run `i` at step `k`, and the target the value at
  step `k`

#### Scenario: Masked steps
- **WHEN** the observed values of the first 3 times of every run are NaN
- **THEN** the loss and its gradient do not depend on the inner outputs at those times, within 1e-12

#### Scenario: Runs of different lengths
- **WHEN** the runs of a batch hold 8, 12, and 20 observation times
- **THEN** the loss equals the sum of the losses of the three runs computed one by one, within 1e-12

#### Scenario: Shape mismatch
- **WHEN** the values of an input or an observed path of a run have a trailing shape different from the shape of its
  parameter, state, or group
- **THEN** the compiler's report holds an issue with the code `learning-data-shape` that names the run, the path, the
  expected shape, and the given shape

#### Scenario: Unknown path
- **WHEN** a run observes a path that the inner system does not have
- **THEN** the compiler's report holds an issue with the code `learning-data-path` that names the path and lists the
  states, observables, and parameters of the inner system

## ADDED Requirements

### Requirement: Losses follow from the observation model
The loss of a `TrainingInteraction` SHALL be an observable of the inner system or one of the names `'squares'`, `'log'`,
and `'likelihood'`. The likelihood SHALL be that of the noise of the observation model of the data, Gaussian, log-normal,
Poisson, or negative binomial, evaluated at the predicted means.

#### Scenario: Poisson likelihood
- **WHEN** the data of an SIR system are observed with Poisson noise and the loss is `'likelihood'`
- **THEN** the loss of one run equals the sum of `mu - y log mu + log y!` over its samples, where `mu` is the predicted
  mean, within 1e-10

#### Scenario: Likelihood without noise
- **WHEN** the loss is `'likelihood'` and the observation model has `noise='none'`
- **THEN** the compiler's report holds an issue with the code `learning-loss` that says a likelihood needs a noise model

### Requirement: Validation runs stop training early
A `TrainingInteraction` SHALL take `validation`, the share of runs held out, and `patience`, and SHALL record the loss of
the held-out runs every epoch as `<interaction path>/validation_loss`, stop when it has not fallen for `patience` epochs,
and keep the parameters of its lowest value.

#### Scenario: Early stop
- **WHEN** a network that overfits 4 training runs is trained with `validation=0.5` and `patience=10`
- **THEN** the training stops before the last epoch, and the recorded parameters of the last frame are those of the
  epoch of lowest validation loss

### Requirement: One optimizer loop serves every training
The epochs, batches, optimizer state, validation, and early stopping of `GradientDescentSolver` SHALL be one private loop
that also runs the training of policies and of learned analyses, given a function from parameters and a batch to a loss
and its gradient.

#### Scenario: Same optimizer path
- **WHEN** a quadratic loss is minimised once by `GradientDescentSolver` and once by the loop with the same batches,
  optimizer, and seed
- **THEN** the parameters of every epoch are equal
