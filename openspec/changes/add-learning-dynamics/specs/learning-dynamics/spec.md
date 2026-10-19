# Spec Delta

## Purpose

Lets a user make the parameters of any differentiable model evolve by learning: a loss defined from the model's own
states over an unrolled run, its exact gradient, an optimizer, mini-batches of data, and epochs as the time of the
learning system, so neural networks, adaptive controllers, learning agents, and parameter fits run through the same
simulation, trajectory, and visualization interface as every other model.

## ADDED Requirements

### Requirement: Learning systems evolve trained parameters over epochs
A learning system SHALL wrap an inner model and name the inner parameters it trains by dotted path. Its states SHALL
be the trained parameters, with their inner names and shapes, and its parameters SHALL be the optimizer's
hyperparameters. One unit of its simulated time SHALL be one epoch, a pass over every sequence of its dataset, and
frame 0 SHALL hold the inner model's initial values.

#### Scenario: Epochs as frames
- **WHEN** a learning system of an Elman network with 4 hidden units runs with `dt=1` until 200
- **THEN** the trajectory has 201 frames at times 0 to 200, `trajectory['elman/readout.W']` has shape `(201, 1, 4)`,
  and its first row equals the readout weights of the inner model

#### Scenario: Time between frames
- **WHEN** a learning system runs with `dt=0.5`
- **THEN** the simulation raises a named error that says the time between frames of a learning system is a positive
  whole number of epochs and names the value

#### Scenario: Untrained parameter path
- **WHEN** a learning system names `readout.V` as trained and the inner model has no such parameter
- **THEN** compiling raises a named error that names the path and lists the parameters of the inner model

### Requirement: Losses and metrics are inner observables
The loss of a learning system SHALL be the label of an observable of the inner model, evaluated at every inner step
and averaged over the steps and sequences of a batch with the dataset's step weights. A learning system SHALL record
the mean training loss of each epoch as the observable `<label>/loss`, and every metric it names, another observable
of the inner model, as `<label>/<metric>`, averaged over the whole dataset with the trained parameters at the end of
the epoch.

#### Scenario: Loss at the start
- **WHEN** a learning system is compiled and its frame 0 is recorded
- **THEN** its metric equal to the loss matches the weighted mean of the loss observable over a NumPy run of the
  inner model on every sequence with the initial parameters, within 1e-10

#### Scenario: Unknown loss
- **WHEN** a learning system names a loss that is not an observable of the inner model
- **THEN** compiling raises a named error that names the loss and lists the observables of the inner model

### Requirement: Datasets feed inner parameters at every step
A dataset SHALL map dotted inner parameter paths to arrays of shape `(sequences, steps)` followed by the shape of the
parameter, and SHALL set each parameter to its value at the current inner step before the step's laws are evaluated,
as a schedule does in a simulation. It SHALL take optional step weights of shape `(sequences, steps)`, and every
sequence SHALL start from the inner model's initial states.

#### Scenario: Inputs and targets
- **WHEN** a dataset maps `inputs.x` and `outputs.target` to arrays of shape `(996, 12, 1)` for an input group and an
  output group of one member each
- **THEN** at inner step `k` of sequence `i` the input reads the value at `[i, k]` and the target the value at `[i, k]`

#### Scenario: Masked steps
- **WHEN** the step weights are 0 on the first 3 steps of every sequence
- **THEN** the loss and its gradient do not depend on the inner outputs at those steps, within 1e-12

#### Scenario: Shape mismatch
- **WHEN** a dataset array has a different number of steps from another, or a trailing shape different from its
  parameter's shape
- **THEN** compiling raises a named error that names the path, the expected shape, and the given shape

### Requirement: Mini-batches are drawn reproducibly
Each epoch SHALL shuffle the sequences with the simulation's random stream, split them into batches of the given size
with a smaller last batch when the size does not divide the count, and apply one optimizer update per batch. Two runs
with the same seed SHALL record identical trajectories, and a batch size of the number of sequences SHALL give full
batch gradient descent with one update per epoch.

#### Scenario: Same seed
- **WHEN** a learning system runs twice for 20 epochs with `random_state=0`
- **THEN** both trajectories are equal element by element

#### Scenario: Updates per epoch
- **WHEN** a dataset holds 996 sequences and the batch size is 16
- **THEN** each epoch applies 63 updates, the last on 4 sequences

### Requirement: Gradients are exact through the unrolled run
The gradient of the loss with respect to the trained parameters SHALL be computed by automatic differentiation of the
unrolled inner run, for inner models whose form is a discrete map or ordinary differential equations. Inner models
of other forms, laws given as opaque callables, and rules written as code SHALL be rejected.

#### Scenario: Gradient against finite differences
- **WHEN** the gradient of the loss of an Elman network over one batch is compared with central finite differences of
  step 1e-6 in double precision
- **THEN** they agree within 1e-6 relative error for every trained parameter

#### Scenario: Gradient through an ODE
- **WHEN** the loss of a Lotka-Volterra model is the squared error to observed prey and predator series at given times
- **THEN** its gradient with respect to the four rate parameters agrees with central finite differences within 1e-5
  relative error

#### Scenario: Not differentiable
- **WHEN** a learning system wraps an agent-based model, a jump process, or a model with a law given as a Python
  callable
- **THEN** compiling raises a named error that names the form or the law, and says that learning needs a discrete map
  or ordinary differential equations written as symbolic laws

### Requirement: Backpropagation through time can be truncated
A learning system SHALL take a truncation, the number of inner steps the gradient flows back through the inner states,
or `None` for full backpropagation through time. A truncation of 1 SHALL treat the previous inner states as inputs,
as Elman's original training does.

#### Scenario: Truncation of one step
- **WHEN** an Elman network is trained with a truncation of 1
- **THEN** the gradient with respect to the recurrent weights equals the gradient computed with the previous hidden
  states held as constants, within 1e-12

#### Scenario: Full and truncated gradients
- **WHEN** the gradients of the same batch are computed with no truncation and with a truncation of 1
- **THEN** the gradients with respect to the input weights differ, and the gradients with respect to the readout
  weights are equal within 1e-12

### Requirement: Optimizers are configured and change while running
A learning system SHALL take an optimizer by name, at least `'sgd'`, `'momentum'`, `'rmsprop'`, and `'adam'`, with its
hyperparameters, and SHALL expose the hyperparameters as its parameters, so `update_parameters` and a simulation's
`schedule` change them between epochs. The optimizer's internal state SHALL persist across frames and SHALL reset
when the simulation is reset.

#### Scenario: Zero learning rate
- **WHEN** a learning system runs for 5 epochs with a learning rate of 0
- **THEN** every trained parameter keeps its initial value exactly

#### Scenario: Learning rate schedule
- **WHEN** a simulation of a learning system has a schedule that halves `elman.learning_rate` every 50 epochs
- **THEN** the recorded parameter `elman.learning_rate` halves at frames 50, 100, and 150

#### Scenario: Unknown optimizer
- **WHEN** a learning system names the optimizer `'lbfgs-b'`
- **THEN** creating it raises a named error that names the optimizer and lists the supported ones

### Requirement: Interactions take matrix-valued parameters
An interaction parameter SHALL accept a shape given by two of its roles, and SHALL then hold a matrix with one row
per member of the first role's group and one column per member of the second's. The weights of a total or mean
reduction SHALL accept such a parameter, so member `i` reads `sum_j W_ij a_j`. It SHALL be addressed by path, recorded
with its shape, changed while running, and trained like a scalar parameter.

#### Scenario: Weighted input
- **WHEN** a hidden group of 4 members reads `total(s.source.x, weights=s.W)` from a source group of 2 members with a
  matrix parameter `W` of shape `(4, 2)`
- **THEN** member `i` reads `(W x)_i` within 1e-12 on the NumPy, Numba, and JAX backends

#### Scenario: Wrong shape
- **WHEN** `update_parameters` sets that parameter to an array of shape `(2, 4)`
- **THEN** it raises a named error that names the parameter, the expected shape `(4, 2)`, and the given shape

### Requirement: The trained model is a system
A learning system SHALL build the inner model with the trained parameters of any recorded state, as a system that
runs with every backend and solver of its form, so the trained model can be simulated, analysed, and visualised like
any model.

#### Scenario: Trained model agrees with training
- **WHEN** the inner model is built from the state of the last frame and run with the NumPy backend on the first
  sequence, its inputs given by a schedule
- **THEN** the loss observable it records equals the one the learning system evaluates for that sequence within 1e-10

### Requirement: Any differentiable model can be fitted
A learning system SHALL accept any inner model of a differentiable form, not only neural networks, so the parameters
of a model of any field can be fitted to observed series by gradient descent.

#### Scenario: Lotka-Volterra fit
- **WHEN** a learning system with Adam fits the four rate parameters of `lotka_volterra`, started 30% off, to noise-free
  prey and predator series of 50 points over 20 time units
- **THEN** after 500 epochs every parameter is within 1% of its true value and the loss has fallen by at least four
  orders of magnitude

### Requirement: Local learning rules stay ordinary laws
A learning rule that changes a weight from quantities at its own synapse, such as Hebb's or Oja's rule, SHALL be
written as update or rate laws on weight states, and SHALL run in the default install without the learning extra.

#### Scenario: Oja's rule
- **WHEN** a neuron with 2 weight states follows Oja's rule as rate laws on inputs with a covariance of principal
  direction `(1, 1) / sqrt(2)`
- **THEN** its weight vector converges to that direction, up to sign, within 1e-2

### Requirement: Learning needs its optional extra
Gradient-based learning SHALL ship in the optional extra `learning`. Without it, compiling a learning system or
selecting the JAX backend SHALL raise a named error that names the extra and how to install it, and every other
capability SHALL work.

#### Scenario: Missing extra
- **WHEN** a user without the extra compiles a learning system
- **THEN** compiling raises a named error whose message names `pip install 'scikit-complexity[learning]'`, and running
  `lotka_volterra` in the same session succeeds
