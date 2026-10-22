# Spec Delta

## Purpose

Lets a user make the parameters of any differentiable model evolve by learning: a loss defined from the model's own
states over an unrolled run, its exact gradient, an optimizer, mini-batches of data, and epochs as the time of the
system that trains, so neural networks, adaptive controllers, learning agents, and parameter fits run through the same
system, simulation, trajectory, and visualization interface as every other model.

## ADDED Requirements

### Requirement: Gradient-step laws evolve adaptive parameters over epochs
A gradient-step law SHALL be a discrete law, returned by an interaction bound to a nested system, that trains
adaptive parameters of that system named by dotted path. A system holding it under discrete time SHALL advance one
epoch, a pass over its dataset, per unit of time, record the trained parameters with their inner paths and shapes, and
hold the inner model's initial values in frame 0, without any subclass of `System`.

#### Scenario: Epochs as frames
- **WHEN** a system `elman` holding an Elman network `net` with 4 hidden units and a training interaction runs with
  `dt=1` until 200
- **THEN** the trajectory has 201 frames at times 0 to 200, `trajectory['elman/net/readout.W']` has shape
  `(201, 1, 4)`, and its first row equals the initial readout weights of the inner model

#### Scenario: Time between frames
- **WHEN** a system holding a gradient-step law runs with `dt=0.5`
- **THEN** the simulation raises a named error that says the time between frames of a system that trains is a positive
  whole number of epochs and names the value

#### Scenario: Untrained parameter path
- **WHEN** a gradient-step law names `readout.V` as trained and the inner model has no such parameter
- **THEN** validation reports an issue whose path is the law and whose message names the path and lists the parameters
  of the inner model

#### Scenario: Parameter not adaptive
- **WHEN** a gradient-step law names a trained parameter that is not declared `adaptive=True`
- **THEN** validation reports an issue whose fix says to declare the parameter adaptive

### Requirement: Losses and metrics are observables of the inner model
The loss of a gradient-step law SHALL be an observable over the symbols of its nested system, evaluated at every inner
step and averaged over the steps and sequences of a batch with the dataset's step weights. The law SHALL record the
mean training loss of each epoch as `<interaction path>/loss`, and every metric it names, another observable of the
nested system, as `<interaction path>/<metric>`, averaged over the whole dataset with the trained parameters at the end
of the epoch.

#### Scenario: Loss at the start
- **WHEN** a system holding a gradient-step law is compiled and its frame 0 is recorded
- **THEN** its metric equal to the loss matches the weighted mean of the loss observable over a NumPy run of the
  inner model on every sequence with the initial parameters, within 1e-10

#### Scenario: Unknown loss
- **WHEN** a gradient-step law is given a loss observable that reads a variable the nested system does not have
- **THEN** validation reports an issue that names the loss and the missing variable

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
- **THEN** validation reports an issue that names the path, the expected shape, and the given shape

### Requirement: Mini-batches are drawn reproducibly
Each epoch SHALL shuffle the sequences with the simulation's random stream, split them into batches of the given size
with a smaller last batch when the size does not divide the count, and apply one optimizer update per batch. Two runs
with the same seed SHALL record identical trajectories, and a batch size of the number of sequences SHALL give full
batch gradient descent with one update per epoch.

#### Scenario: Same seed
- **WHEN** a system holding a gradient-step law runs twice for 20 epochs with `random_state=0`
- **THEN** both trajectories are equal element by element

#### Scenario: Updates per epoch
- **WHEN** a dataset holds 996 sequences and the batch size is 16
- **THEN** each epoch applies 63 updates, the last on 4 sequences

### Requirement: Gradients are exact through the unrolled run
The gradient of the loss with respect to the trained parameters SHALL be computed by automatic differentiation of the
unrolled nested system, for nested systems that hold only rate, update, and constraint laws on real states. The
gradient-step law SHALL declare these requirements, and validation SHALL report a nested system with noise, jump,
rule, or opaque callable laws, before compilation.

#### Scenario: Gradient against finite differences
- **WHEN** the gradient of the loss of an Elman network over one batch is compared with central finite differences of
  step 1e-6 in double precision
- **THEN** they agree within 1e-6 relative error for every trained parameter

#### Scenario: Gradient through an ODE
- **WHEN** the loss of a Lotka-Volterra model is the squared error to observed prey and predator series at given times
- **THEN** its gradient with respect to the four rate parameters agrees with central finite differences within 1e-5
  relative error

#### Scenario: Not differentiable
- **WHEN** a gradient-step law is bound to a nested system with rule laws, jump laws, or a law given as a Python
  callable
- **THEN** validation reports an issue that names the law and says that learning needs a discrete map or ordinary
  differential equations written as symbolic laws

### Requirement: Backpropagation through time can be truncated
A gradient-step law SHALL take a truncation, the number of inner steps the gradient flows back through the inner
states, or `None` for full backpropagation through time. A truncation of 1 SHALL treat the previous inner states as
inputs, as Elman's original training does.

#### Scenario: Truncation of one step
- **WHEN** an Elman network is trained with a truncation of 1
- **THEN** the gradient with respect to the recurrent weights equals the gradient computed with the previous hidden
  states held as constants, within 1e-12

#### Scenario: Full and truncated gradients
- **WHEN** the gradients of the same batch are computed with no truncation and with a truncation of 1
- **THEN** the gradients with respect to the input weights differ, and the gradients with respect to the readout
  weights are equal within 1e-12

### Requirement: Optimizers are configured and change while running
A gradient-step law SHALL take an optimizer by name, at least `'sgd'`, `'momentum'`, `'rmsprop'`, and `'adam'`, with
its hyperparameters, and the hyperparameters SHALL be parameters of the interaction that returns the law, so
`update_parameters` and a simulation's `schedule` change them between epochs. The optimizer's internal state SHALL
persist across frames and SHALL reset when the simulation is reset.

#### Scenario: Zero learning rate
- **WHEN** a system holding a gradient-step law runs for 5 epochs with a learning rate of 0
- **THEN** every trained parameter keeps its initial value exactly

#### Scenario: Learning rate schedule
- **WHEN** a simulation has a schedule that halves `elman/training.learning_rate` every 50 epochs
- **THEN** the recorded parameter `elman/training.learning_rate` halves at frames 50, 100, and 150

#### Scenario: Unknown optimizer
- **WHEN** a gradient-step law names the optimizer `'lbfgs-b'`
- **THEN** creating it raises a named error that names the optimizer and lists the supported ones

### Requirement: The learning solver and the JAX backend declare their support
The learning solver SHALL declare support for discrete time and gradient-step laws, and the JAX backend SHALL declare
support for continuous and discrete time with rate, update, equation, and gradient-step laws on real states. With
`'auto'`, a simulation SHALL choose the learning solver for a system holding a gradient-step law. Validation SHALL
report any other solver chosen for such a system.

#### Scenario: Automatic solver
- **WHEN** a system holding a gradient-step law is simulated with the default solver and backend
- **THEN** the learning solver and the JAX backend are chosen

#### Scenario: Unsupported solver
- **WHEN** a system holding a gradient-step law is validated with the solver `'rk4'`
- **THEN** the report holds one error that says the solver does not support gradient-step laws and names the learning
  solver

### Requirement: Interactions take matrix-valued parameters
An interaction parameter SHALL accept a shape given by two of its roles, and SHALL then hold a matrix with one row
per member of the first role's group and one column per member of the second's. The weights of a total or mean
reduction SHALL accept it, so member `i` reads `sum_j W_ij a_j`, and it SHALL be addressed by path, recorded with its
shape, and changed or trained like a scalar parameter. Validation SHALL report its roles not bound to groups.

#### Scenario: Weighted input
- **WHEN** a hidden group of 4 members reads `total(s.source.x, weights=s.W)` from a source group of 2 members with a
  matrix parameter `W` of shape `(4, 2)`
- **THEN** member `i` reads `(W x)_i` within 1e-12 on the NumPy, Numba, and JAX backends

#### Scenario: Wrong shape
- **WHEN** `update_parameters` sets that parameter to an array of shape `(2, 4)`
- **THEN** it raises a named error that names the parameter, the expected shape `(4, 2)`, and the given shape

#### Scenario: Unbound roles
- **WHEN** a parameter is shaped by a role bound to a single element rather than a group
- **THEN** validation reports an issue whose path is the parameter and whose fix asks for a group

### Requirement: The trained model is a system
The core SHALL build the inner model with the trained parameters of any recorded state, as a system
that runs with every backend and solver that supports it, so the trained model can be simulated, analysed, and
visualised like any model.

#### Scenario: Trained model agrees with training
- **WHEN** the inner model is built from the state of the last frame and run with the NumPy backend on the first
  sequence, its inputs given by a schedule and the loss given as an observable
- **THEN** the loss it records equals the one the gradient-step law evaluates for that sequence within 1e-10

### Requirement: Any differentiable model can be fitted
A gradient-step law SHALL accept any nested system that validates as differentiable, not only neural networks, so the
parameters of a model of any field can be fitted to observed series by gradient descent.

#### Scenario: Lotka-Volterra fit
- **WHEN** a gradient-step law with Adam fits the four adaptive rate parameters of `lotka_volterra`, started 30% off, to
  noise-free prey and predator series of 50 points over 20 time units
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
Gradient-based learning SHALL ship in the optional extra `learning`. Without it, validation SHALL report the JAX
backend and the learning solver as unavailable with a fix that names the extra and how to install it, and every other
capability SHALL work.

#### Scenario: Missing extra
- **WHEN** a user without the extra validates a system holding a gradient-step law
- **THEN** the report holds one error whose fix names `pip install 'scikit-complexity[learning]'`, and running
  `lotka_volterra` in the same session succeeds
