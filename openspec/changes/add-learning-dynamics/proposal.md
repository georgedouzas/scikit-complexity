# Proposal

## Why

Gros (2015 §10.5.1) presents Elman's simple recurrent network. Its hidden units follow the discrete map
`s(t) = σ(A s(t-1) + D x(t))` (Gros 2015 Eq. 10.18), and a readout predicts the next input. The weights `A`, `D`, and
the readout then change by backpropagation, so that the prediction error over a stream falls. The network is a
dynamical system in two times. The inner time steps through the input stream. The outer time steps through training,
where the weights are the states and gradient descent on a loss of the inner states is the law. The same structure
appears outside neural networks:
- learning agents that adapt a policy from the outcomes of their own runs;
- adaptive control that tunes gains against a tracking error;
- fitting the parameters of any model of the framework to data by differentiating through its simulation.

The framework can write the inner map today, with groups, reductions, and update laws. It has no gradient of a loss
through an unrolled run, no optimizer, and no mini-batches. The inventory records `elman_network` as out of scope for
that reason. The user has asked that every such model become first class, so this change adds learning as a general
capability.

**Rebased on `redesign-system-model`.** The redesign's inventory gate mapped this change onto its vocabulary:
parameters fitted by an inner loss are `parameter(adaptive=True)` changed by a law, and the inner model is a nested
system. So this change no longer adds a `LearningSystem` subclass of `System` or a public `Form.LEARNING`: a learning
system is an ordinary `System` under `DiscreteTime()`, whose time is the epoch, holding the inner model as a component
and an interaction whose gradient-step law changes the inner model's adaptive parameters. The redesign also already
records adaptive parameters in trajectories like states, lets schedules and `update_parameters` change parameters, and
moves observables out of systems, so losses and metrics are observables given to the law rather than declared on the
inner model. This change still adds the gradient-step law, datasets and mini-batches, backpropagation through time,
optimizers, matrix-valued parameters, the JAX backend and the learning solver with their declarations and validation,
the matrix visual, and the Elman network.

## What Changes

- Add the `GradientStep` law in `core.laws`, a law of the discrete class. Returned by an interaction bound to a nested
  system, it changes the adaptive parameters of that system it names as trained, once per epoch, by an optimizer on
  the gradient of a loss over unrolled runs of the nested system. The system holding the interaction runs under
  `DiscreteTime()`, one step per epoch. The trained parameters are recorded with their inner paths and shapes, the
  optimizer's hyperparameters are parameters of the interaction, and the law records the training loss and named
  metrics.
- Add datasets: arrays of sequences that set inner parameters at every inner step, the way a schedule sets them in a
  simulation, with step weights that mask which steps count in the loss. Mini-batches are drawn from the simulation's
  random stream.
- Add backpropagation through time with optional truncation, which with a truncation of one step reproduces Elman's
  original training rule.
- Add matrix-valued parameters of interactions, shaped by the sizes of two role groups, which the weights of a
  reduction can read. They are addressable by path, change while running, and train like scalar parameters.
- Add a JAX backend in `core.backends` for discrete maps and ODEs, printed from the same Sage laws as the NumPy and
  Numba backends, and a registered learning solver on JAX and Optax, with diffrax for ODE inner models. Both declare
  what they support, and validation reports an inner model they cannot differentiate. They ship in an optional extra
  `learning`.
- Add a matrix visual that streams matrix-valued states and parameters to Rerun as heat maps, and make the loss series
  the primary view of a system that trains.
- Add the library model `elman_network`, with its gallery example, thumbnail, and analysis plot.

## Capabilities

### New Capabilities

- `learning-dynamics`: the gradient-step law over nested systems, datasets and mini-batches, losses and metrics,
  backpropagation through time, optimizers, matrix-valued parameters, their validation, and the trained model.

### Modified Capabilities

- `rerun-visualization`: matrix-valued states and parameters as heat maps, and the loss as the primary view of a
  system that trains.
- `model-library`: the Elman network.

## Impact

- `skcomplex.core.base` gains `LawKind.GRADIENT_STEP` in the discrete class and `MissingExtraError` if
  `add-constraint-based-models` has not added it first.
- `skcomplex.core.components` gains the `shape` argument of `parameter`.
- `skcomplex.core.laws` gains `GradientStep`, `Dataset`, and `Optimizer`.
- `skcomplex.core.backends` gains `JaxBackend`, registered as `'jax'`, with its private JAX printer dialect.
- `skcomplex.core.solvers` gains `GradientDescent`, registered as `'optax'`, the solver validation chooses for systems
  with a gradient-step law.
- `skcomplex.core.viz` gains `MatrixVisual`, registered as `'matrix'`.
- `skcomplex.models.computer_science.neural_networks` gains `elman_network` and its components.
- New optional extra `learning = ["jax>=0.11.2", "optax>=0.2.8", "diffrax>=0.7.2"]`. The default install does not
  change.
- Depends on `redesign-system-model` for adaptive parameters, nested systems, and validation, on
  `add-group-reductions` for weighted reductions over groups, which matrix-valued parameters extend, and on
  `add-model-gallery` for the example and thumbnail.
