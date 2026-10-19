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

The framework can write the inner map today, with groups, reductions, and update laws. It has no outer time, no
gradient of a loss through an unrolled run, no optimizer, and no mini-batches. The inventory records
`elman_network` as out of scope for that reason. The user has asked that every such model become first class, so this
change adds learning as a general capability.

## What Changes

- Add learning systems: a system that wraps an inner model, names the inner parameters it trains, takes a per-step
  loss from an inner observable, and evolves those parameters by an optimizer. One frame of a learning system is one
  epoch, its states are the trained parameters, its parameters are the optimizer's hyperparameters, and its
  observables are the training loss and named metrics.
- Add datasets: arrays of sequences that set inner parameters at every inner step, the way a schedule sets them in a
  simulation, with step weights that mask which steps count in the loss. Mini-batches are drawn from the simulation's
  random stream.
- Add backpropagation through time with optional truncation, which with a truncation of one step reproduces Elman's
  original training rule.
- Add matrix-valued parameters of interactions, shaped by the sizes of two role groups, which the weights of a
  reduction can read. They are addressable by path, change while running, and train like scalar parameters.
- Add a JAX lowering of discrete maps and ODEs, printed from the same Sage laws as the NumPy and Numba backends, and a
  registered learning stepper on JAX and Optax, with diffrax for ODE inner models. They ship in an optional extra
  `learning`.
- Add a matrix visual that streams matrix-valued states to Rerun as heat maps, and make the loss series the primary
  view of a learning system.
- Add the library model `elman_network`, with its gallery example, thumbnail, and analysis plot.

## Capabilities

### New Capabilities

- `learning-dynamics`: learning systems, datasets and mini-batches, losses and metrics from inner observables,
  backpropagation through time, optimizers, matrix-valued parameters, and the trained model.

### Modified Capabilities

- `rerun-visualization`: matrix-valued states as heat maps, and the loss as the primary view of a learning system.
- `model-library`: the Elman network.

## Impact

- `skcomplex.core.base` gains `Form.LEARNING`, the `shape` argument of `parameter`, and `MissingExtraError` if
  `add-constraint-based-models` has not added it first.
- `skcomplex.core.model` gains `LearningSystem`, `Dataset`, and `Optimizer`.
- `skcomplex.core.compile` gains `JAX_DIALECT` and `JaxBackend`, registered as `'jax'`.
- `skcomplex.core.solvers` gains `GradientDescent`, registered as `'optax'` and the default solver of the learning form.
- `skcomplex.core.viz` gains `MatrixVisual`, registered as `'matrix'`.
- `skcomplex.models.computer_science.neural_networks` gains `elman_network` and its components.
- New optional extra `learning = ["jax>=0.11.2", "optax>=0.2.8", "diffrax>=0.7.2"]`. The default install does not
  change.
- Depends on `add-group-reductions` for weighted reductions over groups, which matrix-valued parameters extend, and on
  `add-model-gallery` for the example and thumbnail.
