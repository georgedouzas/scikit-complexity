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

The core already gives most of the vocabulary: adaptive parameters (`ParameterSpec(adaptive=True)`) recorded in
trajectories and changed by schedules, nested systems (`System(systems=...)`), observables as `ObservableSpec`s given to
simulations and catalog entries, reductions with constant weight matrices (`total(x, weights=matrix)`), the
derivative-free `CalibrationExperiment`, and the compiler's issue reports. So this change adds no system subclass. It
adds what is missing: a law that takes gradient steps on the adaptive parameters of a nested system, datasets and
mini-batches, backpropagation through time, optimizers, parameters shaped by the sizes of role groups, a JAX backend
and a learning solver, a matrix visual, and the Elman network. The inventory records `elman_network` as blocked by this
change, and the generalized Lotka-Volterra model keeps its interaction matrix as a constant because no parameter can
hold a matrix. Shaped parameters of higher order also hold k-body coupling tensors: Sidhom and Galla (2024, arXiv
2409.10990) solve random Lotka-Volterra communities with such tensors, so the reduction that contracts a tensor with
member states and the model `higher_order_random_lotka_volterra` move here from `add-higher-order-interactions`, whose
remaining work does not need shaped parameters.

`GradientInteraction` already exists and is a different operation: it moves the states of its target down the gradient
of a potential in the system's own time and lowers to `RateLaw`s. The training law of this change takes optimizer steps
on the parameters of a nested system, once per epoch, on the gradient of a loss over unrolled runs. The names keep them
apart: the interaction is `TrainingInteraction` and its law `GradientStepLaw`.

## What Changes

- Add `GradientStepLaw` in `skcomplex.core.laws`, a discrete-time law that changes the adaptive parameters of a nested
  system it names as trained, once per epoch, by an optimizer on the gradient of a loss over unrolled runs of that
  system, and records the training loss and named metrics.
- Add `TrainingInteraction` in `skcomplex.core.interactions`, which binds the nested system through its role `model`,
  holds the optimizer's hyperparameters as parameters, and returns the `GradientStepLaw`.
- Add `DatasetSpec` in `skcomplex.specs`: arrays of sequences that set inner parameters at every inner step, as a
  schedule does, with step weights that mask the loss. Mini-batches are drawn from the simulation's random stream.
- Add backpropagation through time with optional truncation, which with a truncation of one step reproduces Elman's
  original training rule.
- Add parameters shaped by the sizes of role groups, `ParameterSpec(shape=('target', 'source'))`, which the weights of
  `total`, `mean`, and `CouplingInteraction(mode='group')` read, which a schedule changes, and which train like scalar
  parameters. A role repeated k times, `shape=('target',) * k`, gives a k-body coupling tensor.
- Add the reduction `contraction(tensor, *expressions)` to `skcomplex.core.laws.utils`, which contracts a shaped
  parameter of order k with one member expression per index after the first, lowered to `numpy.einsum`, a Numba loop,
  and `jax.numpy.einsum`.
- Add `JaxBackend` in `skcomplex.engine.backends` for discrete maps and ODEs, printed from the same Sage laws as the
  NumPy and Numba backends, and `GradientDescentSolver` in `skcomplex.engine.solvers` on JAX and Optax, with diffrax for
  ODE inner models, the default solver of the new `FormKind.LEARNING`. JAX, Optax, and diffrax become runtime
  dependencies, since the engine uses them.
- Add `MatrixVisual` in `skcomplex.engine.viz`, which streams matrix-valued states and parameters to Rerun as heat maps,
  and make the loss series the first view of the default blueprint of a system that trains.
- Add `elman_network` and `higher_order_random_lotka_volterra`, and rebuild `generalized_lotka_volterra` on a shaped
  interaction matrix parameter.

## Capabilities

### New Capabilities

- `learning-dynamics`: the gradient-step law over nested systems, datasets and mini-batches, losses and metrics,
  backpropagation through time, optimizers, shaped parameters, coupling tensors and their contraction, and their
  issues.

### Modified Capabilities

- `rerun-visualization`: matrix-valued states and parameters as heat maps, and the loss as the first view of a system
  that trains.
- `model-library`: the Elman network, the random Lotka-Volterra community with k-body interactions, and the
  generalized Lotka-Volterra community on a matrix parameter.

## Impact

- `skcomplex.core.laws` gains `GradientStepLaw`, `skcomplex.core.laws.utils` gains `contraction`,
  `skcomplex.core.interactions` gains `TrainingInteraction`,
  `skcomplex.specs` gains `DatasetSpec` and the `shape` field of `ParameterSpec`, and `skcomplex.types` gains
  `FormKind.LEARNING`.
- `skcomplex.engine.compile` compiles the nested system of a training interaction into the inner problem of a new
  `LearningProblemSpec` of `skcomplex.engine.problems`, and its pass compiler gains the learning issues.
- `skcomplex.engine.backends` gains `JaxBackend`, `skcomplex.engine.solvers` gains `GradientDescentSolver` in
  `DEFAULT_SOLVERS`, and `skcomplex.engine.viz` gains `MatrixVisual`.
- `skcomplex.models.computer_science.neural_networks` gains `elman_network`, and
  `skcomplex.models.biology.ecology` gains `higher_order_random_lotka_volterra` and rebuilds
  `GeneralizedLotkaVolterraInteraction`.
- `jax`, `optax`, and `diffrax` become runtime dependencies, and the Project Profile names them. `jaxlib` publishes no
  wheels for macOS on Intel, so the supported macOS platform narrows to Apple silicon.
