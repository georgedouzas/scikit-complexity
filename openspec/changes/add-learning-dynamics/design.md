# Design

## Context

See proposal.md for the motivation. The relevant parts of the current core:
- `System(label, elements=, interactions=, systems=, space=, time=)` holds nested systems, and an interaction binds the
  path of a nested system through a role.
- `ParameterSpec(adaptive=True)` declares a parameter that laws may change. It is compiled as a state, recorded in
  trajectories, and changed by the `schedule` of a `Simulation`.
- Every law derives from `BaseLaw` and states what it requires of a system in `_get_requirements`, a `RequirementSpec`
  of time and space classes. Every solver states its `_get_form`, a `FormKind`, and its `_get_support`, a
  `SupportSpec`, and `DEFAULT_SOLVERS` names the solver of each form. The pass compiler reports every problem it finds
  before compiling as an `IssueSpec` of its `report_`, and running it with an error raises one error listing them.
- The compiler prints laws to NumPy and Numba code through `NumpyBackend` and `NumbaBackend`, as functions
  `f(t, x, p)` over a flat state vector and a flat parameter vector.
- `ObservableSpec(label, expression, element, group, reduce, variable)` names an expression of a system, which a
  simulation records and a catalog entry passes through `register_model(observables=...)`.
- `total(x, weights=matrix)` reduces with a constant matrix. `CouplingInteraction(mode='group')` couples every member to
  the whole group, but its pairwise reductions take no matrix weights, so `GeneralizedLotkaVolterraInteraction` stays a
  `RateInteraction` with a constant matrix argument, which no experiment can sweep or fit.
- `GradientInteraction(target, states, potential, ...)` moves states down the gradient of a potential, a gradient flow
  of states in the system's own time.
- `CalibrationExperiment` fits parameters with SciPy least squares on finite differences, derivative-free over the run.
- `ImageVisual` logs lattice states as images, and `RerunObserver` builds a default blueprint from the views of the
  visuals.

Elman's network (Gros 2015 §10.5.1) needs three things the core lacks: trainable weight matrices, the gradient of a
loss over an unrolled run, and a law that changes the weights by that gradient.

## Goals / Non-Goals

**Goals:**

- Fit learning to the system, simulation, and trajectory interface without forcing it. The learner is a dynamical
  system in its own right whose states are the weights and whose time is the epoch.
- Write learning with the existing primitives: adaptive parameters, a nested system, an interaction, and one new law,
  with no system subclass.
- Keep the inner model an ordinary system, written with the same symbolic laws, so it runs, is analysed, and is
  visualised alone, before or after training.
- Take automatic differentiation, optimizers, and differentiable ODE solves from mature libraries, and keep in the core
  only the lowering of Sage laws to them.

**Non-Goals:**

- A deep learning toolkit. Convolutions, attention, dropout, and GPU device management are out of scope.
- Gradients through stochastic inner models, jump processes, rule laws, lattices, and co-simulations, whose sampling is
  not differentiable in the form the engine compiles them.
- Replacing `CalibrationExperiment`. It stays the derivative-free fitter for any system.

## Decisions

- **Adaptive shaped parameters on interactions** (decided on 2026-11-01): interactions accept adaptive parameters only
  when they are shaped, so the Elman couplings learn their weights; they compile as states and are recorded with their
  shape. The exact Jacobian treats their entries as constants, which matters only for implicit solvers.
- **Contractions lower to weighted sums:** `contraction` becomes a weighted `sum` reduction over the product of indices,
  with the tensor viewed as a `(rows, prod(rest))` matrix, so NumPy, Numba, JAX, and the exact Jacobian share one path;
  it equals `einsum` within 1e-12.

- **JAX, Optax, and Diffrax are runtime dependencies.** The Project Profile's platforms narrow to Linux and macOS 13
  or later on Apple silicon, since `jaxlib` publishes no wheels for macOS on Intel. Decided on 2026-11-01.

### Learning is a system whose time is the epoch

A learning system is an ordinary system:

```text
System('elman', systems=[net], interactions=[TrainingInteraction('training', model='net', ...)], time=DiscreteTime())
```

- `net` is the inner model, a nested system whose trained parameters are declared `ParameterSpec(adaptive=True)`.
- `TrainingInteraction` binds it through its role `model` and returns one `GradientStepLaw`.
- The outer time is discrete, one step per epoch.

What the concepts of the framework mean for a learning system:
- Recorded variables: the trained adaptive parameters of the inner model, flattened in row-major order, named by their
  inner paths, such as `elman/net/readout.W`.
- Parameters: the optimizer's hyperparameters, parameters of the training interaction such as
  `elman/training.learning_rate`, so a schedule changes them.
- Simulated time: epochs, so `dt` is a positive whole number.
- One frame: `dt` epochs of mini-batch updates.
- Recorded outputs: `<interaction path>/loss`, the mean batch loss of the epoch, and every named metric over the whole
  dataset at the end of the epoch, as `<interaction path>/<metric>`.
- Solver state: the optimizer's state, such as Adam's moments, kept by the solver and cleared by a reset.

### Two gradients, two names

`GradientInteraction` and the training law are different operations, and the names keep them apart:
- `GradientInteraction` moves the states of its target down the gradient of a potential, `x' = -dV/dx / divisor`, in
  the system's own time, and lowers to `RateLaw`s.
- `GradientStepLaw` takes optimizer steps on the adaptive parameters of a nested system, once per epoch, on the gradient
  of a loss over unrolled runs, computed by automatic differentiation. `TrainingInteraction` returns it.

Only the law and its solver, `GradientDescentSolver`, carry `Gradient` in their names, and both act on the trained
parameters of a nested system, never on the states of the system that holds them.

### The training interaction and its law

`TrainingInteraction(label, *, model, trained, loss, data, optimizer='adam', learning_rate=1e-3, momentum=0.9,
beta1=0.9, beta2=0.999, batch_size=32, truncation=None, metrics=())` in `skcomplex.core.interactions` binds the nested
system through its role `model`. `trained` lists dotted paths of adaptive parameters of that system, `loss` and
`metrics` are `ObservableSpec`s over its symbols, and `data` is a `DatasetSpec`. The optimizer is a string variant
checked in `_check` against the module constant `OPTIMIZERS = ('sgd', 'momentum', 'rmsprop', 'adam')`, since its
algorithm comes from Optax, and its hyperparameters are parameters of the interaction declared with `ParameterSpec`, so
a schedule changes them. An unknown optimizer raises `ValidationError` naming it and the supported ones.

`GradientStepLaw(trained, loss, data, optimizer, batch_size, truncation=None, metrics=())` in `skcomplex.core.laws`
stores its arguments unmodified and requires `RequirementSpec(time=(DiscreteTime,))`.

### A nested system unrolled by a law

The nested system bound to a training interaction is not stepped by the outer time. The law unrolls it over each
sequence of a batch, from its initial states, inside one epoch, so it is not a partition of a co-simulation. The pass
compiler compiles the nested system to its own problem, a discrete map or an ODE, and returns a `LearningProblemSpec` of
`skcomplex.engine.problems`, of the new form `FormKind.LEARNING`, holding the inner problem, the layout of the trained
parameters, and the layout of the hyperparameters. The pass compiler checks the nested system against the law's
requirements instead of against the outer time.

### Issues

The pass compiler reports, each as an `IssueSpec` with the path of the law or parameter and a fix:
- `learning-trained-path`: a trained path that is not a parameter of the nested system, with the parameters it has;
- `learning-not-adaptive`: a trained parameter that is not declared adaptive;
- `learning-not-differentiable`: a nested system holding a law the JAX backend cannot differentiate, such as a
  `NoiseLaw`, `JumpLaw`, `RuleLaw`, or a law given as a Python callable, saying that learning needs a discrete map or
  ordinary differential equations written as symbolic laws;
- `learning-unknown-observable`: a loss or metric that reads a variable the nested system does not have;
- `learning-data-shape`: a dataset array whose number of steps or trailing shape does not fit;
- `parameter-shape`: a shaped parameter whose roles are not bound to groups;
- `solver-unsupported`: the existing issue, for a solver other than `GradientDescentSolver` given to a system holding a
  `GradientStepLaw`.

### Datasets set inner parameters per step

`DatasetSpec(values, weights=None)` in `skcomplex.specs` maps dotted inner parameter paths to arrays
`(sequences, steps, *shape)`. Inputs and targets are ordinary, not adaptive, parameters of the inner model, so no new
law kind is needed for them. At inner step `k` the lowering writes `values[path][:, k]` into the parameter vector before
evaluating the step, the channel a simulation's schedule uses, so the trained model runs alone under a schedule built
from the same arrays. `weights` masks steps out of the loss and selects the steps of a metric. Every sequence starts
from the inner model's initial states.

Shuffling uses the simulation's NumPy `Generator`, never JAX keys, so runs stay reproducible by seed. Batches keep a
smaller last batch. JAX compiles the update once per distinct batch shape, so a run compiles at most twice.

### The loss is an observable of the inner model

The loss and the metrics are `ObservableSpec`s over the nested system's symbols, which the law compiles with the
observable compiler. The loss of a batch is `sum(w * o) / sum(w)` over sequences and steps, where `o` is the observable
at each step and `w` the step weights. The same observables, given to a `Simulation` of the trained model, record the
loss in a forward run.

### JAX lowering of Sage laws

`JaxBackend` in `skcomplex.engine.backends` prints with a `DialectSpec` of its own: the NumPy dialect with `jnp.` in
place of `np.` and `jax.scipy.special` for `erf`, `erfc`, and `gamma`. It lowers the `UpdateLaw` kernels of a discrete
map and the `RateLaw` kernels of an ODE to pure functions `f(t, x, p)` that return new arrays instead of writing into
buffers, which JAX's tracing requires. Reductions lower to `jnp.sum`, `jnp.mean`, and matrix products, and a reduction
whose weights are a shaped parameter lowers to `W @ a`, with `W` a reshaped slice of `p`. `heaviside`, `sign`, `floor`,
and `ceil` lower with zero gradient almost everywhere, which the user guide states once. Its `SupportSpec` declares
continuous and discrete time and the laws `RateLaw`, `UpdateLaw`, `EquationLaw`, and `GradientStepLaw`.

`JaxBackend` is also a forward backend for discrete maps and ODEs, which is how the agreement scenarios compare it with
NumPy and Numba.

### The learning solver

`GradientDescentSolver(precision='float64', random_state=None, until=inf)` in `skcomplex.engine.solvers` solves
`FormKind.LEARNING`, which `DEFAULT_SOLVERS` maps to it. Per frame it:
1. builds once, from the inner problem's JAX lowering, `unroll(theta, batch)`, a `jax.lax.scan` over the inner steps of
   a discrete map, or a `diffrax.diffeqsolve` with `Tsit5`, a PID step size controller, `SaveAt(ts=data times)`, and
   `RecursiveCheckpointAdjoint` for an ODE;
2. builds `loss(theta, batch)` from the unroll and the loss observable, and `step = jax.jit(value_and_grad(loss))`
   followed by the Optax update;
3. runs `dt` epochs of shuffled batches, and records the mean loss and the metrics.

`'sgd'`, `'momentum'`, `'rmsprop'`, and `'adam'` map to `optax.sgd`, `optax.sgd(momentum=...)`, `optax.rmsprop`, and
`optax.adam`, each wrapped in `optax.inject_hyperparams`, so the solver passes the current parameter vector into the
optimizer state before every update and a scheduled learning rate takes effect at the next frame. Truncated
backpropagation through time applies `jax.lax.stop_gradient` to the carried inner states every `truncation` steps. The
solver computes in double precision inside `jax.enable_x64(True)`, a context manager, so no global JAX flag is set and
NumPy and JAX agree within 1e-10. `precision='float32'` trades that agreement for speed.

### Shaped parameters

`ParameterSpec(shape=('target', 'source'))` declares a parameter of an interaction whose shape is the sizes of the
groups two roles bind, and a tuple of one role repeated k times gives a tensor of order k. The layout gives it a block
of the parameter vector in row-major order and records its shape, so trajectory reads and schedules reshape it. With
`adaptive=True` it is trainable. The reductions `total` and `mean` accept it as `weights`, lowered to a NumPy matrix
product, a Numba double loop, and a JAX matrix product.

`CouplingInteraction(mode='group')` accepts a shaped parameter as `weights` with a `function` that reads both members,
so member `i` gets `sum_j W_ij function(ns)`. A function that factors into a term of the member times a term of the
other lowers to the member term times a matrix product, and any other function to a loop over the pairs. This closes
the generalized Lotka-Volterra gap: `GeneralizedLotkaVolterraInteraction` becomes a `CouplingInteraction` subclass whose
interaction matrix `a` is a shaped parameter, with `function=lambda s: s.target.x * s.source.x`, beside the growth
`b x` as a `RateInteraction`, so experiments sweep and fit the matrix.

### Coupling tensors

A k-body tensor is a parameter of an interaction declared `ParameterSpec(shape=('target',) * k)`, whose value a
factory draws with its `random_state`. No core class draws it. `contraction(tensor, *expressions)` in
`skcomplex.core.laws.utils` returns `sum_jk T_ijk a_j b_k` for every member `i`, with one expression per index after the
first, and lowers to `numpy.einsum`, a Numba loop over the indices, and `jax.numpy.einsum`, so a tensor trains like a
matrix. A tensor whose role binds a single element gets the issue `parameter-shape`, as every shaped parameter does,
with the fix of binding a group. Matrices keep `total(x, weights=W)`, and the user guide shows `contraction` for
tensors of order 3 and above.

This part moved from `add-higher-order-interactions`, which planned it on the shaped parameters of this change. Its
model moved with it: `higher_order_random_lotka_volterra` in `biology/ecology`, after Sidhom and Galla (2024, arXiv
2409.10990), is a group of species with logistic growth as a `RateInteraction` and a rate `x_i contraction(T, x, x) / N`
from a random symmetric tensor `T` of order 3 with mean `mu` and standard deviation `sigma`, drawn by the factory from
its `random_state`.

### Local learning rules need nothing new

Hebb's and Oja's rules change a weight from its own pre- and postsynaptic activity, so they are update or rate laws on
weight states, as in the `hopfield` model. The spec states this with Oja's rule, so contributors do not route local
rules through the gradient machinery.

### Visualization

`MatrixVisual` in `skcomplex.engine.viz`, chosen by `VisualSpec('matrix')`, logs a matrix-valued state or parameter as
an `rr.Image` colored by value, with a fixed range when one is given. The visual of a system that holds a
`GradientStepLaw` logs the loss and metrics as scalars over the epoch timeline and every trained matrix through
`MatrixVisual`. The default blueprint of `RerunObserver` places the loss series first and the heat maps next to it, and
leaves out the inner model's states, which do not exist at the epoch time scale.

### The Elman network

`elman_network(n_hidden=4, n_bits=3000, window=12, batch_size=16, learning_rate=0.05, truncation=None,
random_state=0)` in `computer_science/neural_networks` builds:
- the inner discrete map, a nested system under `DiscreteTime()`: an input group with the parameter `x`, a hidden group
  with state `s`, an output group with state `y` and the parameter `target`, coupled by `CouplingInteraction`s
  `input` (`D`), `recurrence` (`A`), and `readout` (`W`) with adaptive shaped parameters, and adaptive biases `b`
  and `c`;
- the loss observable `error = total((y - target)**2)`, and the masks of the XOR steps and the random steps as step
  weights of two metrics;
- the dataset of windows of 12 bits starting at every third bit of a temporal XOR stream drawn from `random_state`;
- the initial weights drawn from `random_state` with scale 0.5;
- the outer system with the inner model, a `TrainingInteraction`, and `DiscreteTime()`.

Its catalog defaults are `{'until': 200, 'dt': 1}`, and its catalog entry declares the loss and the two metrics as its
observables. The trained model of a frame is the nested system with the recorded values set through `set_params`, the
estimator contract every part follows, so it needs no function of its own. The gallery example
`plot_elman_network.py` streams the run with the default blueprint, whose first view, the loss with `xor_error` and
`random_error`, is the thumbnail, and its analysis figure runs the trained model of the first and last frame over a
fresh stream with a schedule, drawing the mean squared error at each position of the bit triple before and after
training.

### Prototype

A prototype trained the temporal XOR network with 4 hidden units, windows of 12 steps, batches of 16, and Adam at 0.05,
on macOS with Apple silicon and Python 3.13:
- JAX 0.12.0 with `lax.scan`, `jit`, and Optax 0.2.8: 200 epochs in 1.1 s, final mean loss 0.171, error 0.003 on the
  XOR bits and 0.25 on the other bits.
- PyTorch 2.14.1 eager with a Python loop over steps: 200 epochs in 8.6 s, final mean loss 0.177.

Both reach the expected loss of about `(0.25 + 0.25 + 0) / 3`. `jax.enable_x64(True)` works as a context manager in
0.12.0, and diffrax 0.7.2 installs and imports with JAX 0.12.0.

### Library survey

Checked on PyPI on 2026-10-19 for Python 3.12 and 3.13 on Linux and macOS:
- JAX and jaxlib 0.12.0, Apache-2.0, monthly releases, `jaxlib` wheels for manylinux x86_64 and aarch64 and macOS 11
  arm64, none for macOS x86_64. Functional `grad`, `jit`, `lax.scan`, and `vmap` over pure functions `f(t, x, p)`,
  which is what the printer emits. Wrapped.
- Optax 0.2.8, Apache-2.0, pure Python, standard optimizers with injectable hyperparameters. Wrapped.
- diffrax 0.7.2, Apache-2.0, pure Python on Equinox, differentiable ODE solves with checkpointed and adjoint
  backpropagation. Wrapped for ODE inner models.
- PyTorch 2.14.1: the Linux wheel pulls the CUDA toolkit, several gigabytes, a CPU build needs a custom index URL, and
  unrolling is a Python loop, eight times slower above. Rejected.
- autograd 1.9.1: no compilation and no `scan`, in maintenance mode. Rejected.
- Equinox and Flax: layer libraries whose layers would duplicate the framework's symbolic laws. Rejected as direct
  dependencies.
- Sage symbolic differentiation: exact for one step, but the unrolled loss grows exponentially as an expression.
  Rejected for the gradient, kept for checking single-step derivatives in tests.
- Numba: no automatic differentiation. Rejected.

### Dependencies

The Project Profile requires every library that `skcomplex.engine` uses to be a runtime dependency, so `jax`, `optax`,
and `diffrax` join the runtime dependencies of `pyproject.toml` and the Project Profile, with no optional extra and no
lazy import. `jaxlib` adds about 60 to 90 MB. It publishes no wheels for macOS on Intel, so the Project Profile's
platforms narrow to Linux and macOS 13 or later on Apple silicon.

## Risks / Trade-offs

- [No `jaxlib` wheels for macOS on Intel] → The install fails there with pip's message, and the Project Profile states
  the platforms. Linux and Apple silicon are covered.
- [JAX's monthly releases change APIs] → The adapter uses only `grad`, `jit`, `lax.scan`, `lax.stop_gradient`,
  `enable_x64`, and `jax.numpy`, and the lower bound is tested in CI.
- [Compile time of the first `jit`] → It is about 0.2 s for the Elman network and is paid once per batch shape.
- [Non-differentiable functions give zero gradients silently] → The user guide states it once, and the gradient
  scenario against finite differences catches a broken lowering.
- [Large matrices make the flat parameter vector long] → Parameters stay one vector so the layout, trajectory, and
  schedules work. JAX reshapes slices without copies inside `jit`.
- [A coupling tensor of order k over N members holds N^k entries] → The documentation states the memory, and
  `higher_order_random_lotka_volterra` keeps N at 100 for order 3, a million entries.
- [The rebuilt generalized Lotka-Volterra model changes the order of sums] → It is compared with its previous trajectory
  within 1e-10.

## Migration Plan

- The inventory entry of `elman_network` moves from `blocked add-learning-dynamics` to implemented when the model lands.
- The inventory entry of `higher_order_random_lotka_volterra`, recorded as `pending add-higher-order-interactions`
  with hypergraph tags, is marked implemented by this change when the model lands, with the tags of the group system it
  builds.
- The Project Profile gains `jax`, `optax`, and `diffrax` as runtime dependencies, the JAX backend, and the narrowed
  macOS platform, in the same PR as the code.
- No public name is removed or renamed.

## Open Questions

- None. The nested system bound to a training interaction is unrolled by its law, and the loss and metrics are outputs
  of the law, recorded as a jump law records its firings.
