# Design

## Context

See proposal.md for the motivation. The relevant parts of the framework today:
- Laws are Sage expressions. The compiler prints them with a `Dialect` to NumPy and Numba code, and backends lower
  the kernels of each law kind to the numeric functions of one `Form`, such as `update` for a discrete map and `drift`
  for an ODE, all of the shape `f(t, x, p)` over a flat state vector and a flat parameter vector.
- A `Simulation` advances a compiled `Problem` frame by frame with the stepper registered for its form, records a
  `Trajectory` of states, parameters, and observables, and hands each `Frame` to observers such as `RerunObserver`.
- `add-group-reductions` adds `total(x, weights=A)` with a constant weight matrix, which is the dense layer of a neural
  network as soon as the matrix can be a parameter.
- `calibrate` in `skcomplex.core.experiments` fits parameters with SciPy least squares on finite differences. It is
  derivative-free over the run and scales poorly beyond a few parameters.
- The numerics audit in `organize-model-library` kept diffrax "for gradients or GPUs later". This is that case.

Elman's network (Gros 2015 §10.5.1) needs three things the framework lacks: trainable weight matrices, the gradient of
a loss over an unrolled run, and an outer process that changes the weights by that gradient.

## Goals / Non-Goals

**Goals:**

- Fit learning to the framework's System, Simulation, and Trajectory interface without forcing it. The learner is a
  dynamical system in its own right whose states are the weights and whose time is the epoch.
- Keep the inner model an ordinary system, written with the same symbolic laws, so it runs, is analysed, and is
  visualised alone, before or after training.
- Take automatic differentiation, optimizers, and adjoint ODE solves from mature libraries, and keep in the core only
  the lowering of Sage laws to them.
- Keep the default install unchanged.

**Non-Goals:**

- A deep learning toolkit. Convolutions, attention, dropout, and GPU device management are out of scope. JAX runs on
  a GPU when the user installs a GPU `jaxlib`, but the framework does not test or configure it.
- Gradients through stochastic inner models, jump processes, agents, lattices, and co-simulations. Their sampling is
  not differentiable in the form the framework compiles them, and reinforcement learning by policy gradients is a
  later change on top of this one.
- Replacing `calibrate`. It stays the derivative-free fitter for any form. A learning system is the gradient fitter for
  the differentiable forms.

## Decisions

### Learning is a system whose time is the epoch

A model that is optimised rather than integrated still has a faithful time axis here, the sequence of parameter values
the optimizer visits. `LearningSystem` is a `System` subclass, as `MesaSystem` is, that compiles to a new
`Form.LEARNING`:

| Framework concept | Learning system |
|---|---|
| State vector `x` | The trained parameters of the inner model, flattened in row-major order, named by their inner paths under the learning system's label, such as `elman/readout.W` |
| Parameter vector `p` | The optimizer's hyperparameters, such as `elman.learning_rate`, so `update_parameters` and `schedule` work unchanged |
| Simulated time | Epochs. `dt` must be a positive whole number |
| One frame | `dt` epochs of mini-batch updates |
| Observables | `<label>/loss`, the mean batch loss of the epoch, and every named metric over the whole dataset at the end of the epoch |
| `run(until=n)` | Trains for `n` epochs. `trajectory_` holds the weights and the losses per epoch |
| Stepper state | The optimizer's state, such as Adam's moments, kept by the stepper and cleared by a reset |

Two other fits were considered:
- **One frame per optimizer update.** It makes the time axis depend on the batch size, and 63 updates per epoch for
  200 epochs exceed the gallery's 300-frame budget. Rejected. A frame of `dt` epochs keeps the axis meaningful.
- **The learner as an external component.** A PyTorch or Equinox module would own the network and the framework would
  only log its outputs. The network's laws would then exist twice, once in Sage for analysis and once in the library's
  layer language, which the constitution forbids. Rejected.

`LearningSystem(label, *, model, trained, loss, data, optimizer, batch_size, truncation=None, metrics=())` stores its
arguments unmodified. `build_model(state)` returns a new inner `System` with the trained parameters set from a state
vector, through `set_params` on the components, so the trained model runs on any backend.

### Datasets set inner parameters per step

`Dataset(values, weights=None)` maps dotted inner parameter paths to arrays `(sequences, steps, *shape)`. Inputs and
targets are parameters of the inner model, so no new law kind is needed. At inner step `k` the lowering writes
`values[path][:, k]` into the parameter vector before evaluating the update, the same channel that
`Simulation(schedule=...)` uses for forward runs. The trained model is therefore driven by a schedule built from the
same arrays when it runs alone. `weights` masks steps out of the loss, which also selects the steps of a metric. Every
sequence starts from the inner model's initial states.

Shuffling uses the simulation's NumPy `Generator`, never JAX keys, so the reproducibility rule of the simulation spec
holds unchanged. Batches keep a smaller last batch. JAX compiles the update once per distinct batch shape, so a run
compiles at most twice.

### The loss is an inner observable

The loss and the metrics are labels of `Observable`s of the inner model, which already compile from Sage expressions
and reductions. The loss of a batch is `sum(w * o) / sum(w)` over sequences and steps, where `o` is the observable at
each step and `w` the step weights. This reuses the observable compiler and keeps the loss visible in a forward run of
the trained model.

### JAX lowering of Sage laws

`JAX_DIALECT` is `NUMPY_DIALECT` with `jnp.` in place of `np.` and `jax.scipy.special` for `erf`, `erfc`, and `gamma`.
`JaxBackend`, registered as `'jax'`, lowers the `Update` kernels of a discrete map and the `Rate` kernels of an ODE to
pure functions `f(t, x, p)` that return new arrays instead of writing into buffers, which JAX's tracing requires.
Reductions lower to `jnp.sum`, `jnp.mean`, and matrix products, and a weighted reduction whose weights are a matrix
parameter lowers to `W @ a`, with `W` a reshaped slice of `p`. Laws that are opaque callables, `Call`,
`NumericFunction`, and `Rule`, raise `BackendError` naming the law, since JAX cannot trace arbitrary Python.
`heaviside`, `sgn`, `floor`, and `ceil` lower, with zero gradient almost everywhere, which the user guide states once.

`JaxBackend` is also a forward backend for discrete maps and ODEs, which is how the agreement scenarios compare it with
NumPy and Numba.

### The learning stepper

`GradientDescent`, registered as `'optax'` and the default solver of `Form.LEARNING`, does this per frame:
1. builds once, from the inner problem's JAX lowering, `unroll(theta, batch)`. For a discrete map it is a
   `jax.lax.scan` over the inner steps. For an ODE it is a `diffrax.diffeqsolve` with `Tsit5`, a PID step size
   controller, `SaveAt(ts=data times)`, and `RecursiveCheckpointAdjoint`;
2. builds `loss(theta, batch)` from the unroll and the observable lowered with `JAX_DIALECT`, and
   `step = jax.jit(value_and_grad(loss))` followed by the Optax update;
3. runs `dt` epochs of shuffled batches, and records the mean loss and the metrics.

The optimizer is `Optimizer(name, **hyperparameters)`, with `'sgd'`, `'momentum'`, `'rmsprop'`, and `'adam'` mapped to
`optax.sgd`, `optax.sgd(momentum=...)`, `optax.rmsprop`, and `optax.adam`. Each is wrapped in
`optax.inject_hyperparams`, so the stepper passes the current parameter vector into the optimizer state before every
update, and a schedule of the learning rate takes effect at the next frame.

Truncated backpropagation through time applies `jax.lax.stop_gradient` to the carried inner states every `truncation`
steps. A truncation of 1 cuts every step, which is Elman's copy of the hidden units into context units.

The stepper computes in double precision inside `jax.enable_x64(True)`, a context manager, so no global JAX flag is
set and NumPy and JAX agree within 1e-10. A `precision='float32'` stepper argument trades that agreement for speed.

### Matrix-valued parameters

`parameter(shape=('target', 'source'))` declares a parameter of an interaction whose shape is the sizes of the groups
two roles bind. The layout gives it a block of the parameter vector in row-major order and records its shape, so
`trajectory[...]`, `update_parameters`, and `build_model` reshape it. The reduction pass of `add-group-reductions`
accepts such a parameter as `weights`, and lowers it to a NumPy matrix product, a Numba double loop, and a JAX matrix
product. This is general, since Kuramoto couplings, payoff matrices, and food webs are matrix parameters too.

### Local learning rules need nothing new

Hebb's and Oja's rules change a weight from its own pre- and postsynaptic activity, so they are update or rate laws on
weight states, as in the `hopfield` model. The spec states this with Oja's rule, so contributors do not route local
rules through the gradient machinery.

### Visualization

`MatrixVisual`, registered as `'matrix'`, logs a matrix-valued state or parameter as an `rr.Image` colored by value,
with a fixed range when one is given. The learning system's visual logs the loss and metrics as scalars over the
epoch timeline and every trained matrix through `MatrixVisual`. `primary_view` returns the loss series for a learning
system, and `default_blueprint` places the heat maps next to it, leaving out the inner model's states, which do not
exist at the epoch time scale.

### The Elman network and its gallery example

`elman_network(n_hidden=4, n_bits=3000, window=12, batch_size=16, learning_rate=0.05, truncation=None,
random_state=0)` builds:
- the inner discrete map: an `Input` group with the parameter `x`, a `Hidden` group with state `s`, an `Output` group
  with state `y` and the parameter `target`, and the interactions `input` (`D`), `recurrence` (`A`), and `readout`
  (`W`) with matrix parameters, and biases `b` and `c`;
- the observables `error = total((y - target)**2)`, and the masks of the XOR steps and the random steps as step
  weights of two metric datasets;
- the dataset of windows of 12 bits starting at every third bit of a temporal XOR stream drawn from `random_state`;
- the initial weights drawn from `random_state` with scale 0.5, as the prototype below did.

Its catalog defaults are `{'until': 200, 'dt': 1}`. The gallery example `plot_elman_network.py` streams it with the
default blueprint, whose primary view, the loss with `xor_error` and `random_error`, is the thumbnail. Its analysis
figure runs `build_model` on the first and the last state over a fresh stream with a schedule, and draws the mean
squared error at each position of the bit triple before and after training. At 200 epochs it records 201 frames and
runs in about a second, inside the gallery budget.

### Prototype

A prototype in the scratchpad trained the temporal XOR network with 4 hidden units, windows of 12 steps, batches of
16, and Adam at 0.05, on macOS with Apple silicon and Python 3.13:

| Library | Version | 200 epochs | Final mean loss | Error on XOR bits, other bits |
|---|---|---|---|---|
| JAX with `lax.scan`, `jit`, and Optax | 0.12.0, 0.2.8 | 1.1 s | 0.171 | 0.003, 0.25 |
| PyTorch eager with a Python loop over steps | 2.14.1 | 8.6 s | 0.177 | not measured |

Both reach the expected loss of about `(0.25 + 0.25 + 0) / 3`. JAX's compiled scan is eight times faster for the same
work, and `jax.enable_x64(True)` works as a context manager in 0.12.0. diffrax 0.7.2 installs and imports with JAX
0.12.0.

## Library survey

Checked on PyPI on 2026-10-19 for Python 3.12 and 3.13 on Linux and macOS.

| Library | Version, date | Licence | Wheels | Fit | Verdict |
|---|---|---|---|---|---|
| JAX and jaxlib | 0.12.0, 2026-10-19, monthly releases | Apache-2.0 | `jaxlib` for cp312 to cp315 on manylinux x86_64 (86 MB) and aarch64 (80 MB), macOS 11 arm64 (60 MB). No macOS x86_64 | Functional `grad`, `jit`, `lax.scan`, and `vmap` over pure functions `f(t, x, p)`, which is exactly what the printer emits. `jax.numpy` mirrors NumPy, so the dialect is a renaming | **Wrapped** in the `learning` extra |
| Optax | 0.2.8, 2026-03-20 | Apache-2.0 | Pure Python | Standard optimizers with injectable hyperparameters, maintained by Google DeepMind | **Wrapped** in the `learning` extra |
| diffrax | 0.7.2, 2026-02-18 | Apache-2.0 | Pure Python, on Equinox, lineax, and optimistix | Differentiable ODE solves with checkpointed and adjoint backpropagation, chosen by the numerics audit for gradients | **Wrapped** in the `learning` extra for ODE inner models |
| PyTorch | 2.14.1, 2026-09-30 | BSD-style, Apache-2.0 | 127 MB on macOS 14 arm64. On Linux the PyPI wheel (554 MB on x86_64) pulls the CUDA toolkit, cuDNN, NCCL, and Triton, several gigabytes. A CPU build needs a custom index URL, which an extra cannot declare | Imperative tensors. Printing Sage to `torch` works, but unrolling is a Python loop, eight times slower above, and `torchdiffeq` and `torchsde` are smaller projects than diffrax | Rejected: install size on Linux, macOS 14 minimum, and speed |
| autograd | 1.9.1, 2026-07-02 | MIT | Pure Python | Differentiates NumPy code, but has no compilation, no `scan`, and is in maintenance mode | Rejected: too slow for unrolled runs |
| Equinox, Flax | 0.13.8, 0.12.10 | Apache-2.0 | Pure Python | Neural network layer libraries on JAX. The layers here come from the framework's symbolic laws, so they would duplicate them. Equinox arrives only as a dependency of diffrax | Rejected as direct dependencies |
| Sage symbolic differentiation | In the core | GPL through passagemath, already a dependency | Present | Exact derivatives of one step, but the unrolled loss over 12 steps and a batch grows exponentially as an expression | Rejected for the gradient. Kept for checking single-step derivatives in tests |
| Numba | In the core | BSD-2-Clause | Present | No automatic differentiation | Rejected |

JAX, Optax, and diffrax are permissive, so the extra is optional for size, not licence. `jaxlib` adds about 60 to 90
MB and is needed by no other capability.

## Fit with the compile pipeline, the runtime, and Rerun

- **Compile:** `compile_system` sees a `LearningSystem`, compiles its inner model to a discrete map or ODE problem as
  today, checks the trained paths, the loss, the metrics, and the dataset shapes, and returns a `Problem` of
  `Form.LEARNING` that holds the inner problem, the layout of the trained parameters, and the hyperparameter layout.
  The JAX lowering is requested from the inner problem lazily by the stepper, so compiling for inspection does not
  need the extra.
- **Runtime:** `Simulation` resolves the default solver of `Form.LEARNING`, checks that `dt` is a whole number of
  epochs, and runs unchanged. `Trajectory.__getitem__` reshapes matrix-valued names using the recorded shape.
- **Rerun:** frames of a learning system carry the trained parameters as states and the loss and metrics as
  observables, so `RerunObserver` logs them through the new visual with no change to the observer.
- **Backends:** the inner problem keeps its NumPy and Numba lowerings for forward runs. Only the stepper uses JAX.

## Optional extra and missing-extra error

`pyproject.toml` declares `learning = ["jax>=0.11.2", "optax>=0.2.8", "diffrax>=0.7.2"]`. The JAX backend and the
learning stepper import JAX inside the function that builds them, the one lazy import the constitution allows, with a
suppression whose reason names the extra. On `ImportError` they raise `MissingExtraError`, a `BackendError` from
`skcomplex.core.base`, whose message names `pip install 'scikit-complexity[learning]'`. The change adds
`MissingExtraError` if `add-constraint-based-models` has not added it first. deptry learns the optional imports. One
CI job runs without the extra and checks the error and that the rest of the suite passes. The Elman example and its
doctests are skipped by a marker without the extra, and the documentation build installs it.

## Risks / Trade-offs

- [No `jaxlib` wheels for macOS on Intel] → The extra fails to install there and the error says so. The default
  install is unaffected, and Linux and Apple silicon are covered.
- [JAX's monthly releases change APIs] → The adapter uses only `grad`, `jit`, `lax.scan`, `lax.stop_gradient`,
  `enable_x64`, and `jax.numpy`, which have been stable for years, and the lower bound is tested in CI.
- [Compile time of the first `jit`] → It is about 0.2 s for the Elman network and is paid once per batch shape.
- [Non-differentiable functions give zero gradients silently] → The user guide states it once, and the gradient
  scenario against finite differences catches a broken lowering.
- [Large matrices make the flat parameter vector long] → Parameters stay one vector so the existing layout,
  trajectory, and `update_parameters` work. JAX reshapes slices without copies inside `jit`.

## Migration Plan

- The inventory entry of `elman_network` moves from out of scope to `pending add-learning-dynamics`, and to
  implemented when the model lands. The out-of-scope list in `organize-model-library`'s design keeps its history.
- The constitution's Project Profile gains the `learning` extra with JAX, Optax, and diffrax, and the JAX printer in
  the symbolic layer bullet, in the same PR as the code.
- No public name is removed or renamed.
