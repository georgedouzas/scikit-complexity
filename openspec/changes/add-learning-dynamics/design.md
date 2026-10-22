# Design

## Context

See proposal.md for the motivation. The relevant parts of the framework after `redesign-system-model`:
- A `System` is its components, interactions, space, and time. A nested system is a component, and an interaction
  binds components through roles and returns laws.
- `parameter(adaptive=True)` declares a parameter that laws of its own component, or of interactions bound to it, may
  change. It is compiled as a state and recorded in trajectories, and it stays settable before a run.
- Laws are Sage expressions classified by time semantics. The compiler prints them with a `Dialect` to NumPy and Numba
  code, and lowers the kernels of each law kind to the numeric functions of an internal form, such as `update` for a
  discrete map and `drift` for an ODE, all of the shape `f(t, x, p)` over a flat state vector and a flat parameter
  vector. Forms are a lowering detail below the system's definition.
- Every law kind, solver, and backend declares `Requires` or `Supports`, and `validate` reports every invalid
  combination with a path, a code, and a fix before compilation.
- A `Simulation` takes the observables to record, advances the compiled problem frame by frame with the chosen
  solver, records a `Trajectory` of states, adaptive parameters, and observables, and hands each `Frame` to observers
  such as `RerunObserver`.
- `add-group-reductions` adds `total(x, weights=A)` with a constant weight matrix, which is the dense layer of a neural
  network as soon as the matrix can be a parameter.
- `calibrate` in `skcomplex.core.experiments` fits parameters with SciPy least squares on finite differences. It is
  derivative-free over the run and scales poorly beyond a few parameters.
- The numerics audit in `organize-model-library` kept diffrax "for gradients or GPUs later". This is that case.

Elman's network (Gros 2015 §10.5.1) needs three things the framework lacks: trainable weight matrices, the gradient of
a loss over an unrolled run, and a law that changes the weights by that gradient.

## Goals / Non-Goals

**Goals:**

- Fit learning to the framework's System, Simulation, and Trajectory interface without forcing it. The learner is a
  dynamical system in its own right whose states are the weights and whose time is the epoch.
- Write learning with the redesign's primitives: adaptive parameters, a nested system, an interaction, and one new
  law, with no system subclass.
- Keep the inner model an ordinary system, written with the same symbolic laws, so it runs, is analysed, and is
  visualised alone, before or after training.
- Take automatic differentiation, optimizers, and adjoint ODE solves from mature libraries, and keep in the core only
  the lowering of Sage laws to them.
- Keep the default install unchanged.

**Non-Goals:**

- A deep learning toolkit. Convolutions, attention, dropout, and GPU device management are out of scope. JAX runs on
  a GPU when the user installs a GPU `jaxlib`, but the framework does not test or configure it.
- Gradients through stochastic inner models, jump processes, rule laws, lattices, and co-simulations. Their sampling
  is not differentiable in the form the framework compiles them, and reinforcement learning by policy gradients is a
  later change on top of this one.
- Replacing `calibrate`. It stays the derivative-free fitter for any system. The gradient-step law is the gradient
  fitter for differentiable inner models.

## Decisions

### Learning is a system whose time is the epoch

A model that is optimised rather than integrated still has a faithful time axis here, the sequence of parameter values
the optimizer visits. A learning system is an ordinary system:

```text
System('elman', components=[net], interactions=[Training(model='net')], time=DiscreteTime())
```

- `net` is the inner model, a nested system whose trained parameters are declared `adaptive=True`.
- `Training` is any interaction with one role bound to the nested system, whose `laws` return one `GradientStep`.
- The outer time is discrete, one step per epoch.

| Framework concept | Learning system |
|---|---|
| Recorded variables | The trained adaptive parameters of the inner model, flattened in row-major order, named by their inner paths, such as `elman/net/readout.W` |
| Parameters | The optimizer's hyperparameters, parameters of the training interaction such as `elman/training.learning_rate`, so `update_parameters` and `schedule` work unchanged |
| Simulated time | Epochs. `dt` must be a positive whole number |
| One frame | `dt` epochs of mini-batch updates |
| Recorded outputs | `<interaction path>/loss`, such as `elman/training/loss`, the mean batch loss of the epoch, and every named metric over the whole dataset at the end of the epoch as `<interaction path>/<metric>`, recorded by the law as `Jump` records its firings |
| `run(until=n)` | Trains for `n` epochs. `trajectory_` holds the weights and the losses per epoch |
| Solver state | The optimizer's state, such as Adam's moments, kept by the solver and cleared by a reset |

`GradientStep(trained, loss, data, optimizer, batch_size, truncation=None, metrics=(), label='training')` stores its
arguments unmodified. `trained` lists dotted paths of adaptive parameters of the bound nested system. A
`trained_model(system, state)` function in `skcomplex.core.laws` returns a new inner `System` with the trained
parameters set from a recorded state, through `set_params` on the components, so the trained model runs on any
backend.

Three other fits were considered:
- **A `LearningSystem` subclass of `System`.** It was this change's first design. The redesign removed execution
  subclasses such as `MesaSystem` and keeps a system to its four parts, and the inventory gate mapped learning onto
  adaptive parameters and a nested system. Rejected.
- **One frame per optimizer update.** It makes the time axis depend on the batch size, and 63 updates per epoch for
  200 epochs exceed the gallery's 300-frame budget. Rejected. A frame of `dt` epochs keeps the axis meaningful.
- **The learner as an external component.** A PyTorch or Equinox module would own the network and the framework would
  only log its outputs. The network's laws would then exist twice, once in Sage for analysis and once in the library's
  layer language, which the constitution forbids. Rejected.

### A nested system unrolled by a law

The nested system bound to a gradient-step law is not stepped by the outer time. The law unrolls it over each
sequence of a batch, from its initial states, inside one epoch, so it is not a partition of a co-simulation. The
compiler compiles the nested system to its own problem, a discrete map or an ODE, and the outer problem holds it as
the law's inner problem. Validation treats the nested system bound to a gradient-step law as unrolled, and checks it
against the law's requirements instead of against the outer time.

### Declarations and validation

| Primitive | Declaration |
|---|---|
| `GradientStep` | `Requires(time={DISCRETE})`; its bound nested system must hold only `Rate`, `Update`, and constraint laws, on real domains, without noise, jumps, or rules |
| `GradientDescent` solver | `Supports(time={DISCRETE}, laws={GRADIENT_STEP}, domains={REAL, VECTOR}, components={ELEMENT, GROUP, SYSTEM})` |
| `JaxBackend` | `Supports(time={CONTINUOUS, DISCRETE}, laws={RATE, UPDATE, EQUATION, GRADIENT_STEP}, domains={REAL, VECTOR}, components={ELEMENT, GROUP, SYSTEM})` |
| matrix-valued parameter | requires an interaction with the two named roles bound to groups |

Validation reports, each as an `Issue` with the path of the law or parameter and a fix:
- `learning-trained-path`: a trained path that is not a parameter of the nested system, with the parameters it has;
- `learning-not-adaptive`: a trained parameter that is not declared `adaptive=True`;
- `learning-not-differentiable`: a nested system holding a law the backend cannot differentiate, such as a `Noise`,
  `Jump`, `Rule`, `Call`, or `NumericFunction`, naming the law and saying that learning needs a discrete map or
  ordinary differential equations written as symbolic laws;
- `learning-unknown-observable`: a loss or metric that is not an observable of the nested system's symbols;
- `learning-data-shape`: a dataset array whose number of steps or trailing shape does not fit;
- `parameter-shape`: a matrix parameter whose roles are not bound to groups;
- `backend-unavailable`: the `learning` extra not installed, whose fix names `pip install 'scikit-complexity[learning]'`.

### Datasets set inner parameters per step

`Dataset(values, weights=None)` maps dotted inner parameter paths to arrays `(sequences, steps, *shape)`. Inputs and
targets are ordinary, not adaptive, parameters of the inner model, so no new law kind is needed for them. At inner step
`k` the lowering writes `values[path][:, k]` into the parameter vector before evaluating the update, the same channel
that `Simulation(schedule=...)` uses for forward runs. The trained model is therefore driven by a schedule built from
the same arrays when it runs alone. `weights` masks steps out of the loss, which also selects the steps of a metric.
Every sequence starts from the inner model's initial states.

Shuffling uses the simulation's NumPy `Generator`, never JAX keys, so the reproducibility rule of the simulation spec
holds unchanged. Batches keep a smaller last batch. JAX compiles the update once per distinct batch shape, so a run
compiles at most twice.

### The loss is an observable of the inner model

Observables are no longer declared on systems. The loss and the metrics are `Observable`s over the nested system's
symbols, given to the gradient-step law, which compiles them with the observable compiler from Sage expressions and
reductions. The loss of a batch is `sum(w * o) / sum(w)` over sequences and steps, where `o` is the observable at each
step and `w` the step weights. The same observables, given to a `Simulation` of the trained model, record the loss in
a forward run.

### JAX lowering of Sage laws

`JaxBackend`, in `skcomplex.core.backends` and registered as `'jax'`, prints with a private dialect: the NumPy dialect
with `jnp.` in place of `np.` and `jax.scipy.special` for `erf`, `erfc`, and `gamma`. It lowers the `Update` kernels of
a discrete map and the `Rate` kernels of an ODE to pure functions `f(t, x, p)` that return new arrays instead of
writing into buffers, which JAX's tracing requires. Reductions lower to `jnp.sum`, `jnp.mean`, and matrix products,
and a weighted reduction whose weights are a matrix parameter lowers to `W @ a`, with `W` a reshaped slice of `p`.
Laws that are opaque callables are outside its declared support, so validation reports them before lowering.
`heaviside`, `sgn`, `floor`, and `ceil` lower, with zero gradient almost everywhere, which the user guide states once.

`JaxBackend` is also a forward backend for discrete maps and ODEs, which is how the agreement scenarios compare it with
NumPy and Numba.

### The learning solver

`GradientDescent`, registered as `'optax'` and chosen by `'auto'` for a system with a gradient-step law, does this per
frame:
1. builds once, from the inner problem's JAX lowering, `unroll(theta, batch)`. For a discrete map it is a
   `jax.lax.scan` over the inner steps. For an ODE it is a `diffrax.diffeqsolve` with `Tsit5`, a PID step size
   controller, `SaveAt(ts=data times)`, and `RecursiveCheckpointAdjoint`;
2. builds `loss(theta, batch)` from the unroll and the loss observable lowered by the JAX backend, and
   `step = jax.jit(value_and_grad(loss))` followed by the Optax update;
3. runs `dt` epochs of shuffled batches, and records the mean loss and the metrics.

The optimizer is `Optimizer(name, **hyperparameters)`, with `'sgd'`, `'momentum'`, `'rmsprop'`, and `'adam'` mapped to
`optax.sgd`, `optax.sgd(momentum=...)`, `optax.rmsprop`, and `optax.adam`. Each is wrapped in
`optax.inject_hyperparams`, so the solver passes the current parameter vector into the optimizer state before every
update, and a schedule of the learning rate takes effect at the next frame.

Truncated backpropagation through time applies `jax.lax.stop_gradient` to the carried inner states every `truncation`
steps. A truncation of 1 cuts every step, which is Elman's copy of the hidden units into context units.

The solver computes in double precision inside `jax.enable_x64(True)`, a context manager, so no global JAX flag is set
and NumPy and JAX agree within 1e-10. A `precision='float32'` solver argument trades that agreement for speed.

### Matrix-valued parameters

`parameter(shape=('target', 'source'))`, in `skcomplex.core.components`, declares a parameter of an interaction whose
shape is the sizes of the groups two roles bind. The layout gives it a block of the parameter vector in row-major
order and records its shape, so `trajectory[...]`, `update_parameters`, and `trained_model` reshape it. With
`adaptive=True` it is trainable. The reduction pass of `add-group-reductions` accepts such a parameter as `weights`,
and lowers it to a NumPy matrix product, a Numba double loop, and a JAX matrix product. This is general, since
Kuramoto couplings, payoff matrices, and food webs are matrix parameters too.

### Local learning rules need nothing new

Hebb's and Oja's rules change a weight from its own pre- and postsynaptic activity, so they are update or rate laws on
weight states, as in the `hopfield` model. The spec states this with Oja's rule, so contributors do not route local
rules through the gradient machinery.

### Visualization

`MatrixVisual`, in `skcomplex.core.viz` and registered as `'matrix'`, logs a matrix-valued state or parameter as an
`rr.Image` colored by value, with a fixed range when one is given. The visual of a system that holds a gradient-step
law logs the loss and metrics as scalars over the epoch timeline and every trained matrix through `MatrixVisual`.
`primary_view` returns the loss series for such a system, and `default_blueprint` places the heat maps next to it,
leaving out the inner model's states, which do not exist at the epoch time scale.

### The Elman network and its gallery example

`elman_network(n_hidden=4, n_bits=3000, window=12, batch_size=16, learning_rate=0.05, truncation=None,
random_state=0)` builds:
- the inner discrete map, a nested system under `DiscreteTime()`: an `Input` group with the parameter `x`, a `Hidden`
  group with state `s`, an `Output` group with state `y` and the parameter `target`, and the interactions `input`
  (`D`), `recurrence` (`A`), and `readout` (`W`) with adaptive matrix parameters, and adaptive biases `b` and `c`;
- the loss observable `error = total((y - target)**2)`, and the masks of the XOR steps and the random steps as step
  weights of two metric datasets;
- the dataset of windows of 12 bits starting at every third bit of a temporal XOR stream drawn from `random_state`;
- the initial weights drawn from `random_state` with scale 0.5, as the prototype below did;
- the outer system with the inner model, a training interaction returning the `GradientStep`, and `DiscreteTime()`.

Its catalog defaults are `{'until': 200, 'dt': 1}`, and its catalog entry declares the loss and the two metrics as its
observables. The gallery example `plot_elman_network.py` streams it with the default blueprint, whose primary view, the
loss with `xor_error` and `random_error`, is the thumbnail. Its analysis figure runs `trained_model` on the first and
the last state over a fresh stream with a schedule, and draws the mean squared error at each position of the bit
triple before and after training. At 200 epochs it records 201 frames and runs in about a second, inside the gallery
budget.

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

- **Validation:** `validate` checks the gradient-step law, its nested system, the trained paths, the observables, the
  dataset shapes, and the solver and backend against their declarations, and reports every problem at once.
- **Compile:** `compile_system` compiles the nested system to a discrete map or ODE problem as today, and returns a
  problem of an internal learning form that holds the inner problem, the layout of the trained parameters, and the
  hyperparameter layout. The JAX lowering is requested from the inner problem lazily by the solver, so compiling for
  inspection does not need the extra.
- **Runtime:** `Simulation` resolves the solver through validation, checks that `dt` is a whole number of epochs, and
  runs unchanged. `Trajectory.__getitem__` reshapes matrix-valued names using the recorded shape.
- **Rerun:** frames of a learning system carry the trained parameters as adaptive parameters and the loss and metrics
  as recorded outputs, so `RerunObserver` logs them through the new visual with no change to the observer.
- **Backends:** the inner problem keeps its NumPy and Numba lowerings for forward runs. Only the learning solver uses
  JAX.

## Optional extra and missing-extra error

`pyproject.toml` declares `learning = ["jax>=0.11.2", "optax>=0.2.8", "diffrax>=0.7.2"]`. The JAX backend and the
learning solver import JAX inside the function that builds them, the one lazy import the constitution allows, with a
suppression whose reason names the extra. Validation's availability check reports the missing extra before any work.
Building the backend directly without it raises `MissingExtraError`, a `BackendError` from `skcomplex.core.base`, whose
message names `pip install 'scikit-complexity[learning]'`. The change adds `MissingExtraError` if
`add-constraint-based-models` has not added it first. deptry learns the optional imports. One CI job runs without the
extra and checks the error and that the rest of the suite passes. The Elman example and its doctests are skipped by a
marker without the extra, and the documentation build installs it.

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
  implemented when the model lands, tagged with discrete time, a nested system, adaptive parameters, and the
  gradient-step law. The out-of-scope list in `organize-model-library`'s design keeps its history.
- The constitution's Project Profile gains the `learning` extra with JAX, Optax, and diffrax, and the JAX backend in
  the backends bullet, in the same PR as the code.
- No public name is removed or renamed.

## Open Questions

- How does the redesign mark a nested system that a law unrolls rather than co-simulates? This design lets the
  gradient-step law claim the nested system bound to it, so validation checks it against the law and the compiler does
  not partition it. The redesign specifies nested systems with other law classes only as coupled partitions in one
  time. A general marker, such as an option on the role or on the nested system, would also serve other inner-loop
  models, and choosing it is a decision on the system vocabulary.
- Should the loss and metrics a gradient-step law records count as observables of the trajectory? This design records
  them as outputs of the law, as `Jump` records its firings, since the redesign moves observables to simulations and
  catalog entries. The alternative is to require the user to pass them to the simulation as observables of the outer
  system, which keeps one recording channel but duplicates the loss in two places.
