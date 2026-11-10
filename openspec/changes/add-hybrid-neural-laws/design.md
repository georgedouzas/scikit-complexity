# Design

## Context

- `GradientStepLaw` and `GradientDescentSolver` (`src/skcomplex/core/engine/solvers/_gradient_descent.py`) train
  adaptive parameters of a nested system through an unrolled run. ODEs advance with one `diffrax.diffeqsolve` per inner
  step, with `Tsit5`, a PID controller, and `RecursiveCheckpointAdjoint`, on the pure function `traced_` that
  `JaxBackend` builds.
- `JaxBackend` (`engine/backends/_jax.py`) lowers rate, update, and equation laws and reductions. It has no form for
  impulses, rules, noise, jumps, or Python callables.
- `DosingInteraction` (`systems/biology/common/pharmacokinetics/_dosing.py`) doses a state by infusions, a rate law
  with `pulse_train`, so it already lowers to JAX. The NumPy path respects the breakpoints of pulses, but the Diffrax
  calls of the JAX path do not pass them to the step size controller, so a short infusion can be stepped over.
  `ImpulseInteraction` resets states through `CrossingLaw`, which has no JAX form.
- `ParameterSpec.shape` (`core/specs/_parameter_spec.py`) holds role names only. Matrix parameters by roles and tensor
  contractions exist.
- Law functions such as `lookup` and `series` live in `components/laws/utils` and return Sage expressions that every
  backend lowers.
- `calibrate` (`core/experiments/_calibration.py`) fits parameters of a known system with SciPy through `simulate`.
- Pharmacokinetics lives in `systems/biology/physiology` (`_pharmacokinetics.py`), with shared parts in
  `systems/biology/common/pharmacokinetics`. There is no `biology/pharmacology` subfield and none is created.

## Goals / Non-Goals

**Goals:**
- One primitive for neural terms, `neural_function`, used by every hybrid law, parameter map, and latent rate.
- One fitting function for systems with learned terms, with shared and per-member parameters, on JAX.
- The trained system is an ordinary system that every backend runs.

**Non-Goals:**
- Amortised encoders of variational autoencoders. The latent form of 2602.03215 is fitted here by maximum a posteriori
  effects per patient, which is the empirical Bayes estimate that NLME practice uses. An encoder can be added later as
  a network of the samples.
- Estimating the variances of the random effects. They are given as priors. Their estimation belongs with the Bayesian
  methods of `calibrate` in `add-bayesian-calibration`.
- The PBPK surrogate of 2310.09167. The reference form of `oral_iv_pk_hybrid` is a reduced compartmental model, which
  the paper names as an option.
- Graph networks of chemical structure. Descriptors are given as numbers per compound.
- torch, sbi, deepxde, and other torch libraries.

## Decisions

### Dependencies

- **Equinox 0.13.8, released on 2026-05-05,** becomes a runtime dependency in `pyproject.toml`. It is the network
  library of the Diffrax authors, so modules are pytrees that `jax.grad` and `diffrax` take directly. Flax was rejected
  because its modules carry state outside the pytree and it is a larger dependency. Haiku is no longer developed.
  Writing networks as symbolic expressions over matrix parameters was rejected: the expression of a network with
  hundreds of weights is slow to compile and gives no access to the layers of Equinox.
- JAX 0.12, Diffrax 0.7.2, and Optax 0.2.8 are already runtime dependencies.

### Neural functions in laws

- `neural_function(module, *inputs, weights)` in a new module `components/laws/utils/_neural.py` returns a list of Sage
  expressions, one per output. Each is a call of a symbolic function that holds the static part of the module from
  `equinox.partition(module, equinox.is_array)` in a registry keyed by the call, as `series` holds its table. The
  weights expression is the vector parameter.
- `neural_weights(module)` returns the flat initial weights with `jax.flatten_util.ravel_pytree`, so a part declares
  `theta: Annotated[Array, ParameterSpec(shape=(size,), adaptive=True)]` with the initial value it returns.
- The JAX backend lowers a call to `equinox.combine(unravel(theta), static)(inputs)[k]`, traced and differentiable. For
  a group, the call is vectorised over members with `jax.vmap`. The NumPy backend calls the same function compiled with
  `jax.jit` on NumPy arrays and converts back. The Numba printer has no form, so the law takes the Python path with the
  existing fall-back warning.
- The compiler checks the length of the weights against the module and reports `neural-weights`.

### Fixed shapes

- `ParameterSpec.shape` accepts a tuple of role names and positive integers. The layout of the parameter vector takes
  the product of the sizes, as it does for role shapes. Trajectories record the shape.

### Doses on JAX

- Doses are infusions of the existing `DosingInteraction`, rate laws that JAX already lowers. Bolus resets by
  `ImpulseInteraction` stay on the NumPy path, since a reset has no JAX form and a short infusion approximates a bolus
  as closely as needed.
- The breakpoints of `pulse` and `pulse_train` inputs, which the compiled problem already collects for the SciPy
  integrators, are passed as `jump_ts` to `diffrax.PIDController` in `GradientDescentSolver` and in
  `neural_identification`, so no step crosses the start or the end of an infusion.

### Neural identification

- `neural_identification(system, data, *, trained, individual=None, loss='squares', optimizer='adam',
  learning_rate=1e-2, epochs=1000, validation=0.0, random_state=None)` lives in
  `core/analysis/_neural_identification.py` and returns `NeuralIdentificationResult`. It is an analysis because it
  learns a system, as `sparse_identification` does. It takes the data format of `calibrate`, a mapping from paths to
  times and values, with one column per member for a group.
- It compiles the system with `JaxBackend`, takes `traced_`, and solves the ODEs with `diffrax.diffeqsolve` and
  `SaveAt(ts=...)` at the union of the sample times, with the breakpoints of the doses as `jump_ts`. The solver,
  tolerances, and adjoint of Diffrax are one fact with those of `GradientDescentSolver`, so they are lifted to the
  shared constants of the core and both read them there.
- The loss is the mean squared error over the samples that are not NaN, or the squared error of logarithms with
  `loss='log'` for concentrations. Each individual parameter adds `sum(eta^2) / (2 sigma^2)` divided by the number of
  samples, which makes the minimum the maximum a posteriori estimate.
- Shared parameters are those in `trained`. Individual parameters are per-member parameters of a group, declared with
  `ParameterSpec(per=True, adaptive=True)`, with one value per member. A group compiles once for any size, so all
  members solve in one call.
- `validation` holds out a share of the members, fits only their individual effects with the shared values fixed, and
  reports the error on their samples.
- The result holds `system`, `estimates` (path to fitted value), `effects` (path to one value per member), `losses`
  (one per epoch), `training_error`, `validation_error`, and `samples`.
- Prediction for new patients is the same function with `trained=()`, which needs no method on the result.

### Systems

- **Placement.** Both systems go in `systems/biology/physiology`, where pharmacokinetics lives, each in its own
  module, `_tacrolimus.py` and `_oral_iv_pharmacokinetics.py`. Both dose through `DosingInteraction` of
  `biology/common/pharmacokinetics`. Parts that both new systems share, such as a first-order transfer chain of
  compartments, go to `biology/common/pharmacokinetics` beside it, and parts that only one uses stay in its module.
- **`tacrolimus_population_pk`.** One `BaseSystem` with `form='linear'`, `'saturable'`, or `'latent'`, `n_patients`,
  `hematocrit`, and `random_state`. The group `patients` holds the states `A1` to `A4`, `Ac`, and `Ap`, and the
  per-member covariates `CYP` and `ST` and effects `eta_Ktr`, `eta_CL`, `eta_Vc`, `eta_Q`, and `eta_Vp`, drawn at
  construction from the values of the supplementary tables of 2602.03215. Doses are a `DosingInteraction` into `A1`
  every 12 hours, an infusion of 0.1 hours with a per-member dose. The observable is `Cc = 1000 Ac / Vc`. The latent
  form holds a depot that receives the doses, latent states with a neural rate of the latent states, the depot, the
  covariates, and per-member effects, and a linear readout as `Cc`. Primitives: `rates`, `time-inputs`, `groups`,
  `draws`, `observables`, and `neural-functions` for the latent form.
- **`oral_iv_pk_hybrid`.** One `BaseSystem` with `form='reference'` or `'neural'`, `n_compounds`, and `random_state`.
  The group `compounds` has per-member descriptors and two arms, intravenous (central and peripheral) and oral (gut,
  liver, central, and peripheral). The absorption rate, the absorbed fraction, and the hepatic and renal clearances
  follow from the descriptors through a fixed smooth map in the reference form and through one neural function in the
  neural form, whose last layer at zero gives the typical rates. Areas under the curves are states with the
  concentration as rate, and the observable `F` is their dose-scaled ratio. Primitives: `rates`, `time-inputs`,
  `groups`, `observables`, and `neural-functions`.
- **`growing_neural_cellular_automaton`.** Added from the survey of complex adaptive systems, since the neural
  cellular automata that Hartl, Levin, and Pio-Lopez (2025, arXiv 2509.11131) review need exactly `neural_function` on
  the sites of a lattice. Placement step 3, one system: `computer_science/automata/_neural_cellular_automaton.py`. One
  `BaseSystem` with `n_channels=16`, `hidden=128`, `size`, `fire_rate=0.5`, `target`, and `random_state`. A site
  element holds the channels, of which the first four are colour and alpha. One update interaction perceives every
  channel through `convolve` with the identity and the two Sobel kernels, feeds the perception to one neural function
  of two layers whose last layer starts at zero, multiplies the change by a `bernoulli` draw of each cell with
  probability `fire_rate`, and keeps a cell alive when the largest alpha in its 3 by 3 neighbourhood, a `maximum`
  reduction, exceeds 0.1 before and after the step. Training is the growing experiment of Mordvintsev et al.: from
  one seed cell, a fixed number of steps, and the squared error of the colour and alpha channels against the target at
  the last step, through `GradientStepLaw` on the unrolled run. The sample-pool and damage experiments are left out.
  The default target is a synthetic ring of colours drawn by the system and labelled as such. Primitives: `lattices`,
  `nonlocal-operators`, `updates`, `draws`, `reductions`, `learning`, and `neural-functions`.
- The inventory gains the primitive `neural-functions`, "Neural networks called from laws, with weights as adaptive
  parameters", with `change = "add-hybrid-neural-laws"`.

## Risks / Trade-offs

- [The registry of static modules makes a Sage expression depend on an object outside it] → The key is derived from
  the module structure, and `clone` and `set_params` of the part keep the module, so compiled caches stay valid. A test
  pickles a system with a neural law and runs it again.
- [Fits of 200 patients over many epochs are slow in tests] → Tests use smaller groups and fewer epochs with relaxed
  thresholds, and the full scenarios run in the gallery examples, which the docs build caches.
- [Maximum a posteriori effects ignore their uncertainty] → The result documents it, and the Bayesian methods of
  `calibrate` give posteriors when needed.
- [The JAX backend in single precision loses accuracy] → Neural identification runs in double precision like
  `GradientDescentSolver`.

## Open Questions

- Should a general latent ODE element move to the core once a second system needs one? Today only the latent form of
  tacrolimus uses it.
