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
- One fitting function for systems with learned terms, with shared and per-individual parameters, on JAX.
- The trained system is an ordinary system that every backend runs.

**Non-Goals:**
- Amortised encoders of variational autoencoders. The latent form of 2602.03215 is fitted here by maximum a posteriori
  effects per individual, which is the empirical Bayes estimate that NLME practice uses. An encoder can be added later
  as a network of the samples.
- Estimating the variances of the random effects. They are given as priors. Their estimation belongs with the Bayesian
  methods of `calibrate` in `add-bayesian-calibration`.
- New catalog systems. The tacrolimus model of 2602.03215 and the rat model of 2310.09167 need no core primitive
  beyond those of this change, which catalog systems already exercise, so they are not added (see Probes).
- Graph networks of chemical structure. Covariates are given as numbers per individual.
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

- `neural_identification(system, data, *, trained, settings=None, maps=None, covariates=None, individual=None,
  loss='squares', optimizer='adam', learning_rate=1e-2, epochs=1000, validation=0.0, random_state=None)` lives in
  `core/analysis/_neural_identification.py` and returns `NeuralIdentificationResult`. It is an analysis because it
  learns a system, as `sparse_identification` does.
- **Individuals are runs.** The data of one individual is the format of `calibrate`, a mapping from paths to times and
  values. A list of such mappings gives several individuals, and `settings` gives each its parameter values, such as
  doses, applied through `set_params`. So any catalog system is a population model without a group, and no part needs
  per-member parameters.
- It compiles the system once with `JaxBackend`, takes `traced_`, and solves the ODEs of every individual with
  `diffrax.diffeqsolve` and `SaveAt(ts=...)` at its sample times, vectorised over individuals with `jax.vmap` on their
  parameter vectors, with the breakpoints of the doses as `jump_ts`. The solver, tolerances, and adjoint of Diffrax
  are one fact with those of `GradientDescentSolver`, so they are lifted to the shared constants of the core and both
  read them there.
- **Covariate maps.** `maps` gives a parameter by an Equinox network of the covariates of each individual. The
  network writes into the parameter vector of the individual before its solve, so the laws of the system do not
  change. The result holds the trained networks in `maps`, which give the parameters of a new individual from its
  covariates.
- **Mixed effects.** `individual` gives a parameter the value `typical * exp(eta)` per individual, the log-normal
  convention of NLME. `typical` is the value of the system, trained when it is also in `trained`. Each `eta` adds
  `sum(eta^2) / (2 sigma^2)` divided by the number of samples to the loss, which makes the minimum the maximum a
  posteriori estimate.
- **Neural terms in laws.** Weights of `neural_function` are parameters like any other and are named in `trained`.
  Universal differential equations and latent ODEs are systems the user writes with neural functions, from the parts of
  a catalog system or from core parts.
- The loss is the mean squared error over the samples that are not NaN, or the squared error of logarithms with
  `loss='log'` for concentrations.
- `validation` holds out a share of the individuals, fits only their effects with the shared values fixed, and reports
  the error on their samples.
- The result holds `system`, `estimates` (path to fitted value), `maps`, `effects` (path to one value per individual),
  `losses` (one per epoch), `training_error`, `validation_error`, and `samples`.
- Prediction for new individuals is the same function with `trained=()`, which needs no method on the result.

### Probes

- A change adds a system only when supporting it needs a change to the core. No system is added here, and the methods
  are tested and shown on catalog systems:
  - `lotka_volterra`: its interaction terms replaced by a neural function, for universal differential equations
  - `drug_toxicity`: weekly infusions through `DosingInteraction`, for doses on JAX and log-normal mixed effects on the
    elimination rate
  - `two_compartment_pharmacokinetics`: the elimination rate mapped from covariates, and a latent neural ODE with
    individual effects fitted on its runs with random rates, standing in for the misspecified tacrolimus model of
    2602.03215
- The inventory gains the primitive `neural-functions`, "Neural networks called from laws, with weights as adaptive
  parameters", with `change = "add-hybrid-neural-laws"`. No inventory entry lists it until a system needs it.

## Risks / Trade-offs

- [The registry of static modules makes a Sage expression depend on an object outside it] → The key is derived from
  the module structure, and `clone` and `set_params` of the part keep the module, so compiled caches stay valid. A test
  pickles a system with a neural law and runs it again.
- [Fits of 200 individuals over many epochs are slow in tests] → Tests use fewer individuals and epochs with relaxed
  thresholds, and the full scenarios run in the gallery examples, which the docs build caches.
- [Maximum a posteriori effects ignore their uncertainty] → The result documents it, and the Bayesian methods of
  `calibrate` give posteriors when needed.
- [The JAX backend in single precision loses accuracy] → Neural identification runs in double precision like
  `GradientDescentSolver`.

## Open Questions

- Should a general latent ODE element join the core? Today latent systems are written from core parts in the tests and
  the gallery, and a catalog system that needs one would decide it.
