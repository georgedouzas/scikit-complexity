# Tasks

## 1. Dependency and fixed shapes

- [ ] 1.1 Add `equinox>=0.13.8` to the runtime dependencies in `pyproject.toml` and to the dependency list of the docs
  and the freshness check, and verify `pdm install` resolves and `import equinox` works
- [ ] 1.2 Accept positive integers in `ParameterSpec.shape`, lay out and record such parameters, and verify the weight
  vector and non-positive shape scenarios in `tests/core/test_specs.py` and the simulation tests

## 2. Neural functions in laws

- [ ] 2.1 Add `neural_function` and `neural_weights` in `src/skcomplex/core/components/laws/utils/_neural.py`, export
  them from `skcomplex.core.components.laws.utils`, and verify their doctests
- [ ] 2.2 Lower neural calls in `JaxBackend`, vectorised over group members, and in the NumPy backend through a jitted
  JAX function, and verify the rate scenario on both backends and the changed-weights scenario
- [ ] 2.3 Report `neural-weights` in the compiler for a weight vector of the wrong length, and verify the scenario
- [ ] 2.4 Make the Numba path fall back to Python with the existing warning, and verify the 2000-member scenario
- [ ] 2.5 Tag systems with neural laws as differentiable, and verify the gradient of the weights against finite
  differences, also through `GradientStepLaw`
- [ ] 2.6 Document neural functions in the section on writing laws of the user guide with a runnable example

## 3. Doses on JAX

- [ ] 3.1 Lift the Diffrax solver, tolerances, and adjoint of `GradientDescentSolver` to shared constants, pass the
  breakpoints of pulse inputs as `jump_ts`, and verify the existing learning-dynamics tests still pass
- [ ] 3.2 Verify the dosing schedule and gradient-through-doses scenarios on `drug_toxicity` on the NumPy and JAX
  backends

## 4. Neural identification

- [ ] 4.1 Add `neural_identification` and `NeuralIdentificationResult` in
  `src/skcomplex/core/analysis/_neural_identification.py` with shared parameters, the squares and log losses, and
  Optax optimizers, and verify the universal differential equation, known rates, and reproducibility scenarios on
  `lotka_volterra` in `tests/core/analysis/test_neural_identification.py`
- [ ] 4.2 Add several individuals with `settings`, vectorised solves, and missing samples, and verify the sparse
  samples, unknown path, and settings-length scenarios
- [ ] 4.3 Add the checks of parameter names and differentiable tags before the fit, and verify the unknown parameter
  and stochastic system scenarios
- [ ] 4.4 Add `maps` and `covariates` with Equinox networks, and verify the covariate map and missing covariates
  scenarios on `two_compartment_pharmacokinetics`
- [ ] 4.5 Add `individual` with log-normal effects, Gaussian priors, refits with `trained=()`, and `validation` over
  individuals, and verify the typical rate, prior, and three-sample scenarios on `drug_toxicity`
- [ ] 4.6 Verify the latent neural ODE scenario on runs of `two_compartment_pharmacokinetics`, with a latent system
  written from core parts in the test
- [ ] 4.7 Export the function and result from `skcomplex.core.analysis`, declare their analysis tags, and document
  hybrid laws, covariate maps, mixed effects, and refits of new individuals in the learned surrogates section of
  `docs/overview/user_guide/analyse.md` with runnable examples

## 5. Gallery and inventory

- [ ] 5.1 Add a gallery example in `docs/examples/biology/ecology/` beside the `lotka_volterra` example that fits a
  universal differential equation, and verify the gallery build runs it
- [ ] 5.2 Add a gallery example in `docs/examples/biology/physiology/` beside the `drug_toxicity` example that fits
  mixed effects to a population and predicts new individuals from three samples, and verify the gallery build runs it
- [ ] 5.3 Add the primitive `neural-functions` with `change = "add-hybrid-neural-laws"` to the inventory, and verify
  the inventory check passes

## 6. Checks

- [ ] 6.1 Run `nox` (formatting, ruff, mypy, pydoclint, import contracts, tests of the touched modules, doctests, the
  inventory check, and the strict docs build) and verify every session passes
