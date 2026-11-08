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
- [ ] 3.2 Verify the dosing schedule and gradient-through-doses scenarios with `DosingInteraction` of
  `biology/common/pharmacokinetics` on the NumPy and JAX backends

## 4. Neural identification

- [ ] 4.1 Add `neural_identification` and `NeuralIdentificationResult` in
  `src/skcomplex/core/analysis/_neural_identification.py` with shared parameters, the squares and log losses, missing
  samples, and Optax optimizers, and verify the universal differential equation, known rates, and reproducibility
  scenarios on Lotka-Volterra in `tests/core/analysis/test_neural_identification.py`
- [ ] 4.2 Verify the sparse-samples and unknown-path scenarios
- [ ] 4.3 Add the checks of trained names, adaptivity, and differentiable tags before the fit, and verify the
  not-adaptive and stochastic-system scenarios
- [ ] 4.4 Add `individual` with Gaussian priors, per-member effects, and `validation` over members, and verify the
  prior-not-positive scenario and the recovery of effects on a small group
- [ ] 4.5 Export the function and result from `skcomplex.core.analysis`, declare their analysis tags, and document
  hybrid laws, covariate maps, mixed effects, and refits of new members in the learned surrogates section of
  `docs/overview/user_guide/analyse.md` with runnable examples

## 5. Pharmacokinetic systems

- [ ] 5.1 Add `TacrolimusPopulationPKSystem` in `src/skcomplex/systems/biology/physiology/_tacrolimus.py` with the
  linear and saturable forms, dosed by `DosingInteraction`, export it from the subfield, and verify the superposition,
  covariate, and reproducible population scenarios in `tests/systems/biology/physiology/`
- [ ] 5.2 Add the latent form of the tacrolimus system with its neural rate and readout, and verify it compiles on JAX
  and runs on NumPy
- [ ] 5.3 Add `OralIVPKHybridSystem` in `src/skcomplex/systems/biology/physiology/_oral_iv_pharmacokinetics.py` with
  the reference and neural forms, moving any part shared with the tacrolimus system to
  `biology/common/pharmacokinetics`, and verify the bioavailability and untrained neural form scenarios
- [ ] 5.4 Verify the covariate map scenario on held-out compounds and the population clearance, three-sample exposure,
  and misspecified elimination scenarios, with groups and epochs small enough for the test suite
- [ ] 5.5 Add `docs/examples/biology/physiology/plot_tacrolimus_population_pk.py` with the population fit and the
  exposure of new patients, and `plot_oral_iv_pk_hybrid.py` with the learned bioavailability, and verify the gallery
  build runs them
- [ ] 5.6 Add the inventory entries `tacrolimus_population_pk` and `oral_iv_pk_hybrid` under `[biology.physiology]`
  citing 2602.03215 and 2310.09167, and the primitive `neural-functions` with `change = "add-hybrid-neural-laws"`, and
  verify the inventory check passes

## 6. Checks

- [ ] 6.1 Run `nox` (formatting, ruff, mypy, pydoclint, import contracts, tests of the touched modules, doctests, the
  inventory check, and the strict docs build) and verify every session passes
