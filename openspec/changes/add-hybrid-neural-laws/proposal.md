# Proposal

## Why

A law that is partly known is common. The mechanism of a pharmacokinetic model or of a population model is trusted, and
one term or one parameter is not. The core can train adaptive parameters through an unrolled run with
`GradientStepLaw`, but a law cannot hold a neural network, a parameter cannot be a learned function of covariates, and
no fit separates parameters shared by a population from the effects of each individual. Universal differential
equations (2306.10335), parameter maps of chemical structure (2310.09167), and latent neural ODEs of patients
(2602.03215) need all three, and they beat purely mechanistic fits when the mechanism is misspecified.

## What Changes

- New law function `neural_function(module, *inputs, weights=...)` that calls an Equinox network inside a law. Its
  weights are one adaptive vector parameter of the part, so `set_params`, the `schedule` of `simulate`, and
  `GradientStepLaw` read and change them like any parameter. It lowers to JAX and NumPy, and other backends fall back
  to Python with the warning they already give.
- `ParameterSpec(shape=...)` accepts whole numbers as fixed sizes beside role names, so a weight vector of a given
  length is declared.
- The JAX path of the ODE solvers steps on the breakpoints of pulse inputs, so dosing schedules such as those of the
  existing `DosingInteraction` of `biology/common/pharmacokinetics` are differentiable.
- New analysis `neural_identification(system, data, trained=..., individual=..., maps=...)` that fits a
  differentiable system to observed series of one or several individuals through Diffrax with an Optax optimizer.
  `trained` names the shared parameters, neural weights or mechanistic rates. `maps` gives parameters by neural
  networks of the covariates of each individual. `individual` names parameters with log-normal random effects, fitted
  under a Gaussian prior. It returns a frozen result with the fitted `system`, the estimates, the trained maps, the
  effects, and the loss of every epoch. Called on new individuals with no shared parameter trained, it gives their
  maximum a posteriori effects from few samples.
- No new system. The methods are tested and shown on catalog systems: `lotka_volterra` for universal differential
  equations, `two_compartment_pharmacokinetics` for covariate maps and latent neural ODEs, and `drug_toxicity`, dosed by
  infusions, for doses on JAX and mixed effects.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `learning-dynamics`: laws call neural networks, parameters take fixed shapes, and infusion schedules are
  differentiable.
- `data-driven-dynamics`: hybrid neural surrogates fitted by `neural_identification`, with parameter maps of covariates
  and mixed effects.

## Impact

- Code: `src/skcomplex/core/components/laws/utils/` (new `_neural.py`), `src/skcomplex/core/specs/_parameter_spec.py`,
  `src/skcomplex/core/engine/backends/_jax.py` and `_numpy.py`, `core/engine/solvers/_gradient_descent.py`, and a new
  `src/skcomplex/core/analysis/_neural_identification.py`. No module of `skcomplex.systems` changes.
- Dependencies: Equinox becomes a runtime dependency. JAX, Diffrax, and Optax already are.
- No dependency on the other changes of this set. `calibrate` of `add-bayesian-calibration` stays the function for
  parameters of a known system, and `neural_identification` is the function for systems with learned terms.
- Docs: a user guide section on hybrid laws, gallery examples of the probe systems, and the primitive
  `neural-functions` of the inventory.
