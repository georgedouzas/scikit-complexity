# Proposal

## Why

A law that is partly known is common: the mechanism of a pharmacokinetic model or of a population model is trusted, and
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
- The JAX path of the ODE solvers steps on the breakpoints of pulse inputs, so dosing schedules of the existing
  `DosingInteraction` of `biology/common/pharmacokinetics` are differentiable.
- New analysis `neural_identification(system, data, trained=..., individual=...)` that fits a differentiable system to
  observed series through Diffrax with an Optax optimizer. `trained` names the parameters shared by every member,
  neural weights or mechanistic rates. `individual` names per-member parameters, the random effects, fitted under a
  Gaussian prior. It returns a frozen result with the fitted `system`, the estimates, the effects, and the loss of
  every epoch. Called again on a new member with no shared parameter trained, it gives the maximum a posteriori
  effects of that member from few samples.
- New systems in `biology/physiology`, beside the existing two-compartment pharmacokinetics, dosed through
  `DosingInteraction` of `biology/common/pharmacokinetics`:
  - `tacrolimus_population_pk`: a group of patients with transit absorption, two compartments, covariates, and
    log-normal random effects, with linear or saturable elimination, and a latent neural form
  - `oral_iv_pk_hybrid`: compounds given orally and intravenously, whose rates follow from descriptors through a fixed
    reference map or through a neural map
- New system in `computer_science/automata`, added from the survey of complex adaptive systems (2509.11131):
  - `growing_neural_cellular_automaton`: the growing neural cellular automaton of Mordvintsev et al. (2020), whose 16
    channels per cell are updated by a neural function of the perceived neighbourhood (the identity and two Sobel
    convolutions of every channel) with a random mask of cells and an alive mask, trained through the unrolled run to
    grow a target pattern from one seed

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `learning-dynamics`: laws call neural networks, parameters take fixed shapes, and infusion schedules are
  differentiable.
- `data-driven-dynamics`: hybrid neural surrogates fitted by `neural_identification`, with parameter maps of covariates
  and mixed effects.
- `model-library`: the systems `tacrolimus_population_pk`, `oral_iv_pk_hybrid`, and
  `growing_neural_cellular_automaton`.

## Impact

- Code: `src/skcomplex/core/components/laws/utils/` (new `_neural.py`), `src/skcomplex/core/specs/_parameter_spec.py`,
  `src/skcomplex/core/engine/backends/_jax.py` and `_numpy.py`, `core/engine/solvers/_gradient_descent.py`, a new
  `src/skcomplex/core/analysis/_neural_identification.py`, two modules in
  `src/skcomplex/systems/biology/physiology/`, and one module in `src/skcomplex/systems/computer_science/automata/`.
- Dependencies: Equinox becomes a runtime dependency. JAX, Diffrax, and Optax already are.
- No dependency on the other changes of this set. `calibrate` of `add-bayesian-calibration` stays the function for
  parameters of a known system, and `neural_identification` is the function for systems with learned terms.
- Docs: a user guide section on hybrid laws, gallery examples, and the inventory.
