# Proposal

## Why

A law that is partly known is common: the mechanism of a pharmacokinetic model or of a population model is trusted, and
one term or one parameter is not. `add-learning-foundation` lets a law call a learned function and lets `identify` fit
its weights to observed runs, but no fit separates parameters shared by a population from the effects of each
individual, and short infusions are stepped over by the JAX path of the ODE solvers. Universal differential equations
(2306.10335), parameter maps of chemical structure (2310.09167), and latent neural ODEs of patients (2602.03215) need
both, and they beat purely mechanistic fits when the mechanism is misspecified. Neural cellular automata (2509.11131)
need learned functions on the sites of a lattice, trained through the unrolled run.

## What Changes

- The JAX path of the ODE solvers steps on the breakpoints of pulse inputs, so dosing schedules of the existing
  `DosingInteraction` of `biology/common/pharmacokinetics` are differentiable.
- `identify` of `add-learning-foundation` gains `individual`, a mapping from per-member parameters of a group to the
  standard deviation of their Gaussian prior, the random effects, fitted beside the shared parameters named by
  `trained`. `IdentificationResult` gains `effects`, one value per member. Called with `trained=()` on new members, it
  gives their maximum a posteriori effects from few samples. Its `validation` holds out members when `individual` is
  set.
- Parameters mapped from covariates are `learned_function` calls of per-member covariates, so one `identify` call learns
  the map across members, and a new member gets its parameter from its covariates.
- New systems in `biology/physiology`, beside the existing two-compartment pharmacokinetics, dosed through
  `DosingInteraction` of `biology/common/pharmacokinetics`:
  - `tacrolimus_population_pk`: a group of patients with transit absorption, two compartments, covariates, and
    log-normal random effects, with linear or saturable elimination, and a latent neural form
  - `oral_iv_pk_hybrid`: compounds given orally and intravenously, whose rates follow from descriptors through a fixed
    reference map or through a learned map
- New system in `computer_science/automata`, added from the survey of complex adaptive systems (2509.11131):
  - `growing_neural_cellular_automaton`: the growing neural cellular automaton of Mordvintsev et al. (2020), whose 16
    channels per cell are updated by a learned function of the perceived neighbourhood (the identity and two Sobel
    convolutions of every channel) with a random mask of cells and an alive mask, trained through the unrolled run to
    grow a target pattern from one seed

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `learning-dynamics`: infusion schedules are differentiable.
- `experiments`: mixed effects, covariate maps, and refits of new members in `identify`.
- `model-library`: the systems `tacrolimus_population_pk`, `oral_iv_pk_hybrid`, and
  `growing_neural_cellular_automaton`.

## Impact

- Code: `core/engine/solvers/_gradient_descent.py` and the Diffrax constants it shares,
  `core/experiments/_identification.py`, two modules in `src/skcomplex/systems/biology/physiology/`, and one module in
  `src/skcomplex/systems/computer_science/automata/`.
- Dependencies: none new. Equinox comes with `add-learning-foundation`.
- Depends on `add-learning-foundation`, which adds `learned_function`, fixed shapes of `ParameterSpec`, observed runs
  with per-member columns and missing values, the shared loop, and `identify`. `calibrate` stays the function for
  posteriors of known parameters.
- Docs: a user guide section on mixed effects, gallery examples, and the inventory.
