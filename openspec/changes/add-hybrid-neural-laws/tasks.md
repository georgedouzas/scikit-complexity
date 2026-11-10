# Tasks

## 1. Prerequisite

- [ ] 1.1 Check that `add-learning-foundation` is applied, so `learned_function`, fixed shapes, observed runs, the
  shared loop, and `identify` exist, and stop if it is not

## 2. Doses on JAX

- [ ] 2.1 Pass the breakpoints of pulse inputs as `jump_ts` through the shared Diffrax constants, and verify the
  existing learning-dynamics and identification tests still pass
- [ ] 2.2 Verify the dosing schedule and gradient-through-doses scenarios with `DosingInteraction` of
  `biology/common/pharmacokinetics` on the NumPy and JAX backends

## 3. Mixed effects

- [ ] 3.1 Add `individual` to `identify` with Gaussian priors and the field `effects` of `IdentificationResult`, and
  verify the prior-not-positive scenario and the recovery of effects on a small group in
  `tests/core/experiments/test_identification.py`
- [ ] 3.2 Hold out members in `validation` when `individual` is set, and verify the held-out error is reported
- [ ] 3.3 Verify that `identify` with `trained=()` keeps the shared values and fits only the effects of new members
- [ ] 3.4 Document mixed effects, covariate maps, and refits of new members in the calibrate page of the user guide,
  beside `identify`, with runnable examples

## 4. Pharmacokinetic systems

- [ ] 4.1 Add `TacrolimusPopulationPKSystem` in `src/skcomplex/systems/biology/physiology/_tacrolimus.py` with the
  linear and saturable forms, dosed by `DosingInteraction`, export it from the subfield, and verify the superposition,
  covariate, and reproducible population scenarios in `tests/systems/biology/physiology/`
- [ ] 4.2 Add the latent form of the tacrolimus system with its learned rate and readout, and verify it compiles on JAX
  and runs on NumPy
- [ ] 4.3 Add `OralIVPKHybridSystem` in `src/skcomplex/systems/biology/physiology/_oral_iv_pharmacokinetics.py` with
  the reference and learned forms, moving any part shared with the tacrolimus system to
  `biology/common/pharmacokinetics`, and verify the bioavailability and untrained learned form scenarios
- [ ] 4.4 Verify the covariate map scenario on held-out compounds and the population clearance, three-sample exposure,
  and misspecified elimination scenarios, with groups and epochs small enough for the test suite
- [ ] 4.5 Add `docs/examples/biology/physiology/plot_tacrolimus_population_pk.py` with the population fit and the
  exposure of new patients, and `plot_oral_iv_pk_hybrid.py` with the learned bioavailability, and verify the gallery
  build runs them
- [ ] 4.6 Add the inventory entries `tacrolimus_population_pk` and `oral_iv_pk_hybrid` under `[biology.physiology]`
  citing 2602.03215 and 2310.09167 with the primitive `learned-functions`, and verify the inventory check passes

## 5. Neural cellular automaton

- [ ] 5.1 Verify that `learned_function` lowers on the sites of a lattice with `convolve` inputs on JAX and NumPy, with
  a test of a one-layer network over the identity and Sobel perception of a 16 by 16 field
- [ ] 5.2 Add `GrowingNeuralCellularAutomatonSystem` in
  `src/skcomplex/systems/computer_science/automata/_neural_cellular_automaton.py`, export it, and verify the
  untrained-network and dead-cells scenarios in `tests/systems/computer_science/automata/`
- [ ] 5.3 Verify the training scenario on JAX, marked slow
- [ ] 5.4 Add the inventory entry `growing_neural_cellular_automaton` under `[computer_science.automata]` citing
  2509.11131 with the primitive `learned-functions`, and the gallery example
  `docs/examples/computer_science/automata/plot_growing_neural_cellular_automaton.py` that trains and grows the
  synthetic ring, and verify the inventory check and the gallery build pass

## 6. Checks

- [ ] 6.1 Run `nox` (formatting, ruff, mypy, pydoclint, import contracts, tests of the touched modules, doctests, the
  inventory check, and the strict docs build) and verify every session passes
