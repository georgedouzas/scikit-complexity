## 1. Derivatives of reductions

- [ ] 1.1 Add `_derivative_` hooks to `total`, `mean`, and `variance` in
  `src/skcomplex/core/components/laws/utils/_reductions.py`, with the error for the states of the reduced member, and
  verify the mean-distance scenario on the NumPy backend in `tests/core/components/laws/utils/test_reductions.py`
- [ ] 1.2 Verify the mean-distance scenario on the Numba and JAX backends
- [ ] 1.3 Raise `ValidationError` for derivatives of the other reductions, and verify the scenario
- [ ] 1.4 Verify the gradient-flow scenario with `GradientInteraction` in
  `tests/core/components/interactions/test_gradient.py`
- [ ] 1.5 Document derivatives of reductions in the user guide on writing laws with a runnable example, and verify the
  markdown tests run it

## 2. Active-inference flock

- [ ] 2.1 Add `FreeEnergyInteraction` and `ActiveInferenceFlockSystem` in
  `src/skcomplex/systems/biology/collective_behaviour/_active_inference_flock.py`, export the system, and verify the
  one-sector, set-point, and defaults scenarios in `tests/systems/biology/collective_behaviour/`
- [ ] 2.2 Add the observables and the milling configuration, and verify the collective-states scenario, marked slow
- [ ] 2.3 Add the inventory entry `active_inference_flock` with the source 2307.14804 and the primitive
  `reduction-derivatives` with `change = "add-active-inference"`, and verify the inventory check passes
- [ ] 2.4 Add the gallery example `docs/examples/biology/collective_behaviour/plot_active_inference_flock.py` with the
  polarized and milling configurations and the polarization over a sweep of `Gamma_z`, and verify the strict gallery
  build runs it

## 3. Integration

- [ ] 3.1 Commit the work as conventional `feat` commits and verify `nox -s changelog` lists them
- [ ] 3.2 Run `nox` formatting, checks, tests of the touched modules, and the strict docs build, and fix every finding
