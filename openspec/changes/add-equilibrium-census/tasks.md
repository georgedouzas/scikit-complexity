## 1. Deflated root search

- [ ] 1.1 Add a private deflated search to `src/skcomplex/core/analysis/_stability.py` that runs `scipy.optimize.root` on `M(x) f(x)` from seeded uniform starts within bounds, and verify with a test that `dx/dt = sin(x)` from 50 starts in `[-10, 10]` gives the seven roots `k pi` once each
- [ ] 1.2 Replace `_numeric_points` with the deflated search in the fallback of `'fixed-points'`, and verify that the existing tests of `tests/core/analysis/test_stability.py` pass unchanged

## 2. Census kind

- [ ] 2.1 Add the numeric drift and Jacobian of a compiled problem, from `problem.drift` and `problem.jacobian` or `scipy.differentiate.jacobian` for callable laws, and verify with a test that both Jacobians agree within 1e-6 on a symbolic system
- [ ] 2.2 Add the kind `'census'` with the settings `parameters`, `starts`, `bounds`, `realizations`, and `random_state`, the index of every point, and the counts by index, and verify the bistable flow, the saddle of a map, and the same-seed scenarios with tests
- [ ] 2.3 Raise `AnalysisError` for forms other than ODE and map, and verify the SDE scenario with a test
- [ ] 2.4 Verify with tests that a group of 200 relaxing members and a system with a callable rate law are accepted by the census
- [ ] 2.5 Once `add-learning-foundation` is applied, take the Jacobian of systems with learned laws by `jax.jacfwd` of the JAX lowering, and verify the identified-system-with-a-learned-law scenario
- [ ] 2.6 Add realizations by `set_params(random_state=seed)` with the mean count, typical count, and both complexities, raising `AnalysisError` for a system without `random_state`, and verify both behaviours with tests
- [ ] 2.7 Add the new `StabilityResult` fields with their docstrings and a doctest of the census on the bistable flow, and verify the doctest runs
- [ ] 2.8 Document the census in the stability section of `docs/overview/user_guide/analyse.md`, stating that counts are lower bounds that grow with the starts, and verify the page builds

## 3. Random force field relaxation system

- [ ] 3.1 Add `RandomForceFieldRelaxationSystem` in a new module of `src/skcomplex/systems/physics/statistical_mechanics/` with `size`, `mu`, `gamma`, `tau`, `features`, and `random_state`, exported from the subfield, and verify its doctest
- [ ] 3.2 Add tests in `tests/systems/physics/statistical_mechanics/` for the symmetric Jacobian at `tau = 1`, the zero divergence at `tau = 0`, and the decay at `gamma = 0`
- [ ] 3.3 Add tests of the census of the system for the trivial phase at `gamma = 0.5` and for a typical count below the mean count at `gamma = 3` and `tau = 1`
- [ ] 3.4 Add the inventory entry `random_force_field_relaxation` under `physics.statistical_mechanics` with source `2611.08324`, its primitives, and status `implemented`, and verify the inventory check passes
- [ ] 3.5 Add the gallery example `docs/examples/physics/statistical_mechanics/plot_random_force_field_relaxation.py` that plots the quenched and annealed complexities against `gamma` for several `tau`, and verify the gallery check passes
- [ ] 3.6 Commit the census kind and the new system as conventional `feat` commits, and verify `nox -s changelog` lists them

## 4. Verification

- [ ] 4.1 Run `nox -s checks`, the tests of the touched modules, and the strict docs build, and fix every finding
