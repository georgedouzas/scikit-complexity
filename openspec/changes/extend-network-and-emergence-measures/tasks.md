## 1. Core-periphery detection

- [ ] 1.1 Make `METHODS` of `_networks.py` a mapping from kind to methods with `method=None` taking the first, raising `AnalysisError` that lists the methods of the kind, and verify the default-communities and wrong-method scenarios with tests in `tests/core/analysis/test_networks.py`
- [ ] 1.2 Add the kind `'core-periphery'` with the Borgatti-Everett search and the k-core method, the fields `core`, `coreness`, and `quality`, and verify the star, planted-core, and same-seed scenarios with tests
- [ ] 1.3 Symmetrise directed and weighted graphs by summed reciprocal weights, and verify the transfer-counts scenario with a test
- [ ] 1.4 Add a doctest of the kind on a star and a paragraph in the network section of `docs/overview/user_guide/analyse.md`, and verify both run

## 2. Bootstrap of the causal emergence gain

- [ ] 2.1 Add `resamples`, `confidence`, and `random_state` to `emergence` for `'causal'`, with the bootstrap through `scipy.stats.bootstrap` and the fields `gains`, `interval`, and `p_value`, and verify the emergent-chain, no-emergence, and same-seed scenarios with tests in `tests/core/analysis/test_emergence.py`
- [ ] 2.2 Raise `AnalysisError` for resamples of a matrix that is not whole counts, and verify it with a test
- [ ] 2.3 Add a doctest of the bootstrap and a paragraph in the emergence section of `docs/overview/user_guide/analyse.md`, and verify both run

## 3. Surviving by serving

- [ ] 3.1 Add `SurvivingByServingSystem` and its elements and interactions in a new module of `src/skcomplex/systems/biology/evolution/`, exported from the subfield, with the defaults of Table 1 of 2606.26733, and verify its doctest
- [ ] 3.2 Add tests in `tests/systems/biology/evolution/` for the pool without agents, the exclusion rules, the credit balance, and the Hill adaptation
- [ ] 3.3 Add the inventory entry `surviving_by_serving` under `biology.evolution` with source `2606.26733` and status `implemented`, and verify the inventory check passes
- [ ] 3.4 Add the gallery example `docs/examples/biology/evolution/plot_surviving_by_serving.py` that runs 24 agents for 1000 episodes and detects the core of the transfer network over time windows, and verify the gallery check passes
- [ ] 3.5 Commit the work as conventional `feat` commits, with the breaking `method` default marked, and verify `nox -s changelog` lists them

## 4. Causal emergence across scales

- [ ] 4.1 Add the kind `'scales'` to `emergence` with both pairs of causal primitives, the consistency check, the given
  and searched paths, and the fields `scale_values`, `contributions`, `path`, and `complexity`, and verify the
  chain-of-the-paper, equal-gains, mesoscale, and inconsistent-partition scenarios in
  `tests/core/analysis/test_emergence.py`
- [ ] 4.2 Add a doctest of the kind on a small chain and a paragraph in the emergence section of
  `docs/overview/user_guide/analyse.md`, and verify both run

## 5. Group coordination

- [ ] 5.1 Add the kinds `'synergy-capacity'` and `'coalition'` to `emergence` on `partial_information` and the
  information estimators, reading trajectories and, once `add-learning-foundation` is applied, pooled observed runs,
  and verify the exclusive-or, pooled-runs, independent-copies, and three-way scenarios
- [ ] 5.2 Add `method` to `surrogate_test` with `'shift'`, `'swap'`, and `'block'`, drawn within every run of observed
  runs, and verify the shifts, swaps, and default-unchanged scenarios in `tests/core/analysis/test_surrogates.py`
- [ ] 5.3 Add a paragraph and a runnable example of a synergy test against swap surrogates in the emergence section of
  `docs/overview/user_guide/analyse.md`, and verify it runs

## 6. Causal networks

- [ ] 6.1 Add `tigramite>=5.2.10.1`, `pyEDM>=2.5.7`, and `statsmodels>=0.15.0` (unless `add-learning-bridges` added it
  first) to the runtime dependencies in `pyproject.toml` and the Project Profile with their release dates and licences,
  record the GPL-3.0 decision of 2026-11-10 on the installation page, and verify `pdm install` resolves and all import
- [ ] 6.2 Add `causal_network` and `CausalNetworkResult` in `src/skcomplex/core/analysis/_causal_network.py` with
  `'pcmci'` on tigramite, the graph, and the pooling of runs, export them with their analysis tags, and verify the
  known-linear-links scenario (marked slow) and the graph-for-network-measures scenario in
  `tests/core/analysis/test_causal_network.py`
- [ ] 6.3 Add `'granger'` on statsmodels and `'transfer'` on the estimators of `information` with shift surrogates, and
  verify the Granger-on-the-same-model scenario and a transfer entropy result on the same model
- [ ] 6.4 Add `'ccm'` on pyEDM with the embedding chosen by false nearest neighbours, the skills by library size, and
  the surrogate p-value, and verify the cross-mapping-of-Lorenz-63 and independent-systems scenarios
- [ ] 6.5 Add a paragraph and a runnable example of `causal_network` in the network section of
  `docs/overview/user_guide/analyse.md`, and a gallery example
  `docs/examples/physics/climate/plot_causal_network.py` that finds the links of a coupled low-order climate model, and
  verify both run

## 7. Verification

- [ ] 7.1 Run `nox -s checks`, the tests of the touched modules, and the strict docs build, and fix every finding
