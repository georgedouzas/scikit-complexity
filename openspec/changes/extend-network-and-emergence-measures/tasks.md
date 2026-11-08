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

## 4. Verification

- [ ] 4.1 Run `nox -s checks`, the tests of the touched modules, and the strict docs build, and fix every finding
