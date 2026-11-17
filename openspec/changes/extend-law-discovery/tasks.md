# Tasks

## 1. Prerequisite

- [ ] 1.1 Check that `add-learning-foundation` is applied, so observed runs, `observe`, and `split_runs` exist, and stop
  if it is not

## 2. Sparse identification options

- [ ] 2.1 Add `weak`, `subdomains`, and `test_degree` to `sparse_identification` with `pysindy.WeakPDELibrary`, and
  verify the noisy and noise-free Lorenz scenarios in `tests/core/analysis/test_sparse_identification.py`
- [ ] 2.2 Add `ensemble`, `drop_terms`, and `random_state` with `pysindy.EnsembleOptimizer`, set `inclusion` and the
  median coefficients, and verify the inclusion, reproducibility, and invalid-size scenarios
- [ ] 2.3 Add `parameters`, read from the conditions of observed runs, and `parameter_degree` with
  `pysindy.ParameterizedLibrary`, declare the parameters on the surrogate element, and verify the Lorenz over `rho` and
  run-without-the-condition scenarios
- [ ] 2.4 Move `_input_declarations` into `_sparse_identification.py`, add `inputs` to `sparse_identification` and read
  the inputs of observed runs, and verify the forced oscillator and input schedule scenarios, with the mode
  decomposition tests still passing
- [ ] 2.5 Verify the mean-of-stochastic-replicates and swept-parameter-as-a-condition scenarios on runs built by
  `observe`, and add `training_error` and `validation_error` to `SparseIdentificationResult`
- [ ] 2.6 Update the docstring examples of `sparse_identification` and `SparseIdentificationResult`, the learned
  surrogates section of `docs/overview/user_guide/analyse.md`, and the changelog entry for the new fields, and verify
  the doctests run

- [ ] 2.7 Add `system` and `learned` to `sparse_identification`, evaluating the trained models at the states of the
  source and replacing the recovered terms in the returned system, and verify the system-without-learned-functions
  scenario and, once the foundation's `UniversalLotkaVolterraSystem` exists, the
  interaction-terms-of-Lotka-Volterra scenario, marked slow
- [ ] 2.8 Document the recovery of learned terms in the learned surrogates section of
  `docs/overview/user_guide/analyse.md`, and extend the gallery example
  `docs/examples/biology/ecology/plot_universal_lotka_volterra.py` with the recovered equation, and verify both run

## 3. Mode decomposition extensions

- [ ] 3.1 Add `eigenfunctions`, `eigenfunction_errors`, `valid`, and `eigenfunction_tolerance` from the left
  eigenvectors, and verify the slow manifold and Duffing scenarios in `tests/core/analysis/test_mode_decomposition.py`
- [ ] 3.2 Fit the readout of the variables, declare one observable per variable on the lifted system, and verify the
  lifted run records `x` and `y`
- [ ] 3.3 Add `delays` with stacked lagged states, and verify the sine series and negative-delay scenarios
- [ ] 3.4 Accept a list of dictionaries, keep the one of smallest error on runs held out by `split_runs`, set
  `dictionary_errors`, and verify the polynomial choice and zero-validation scenarios
- [ ] 3.5 Document eigenfunctions, delays, and dictionary choice in the user guide with a runnable example, and extend
  `docs/examples/engineering/control/plot_koopman_model_predictive_control.py` or add a slow manifold example that the
  gallery build runs

- [ ] 3.6 Add `deeptime>=0.5.0` to the runtime dependencies in `pyproject.toml` and the Project Profile with its
  release date and licence, and verify `pdm install` resolves and it imports
- [ ] 3.7 Add the kinds `'tica'` and `'vamp'` to `embedding` with `lag`, the implied timescales, the singular values,
  and the VAMP-2 score on deeptime, and verify the two-relaxation-rates and VAMP-of-a-reversible-process scenarios in
  `tests/core/analysis/test_embedding.py`

## 4. Cluster expansion

- [ ] 4.1 Write the 2D one-particle and centre bases, the charge rule, and the pooled features with pure body orders in
  `src/skcomplex/core/analysis/_cluster_expansion.py`, and verify equivariance, permutation invariance, and vanishing
  of each body order on random environments in `tests/core/analysis/test_cluster_expansion.py`
- [ ] 4.2 Add the 3D heading-frame basis with perpendicular and axial outputs, and verify 3D equivariance on random
  environments
- [ ] 4.3 Read particle snapshots, form environments within `cutoff` and accelerations from velocities, fit with
  `BayesianRidge` and the smoothness prior, and verify the reproducibility and uncertainty scenarios
- [ ] 4.4 Build the result system with a 1-body rate law and a private pair interaction for body order 2, and verify
  the law and system agree within 1e-8 and the `bird_swarm` pairwise and free-run scenarios
- [ ] 4.5 Add the 3-body and 4-body interactions on the k-body neighbour roles of `add-many-body-neighbour-roles`, and
  verify the `many_body_alignment_swarm` scenario
- [ ] 4.6 Add the group-without-velocities error and the argument checks, export `cluster_expansion` and
  `ClusterExpansionResult` from `skcomplex.core.analysis`, declare its analysis tags, and verify the error scenario
- [ ] 4.7 Document `cluster_expansion` in the user guide and add a gallery example in
  `docs/examples/biology/collective_behaviour/` that learns the law of a swarm and plots the contribution of each body
  order, and verify the gallery build runs it

## 5. Online and offline activity

- [ ] 5.1 Add `OnlineOfflineActivityNetworkSystem` in `src/skcomplex/systems/social/spreading/_online_offline.py` with
  its stochastic and mean-field forms, export it from the subfield, and verify the balance and fully mixed scenarios
  in `tests/systems/social/spreading/`
- [ ] 5.2 Verify the weak-form scenario of five noisy mean-field runs in the same tests
- [ ] 5.3 Add `docs/examples/social/spreading/plot_online_offline_activity_network.py` with the stochastic runs, their
  mean, and the weak-form fit, and verify the gallery build runs it
- [ ] 5.4 Add the inventory entry `online_offline_activity_network` under `[social.spreading]` citing 2605.30432, add
  the primitive `learned-interaction-laws` with `change = "extend-law-discovery"`, and verify the inventory check passes

## 6. Checks

- [ ] 6.1 Run `nox` (formatting, ruff, mypy, pydoclint, import contracts, tests of the touched modules, doctests, the
  inventory check, and the strict docs build) and verify every session passes
