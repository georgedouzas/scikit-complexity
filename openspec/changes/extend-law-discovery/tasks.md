# Tasks

## 1. Sparse identification options

- [ ] 1.1 Add `weak`, `subdomains`, and `test_degree` to `sparse_identification` with `pysindy.WeakPDELibrary`, and
  verify the noisy and noise-free Lorenz scenarios in `tests/core/analysis/test_sparse_identification.py`
- [ ] 1.2 Add `ensemble`, `drop_terms`, and `random_state` with `pysindy.EnsembleOptimizer`, set `inclusion` and the
  median coefficients, and verify the inclusion, reproducibility, and invalid-size scenarios
- [ ] 1.3 Add `parameters` and `parameter_degree` with `pysindy.ParameterizedLibrary`, declare the parameters on the
  surrogate element, and verify the Lorenz over `rho` and wrong-length scenarios
- [ ] 1.4 Move `_input_declarations` into `_sparse_identification.py`, add `inputs` to `sparse_identification`, and
  verify the forced oscillator and input schedule scenarios, with the mode decomposition tests still passing
- [ ] 1.5 Read `ExperimentResult` sources in `_read_samples` with `replicates='mean'` or `'each'`, and verify the
  ensemble mean and unknown-option scenarios
- [ ] 1.6 Update the docstring examples of `sparse_identification` and `SparseIdentificationResult`, the learned
  surrogates section of `docs/overview/user_guide/analyse.md`, and the changelog entry for the new fields, and verify
  the doctests run

## 2. Mode decomposition extensions

- [ ] 2.1 Add `eigenfunctions`, `eigenfunction_errors`, `valid`, and `eigenfunction_tolerance` from the left
  eigenvectors, and verify the slow manifold and Duffing scenarios in `tests/core/analysis/test_mode_decomposition.py`
- [ ] 2.2 Fit the readout of the variables, declare one observable per variable on the lifted system, and verify the
  lifted run records `x` and `y`
- [ ] 2.3 Add `delays` with stacked lagged states, and verify the sine series and negative-delay scenarios
- [ ] 2.4 Accept a list of dictionaries, keep the one of smallest validation error, set `dictionary_errors`, and verify
  the polynomial choice and zero-validation scenarios
- [ ] 2.5 Document eigenfunctions, delays, and dictionary choice in the user guide with a runnable example, and extend
  `docs/examples/engineering/control/plot_koopman_model_predictive_control.py` or add a slow manifold example that the
  gallery build runs

## 3. Cluster expansion

- [ ] 3.1 Write the 2D one-particle and centre bases, the charge rule, and the pooled features with pure body orders in
  `src/skcomplex/core/analysis/_cluster_expansion.py`, and verify equivariance, permutation invariance, and vanishing
  of each body order on random environments in `tests/core/analysis/test_cluster_expansion.py`
- [ ] 3.2 Add the 3D heading-frame basis with perpendicular and axial outputs, and verify 3D equivariance on random
  environments
- [ ] 3.3 Read particle snapshots, form environments within `cutoff` and accelerations from velocities, fit with
  `BayesianRidge` and the smoothness prior, and verify the reproducibility and uncertainty scenarios
- [ ] 3.4 Build the result system with a 1-body rate law and a private pair interaction for body order 2, and verify
  the law and system agree within 1e-8 and the `bird_swarm` pairwise and free-run scenarios
- [ ] 3.5 Add the 3-body and 4-body interactions on the k-body neighbour roles of `add-many-body-neighbour-roles`, and
  verify the `many_body_alignment_swarm` scenario
- [ ] 3.6 Add the group-without-velocities error and the argument checks, export `cluster_expansion` and
  `ClusterExpansionResult` from `skcomplex.core.analysis`, declare its analysis tags, and verify the error scenario
- [ ] 3.7 Document `cluster_expansion` in the user guide and add a gallery example in
  `docs/examples/biology/collective_behaviour/` that learns the law of a swarm and plots the contribution of each body
  order, and verify the gallery build runs it

## 4. Probes and inventory

- [ ] 4.1 Verify the pooled noisy `sir` scenario and the `maki_thompson` replicate-mean scenario in
  `tests/core/analysis/test_sparse_identification.py`
- [ ] 4.2 Add a gallery example in `docs/examples/social/spreading/` beside the `maki_thompson` example that fits the
  weak form to the means of stochastic replicates from several initial states, and verify the gallery build runs it
- [ ] 4.3 Add the primitive `learned-interaction-laws` with `change = "extend-law-discovery"` to the inventory, and
  verify the inventory check passes

## 5. Checks

- [ ] 5.1 Run `nox` (formatting, ruff, mypy, pydoclint, import contracts, tests of the touched modules, doctests, the
  inventory check, and the strict docs build) and verify every session passes
