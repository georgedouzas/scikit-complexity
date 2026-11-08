# Tasks

## 1. Dependencies

- [ ] 1.1 Add `flowjax>=19.2.1` and, unless an earlier change of the wave added it, `equinox>=0.13.8` to the runtime
  dependencies in `pyproject.toml`, and verify `pdm install` resolves and both import

## 2. Independence proposals in Monte Carlo

- [ ] 2.1 Add `ProposalSpec` in `core/specs/_proposal_spec.py` with its `_check`, verified by
  `tests/core/specs/test_proposal_spec.py` and its doctest
- [ ] 2.2 Add `energy` to `SamplingInteraction` and `SampleLaw`, print it in the lattice pass, and record
  `<label>.energy`, verified by `tests/core/components/interactions/test_sampling.py` (equipartition of the free field)
- [ ] 2.3 Add the proposal path to `MonteCarloKernel` with the independence Metropolis rule and the recorded
  acceptance rate, and the validation errors for a missing energy or another rule, verified by tests of the exact
  proposal and of the wide Gaussian proposal
- [ ] 2.4 Document global proposals and site energies in the lattice systems guide and verify its code blocks run

## 3. Lattice φ⁴ system

- [ ] 3.1 Add `LatticePhi4System` in `systems/physics/statistical_mechanics/_phi4.py`, verified by
  `tests/systems/physics/statistical_mechanics/test_phi4.py` (free propagator, broken symmetry) and its doctest
- [ ] 3.2 Register `lattice_phi4`, add its inventory entry and the `generative-samplers` primitive, and verify
  `tests/systems/test_catalog.py` and `inventory_problems()` report no problem

## 4. Generative sampler

- [ ] 4.1 Add the private flow module with the coupling and spline flows, reverse and forward training, and the JAX
  energy over the neighbour table, verified by unit tests on a Gaussian target with a known normaliser
- [ ] 4.2 Add the flow-matching sampler on equinox and diffrax with exact divergences, verified by a test that its log
  densities of a Gaussian target match the exact ones within 1e-6
- [ ] 4.3 Add `generative_sampler` and `GenerativeSamplerResult` with weights, effective sample size, free energy,
  estimates, the learned system, and the errors for a missing energy, discrete states, and reverse flow matching,
  verified by `tests/core/analysis/test_generative_sampler.py` and the doctest
- [ ] 4.4 Add the probe tests of the free field (effective sample size, reweighted `φ²`, free energy), of flow matching
  on chain samples, and of flow proposals in a run of the interacting field, marking the 8 × 8 ones as slow
- [ ] 4.5 Export the function from `skcomplex.core.analysis`, add a section on generative samplers to
  `docs/overview/user_guide/analyse.md`, and verify its code blocks run

## 5. Gallery

- [ ] 5.1 Add the gallery example of `lattice_phi4` comparing the autocorrelation of local and flow proposals, verified
  by `gallery_tools coverage --strict` and a run of the example

## 6. Checks

- [ ] 6.1 Run `nox -s checks` and `nox -s tests` with the thread limits set to 1, and verify they pass
