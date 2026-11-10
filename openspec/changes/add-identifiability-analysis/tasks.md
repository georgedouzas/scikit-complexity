## 1. Dependency

- [x] 1.1 Confirm `add-bayesian-calibration` has landed, so `_CalibrationObjective` and prior bounds exist in `skcomplex.core.experiments`, and verify its tests pass
- [x] 1.2 Add `pypesto>=0.8.0` to the runtime dependencies in `pyproject.toml` and to the Project Profile in `openspec/constitution.md`, and verify `pdm install` resolves and `import pypesto` works

## 2. Identifiability

- [x] 2.1 Add `identifiability` and `IdentifiabilityResult` in `src/skcomplex/core/experiments/_identifiability.py`, export them, and test the error for an unknown method
- [x] 2.2 Add `method='profile'` on pyPESTO with the chi-square interval and verdicts, and test the saturating and the early logistic data
- [x] 2.3 Add `method='refits'` over noisy copies with the relative errors and verdicts of Liyanage et al., and test the saturating logistic data and reproducibility with a seed
- [x] 2.4 Write the docstring of `identifiability` with a runnable example, and verify the doctest passes
- [x] 2.5 Add a user guide section on identifiability in `docs/overview/user_guide/run_experiments.md`, and verify its examples run

## 3. QSP system

- [ ] 3.1 Add `ImmunoOncologyQspSystem` in `biology/oncology/_immuno_oncology_qsp.py` from `CompartmentElement`, `ReactionInteraction`, and `DosingInteraction`, and test the treatment and the antibody kinetics scenarios
- [ ] 3.2 Add the inventory entry `immuno_oncology_qsp` with the source 2403.03335 and its primitives, and verify the inventory check passes

## 4. Virtual populations

- [ ] 4.1 Add `virtual_population` and `VirtualPopulationResult` in `src/skcomplex/core/experiments/_virtual_population.py` on `sweep`, export them, and test the plausible patients and the error when none is plausible
- [ ] 4.2 Add selection by the probability of inclusion with `target`, and test the Kolmogorov-Smirnov match on the QSP system
- [ ] 4.3 Write the docstring with a runnable example and a user guide paragraph, and verify both run

## 5. Per-individual calibration

- [ ] 5.1 Add `calibrate_individuals` and `IndividualCalibrationResult` in `src/skcomplex/core/experiments/_individuals.py`, export them, and test the digital twins of five virtual patients and the error for a known parameter that is also fitted
- [ ] 5.2 Write the docstring with a runnable example and a user guide paragraph, and verify both run
- [ ] 5.3 Add the gallery example `docs/examples/biology/oncology/plot_immuno_oncology_qsp.py` of `ImmunoOncologyQspSystem` with a virtual population, a profile of one patient, and the patient's calibration, and verify strict gallery coverage passes

## 6. Integration

- [ ] 6.1 Leave `CHANGELOG.md` to the release, which writes it from the commits, and verify the docs build
- [ ] 6.2 Run `nox` checks, tests, and the strict docs build, and fix every finding
