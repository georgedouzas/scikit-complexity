# Tasks

## 1. Analysis base and entropy pilot

- [ ] 1.1 Add `BaseAnalysis` to `analysis/_base.py`.
  - It defines `check`, `prepare`, `run`, the hooks `_check`, `_prepare`, `_run`, and `_get_data_kind`.
  - It holds the private readers `_read_series`, `_read_path`, `_read_fields`, `_read_snapshots`, `_read_networks`, moved from `_adapters.py`, and the private record values.
  - Verify with the doctest of `BaseAnalysis` and `tests/analysis/test_base.py`, which covers:
    - reading a trajectory, an array, and a graph;
    - the missing-kind error;
    - run before prepare;
    - cloning.
- [ ] 1.2 Write `EntropyAnalysis` in `analysis/_entropy.py` with the choices listed in the design, and delete the entropy functions and their result classes.
  - Verify with `tests/analysis/test_entropy.py`, which covers:
    - every kind against the values of the deleted functions;
    - the unknown kind;
    - a setting the kind ignores.
- [ ] 1.3 Convert every caller of the deleted entropy functions in `src`, `tests`, and `docs`, and verify that `pytest tests/analysis tests/models` and the model doctests pass.
- [ ] 1.4 Stop for the user's review of the pilot before converting further families.

## 2. Remaining analysis families

- [ ] 2.1 Convert the information families: `InformationAnalysis`, `LocalInformationAnalysis`, `PartialInformationAnalysis`, `EmergenceAnalysis`, and `EpsilonMachineAnalysis`. Verify that each has a test module mirroring its source and reproduces the deleted functions' values.
- [ ] 2.2 Convert the series and scaling families: `CorrelationAnalysis`, `SpectrumAnalysis`, `ScalingAnalysis`, `CollapseAnalysis`, `RecurrenceAnalysis`, `EmbeddingAnalysis`, `EarlyWarningAnalysis`, and `EventAnalysis`. Verify with their test modules.
- [ ] 2.3 Convert the fit and criticality families: `FitAnalysis`, `AvalancheAnalysis`, `ClusterAnalysis`, and `BinderAnalysis`.
  - Map every method of `PowerLawFit` and the other fit result classes to a setting or an attribute.
  - Verify with their test modules.
- [ ] 2.4 Convert the field, agent, and network families: `DomainAnalysis`, `OrderAnalysis`, `DisplacementAnalysis`, `NeighborAnalysis`, `FluctuationAnalysis`, `NetworkAnalysis`, and `MotifAnalysis`. Verify with their test modules.
- [ ] 2.5 Convert the stochastic families: `MarkovAnalysis`, `PassageAnalysis`, `IrreversibilityAnalysis`, `LargeDeviationAnalysis`, `ErgodicityAnalysis`, and `KramersMoyalAnalysis`. Verify with their test modules.
- [ ] 2.6 Convert the families that take a system as their source: `StabilityAnalysis`, `LinearizationAnalysis`, `LyapunovAnalysis`, `SectionAnalysis`, `LoopAnalysis`, `SymbolicAnalysis`, `AutomatonAnalysis`, and `QuantumAnalysis`. Verify with their test modules, including the Lotka-Volterra stability scenario.
- [ ] 2.7 Convert the surrogates and the remaining analyses:
  - `SparseIdentificationAnalysis`, `ModeDecompositionAnalysis`, `ReservoirAnalysis`, `SurrogateTestAnalysis`, and `SignatureAnalysis`;
  - discovery over analysis classes.
  - Delete `BaseSurrogate`, the public adapters, and `DataKind`'s registry.
  - Verify that `skcomplex.analysis.__all__` holds only `BaseAnalysis` and classes ending in `Analysis`.
- [ ] 2.8 Remove every import of `skcomplex.experiments` from `src/skcomplex/analysis`. Update the gallery tools and the gallery examples that plot analysis results. Verify that `pytest tests/analysis tests/models docs` and the analysis and model doctests pass.

## 3. Experiments

- [ ] 3.1 Add `BaseDesign` and `BaseExperiment` to `experiments/_base.py`, and move the designs onto `BaseDesign` with `prepare(bounds).run()` returning rows. Verify with `tests/experiments/test_designs.py`, including the Sobol design scenario.
- [ ] 3.2 Make `Experiment` take `analyses`, clone and run them per run into `result_`, and give `ExperimentResult` a named method for the ensemble of a series.
  - Verify with `tests/experiments/test_experiment.py`, which covers:
    - the SIR analyses scenario;
    - the Ornstein-Uhlenbeck ensemble scenario;
    - serial and process runs giving identical results.
- [ ] 3.3 Replace `calibrate` with `CalibrationExperiment` and `sensitivity` with `SensitivityAnalysis` on an experiment result.
  - Verify with:
    - the logistic calibration scenario;
    - the Ishigami Sobol scenario.
- [ ] 3.4 Move `basin_stability` to `BasinStabilityExperiment`, and make `Task`, `run_task`, `resolve_source`, `RunResult`, and `replicate` private.
  - Verify with:
    - the double-well scenario;
    - a test that `skcomplex.experiments.__all__` holds no function.
- [ ] 3.5 Update the docs and examples that call experiments, and verify that `pytest tests/experiments docs` and the experiments doctests pass.

## 4. Engine surface

- [ ] 4.1 For `engine.compile`, list every re-exported name with its importers outside the package. Make the names used only inside the package private, and turn the single-class helpers into private methods. Verify that `pytest tests/engine/compile` passes and the surface test names no helper.
- [ ] 4.2 Do the same for `engine.problems`. Verify with `pytest tests/engine/problems`.
- [ ] 4.3 Do the same for `engine.solvers`, `engine.backends`, `engine.runtime`, and `engine.viz`. Verify with `pytest tests/engine`.

## 5. Layering and integration

- [ ] 5.1 Set the import-linter layers to `experiments`, `analysis`, `engine`, `core`, and verify that `lint-imports` passes and fails on a planted import of `experiments` from `analysis`.
- [ ] 5.2 Update the user guide and the README to the classes, and verify that their code blocks run in the test session.
- [ ] 5.3 Run the full test session with doctests, the checks session, and the gallery build, and verify that all pass.
