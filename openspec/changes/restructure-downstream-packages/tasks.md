# Tasks

## 1. Analysis base and entropy pilot

- [x] 1.1 Add `BaseAnalysis` to `analysis/_base.py`.
  - It defines `check`, `prepare`, `run`, the hooks `_check`, `_prepare`, `_run`, and `_get_data_kind`.
  - It holds the private readers `_read_series`, `_read_path`, `_read_fields`, `_read_snapshots`, `_read_networks`, moved from `_adapters.py`, and the private record values.
  - Verify with the doctest of `BaseAnalysis` and `tests/analysis/test_base.py`, which covers:
    - reading a trajectory, an array, and a graph;
    - the missing-kind error;
    - run before prepare;
    - cloning.
- [x] 1.2 Write `EntropyAnalysis` in `analysis/_entropy.py` with the choices listed in the design, and delete the entropy functions and their result classes.
  - Verify with `tests/analysis/test_entropy.py`, which covers:
    - every kind against the values of the deleted functions;
    - the unknown kind;
    - a setting the kind ignores.
- [x] 1.3 Convert every caller of the deleted entropy functions in `src`, `tests`, and `docs`, and verify that `pytest tests/analysis tests/models` and the model doctests pass.
- [x] 1.4 Stop for the user's review of the pilot before converting further families. The user waived the stop and asked to run to the end.

## 2. Remaining analysis families

- [x] 2.1 Convert the information families: `InformationAnalysis`, `LocalInformationAnalysis`, `PartialInformationAnalysis`, `EmergenceAnalysis`, and `EpsilonMachineAnalysis`. Verify that each has a test module mirroring its source and reproduces the deleted functions' values.
- [x] 2.2 Convert the series and scaling families: `CorrelationAnalysis`, `SpectrumAnalysis`, `ScalingAnalysis`, `CollapseAnalysis`, `RecurrenceAnalysis`, `EmbeddingAnalysis`, `EarlyWarningAnalysis`, and `EventAnalysis`. Verify with their test modules.
- [x] 2.3 Convert the fit and criticality families: `FitAnalysis`, `AvalancheAnalysis`, `ClusterAnalysis`, and `BinderAnalysis`.
  - Map every method of `PowerLawFit` and the other fit result classes to a setting or an attribute.
  - Verify with their test modules.
- [x] 2.4 Convert the field, agent, and network families: `DomainAnalysis`, `OrderAnalysis`, `DisplacementAnalysis`, `NeighborAnalysis`, `FluctuationAnalysis`, `NetworkAnalysis`, and `MotifAnalysis`. Verify with their test modules.
- [x] 2.5 Convert the stochastic families: `MarkovAnalysis`, `PassageAnalysis`, `IrreversibilityAnalysis`, `LargeDeviationAnalysis`, `ErgodicityAnalysis`, and `KramersMoyalAnalysis`. Verify with their test modules.
- [x] 2.6 Convert the families that take a system as their source: `StabilityAnalysis`, `LinearizationAnalysis`, `LyapunovAnalysis`, `SectionAnalysis`, `LoopAnalysis`, `SymbolicAnalysis`, `AutomatonAnalysis`, and `QuantumAnalysis`. Verify with their test modules, including the Lotka-Volterra stability scenario.
- [x] 2.7 Convert the surrogates and the remaining analyses:
  - `SparseIdentificationAnalysis`, `ModeDecompositionAnalysis`, `ReservoirAnalysis`, `SurrogateTestAnalysis`, and `SignatureAnalysis`;
  - discovery over analysis classes.
  - Delete `BaseSurrogate`, the public adapters, and `DataKind`'s registry.
  - Verify that `skcomplex.analysis.__all__` holds only `BaseAnalysis` and classes ending in `Analysis`.
- [x] 2.8 Remove every import of `skcomplex.experiments` from `src/skcomplex/analysis`. Update the gallery tools and the gallery examples that plot analysis results. Verify that `pytest tests/analysis tests/models docs` and the analysis and model doctests pass.

## 3. Experiments

- [x] 3.1 Add `BaseDesign` and `BaseExperiment` to `experiments/_base.py`, and move the designs onto `BaseDesign` with `prepare(bounds).run()` returning rows. Verify with `tests/experiments/test_designs.py`, including the Sobol design scenario.
- [x] 3.2 Make `Experiment` take `analyses`, clone and run them per run into `result_`, and give `ExperimentResult` a named method for the ensemble of a series.
  - Verify with `tests/experiments/test_experiment.py`, which covers:
    - the SIR analyses scenario;
    - the Ornstein-Uhlenbeck ensemble scenario;
    - serial and process runs giving identical results.
- [x] 3.3 Replace `calibrate` with `CalibrationExperiment` and `sensitivity` with `SensitivityAnalysis` on an experiment result.
  - Verify with:
    - the logistic calibration scenario;
    - the Ishigami Sobol scenario.
- [x] 3.4 Move `basin_stability` to `BasinStabilityExperiment`, and make `Task`, `run_task`, `resolve_source`, `RunResult`, and `replicate` private.
  - Verify with:
    - the double-well scenario;
    - a test that `skcomplex.experiments.__all__` holds no function.
- [x] 3.5 Update the docs and examples that call experiments, and verify that `pytest tests/experiments docs` and the experiments doctests pass.

## 4. Engine surface

- [x] 4.1 For `engine.compile`, list every re-exported name with its importers outside the package. Make the names used only inside the package private, and turn the single-class helpers into private methods. Verify that `pytest tests/engine/compile` passes and the surface test names no helper.
- [x] 4.2 Do the same for `engine.problems`. Verify with `pytest tests/engine/problems`.
- [x] 4.3 Do the same for `engine.solvers`, `engine.backends`, `engine.runtime`, and `engine.viz`. Verify with `pytest tests/engine`.
- [x] 4.4 In every engine package, make each module-level function that classes use a private method of the class it belongs to, or of the package's base when several classes share it. Keep module-level only the Numba-jitted kernels and the functions that generated kernel source imports by name. Verify that `pytest tests/engine` and the engine doctests pass.
- [x] 4.5 Split every engine module that holds several classes into one module per class, keeping `_base.py` for bases only. Verify with `pytest tests/engine` and `lint-imports`.

## 5. General operations and field realizations

- [x] 5.1 Regroup the analysis classes named after a field into the general operations of the design's table: add `AlignmentAnalysis` and `RollingAnalysis`, add the kinds moments, moment-ratio, dispersion-index, and total-variation to `DistributionAnalysis`, overlap to `CorrelationAnalysis`, variance-mean and power to `ScalingAnalysis`, crossings to `CollapseAnalysis`, excursions and branching to `EventAnalysis`, and dispersion to `StabilityAnalysis`, and delete `AvalancheAnalysis`, `BinderAnalysis`, `FluctuationAnalysis`, `OrderAnalysis`, `MixingAnalysis`, and `EarlyWarningAnalysis`. Make each general class read every data kind its operation is defined on. Verify with the test modules of the changed classes, including the alignment scenario on three domains and the moment-ratio scenario.
- [x] 5.2 Generalise `LoopAnalysis` to read feedback loops from the signed dependency graph of any system, and verify it on a stock-and-flow model and on an ODE model of another field.
- [x] 5.3 Add the field realizations of the design's table to the models library, each a subclass that only fixes settings, and convert the models, tests, gallery examples, and user guide that used the deleted classes. Verify with a test per realization that it equals its general analysis.
- [x] 5.4 Audit the public classes and choices of the core, the engine, and experiments for names taken from one application field, list them with a general name and a realization, and convert them the same way. Verify that the surface tests of every package pass.
- [x] 5.5 Place every realization of every field by its users: a realization used by one model package lives in that package beside its models, one shared by several packages of a field lives in a family package of the field's `common` package, and fields with nothing shared have no `common` package. Delete field-measure classes that only fix a choice of a general analysis, and move specialized code with logic of its own instead of removing it. Verify with lint-imports, the full test session, and the user guide.

## 7. Engine design

- [ ] 7.1 List the dead code of the engine: classes, methods, functions, constants, and attributes that nothing in `src`, `tests`, or `docs` reaches, found with `vulture` and with the coverage of the full test session, and delete it. Verify with the full test session and the gallery build.
- [x] 7.2 Write the engine's concept model in the design: what the engine is (system, compiled problem, trajectory), its bases (compiler and passes, problem values, solvers, backends, observers and visuals, simulation), the class tree, and a verdict for every engine class and dataclass: a class of the tree, a frozen value, merged into another class, or removed.
- [ ] 7.3 Stop for the user's review of the concept model.
- [ ] 7.4 Restructure the engine to the reviewed class tree: every class under one base of its package, dataclasses only for frozen values, and no public function outside the pipeline the design keeps. Verify with the full test session, the user guide, lint-imports, and the gallery build.

## 6. Layering and integration

- [x] 6.1 Set the import-linter layers to `experiments`, `analysis`, `engine`, `core`, and verify that `lint-imports` passes and fails on a planted import of `experiments` from `analysis`.
- [x] 6.2 Update the user guide and the README to the classes, and verify that their code blocks run in the test session.
- [ ] 6.3 Run the full test session with doctests, the checks session, and the gallery build, and verify that all pass.
