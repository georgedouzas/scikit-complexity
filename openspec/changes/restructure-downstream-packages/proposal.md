# Proposal

## Why

The core now has one shape:
- a few base classes with the verbs `check()`, `prepare(context)`, and `run(inputs)`;
- constructors that only store their arguments;
- derived state in attributes ending in `_`.

The packages that work on a system do not follow it:
- `analysis` exports 198 functions and 76 result classes, each with its own input and output conventions;
- `engine.compile`, `engine.problems`, and `engine.solvers` export about 200 names, most of them helpers of one pass or one solver;
- `experiments` mixes free functions (`calibrate`, `sensitivity`, `run_task`, `resolve_source`) with estimator classes;
- `analysis` imports `experiments`, so the layering has a cycle in meaning.

A modeller has to learn many conventions instead of one. This change gives `engine`, `analysis`, and `experiments` the same shape as `core`.

## What Changes

The whole package becomes one pipeline, `System → Problem → Trajectory → measures`, plus runs of many systems.

### `engine`
- **BREAKING** The public surface shrinks. It keeps:
  - `Simulation` and `Trajectory`;
  - `compile_system`;
  - the concrete solvers, backends, observers, and clocks;
  - the bases a user subclasses: `BasePass`, `BaseSolver`, `BaseBackend`, `BaseObserver`, `BaseClock`.
- **BREAKING** Every helper used by one pass or one solver becomes a private method of that class. Every helper shared by several becomes a private function of a `_utils` module. Neither is re-exported.
- `Problem` and the compiled records stay frozen values with only `check()`.

### `analysis`
- **BREAKING** One base, `BaseAnalysis`, replaces the 198 analysis functions and the 76 result classes.
  - An analysis stores its settings in `__init__`, validates them in `check()`, reads its input in `prepare(source)`, and computes in `run()`.
  - Its results are fitted attributes ending in `_`, as in scikit-learn. For example, `TransferEntropyAnalysis(lag=1).check().prepare(trajectory).run()` sets `transfer_entropy_`.
- **BREAKING** Analyses that differ only by a choice merge into one class with a string argument, as `MoveSpec('flip')` does in core. For example:
  - `EntropyAnalysis(kind='shannon')`;
  - `InformationAnalysis(kind='transfer')`;
  - `FitAnalysis(law='power')`.
  About 270 public names become a few dozen classes.
- **BREAKING** The adapters stop being public. These become private methods of `BaseAnalysis`, which every analysis uses in `prepare`:
  - `series_of`, `path_of`, `fields_of`, `snapshots_of`, `networks_of`;
  - `data_kinds`, `analyses_for`, `analysis`.
- **BREAKING** The analyses that run a system, rather than read data, take the system as their source:
  - `StabilityAnalysis`;
  - `LyapunovAnalysis`;
  - `LinearizationAnalysis`.

### `experiments`
- **BREAKING** Two bases, `BaseDesign` and `BaseExperiment`, with the three verbs:
  - **Designs:** `GridDesign`, `RandomDesign`, `LatinHypercubeDesign`, `SobolDesign`, and `RowsDesign` prepare against parameter bounds and run to rows of values.
  - **Experiments:**
    - `Experiment(source, design, scenarios, replicates, analyses)` runs many systems and applies analyses to each run.
    - `CalibrationExperiment(source, data, bounds)` fits parameters to data.
- **BREAKING** `sensitivity` becomes `SensitivityAnalysis`, which reads an experiment result.
- **BREAKING** `basin_stability` and `ensemble_of` move from `analysis` to `experiments`, because they run or read many systems:
  - `basin_stability` becomes `BasinStabilityExperiment`;
  - `ensemble_of` becomes a named method of the experiment result.
- **BREAKING** `Task`, `run_task`, `resolve_source`, `RunResult`, `Scenario` internals, and `replicate` become private.
  - `Scenario` stays public, as `ScenarioSpec` in `skcomplex.specs`.
  - `ExperimentResult` stays public with named methods.

### Layering
- **BREAKING** Imports run `experiments → analysis → engine → core`, and import-linter enforces it. `analysis` never imports `experiments`.

### Order of work
- Pilot the entropy family first so the style can be judged.
- Then convert the rest of `analysis`, then `experiments`, then the engine surface.

## Capabilities

### New Capabilities
- `analysis-interface`: the contract of every analysis.
  - The base `BaseAnalysis` and its three verbs.
  - The sources an analysis reads (a trajectory, an array, a graph, or a system).
  - Results as fitted attributes.
  - Families that are merged under one class with a choice argument.

### Modified Capabilities
- `analysis-adapters`: the adapters become private readers of `BaseAnalysis`.
  - The ensemble adapter moves to experiment results.
  - Discovery lists analysis classes instead of functions.
- `experiments`:
  - Designs and experiments become estimators with the three verbs.
  - Sensitivity becomes an analysis of an experiment result.
  - Calibration becomes an experiment.
  - Basin stability moves in from `analysis`.
- `package-layout`:
  - The dependency order becomes `experiments → analysis → engine → core`.
  - The top-level subpackages are `core`, `engine`, `analysis`, `experiments`, and `models`.
  - The engine surface is limited to its pipeline classes.
- `base-interfaces`: analyses and experiments join the configurable bases that share the three verbs.

## Impact

- **Code:**
  - `src/skcomplex/analysis` (54 modules), `src/skcomplex/experiments` (10 modules), and the `__init__` of every `src/skcomplex/engine` subpackage.
  - The engine modules whose helpers become private methods.
- **Tests:**
  - `tests/analysis`, `tests/experiments`, and `tests/engine`, rewritten for the classes.
  - The test tree keeps mirroring `src`.
- **Docs:** the gallery examples and their analysis figures, the user guide, and the README, all of which call analysis functions today.
- **Models:** model modules that call analysis or experiment functions, mainly in `docs` and in model doctests.
- **Configuration:** the import-linter contracts in `pyproject.toml`.
- **Dependencies:** no new dependencies. The adapters keep using the libraries they use today (`antropy`, `powerlaw`, `deeptime`, `SALib`, `networkx`).
- **Compatibility:** no legacy aliases are kept, because breaking changes are allowed.
