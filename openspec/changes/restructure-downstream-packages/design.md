# Design

## Context

- `core` already has the target shape: `BasePart` in `core/base.py` defines `check()`, `prepare(context)`, and `run(inputs)` once and calls the hooks `_check`, `_prepare`, and `_run`. Its subclasses only store arguments in `__init__`.
- `engine` is close to that shape. `BasePass`, `BaseSolver`, `BaseBackend`, `BaseObserver`, `BaseClock`, and `Simulation` already use the three verbs. Its problem is the surface: `engine.compile` re-exports about 90 names, `engine.problems` about 60, and `engine.solvers` about 50. Most of them are helpers of one pass or one solver.
- `analysis` is 54 modules and about 15,600 lines.
  - Every analysis is a function that returns its own frozen result class.
  - The adapters in `_adapters.py` turn a trajectory into the data an analysis reads.
  - `_kinds.py` holds `DataKind` and a `Registry` of analyses per kind.
  - `_bifurcation.py`, `_adapters.py`, and `_signatures.py` import `experiments`.
- `experiments` is 10 modules.
  - `Experiment`, `Scenario`, and the designs are already estimators.
  - `calibrate`, `sensitivity`, `run_task`, and `resolve_source` are functions.
- The constitution and the user's rules bind the design:
  - one class per module, with `_base.py` holding only bases;
  - no base derives from another base;
  - no new class variables;
  - no factories, no second way of doing one thing, no `typing.Any`;
  - helpers become private methods of the class that uses them;
  - mature libraries are wrapped rather than reimplemented;
  - tests and docs are clients, so a name they reach stays public.

## Goals / Non-Goals

**Goals:**
- One contract for every analysis, as specified in `specs/analysis-interface`.
- Analyses and experiments read as scikit-learn estimators.
- A small engine surface whose names a modeller recognises as the pipeline.
- A downward import order that import-linter checks.

**Non-Goals:**
- No change to the numerics of any analysis, solver, or pass. Every converted analysis returns the values its function returned.
- No change to `core`.
- No new analyses, and no removal of an analysis a model, test, or gallery example uses.
- No change to how models are written.

## Decisions

### `BaseAnalysis` is its own base, not a `BasePart`
`BasePart` carries a label and `_get_laws`, which an analysis does not have. `BaseAnalysis` in `analysis/_base.py` derives from `BaseEstimator` and `ABC` only. It defines:
- `check()`, which validates the settings and marks the analysis as checked;
- `prepare(source)`, which reads the source, sets the data attributes, and returns the analysis;
- `run(inputs=None)`, which computes and returns the analysis.

Each verb calls the hooks `_check`, `_prepare`, and `_run`.

*Alternative considered:* a function-style `fit(X)`. The user rejected a fourth verb, so `prepare` is the reading step and `run` the computing step.

### The data kind is a hook, not a class variable
Each analysis returns its kind from `_get_data_kind()`, as core spaces return what they lack from `_get_lacks()`. Discovery builds every analysis with its default settings and asks it.

*Alternative considered:* a `ClassVar` `kind`. The user forbade new class variables.

### Readers are private methods of `BaseAnalysis`
The public adapters become private methods of the base: `series_of`, `path_of`, `fields_of`, `snapshots_of`, and `networks_of` turn into `_read_series`, `_read_path`, `_read_fields`, `_read_snapshots`, and `_read_networks`.
- A subclass calls the one it needs in `_prepare`, with the variable names from its settings.
- An array or graph passes through after a shape check.
- The record types `Series`, `JumpPath`, and `Snapshots` stay private values of `_base.py`, because a frozen value is not a helper.

*Alternative considered:* a reader object per kind that an analysis holds. It adds classes that only one base uses.

### Results are fitted attributes
Every field of today's result class becomes an attribute of the analysis, named after the field, ending in `_`. Plotting code in `gallery_tools` and `viz` reads those attributes. Result classes with methods (`PowerLawFit.compare`, `MarkovStateModel.timescales`) lose the methods. Each method becomes either a setting of the analysis (`compare_with='lognormal'`) or a fitted attribute computed in `run`.

### Families merge under one class with a choice argument
A family is merged only when its members share their data kind, their settings, and the name of their main result. The choice argument is `kind`, or `law` or `method` where the literature uses that word. `check()` validates it against a module constant tuple.

Target grouping (the entropy pilot validates the style, and later families may split):

| Class | Choices | Replaces |
|---|---|---|
| `EntropyAnalysis` | shannon, renyi, tsallis, permutation, sample, approximate, spectral, multiscale, block, rate, excess, lempel-ziv | the entropy functions of `_complexity`, `_series`, `_information` |
| `InformationAnalysis` | mutual, auto-mutual, transfer, active-storage, total-correlation, dual-total-correlation, o-information, kl-divergence | `_information`, `_multivariate` |
| `LocalInformationAnalysis` | mutual, transfer, entropy | `_local_information` |
| `PartialInformationAnalysis` | | `partial_information` |
| `EmergenceAnalysis` | effective-information, causal, criteria | `_emergence`, `_emergence_criteria` |
| `EpsilonMachineAnalysis` | | `_computational_mechanics` |
| `FitAnalysis` | power, stable, extreme-value, avalanche, order-exponent | `_statistics`, `_extremes`, `_criticality` fits |
| `CorrelationAnalysis` | auto, two-time, function, length | `_series`, `_aging`, `_spatial` |
| `SpectrumAnalysis` | power, structure-factor, dispersion | `_series`, `_spatial`, `_fields` |
| `ScalingAnalysis` | hurst, detrended-fluctuation, box-counting, correlation-dimension | `_series`, `_scaling` |
| `CollapseAnalysis` | scaling, aging, distribution | `_scaling`, `_aging` |
| `RecurrenceAnalysis` | | `_recurrence` |
| `EmbeddingAnalysis` | delay, false-nearest-neighbors | `_embedding` |
| `EarlyWarningAnalysis` | | `early_warning` |
| `AvalancheAnalysis` | | `avalanches`, `branching_ratio`, `crackling_noise` |
| `ClusterAnalysis` | sizes, percolation, spanning | `_percolation` |
| `BinderAnalysis` | cumulant, crossings | `_criticality` |
| `DomainAnalysis` | size, growth, wavelength, interface-density | `_fields` |
| `OrderAnalysis` | polar, milling, nematic | `_agents` |
| `DisplacementAnalysis` | | `mean_squared_displacement` |
| `NeighborAnalysis` | distances, clark-evans, radial-distribution | `_agents`, `_spatial` |
| `FluctuationAnalysis` | number, relative, fano, kinetic-temperature, maxwell-boltzmann | `_fluctuations` |
| `NetworkAnalysis` | degrees, assortativity, centrality, communities, neighbor-degree, summary | `_networks` |
| `MotifAnalysis` | counts, z-scores | `_motifs` |
| `MarkovAnalysis` | transition-matrix, stationary, state-model, timescales, spectral-gap | `_markov` |
| `PassageAnalysis` | first-passage, mean-first-passage, hitting, committor, return-times, records | `_passage` |
| `IrreversibilityAnalysis` | entropy-production, detailed-balance, fluctuation-theorem, fluctuation-dissipation | `_irreversibility` |
| `LargeDeviationAnalysis` | tilted, cloning, histogram, block | `_large_deviations` |
| `ErgodicityAnalysis` | breaking, distance, invariant-density, ulam | `_ergodicity`, `_mixing` |
| `EventAnalysis` | detect, counts, laminar-durations | `_events` |
| `KramersMoyalAnalysis` | | `kramers_moyal` |
| `StabilityAnalysis` | | `fixed_points`, `stability`, `jacobian`, `nullclines` |
| `LinearizationAnalysis` | | `linearize`, `controllability`, `observability`, `lqr` |
| `LyapunovAnalysis` | spectrum, largest, kaplan-yorke | `_lyapunov` |
| `SectionAnalysis` | poincare, stroboscopic, orbit-diagram | `_sections`, `_bifurcation` |
| `LoopAnalysis` | loops, dominance | `_loops` |
| `SymbolicAnalysis` | equations, solve, hamiltonian, conservation | `_symbolic`, `_invariants` |
| `AutomatonAnalysis` | injective, pre-injective, surjective, garden-of-eden, inverse, compose | `_automata` |
| `QuantumAnalysis` | entanglement, purity, spectrum, ground-state | `_quantum` |
| `SparseIdentificationAnalysis`, `ModeDecompositionAnalysis`, `ReservoirAnalysis` | | the three surrogates, each with the learned system in `system_` |
| `SurrogateTestAnalysis` | | `surrogate_test` |
| `SignatureAnalysis` | | `signature_analysis` |

### Surrogates become analyses
A surrogate learns a system from data. As an analysis it prepares on a trajectory or an array and keeps the learned system in `system_`. This drops the separate `BaseSurrogate` and keeps `Surrogate` out of the name set the spec allows.

### Experiments have two bases
`experiments/_base.py` holds `BaseDesign` and `BaseExperiment`, each on `BaseEstimator`.
- **`Experiment`:**
  - Takes `analyses` as a list of unprepared analyses.
  - For each run, clones every analysis, prepares the clone on the run's trajectory, and runs it.
  - Copies the fitted attributes into the run's row.
- **`CalibrationExperiment`:** wraps `scipy.optimize`, as `calibrate` does today.
- **`BasinStabilityExperiment`:** wraps `Experiment` with a random design over the initial states and a classifier setting.
- **`SensitivityAnalysis`:** a `BaseAnalysis` in `experiments`. It prepares on an `ExperimentResult` and wraps SALib as `sensitivity` does today.

### The engine surface is cut by client use
For every name an engine package re-exports today, the conversion lists the modules that import it outside the package.
- A name used by `core`, another engine package, `analysis`, `experiments`, `models`, the docs, or a test of the public behaviour stays public.
- A name used only inside its package becomes private.
- A helper used by one class becomes a private method of that class.

Tests of a helper are rewritten to test the owning class through its verbs.

### Layering is checked by import-linter
The `layers` contract in `pyproject.toml` becomes `experiments`, `analysis`, `engine`, `core`, with `models` allowed to import all of them.

## Risks / Trade-offs

- **[Risk]** The change touches about 280 public names, and the gallery, the docs, and the model doctests call many of them. → Mitigation: convert in the order the proposal gives, one family per commit, and run the full test session and the gallery build after each package.
- **[Risk]** A merged family can hide settings that only one member uses, so `check()` must reject a setting that the chosen kind ignores. → Mitigation: each family class validates its settings per kind in `_check`, and its tests cover one ignored setting.
- **[Risk]** Results that were methods of result classes, such as goodness-of-fit comparisons, can lose behaviour. → Mitigation: the conversion lists every public method of every result class, and each one maps to a setting or an attribute before its class is deleted.
- **[Trade-off]** Discovery builds every analysis with default settings to read its data kind. This is cheap, because constructors only store arguments.
- **[Risk]** Analyses sent to worker processes must pickle. → Mitigation: analyses hold only their settings until prepared, and the experiment clones them inside each task.

## Migration Plan

1. Pilot: `BaseAnalysis` and `EntropyAnalysis`, with their tests and doctests. Stop for the user's review.
2. Convert the remaining analysis families, one commit each. Delete each function and result class as its family lands.
3. Move `basin_stability`, the ensemble adapter, and sensitivity to `experiments`. Add `BaseExperiment`, `CalibrationExperiment`, and `BasinStabilityExperiment`, and make the task helpers private.
4. Cut the engine surface per package, starting with `compile`, then `problems`, then `solvers`.
5. Update the import-linter contracts, the gallery tools, the gallery examples, the user guide, and the README.

No legacy aliases are kept. A rollback reverts the branch.
