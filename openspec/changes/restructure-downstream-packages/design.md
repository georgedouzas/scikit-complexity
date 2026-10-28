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

Target grouping. Every class is one mathematical operation, named after the operation, and its choices are variants of that operation (see "Library packages hold general operations"):

| Class | Choices | Replaces |
|---|---|---|
| `EntropyAnalysis` | shannon, renyi, tsallis, permutation, sample, approximate, spectral, multiscale, block, rate, excess, lempel-ziv, ensemble | the entropy functions of `_complexity`, `_series`, `_information` |
| `InformationAnalysis` | mutual, auto-mutual, transfer, active-storage, correlation-information, total-correlation, dual-total-correlation, o-information, o-information-gradients, kl-divergence | `_information`, `_multivariate` |
| `LocalInformationAnalysis` | mutual, transfer, entropy | `_local_information` |
| `PartialInformationAnalysis` | | `partial_information` |
| `ComplexityAnalysis` | | `complexity_entropy` |
| `EmergenceAnalysis` | effective-information, causal, criteria, closure | `_emergence`, `_emergence_criteria` |
| `EpsilonMachineAnalysis` | | `_computational_mechanics` |
| `DistributionAnalysis` | histogram, ccdf, moments, moment-ratio, dispersion-index, gini, tail-index, records, total-variation | `_statistics`, `_extremes`; the Binder cumulant and sweep, the Fano factor, the kinetic temperature, and the dissimilarity index |
| `FitAnalysis` | power, stable, extreme-value, and the named laws exponential, normal, lognormal, gamma, maxwell, rayleigh, uniform | `_statistics`, `_extremes` fits, `compare_with_law`, the Maxwell-Boltzmann test |
| `CorrelationAnalysis` | auto, two-time, function, length, overlap | `_series`, `_aging`, `_spatial`; the replica overlap |
| `SpectrumAnalysis` | power, structure-factor | `_series`, `_spatial`, `_fields` |
| `ScalingAnalysis` | detrended-fluctuation, rescaled-range, box-counting, correlation-dimension, variance-mean, power | `_series`, `_scaling`; number and relative fluctuations, the order-parameter exponent |
| `CollapseAnalysis` | scaling, aging, distribution, crossings | `_scaling`, `_aging`; the Binder crossings |
| `RecurrenceAnalysis` | | `_recurrence` |
| `EmbeddingAnalysis` | delay, false-nearest-neighbors | `_embedding` |
| `RollingAnalysis` | variance, autocorrelation, skewness | `early_warning` |
| `EventAnalysis` | detect, counts, laminar-durations, excursions, branching | `_events`; the avalanches and the branching ratio |
| `ClusterAnalysis` | labels, sizes, percolation, spanning | `_percolation` |
| `DomainAnalysis` | size, growth, wavelength, interface-density | `_fields` |
| `AlignmentAnalysis` | mean-vector, mean-rotation, axial, mean-phase | the polar, milling, nematic, and phase orders, read from agents, fields, or series |
| `DisplacementAnalysis` | | `mean_squared_displacement` |
| `NeighborAnalysis` | distances, clark-evans, radial-distribution, like-share | `_agents`, `_spatial`, `_mixing` |
| `CoarseGrainAnalysis` | | `coarse_grain` |
| `NetworkAnalysis` | degrees, assortativity, centrality, communities, neighbor-degree, summary | `_networks` |
| `MotifAnalysis` | counts, z-scores | `_motifs` |
| `MarkovAnalysis` | transition-matrix, stationary, state-model, timescales, spectral-gap | `_markov` |
| `PassageAnalysis` | first-passage, mean-first-passage, hitting, committor, return-times | `_passage` |
| `IrreversibilityAnalysis` | entropy-production, jump-entropy-production, langevin-entropy-production, count-entropy-production, detailed-balance, fluctuation-theorem, uncertainty-relation, response, fluctuation-dissipation | `_irreversibility`, `_aging` |
| `LargeDeviationAnalysis` | tilted, block, cloning, histogram, reweight | `_large_deviations` |
| `ErgodicityAnalysis` | breaking, metric, distance, invariant-density, ulam | `_ergodicity`, `_invariants` |
| `KramersMoyalAnalysis` | | `kramers_moyal` |
| `StabilityAnalysis` | fixed-points, dispersion | `fixed_points`, `stability`, `jacobian`, `nullclines`, `dispersion_relation` |
| `LinearizationAnalysis` | | `linearize`, `controllability`, `observability`, `lqr` |
| `LyapunovAnalysis` | spectrum, largest | `_lyapunov`, `_invariants` |
| `SectionAnalysis` | poincare, stroboscopic | `_sections` |
| `LoopAnalysis` | loops, dominance, flow-matrix | `_loops`, read from the signed dependency graph of any system |
| `SymbolicAnalysis` | equations, solve, hamiltonian, conservation | `_symbolic`, `_invariants` |
| `AutomatonAnalysis` | injective, pre-injective, surjective, garden-of-eden, inverse, compose, difference, entropy | `_automata` |
| `QuantumAnalysis` | spectrum, ground-state, purity, entanglement | `_quantum` |
| `SparseIdentificationAnalysis`, `ModeDecompositionAnalysis`, `ReservoirAnalysis` | | the three surrogates, each with the learned system in `system_` |
| `SurrogateTestAnalysis` | | `surrogate_test` |
| `DiscoveryAnalysis` | | `analyses_for`, `data_kinds` |
| `SignatureAnalysis` | | `signature_analysis` |

`AvalancheAnalysis`, `BinderAnalysis`, `FluctuationAnalysis`, `OrderAnalysis`, `MixingAnalysis`, and `EarlyWarningAnalysis` are classes named after a field. They dissolve into the general operations above, and the field measures they named become realizations in the models library:

| Realization | Field package | Subclass of | Fixed settings |
|---|---|---|---|
| `BinderCumulantAnalysis` | `models.physics.realizations` | `DistributionAnalysis` | kind moment-ratio, orders 4 and 2 |
| `SusceptibilityAnalysis` | `models.physics.realizations` | `DistributionAnalysis` | kind moments, the variance times the size |
| `ReplicaOverlapAnalysis` | `models.physics.realizations` | `CorrelationAnalysis` | kind overlap |
| `KineticTemperatureAnalysis` | `models.physics.realizations` | `DistributionAnalysis` | kind moments of the speeds, with the mass |
| `MaxwellBoltzmannAnalysis` | `models.physics.realizations` | `FitAnalysis` | law maxwell, on the speeds |
| `GiantNumberFluctuationAnalysis` | `models.physics.realizations` | `ScalingAnalysis` | kind variance-mean, of box counts |
| `CriticalExponentAnalysis` | `models.physics.realizations` | `ScalingAnalysis` | kind power, around the critical point |
| `AvalancheAnalysis` | `models.physics.realizations` | `EventAnalysis` | kind excursions |
| `KuramotoOrderAnalysis` | `models.physics.realizations` | `AlignmentAnalysis` | kind mean-phase |
| `PolarOrderAnalysis`, `MillingOrderAnalysis`, `NematicOrderAnalysis` | the realizations of the collective-behaviour models' field | `AlignmentAnalysis` | kinds mean-vector, mean-rotation, axial |
| `DissimilarityIndexAnalysis` | `models.social.realizations` | `DistributionAnalysis` | kind total-variation, between two groups across units |
| `EarlyWarningAnalysis` | the realizations of the field whose models use it | `RollingAnalysis` | the variance, the lag-one autocorrelation, and the skewness |

A realization only fixes settings in its constructor and documents the field's name. It adds no formula, no verb, and no attribute, so the general operation's tests cover it. The crackling-noise relation, which combines two fitted exponents, is computed in the gallery example that shows it, not as a class.

### Library packages hold general operations
The core, the engine, analysis, and experiments hold general operations named mathematically: a class names what it computes or simulates, and its choices name variants of that operation. Eponymous names are kept where mathematics uses them for the operation itself, such as Shannon, Rényi, Gini, Lyapunov, or Kramers-Moyal. A name taken from one application field, such as the Binder cumulant, the polar order, or the segregation index, belongs to the models library, which subclasses the general class in the realizations of that field and fixes its settings. This is the rule the core already follows for laws and interactions, applied to every package.

*Alternative considered:* field names as choices of the general class, such as `DistributionAnalysis('binder')`. It keeps field vocabulary in the library and makes the choice list grow with every field.

#### Field names outside the models library
An audit of the public classes and choices of `core`, `engine`, and `experiments` for names taken from one application field. Mathematical and numerical names, eponymous ones included (Gillespie, Lindblad, Metropolis, Glauber, Rusanov, Godunov, Itô, Stratonovich, Morris, Saltelli, Sobol, Moore, von Neumann, Cayley), general simulation vocabulary, and the kinds of dynamics the core defines as primitives (mass-action and encounter kinetics, quantum operators, agents) are not listed. No name of the core or of experiments is unambiguous field jargon, so none is converted: the debatable ones are proposed, and the engine's wait for its restructure.

| Package | Name | Field | Proposed general name | Proposed realization | Decision |
|---|---|---|---|---|---|
| `core.laws` | `ProliferationLaw` | mathematical biology, the kinetic theory of active particles | `EncounterBirthLaw` | `ProliferationLaw` in `models.biology.realizations`, used by `oncology` | proposed: proliferative and destructive encounters are the terms of the kinetic theory of active particles, a mathematical framework the core adopts with `EncounterLaw`, and renaming two of its three laws splits the family |
| `core.laws` | `DestructionLaw` | mathematical biology, the kinetic theory of active particles | `EncounterDeathLaw` | `DestructionLaw` in `models.biology.realizations` | proposed, as `ProliferationLaw` |
| `core.specs` | `SamplerSpec` kind `'heat-bath'` | statistical physics | `'gibbs'`, the Gibbs sampler | none: `models.physics.statistical_mechanics` passes `'gibbs'` and documents the physics name | proposed: heat-bath is also the usual name in Monte Carlo texts, and it sits beside `'metropolis'` and `'glauber'` |
| `core.specs` | `SamplerSpec.temperature`, `EnergySpec` | statistical physics | | | keep: energy and temperature are the parameters of the Gibbs measure the sampler draws from, as in simulated annealing |
| `core.laws.utils` | `distributed_delay` symbols `material_delay<n>` | system dynamics, material against information delays | `conserved_delay<n>` | | proposed: the public setting is already `conserved`; only the generated symbol, which printed equations of the system-dynamics models show, carries the field name |
| `core.elements` | `FieldElement` boundary `'zero-flux'` | transport in physics and chemistry | `'neumann'` | | keep: zero-flux names what the homogeneous Neumann condition imposes on a conservation law, and is the usual term in the mathematics of reaction-diffusion |
| `core.laws` | `DriveLaw`, `RelaxationLaw` | self-organized criticality | | | keep: slow drive and fast relaxation describe any threshold dynamics; the sandpile grain appears only in an example |
| `core.laws` | `ExtremalLaw` | evolution, the Bak-Sneppen model | | | keep: extremal dynamics is a general scheme, also of extremal optimization |
| `core.elements`, `core.interactions`, `core.laws` | `ParticleElement`, `PotentialInteraction`, `PairPotentialInteraction`, `FluxLaw` | mechanics | | | keep: general simulation vocabulary of particle methods, gradient systems, and conservation laws |
| `core.laws`, `core.specs` | `Source`, `Sink`, `TransferLaw`, `LayerSpec` | | | | keep: field words, such as a bank, contacts, or employment, appear only in examples |
| `experiments` | `BasinStabilityExperiment` | network dynamics | | | keep: basin stability is defined for any dynamical system with several attractors |
| `engine.problems`, `engine.compile` | `StockFlow`, `StockFlowPass`, the stock and flow structure of `FlowsProblemPass` | system dynamics | `TransferNetwork`, `TransferPass`, after the core's `TransferLaw` | none, engine internals | proposed, after the engine restructure |
| `engine.viz` | `TraitsVisual`, kind `'traits'` | ecology and agent-based biology | `HistogramVisual`, kind `'histograms'`, after what it draws | none | proposed, after the engine restructure |
| `engine.viz` | `BodiesVisual`, kind `'bodies'` | mechanics | | | keep: a body with rods, trails, and outlines is general geometric vocabulary |
| `engine.viz` | `PhaseRingVisual`, `ExpectationVisual` | oscillators, quantum mechanics | | | keep: phases on the circle and expectations of operators are mathematical |
| `engine.solvers`, `engine.problems` | `KineticSolver`, `KineticProblem`, `CompiledEncounter` | kinetic theory | | | keep: kinetic equations are a mathematical class, and direct simulation Monte Carlo a numerical method |
| `engine.solvers` | `LammpsSolver`, `LammpsFix`, `MesaSolver`, `LeptonPrinter` | molecular dynamics, agent-based modelling | | | keep: they name the library they wrap, not a field |
| `engine.problems` | `MODES`, `LATTICE_BOUNDARIES` key `'zero-flux'` | transport | | | keep, following `FieldElement` |

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
