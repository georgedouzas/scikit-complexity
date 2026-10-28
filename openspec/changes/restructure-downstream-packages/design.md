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

`AvalancheAnalysis`, `BinderAnalysis`, `FluctuationAnalysis`, `OrderAnalysis`, `MixingAnalysis`, and `EarlyWarningAnalysis` are classes named after a field. They dissolve into the general operations above. A measure that only picks a choice of a general operation, such as the Binder cumulant (`DistributionAnalysis('moment-ratio')`), the polar order (`AlignmentAnalysis('mean-vector')`), or the avalanches (`EventAnalysis('excursions')`), gets no class: a model, an example, or a test uses the general operation, and its docstring names the field measure. A measure with logic of its own moves to the models library, into the package of the models it measures:

| Class | Package | Subclass of | Its own logic |
|---|---|---|---|
| `SusceptibilityAnalysis` | `models.physics.statistical_mechanics` | `DistributionAnalysis` | the variance times the size over the temperature, and the refined peak of a sweep |
| `KineticTemperatureAnalysis` | `models.physics.statistical_mechanics` | `DistributionAnalysis` | the mean squared speed times the mass over the dimension |
| `MaxwellBoltzmannAnalysis` | `models.physics.statistical_mechanics` | `FitAnalysis` | the test of the speeds against the chi law of their dimension |
| `CracklingNoiseAnalysis` | `models.physics.criticality` | `EventAnalysis` | the scaling relation between the size and duration exponents, and the collapse of mean shapes |
| `EarlyWarningAnalysis` | `models.earth.climate` | `RollingAnalysis` | the variance, autocorrelation, skewness, and reddening with their trends |

Code that one model package uses lives in that package, next to the model definitions. Code that several packages of one field share lives in the field's `common` package, grouped in family packages such as `physics.common.particles` or `economics.common.markets`, and a field with no shared code has no `common` package. No specialized code with logic of its own is removed by this regrouping, and the models library also serves as worked examples for developers who extend the library.

### Library packages hold general operations
The core, the engine, analysis, and experiments hold general operations named mathematically: a class names what it computes or simulates, and its choices name variants of that operation. Eponymous names are kept where mathematics uses them for the operation itself, such as Shannon, Rényi, Gini, Lyapunov, or Kramers-Moyal. A name taken from one application field, such as the Binder cumulant, the polar order, or the segregation index, belongs to the models library, which subclasses the general class next to the models of that field that use it and fixes its settings. This is the rule the core already follows for laws and interactions, applied to every package.

*Alternative considered:* field names as choices of the general class, such as `DistributionAnalysis('binder')`. It keeps field vocabulary in the library and makes the choice list grow with every field.

#### Field names outside the models library
An audit of the public classes and choices of `core`, `engine`, and `experiments` for names taken from one application field. Mathematical and numerical names, eponymous ones included (Gillespie, Lindblad, Metropolis, Glauber, Rusanov, Godunov, Itô, Stratonovich, Morris, Saltelli, Sobol, Moore, von Neumann, Cayley), general simulation vocabulary, and the kinds of dynamics the core defines as primitives (mass-action and encounter kinetics, quantum operators, agents) are not listed. No name of the core or of experiments is unambiguous field jargon, so none is converted: the debatable ones are proposed, and the engine's wait for its restructure.

| Package | Name | Field | Proposed general name | Proposed realization | Decision |
|---|---|---|---|---|---|
| `core.laws` | `ProliferationLaw` | mathematical biology, the kinetic theory of active particles | `EncounterBirthLaw` | `ProliferationLaw` in `models.biology.oncology`, its only user | proposed: proliferative and destructive encounters are the terms of the kinetic theory of active particles, a mathematical framework the core adopts with `EncounterLaw`, and renaming two of its three laws splits the family |
| `core.laws` | `DestructionLaw` | mathematical biology, the kinetic theory of active particles | `EncounterDeathLaw` | `DestructionLaw` in `models.biology.oncology` | proposed, as `ProliferationLaw` |
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
The `layers` contract in `pyproject.toml` becomes `experiments`, `analysis`, `engine`, `core`, then the root modules `base | types`, `specs`, and `errors`, with `models` allowed to import all of them.

### Shared modules live at the root
`BaseConfig` is the root of every configurable class of every package, so it lives in `skcomplex.base`, not in `skcomplex.core.base`, which keeps `BasePart` and the bases of a system's parts. The errors, the specs, and the type aliases are used by every package too, so they move with it to `skcomplex.errors`, `skcomplex.specs`, and `skcomplex.types`. A root module imports no subpackage.

`skcomplex.types` holds every alias built from Python, NumPy, SciPy, and Sage types, and one alias per type: `Array` replaces the local `Array`, `Vector`, `Matrix`, `Floats`, `Values`, and `Couplings`; `IntArray` replaces `Counts`, `Codes`, `Labels`, `Bits`, and `Integers`; `AnyArray` replaces `Columns`; `ComplexArray` replaces `Complex`; `Indices` and `Offset` lose their copies. Aliases that name a class of the library, such as the source of an analysis or the step of a lattice, stay beside the class, since the root cannot import a package. The engine's `KernelSpec` and `SupportSpec` stay in the engine for the same reason. The models keep their aliases, apart from the array aliases, which they import from the root.

### The engine's concept model
This section is task 7.2. It is the class tree the engine is restructured to in task 7.4, written for review before any engine code changes. It is read from the `engine-methods` branch, where helpers already moved into methods, modules hold one class each, `StockFlow` is `TransferNetwork`, and `TraitsVisual` is `HistogramVisual`.

#### What the engine is
The engine turns a core `System` into a trajectory.
1. A **compiler** reads the system and runs **passes** over a **compilation**, the state they share. Passes declare the variables, lower every law into **kernels**, and ask a **backend** to fuse the kernels into numeric functions.
2. The result is a **problem**: a frozen value of one mathematical form, with the layouts of its states and parameters, its kernels, and its numeric functions.
3. A **solver** of that form advances the problem's state through time.
4. A **simulation** composes the three. It compiles the system, prepares the solver, advances it frame by frame, and hands every **frame** to its **observers**. A **clock** paces the frames. The recorder observer builds the **trajectory**, and the Rerun observer draws the frame with **visuals**.

A modeller meets `Simulation`, a solver, and maybe a backend, a clock, or an observer. A developer extends the engine by subclassing one base: a pass, a kernel, a solver, a backend, an observer, or a visual.

Today the engine has 295 classes, 131 of them dataclasses (40 mutable), and 78 of the dataclasses carry behaviour. It also has 9 `Protocol` classes that stand in for types an import would cycle on, 102 private module-level functions, and 43 public functions. Eight bases exist: `BasePass`, `BaseSystemCompiler`, `BaseProblem`, `BaseBackend`, `BaseSolver`, `BaseObserver`, `BaseClock`, and `BaseVisual`. The kernels, the lowerings, the neighbour searches, the printers, and the records have none: 37 classes of those five concepts stand alone, and protocols type them where they meet.

#### Rules the tree follows
- One root, `BaseEstimator`, moves from `engine/utils.py` to `engine/base.py`, as `BasePart` sits in `core/base.py`. Its signatures take `object` instead of `Any`.
- One base per concept, in the `_base.py` of its package. Every other class derives from one base, directly or through a concrete class, as `MooreNeighborhood` derives from `LatticeNeighborhood` in the core.
- A class that acts is configurable. Its `__init__` only stores its arguments, `check()` validates them, `prepare(context)` sets the attributes that end in `_`, and `run(inputs)` acts.
- A dataclass is a frozen value with `check()` and plain readers only. A dataclass that acts today becomes a class of the tree, or its behaviour moves to the class that uses it.
- A name ends in its root concept, such as `RuleKernel`, `FieldLowering`, or `PairSearch`.
- Specialized code moves to `skcomplex.models`. The engine audit of "Field names outside the models library" found nothing more after the two renames, so only two functions move.

#### The bases
| Base | Module | Role | Public verbs and readers | Hooks | A developer subclasses it to |
|---|---|---|---|---|---|
| `BaseEstimator` | `engine/base.py` | Root of every configurable engine class. | `check()`, `prepare(*context)`, `run(*inputs, final=False)`, `get_params`, `set_params` | `_check`, `_prepare`, `_run`, `_release` | Only through a concept base. |
| `BaseProblem` | `problems/_base.py` | Frozen value: a system compiled to one form, with layouts, kernels, and numeric functions. | `check()`; readers `stochastic`, `adaptive_parameters`, `lattice_shape`, `with_initial_state` | none | Add a form whose compiler returns it, as `QuantumProblem`. |
| `BaseKernel` | `problems/_base.py` | A law lowered to numbers, the part of a problem that a solver runs at a state. It replaces the `Compiled*` dataclasses and the protocols `_SiteStep`, `_RenewalProcess`, `_TopologyChanges`, and `Jacobian`. | `check()`, `prepare(problem, rng)` for run state such as renewal clocks, `run(t, x, p)` | `_check`, `_prepare`, `_run` | Add a law kind, with the pass that builds the kernel and a solver that runs it. |
| `BaseSearch` | `problems/_base.py` | The neighbours a kernel reads: static rows, network edges, metric cells, or pairs within a cutoff. | `check()`, `prepare(problem)`, `run(x, p)` returns rows | `_prepare`, `_run` | Add a kind of neighbourhood. |
| `BasePrinter` | `problems/_base.py` | Prints Sage expressions as code of one target. | `check()`, `prepare(symbols)`, `run(expressions)` returns code or a function | `_run` | Add a code target. |
| `BaseRecord` | `problems/_base.py` | Frozen value: something that happened at one time of a run. It replaces the protocol `Record`. | `check()`; readers `label`, `time` | none | Add a record that a kernel emits. |
| `BaseStructure` | `problems/_base.py` | Frozen value: the structure of a rewriting element at one generation. It is `Structure` renamed. | `check()`; readers `parts`, `segments`, `polygons`, `geometry`, `raster` | none | Add a kind of rewriting. |
| `BaseBackend` | `backends/_base.py` | Fuses the law kernels of a problem into numeric functions. | `check()`, `prepare(form, size)`, `run(kernels, changes, reductions=..., draws=...)` returns `LoweredFunctions` | `_run` | Add a code generator. |
| `BaseCompiler` | `compile/_base.py` | Compiles a system into a problem. It is `BaseSystemCompiler` renamed and widened to the general compiler. | `check()`, `prepare(system)` sets `claims_` and `report_`, `run(initial_state=None)` raises the report's errors and returns the problem | `_check`, `_prepare`, `_run` | Add a whole-system formalism, as quantum or rewriting systems. |
| `BasePass` | `compile/_base.py` | One step of `PassCompiler`: it reads fields of the compilation and writes others. | `check()`, `run(compilation)` | `_run`; the fields it reads and writes | Add a compilation step. |
| `BaseLowering` | `compile/_base.py` | Lowers one family of laws. One pass creates it, later passes finish it, and the compilation keeps it in between. | `check()`, `prepare(compilation)`, `run(laws)` returns the lowered laws | `_prepare`, `_run` | Add a law family that needs state across passes. |
| `BaseSolver` | `solvers/_base.py` | Advances a problem of one form through time. | `check()`, `prepare(problem, rng, until)`, `run(t, x, p, until, final=False)`; attributes `events_`, `snapshot_`, `agents_` | `_advance`, `_collect`, `_prepare` | Add a numerical method, or wrap a library through `ExternalSolver`. |
| `BaseObserver` | `runtime/_base.py` | Receives the frames of a simulation. | `check()`, `prepare(problem)`, `run(frame, final=False)` | `_prepare`, `_run`, `_release` | Add an output of a run. |
| `BaseClock` | `runtime/_base.py` | Paces frames against wall time. | `check()`, `prepare()`, `run(elapsed)` | `_prepare`, `_run` | Add a way of pacing. |
| `BaseVisual` | `viz/_base.py` | Logs one element or agent group to a Rerun recording. | `check()`, `prepare(element, problem)`, `run(frame, recording)`; attribute `views_` | `_log`, `_log_agents`, `_views` | Add a drawing. |

`Simulation` is the composer and derives from `BaseEstimator` directly, as `System` composes the core. `Compilation` and `ModelCatalog` are plain classes with no verbs: the first is the input every pass runs on, and the second is a collection read with named methods, as `Trajectory` is.

`problems/_base.py` holds six bases because the problem and its parts form one cycle. A problem holds kernels, and kernels read the values of the problem, so a separate `kernels` package would import `problems` both ways.

#### The class tree
Verdicts:
- **keep**: a class of the tree under the base named, renamed where shown.
- **frozen value**: a frozen dataclass with `check()` and readers only.
- **merge into X**: its behaviour becomes private methods of `X`, and the class goes.
- **move to models**: specialized code.
- **remove**: dead or duplicate.

Classes that move to another package are listed under the package they come from, with the destination in the reason.

##### `engine/utils.py` (2)
| Class | Base | What it is | Verdict | Reason |
|---|---|---|---|---|
| `BaseEstimator` | sklearn `BaseEstimator` | Root of configurable classes | keep | Moves to `engine/base.py`, the root as in the core. |
| `Registry` | none | Generic key to implementation map | merge into `ModelCatalog` | Its only user is `MODELS`. |

##### `engine.problems` (61)
| Class | Base | What it is | Verdict | Reason |
|---|---|---|---|---|
| `BaseProblem` | none | Base of compiled problems | frozen value | The value base; gains a field `observables`, see the function homes. |
| `Problem` | `BaseProblem` | Compiler's problem with flat vectors | frozen value | The pipeline's main value. |
| `Kernel` | none, mutable dataclass | Laws of one component and kind over members or pairs | keep, `LawKernel(BaseKernel)` | The kernel the backends fuse. |
| `_Processes` | none, mutable dataclass | Wiener processes of noise outputs | merge into `LawKernel` | One user. |
| `CompiledReduction` | none, mutable dataclass | Reduction that laws read | keep, `ReductionKernel(BaseKernel)` | It evaluates at a state. |
| `CompiledNeighborhood` | none, mutable dataclass | Neighbourhood reduction per member | keep, `NeighborhoodKernel(BaseKernel)` | It evaluates at a state. |
| `CompiledEvent` | none | Guard and reset functions of an event | keep, `EventKernel(BaseKernel)` | It runs a guard and a reset. |
| `_MarkovMatrix` | none | Markov matrix evaluated from the parameters | keep, `MarkovKernel(BaseKernel)` | It evaluates and checks at a state. |
| `Draws` | none | Random values a stochastic map draws per step | keep, `DrawKernel(BaseKernel)` | `fill` acts at every step, so it becomes `run`. |
| `StaticSearch` | none | Neighbours that do not change | keep, `StaticSearch(BaseSearch)` | Search hierarchy. |
| `EdgeRows` | none | Neighbours along changing edges | keep, `EdgeRowSearch(BaseSearch)` | Search hierarchy, named by its root. |
| `EdgeSearch` | none, mutable dataclass | Edges an interaction runs on | keep, `EdgeSearch(BaseSearch)` | Search hierarchy. |
| `MetricSearch` | none | Neighbours in a continuous space | keep, `MetricSearch(BaseSearch)` | Search hierarchy. |
| `PairSearch` | none, mutable dataclass | Pairs closer than a cutoff | keep, `PairSearch(BaseSearch)` | Search hierarchy. |
| `_Printer` | none, mutable dataclass | Printer of one dialect | keep, `ExpressionPrinter(BasePrinter)` | It absorbs `print_expressions` and `numpy_function`; the dialect is a choice argument (`'numpy'`, `'scalar'`, `'lepton'`). |
| `Dialect` | none | Syntax of a target language | frozen value | Private data of `ExpressionPrinter`'s choices. |
| `Record` | `Protocol` | Something that happened at one time | frozen value, `BaseRecord` | A real value base replaces the protocol. |
| `EventRecord` | none | Event that happened | frozen value | Under `BaseRecord`. |
| `RenewalRecord` | none | Renewals that fired in one pass | frozen value | Under `BaseRecord`. |
| `_RenewalProcess` | `Protocol` | Stand-in for the renewal kernel | remove | `RenewalKernel` is a real type now. |
| `_TopologyChanges` | `Protocol` | Stand-in for the topology kernel | remove | `TopologyKernel` is a real type now. |
| `_SiteStep` | `Protocol` | Stand-in for lattice steps | remove | `BaseKernel`. |
| `_SiteState` | `Protocol` | Stand-in for `LatticeState` | remove | `LatticeState` moves to problems. |
| `Jacobian` | `Protocol` | Stand-in for the exact Jacobian | remove | `JacobianKernel`. |
| `Layout` | none | Named variables in one flat vector | frozen value | A frozen dataclass of its entries with readers (`entry`, `locate`, `read`, `values`); its `__init__` stops computing. |
| `Entry` | none | One named variable in a flat vector | frozen value | It absorbs `encode`, `decode`, `contains`, `labels_of`, `wrap_angle`, and `domain_kind` as private methods, and gains the reader `kind`. |
| `Reference` | none | Component a symbol stands for | frozen value | Index readers only. |
| `Variable` | none | Declaration of a state, parameter, or role | frozen value | |
| `Output` | none, mutable dataclass | Value a kernel produces for one target | frozen value | Built once by `KernelsPass`. |
| `Read` | none | Values a reduction symbol stands for | frozen value | |
| `PairRead` | none | How a reduction reads a pair | frozen value | |
| `NeighborRead` | none | How a symbol reads a neighbouring site | frozen value | |
| `Heading` | none | State holding the heading of members | frozen value | Read by `MetricSearch` and `NeighborhoodKernel`. |
| `SampledDraw` | none | One `scipy.stats` draw | frozen value | Held by `DrawKernel`. |
| `NoiseProcess` | none | Named Wiener process | frozen value | |
| `WienerBlock` | none | Correlated Wiener processes | frozen value | |
| `FractionalRows` | none | Noise rows of a fractional law | frozen value | |
| `_JumpChannels` | none | Random events of one jump law | frozen value | |
| `_Delay` | none | Delayed state read through a parameter | frozen value | |
| `_Breakpoints` | none | Times where inputs jump | frozen value | |
| `Accelerated` | none | Compiled stepper loops | frozen value | A backend output the problem holds. |
| `Lowered` | none, mutable dataclass | Kernels of every law kind | frozen value, `LoweredKernels` | Its name clashed with three other lowering names. |
| `ReductionSchedule` | none, mutable dataclass | Reductions in dependency order | frozen value | `extend` returns a new schedule. |
| `Partition` | none | Part of a co-simulation problem | frozen value | |
| `CompiledElement` | none | Element with dotted variable names | frozen value | Read by visuals and analysis. |
| `CompiledInteraction` | none | Interaction with its bound paths | frozen value | |
| `CompiledAgents` | none | Agent group of an agent problem | frozen value | |
| `NetworkSnapshot` | none | Network layer at one frame | frozen value | Read by analysis. |
| `TransferNetwork` | none | Quantities, flows, and records | frozen value | Read by `SymbolicAnalysis`. |
| `_CompiledFlow` | none | One transfer between holders | frozen value | |
| `_Recorded` | none | Expression recorded every frame | frozen value | |
| `_CompiledDelay` | none | Compiled delay or smoothing | frozen value | |
| `LoopTerm` | none | Node of the feedback graph | frozen value | Read by `LoopAnalysis`. |
| `SupportSpec` | none | What a solver or backend supports | frozen value | |
| `Form` | `StrEnum` | Mathematical form | frozen value | |
| `DomainKind` | `StrEnum` | Kind of value a state takes | frozen value | |
| `ElementKind` | `StrEnum` | Kind of element that carries states | frozen value | |
| `NoiseKind` | `StrEnum` | Structure of Wiener processes | frozen value | |
| `OrderKind` | `StrEnum` | Order of a time derivative | frozen value | |
| `VariableKind` | `Enum` | Role of a declared argument | frozen value | |
| `Source` | `Enum` | How a symbol is read per evaluation index | frozen value, `ReadKind` | `Source` clashes with the core's `Source` law. |

##### `engine.backends` (5)
| Class | Base | What it is | Verdict | Reason |
|---|---|---|---|---|
| `BaseBackend` | `BaseEstimator` | Base of backends | keep | |
| `NumpyBackend` | `BaseBackend` | Reference NumPy backend | keep | |
| `NumbaBackend` | `BaseBackend` | Fused Numba kernels and stepper loops | keep | |
| `_Source` | none, mutable dataclass | Generated Numba source with its slots | keep, `NumbaPrinter(BasePrinter)` | It prints kernels as code, 680 lines that do not fit inside `NumbaBackend`. |
| `Lowering` | none, mutable dataclass | Numeric functions of a form | frozen value, `LoweredFunctions` | `run`'s result, built once; the name is freed for `BaseLowering`. |

##### `engine.compile` (73 outside `passes`)
Compiler, compilation, and lowerings:

| Class | Base | What it is | Verdict | Reason |
|---|---|---|---|---|
| `BaseSystemCompiler` | `BaseEstimator` | Compiler of a claimed whole system | keep, `BaseCompiler` | One base for every compiler, the general one included. |
| `BasePass` | `BaseEstimator` | Base of the steps of a compilation | keep | Moves from `_pipeline.py` to `compile/_base.py`. |
| `Compilation` | none, mutable dataclass | State the passes read and write | keep | A plain class. Its `__init__` stores the system, backend, solver, owners, and held paths, and the passes set the rest. The `facts` dict (13 keys) becomes named attributes. |
| `ValidationReport` | none, mutable dataclass | Issues with the compilation | frozen value | Only `issues`. The compilation moves to the compiler's `compilation_`. Readers `errors`, `warnings`, `ok`. |
| `Issue` | none | One problem, with path and fix | frozen value | |
| `ValidationWarning` | `UserWarning` | Warning of a non-fatal issue | keep | |
| `Declarations` | none, mutable dataclass | Entries and blocks collected from components | keep, `LayoutLowering(BaseLowering)` | It lowers declared variables to layouts across passes. It absorbs `declared_variables`, `declarations_of`, and `default_of`. |
| `DrawLowering` | none, mutable dataclass | Draws lowered to parameter slots | keep, `DrawLowering(BaseLowering)` | |
| `FieldLowering` | none, mutable dataclass | Stencils of operators of a component | keep, `FieldLowering(BaseLowering)` | |
| `MemberShifts` | none, mutable dataclass | Member offset reads | keep, `MemberLowering(BaseLowering)` | Named by its root. |
| `TransferPass` | none | Flows and delays expanded, then lowered | keep, `TransferLowering(BaseLowering)` | It is not a pass: `ExpandFlowsPass` creates it and `LowerFlowsPass` finishes it. |
| `Observer` | none, mutable dataclass | Observables that laws read | keep, `ObservableLowering(BaseLowering)` | `Observer` reads as a runtime observer. |
| `_MemberReductions` | none, mutable dataclass | Reductions in member observables | merge into `ObservableLowering` | One user. |
| `LatticeContext` | none, mutable dataclass | What lattice laws compile against | merge into `LatticeProblemPass` | One user. |
| `TableValues` | none, mutable dataclass | Selected table entries at parameters | merge into `EncounterKernel` | Its users are the encounter kernel and the pass that builds it. |
| `Block` | none, mutable dataclass | Variables of a component with symbols | frozen value | The passes that change it return a copy. |
| `KernelSpec` | none, mutable dataclass | Namespace and references of one component's laws | frozen value | Same: `RowsPass` and `ReductionsPass` return copies. |
| `Census` | none | Laws of every component, with failures | frozen value | |
| `CensusRecord` | none | One law with its element and targets | frozen value | |
| `Node` | none | Element or interaction in the tree | frozen value | |
| `BoundRenewal` | none | Renewal law with its group | frozen value | |
| `MatrixLaw` | none | Markov matrix of a law | frozen value | |
| `SetSymbol` | none | Symbol of one element of a set | frozen value | |
| `Observed` | none | Observable a role reads | frozen value | |
| `_HeldReduction` | none | Reduction a co-simulation holds | frozen value | |
| `_HeldElement` | core `Element` | Element holding a held reduction | keep | An engine-internal core element. |
| `_Owners` | none | Law kinds of the owners | frozen value | |
| `_Plan` | none | Partitions and their issues | frozen value | |
| `_PlannedPartition` | none | One planned subsystem | frozen value | |
| `Parameters` | none | Parameter symbols of rewritten elements | frozen value | Read by the rewriting compiler and kernels. |
| `_Request` | none, mutable dataclass | Draw needing a value per row | frozen value | |
| `_Record`, `_Born` | none, mutable dataclasses | Transfer records of `TransferLowering` | frozen value (2) | |
| `_Declared`, `_Spec` | `Protocol` | Stand-ins in the transfer module | remove (2) | The real types are importable inside `compile`. |
| `_ConservationLaw` | `Protocol` | Stand-in for `FiniteVolume` | remove | `FiniteVolumeKernel`. |
| `RenewalLaw` | core `BaseLaw` | Synthetic renewal law | keep | An engine-internal core law. |

Kernels and their values, all moving to `engine.problems` under `BaseKernel`:

| Class | Base | What it is | Verdict | Reason |
|---|---|---|---|---|
| `CompiledRule` | none | Local rule on a lattice | keep, `RuleKernel` | `apply` becomes `run`. |
| `CompiledDrive` | none | Drive of a lattice state | keep, `DriveKernel` | |
| `CompiledRelaxation` | none | Relaxation within a step | keep, `RelaxationKernel` | |
| `CompiledExtremal` | none | Extremal update | keep, `ExtremalKernel` | |
| `CompiledMonteCarlo` | none | Monte Carlo sweep | keep, `MonteCarloKernel` | |
| `CompiledRenewal` | none | Renewal draws and resets | keep, `RenewalKernel` | `start` becomes `prepare`, `fire` becomes `run`. |
| `_MemberFunction`, `_Sampler` | none | Parts of a renewal | merge into `RenewalKernel` (2) | One user. |
| `CompiledEncounter` | none, mutable dataclass | Encounter law as sparse contractions | keep, `EncounterKernel` | |
| `Topology` | none, mutable dataclass | Topology requests between steps | keep, `TopologyKernel` | `decide`, `apply`, and `fire` become `run` and private methods. |
| `NetworkRows` | none | Variable groups of one network | merge into `TopologyKernel` | Row bookkeeping of one kernel. |
| `CapacityRows` | none | Rows a new node takes | merge into `TopologyKernel` | Same. |
| `CompiledRequest` | none | One request with its rows | frozen value | |
| `ExactJacobian` | none, mutable dataclass | Exact Jacobian and its pattern | keep, `JacobianKernel` | `evaluate` becomes `run`. |
| `_Partials` | none, mutable dataclass | Compiled partial derivatives | merge into `JacobianKernel` | |
| `_TooDense` | `Exception` | Internal size-limit signal | keep | Private control-flow error. |
| `FiniteVolume` | none, mutable dataclass | Conservation laws of a field | keep, `FiniteVolumeKernel` | `rates` becomes `run`. |
| `DerivedRead` | none, mutable dataclass | Spectral multiplier or finite element matrix | keep, `FieldOperatorKernel` | It evaluates at a state. |
| `DiffusiveLimit` | none | Stable step of explicit diffusion | frozen value | |
| `LinearTerm` | none | Linear part of a field rate | frozen value | |
| `CompiledWord` | none | String rewriting system | keep, `WordKernel` | `initial` becomes `prepare`, `rewrite` becomes `run`. |
| `_Group`, `_Successor` | none, mutable dataclasses | Productions of a word | frozen value (2) | Private values of `WordKernel`. |
| `ObservableEvaluator` | none | Evaluator of every observable | keep, `ObservableKernel` | Built by a pass, so `Simulation` stops compiling observables itself. |
| `Arguments` | none | What a lattice law reads | frozen value | |
| `LatticeState` | none | Lattice group in the state vector | frozen value | |
| `Lookup` | none | Lookup array of a table rule | frozen value | |
| `ExtremalRecord` | none | One extremal application | frozen value | Under `BaseRecord`. |
| `RelaxationRecord` | none | One relaxation | frozen value | Under `BaseRecord`. |
| `Structure` | none | Base of rewriting structures | frozen value, `BaseStructure` | |
| `WordStructure` | `Structure` | Word at one generation | frozen value | |
| `PatchStructure` | `Structure` | Patch of tiles | frozen value | |
| `GraphStructure` | `Structure` | Host graph | frozen value | |
| `Operator` | none | Sum of products of local operators | frozen value | Operator arithmetic is the allowed exception to magic methods. Moves to problems with `QuantumProblem`. |
| `LocalOperator` | none | Operator on one subsystem | frozen value | Same. |
| `QuantumView` | none | Element with a Hilbert space state | frozen value | Read by `ExpectationVisual`. |

`BasePass._values_of` duplicates the core's `BaseLaw._get_values`; see the dead code.

##### `engine.compile.passes` (70)
All 52 passes stay under `BasePass` (keep). They are the steps of the pipeline, ordered by the fields they read and write. Merging the small ones, such as `CollectPass`, `DrawsPass`, or `EncounterJacobianPass`, would change that order, so none merge. They become private: a user reaches them through `PassCompiler(passes=...)`.

| Passes (keep, `BasePass`) |
|---|
| `FlattenPass`, `CensusPass`, `PlanPass`, `LayoutPass`, `SpecsPass`, `CollectPass`, `LowerComplexPass`, `MarkovPass`, `RowsPass`, `OrdersPass`, `DelaysPass`, `NoisePass`, `MemberShiftPass`, `OwnersPass`, `DrawsPass`, `FormPass`, `ReductionsPass`, `KernelsPass`, `FunctionsPass`, `JacobianPass`, `ProblemPass`, `UnpackRenewalsPass`, `DeclareRenewalsPass`, `SetRenewalsApartPass`, `RenewalProblemPass`, `ExpandFlowsPass`, `LowerFlowsPass`, `FlowsProblemPass`, `BoundaryLawsPass`, `AgentsPass`, `UnpackMatchesPass`, `SetRulesApartPass`, `AgentsProblemPass`, `LatticeProblemPass`, `LowerFieldsPass`, `ConservedNoisePass`, `FieldFunctionsPass`, `FieldJacobianPass`, `FieldProblemPass`, `CheckFractionalNoisePass`, `FractionalFormPass`, `FractionalProblemPass`, `CompileEncountersPass`, `EncounterFunctionsPass`, `EncounterJacobianPass`, `EncounterProblemPass`, `SplitRequestsPass`, `MaskRequestsPass`, `TowardsPass`, `RequestJumpsPass`, `TopologyFunctionsPass`, `TopologyProblemPass` |

| Class | Base | What it is | Verdict | Reason |
|---|---|---|---|---|
| `QuantumCompiler` | `BaseSystemCompiler` | Compiler of quantum systems | keep, under `BaseCompiler` | |
| `RewritingCompiler` | `BaseSystemCompiler` | Compiler of rewriting systems | keep, under `BaseCompiler` | |
| `_Lowering` | none | Quantum operators lowered to sparse matrices | keep, `QuantumLowering(BaseLowering)` | Third lowering family; private. |
| `_Reductions` | none, mutable dataclass | Reductions collected from laws | keep, `ReductionLowering(BaseLowering)` | |
| `QuantumTerm` | none | Sparse matrix with a symbolic coefficient | keep, `QuantumKernel(BaseKernel)` | `value(t, p)` becomes `run`. Moves to problems. |
| `CompiledGraph` | none | Graph rewriting system | keep, `GraphKernel(BaseKernel)` | Moves to problems. |
| `CompiledPatch` | none | Substitution system of a patch | keep, `PatchKernel(BaseKernel)` | Moves to problems. |
| `QuantumProblem` | `BaseProblem` | Problem of the quantum form | frozen value | Moves to problems. |
| `RewritingProblem` | `BaseProblem` | Problem of the rewriting form | frozen value | Moves to problems. |
| `KineticProblem` | `Problem` | Problem with encounter kernels | frozen value | Moves to problems. |
| `HilbertSpaceError` | core `SkcomplexError` | Hilbert space over its limit | keep | |
| `FractionalState` | none | State of a fractional order | frozen value | |
| `_NoiseTerm` | none | Noise amplitude on one component | frozen value | |
| `_Orders`, `_Relation` | none | Orders and relations of states | frozen value (2) | |
| `_Laws` | none, mutable dataclass | Hamiltonian and dissipator laws collected | frozen value | |
| `_Subsystem` | none | One quantum subsystem | frozen value | |
| `_Weight` | none | Weight target of a request | frozen value | |

The general compiler is a new class, `PassCompiler(BaseCompiler)`, with the arguments `passes`, `backend`, `solver`, `observables`, and `partition` (`'plan'` or `'subsystems'`). Its `prepare` is today's `validate`, and its `run` is today's planned or partitioned compile. The private functions of `_compile.py`, `_pipeline.py`, `_validation.py`, and `checks/` become its private methods, and the `checks` package goes. A pass may add issues by refining an `issues` field, as `CheckFractionalNoisePass` does today.

##### `engine.solvers` (59)
| Class | Base | What it is | Verdict | Reason |
|---|---|---|---|---|
| `BaseSolver` | `BaseEstimator` | Base of solvers | keep | |
| `Solver` | `BaseSolver` | Solvers of the compiler's problem | keep | The concrete parent of the general solvers, as `Element` is in the core. |
| `SDESolver` | `Solver` | Fixed-step SDE schemes | keep | Three children. |
| `CaputoSolver` | `Solver` | Caputo fractional steppers | keep | Two children. |
| `ConvolutionSolver` | `CaputoSolver` | Explicit convolution schemes | keep | Two children. |
| `_SplitSolver` | `Solver` | Linear part apart from the rest | keep, `SplitSolver` | Four children. |
| `OptimalControlSolver` | `Solver` | Optimal control solvers | keep | Two children. |
| `ExternalSolver` | `Solver` | Solvers that wrap a library | keep | `MesaSolver` moves under it, so it has two children. |
| `QuantumSolver` | `BaseSolver` | QuTiP steppers | keep | Three children. |
| `RK4Solver`, `ScipyODESolver`, `SSPRKSolver`, `VelocityVerletSolver`, `KineticSolver`, `GillespieSolver`, `RenewalSolver`, `IterationSolver`, `LatticeSolver`, `MonteCarloSolver`, `CoSimulationSolver`, `FractionalEulerMaruyamaSolver` | `Solver` or `LatticeSolver` | Concrete methods | keep (12) | |
| `EulerMaruyamaSolver`, `HeunSolver`, `MilsteinSolver` | `SDESolver` | SDE schemes | keep (3) | |
| `FractionalABMSolver`, `GrunwaldLetnikovSolver`, `L1Solver` | `CaputoSolver` | Fractional schemes | keep (3) | |
| `CrankNicolsonSolver`, `ETDRK4Solver`, `IMEX2Solver`, `IMEXEulerSolver` | `SplitSolver` | Split schemes | keep (4) | |
| `CollocationSolver`, `PontryaginSolver` | `OptimalControlSolver` | Optimal control | keep (2) | |
| `LammpsSolver`, `MesaSolver` | `ExternalSolver` | LAMMPS and Mesa | keep (2) | `MesaSolver` moves from `Solver` to `ExternalSolver`. |
| `LindbladSolver`, `QuantumTrajectoriesSolver`, `UnitarySolver` | `QuantumSolver` | Quantum solvers | keep (3) | |
| `RewritingSolver` | `BaseSolver` | Rewriting solver | keep | |
| `EventLocator` | none | Locates events in a step | merge into `BaseSolver` | Helper of the base; `events_` is already there. |
| `History` | none | Past states for delays | merge into `BaseSolver` | Helper of the base, set as `history_`. |
| `_Rows` | none | Growing rows of values | merge into `BaseSolver` | Helper of `History`. |
| `RenewalClock` | none | Next renewal per member | merge into `Solver` | One user. |
| `SplitOperator` | none, mutable dataclass | Linear part in Fourier blocks | merge into `SplitSolver` | One user. |
| `_SpectralBlock` | none, mutable dataclass | One spectral block | merge into `SplitSolver` | Same. |
| `_ControlProblem` | none | Control problem of a system | merge into `OptimalControlSolver` | One user, with behaviour. |
| `_Control` | none | One control | frozen value | |
| `ControlSolution` | none | Optimal control on a grid | frozen value | The solvers' `solution_`. |
| `LeptonPrinter` | none | Lepton printer for LAMMPS | merge into `ExpressionPrinter` | The `'lepton'` dialect. |
| `_Atoms` | none, mutable dataclass | Elements as LAMMPS atom types | frozen value | |
| `LammpsFix` | none | Values a native fix reads | frozen value | Models build it. |
| `_Segment` | none | One accepted step with dense output | frozen value | |
| `AgentSnapshot` | none | Agents of one group at a frame | frozen value | |
| `AgentState` | none | Agents and reporters at a frame | frozen value | |
| `StructureState` | none | Structures at a frame | frozen value | |
| `ModelView` | none | Model as Mesa rules see it | keep | A public view that model rules receive (73 model modules). No base: it is neither configurable nor a value. |
| `AgentView` | none | Agent as Mesa rules see it | keep | Same (63 model modules). |
| `_AgentModel` | `mesa.Model` | Mesa model running the rules | keep | Wraps Mesa. |
| `HorizonError` | core `SolverError` | Step beyond a fixed horizon | keep | |

##### `engine.runtime` (11)
| Class | Base | What it is | Verdict | Reason |
|---|---|---|---|---|
| `Simulation` | `BaseEstimator` | Composer of the pipeline | keep | |
| `Trajectory` | none | Recorded run with named readers | keep | The pipeline's output, read with `values`, `member`, `agents`, and the like. |
| `Frame` | none | State at one time | frozen value | |
| `BaseObserver` | `BaseEstimator` | Base of observers | keep | Moves to `runtime/_base.py`. |
| `RecorderObserver` | `BaseObserver` | Keeps every frame | keep | |
| `ConservationObserver` | `BaseObserver` | Checks flows balance | keep | |
| `InvariantObserver` | `BaseObserver` | Checks invariants | keep | |
| `BaseClock` | `BaseEstimator` | Base of clocks | keep | Moves to `runtime/_base.py`. |
| `AsFastAsPossibleClock` | `BaseClock` | Never waits | keep | |
| `RealTimeClock` | `BaseClock` | Fixed speed against wall time | keep | |
| `ModelEntry` | none | Registered model | frozen value | |

A new plain class, `ModelCatalog`, replaces `Registry[ModelEntry]` and the functions `catalog`, `describe`, `create`, `run`, and `summarize`. It has the named methods `register(entry)`, `entry(name)`, and `entries(field=None, tag=None)`, and one instance, `MODELS`.

##### `engine.viz` (14)
| Class | Base | What it is | Verdict | Reason |
|---|---|---|---|---|
| `BaseVisual` | `BaseEstimator` | Base of visuals | keep | Moves from `_visuals.py` to `viz/_base.py`. |
| `AgentsVisual`, `BodiesVisual`, `CayleyVisual`, `ImageVisual`, `MeshVisual`, `PointsVisual`, `GeometryVisual`, `PhaseRingVisual`, `ScalarsVisual`, `SiteVisual` | `BaseVisual` | Drawings | keep (10) | |
| `HistogramVisual` | `AgentsVisual` | Histograms of traits | keep | |
| `ExpectationVisual` | `ImageVisual` | Local quantum expectations | keep | |
| `RerunObserver` | `BaseObserver` | Logs frames to Rerun | keep | It absorbs the blueprint functions, `resolve_visual`, and `agents_visual_of` as private methods. |

##### Count by verdict
Each class counts once. A row that names several classes counts each.

| Package | keep | frozen value | merge | move to models | remove | Total |
|---|---|---|---|---|---|---|
| `utils.py` | 1 | 0 | 1 | 0 | 0 | 2 |
| `problems` | 12 | 43 | 1 | 0 | 5 | 61 |
| `backends` | 4 | 1 | 0 | 0 | 0 | 5 |
| `compile` (outside `passes`) | 26 | 36 | 8 | 0 | 3 | 73 |
| `compile.passes` | 60 | 10 | 0 | 0 | 0 | 70 |
| `solvers` | 43 | 8 | 8 | 0 | 0 | 59 |
| `runtime` | 9 | 2 | 0 | 0 | 0 | 11 |
| `viz` | 14 | 0 | 0 | 0 | 0 | 14 |
| **Total** | **169** | **100** | **18** | **0** | **8** | **295** |

Of the 169 classes kept, 37 are renamed into a new hierarchy: 22 kernels, 8 lowerings, 5 searches, and 2 printers. No class moves to the models library, because the field-name audit left none; the specialized code is two functions, listed with the function homes. The new classes are `PassCompiler`, `ModelCatalog`, and the bases `BaseKernel`, `BaseSearch`, `BasePrinter`, `BaseRecord`, and `BaseLowering`. Of the 131 dataclasses, 86 stay frozen values, 32 become classes of the tree (kernels, lowerings, searches, and printers), `Compilation` becomes a plain class, and 12 merge into the class that uses them.

#### The public surface after the restructure
| Package | Public classes | Public functions and values |
|---|---|---|
| `skcomplex.engine.base` | `BaseEstimator` | none |
| `engine.problems` | `BaseProblem`, `Problem`, `KineticProblem`, `QuantumProblem`, `RewritingProblem`; `BaseKernel`, `BaseSearch`, `BasePrinter`, `ExpressionPrinter`; `BaseRecord`, `EventRecord`, `RenewalRecord`, `ExtremalRecord`, `RelaxationRecord`; `BaseStructure`, `WordStructure`, `PatchStructure`, `GraphStructure`; `Layout`, `Entry`, `Variable`, `SupportSpec`, `NetworkSnapshot`, `TransferNetwork`, `LoopTerm`, `Operator`, `LocalOperator`; the enums `Form`, `DomainKind`, `VariableKind`, `NoiseKind`, `OrderKind` | none |
| `engine.backends` | `BaseBackend`, `NumpyBackend`, `NumbaBackend`, `LoweredFunctions` | none |
| `engine.compile` | `BaseCompiler`, `PassCompiler`, `QuantumCompiler`, `RewritingCompiler`, `BasePass`, `BaseLowering`, `Compilation`, `ValidationReport`, `Issue`, `ValidationWarning`, `HilbertSpaceError` | `compile_system` |
| `engine.solvers` | `BaseSolver`, `Solver`, `SDESolver`, `SplitSolver`, `CaputoSolver`, `OptimalControlSolver`, `ExternalSolver`, `QuantumSolver`, every concrete solver, `ControlSolution`, `AgentSnapshot`, `AgentState`, `StructureState`, `LammpsFix`, `ModelView`, `AgentView`, `HorizonError` | none |
| `engine.runtime` | `Simulation`, `Trajectory`, `Frame`, `BaseObserver`, `RecorderObserver`, `ConservationObserver`, `InvariantObserver`, `BaseClock`, `AsFastAsPossibleClock`, `RealTimeClock`, `ModelEntry`, `ModelCatalog` | `MODELS`, `register_model` |
| `engine.viz` | `BaseVisual`, the twelve visuals, `RerunObserver` | none |

Two public functions remain.
- `compile_system` is the pipeline entry. It asks each compiler whether it claims the system and returns `compiler.check().prepare(system).run(initial_state)`. Analysis and 75 test modules call it.
- `register_model` is the decorator that 271 model modules use to define a model. It calls `MODELS.register`.

The kernels other than `BaseKernel`, the lowerings, the searches, and the passes become private. A test reaches them through `PassCompiler` and the problem it returns.

Today's public module constants get these homes:
- `LATTICE_BOUNDARIES` and `MODES` become private constants of their module.
- `NUMPY_DIALECT` and `SCALAR_DIALECT` become the dialect choices of `ExpressionPrinter`.
- `DEFAULT_MAX_DIMENSION` becomes the default of `QuantumCompiler(max_dimension=...)`.
- `DEFAULT_SOLVERS` becomes the private method `Simulation._default_solver(form)`.
- `VISUALS` becomes private to `RerunObserver`.
- `Getter` and `VectorFunction` become private type aliases.

##### A home for every public function
The engine has 43 public functions besides Numba code, and 8 public Numba-jitted functions.

| Function | Package | Home |
|---|---|---|
| `compile_system` | `compile` | Stays, as the pipeline entry above. |
| `validate` | `compile` | `BaseCompiler.prepare(system)` sets `report_`; `Simulation.check()` raises its errors. |
| `claiming_compiler` | `compile.passes` | `BaseCompiler.prepare` sets `claims_`; `compile_system` asks each compiler. |
| `partition_system` | `compile` | `PassCompiler(partition='subsystems')`. |
| `order_passes`, `run_passes` | `compile` | Private methods of `PassCompiler`. |
| `compiled_problem` | `compile` | A private method of `BaseAnalysis`, its only client. |
| `observables_of` | `compile` | A field `observables` of `BaseProblem`, the paths and specs the compiler found, which analysis and experiments read. The tree walk becomes a private method of `ObservableLowering`. |
| `probe_laws` | `compile._probe` | A private method of `CensusPass`. Experiments read the law classes of each owner from a new problem field `kinds`, today `Compilation.kinds`. |
| `probability_issues`, `request_issues` | `compile.checks` | Private methods of `PassCompiler`. |
| `distribution_from`, `members_from` | `compile` | Move to models. They build core groups from data for kinetic models and compile nothing. Their users are one model test and the user guide. |
| `encode`, `decode`, `contains`, `labels_of`, `wrap_angle`, `domain_kind` | `problems` | Private methods of `Entry`, which already has `encode` and `decode`, plus the reader `Entry.kind`. Experiments check a value by `entry.encode(value)`, which raises outside the domain. |
| `symbol`, `vector`, `components_of`, `symbols_of`, `domain_of` | `problems` | Private functions of `engine/_utils.py`. Every engine package shares them, so no class owns them, which is the `_utils` rule. Analysis reads symbols from `problem.symbols` and keeps its own private helpers. |
| `declared_variables`, `declarations_of`, `default_of` | `problems` | Private methods of `LayoutLowering`. Clients read declarations from the compiled `Layout` and `Variable`. One client is the model module `economics/markets/_queues.py`. |
| `print_expressions`, `numpy_function` | `problems` | `ExpressionPrinter(dialect).check().prepare(symbols).run(expressions)`. A `compiled=True` setting returns a NumPy function. |
| `step_count` | `problems` | A private `register_jitable` function, kept at module level because Numba loops call it. |
| `fractional_gaussian_noise` | `solvers` | A private method of `FractionalEulerMaruyamaSolver`. |
| `resolve_backend` | `backends` | A private method of `BaseCompiler`, which picks the backend by size. |
| `register_model` | `runtime` | Stays, as above. |
| `catalog`, `describe`, `create`, `run` | `runtime` | Removed from the engine. `skcomplex.models` already wraps them as its public functions, so the engine copies are a second way. `ModelCatalog.entries` and `ModelCatalog.entry` serve the models' versions. |
| `summarize` | `runtime` | A private method of the models' catalog, its only user. |
| `default_blueprint`, `primary_view`, `save_blueprint` | `viz` | `RerunObserver` settings `blueprint='default'` and `primary`, the prepared reader `blueprint_`, and private methods. Saving goes through `save_path`. |
| `resolve_visual`, `agents_visual_of` | `viz` | Private methods of `RerunObserver`. |
| `cell_coordinate`, `build_grid`, `radius_neighbours`, `cone_neighbours`, `nearest_neighbours`, `edge_neighbours`, `neighbor_value`, `neighbors` | `backends`, `compile` | They stay at module level because generated kernel source imports them by name. They get private names. |

The 102 private module-level functions follow their callers:
- the validation helpers to `PassCompiler`;
- `_complex.py` to `LowerComplexPass`;
- `_observables.py` to `ObservableLowering`;
- `_blueprints.py` and `_kinds.py` to `RerunObserver`;
- `_conserved_noise`, `_fractional_form`, and `_lattices` helpers to their passes;
- `_domains` to `Entry`;
- `_variable` to `LayoutLowering`;
- `_helpers_fingerprint` to `NumbaBackend`.

#### Dead code for task 7.1
These are candidates. Nothing is deleted here. An AST scan of `src`, `tests`, and `docs` for reads of each name found them. Vulture cannot parse the Python 3.12 syntax of the package, so 7.1 should rely on coverage.
- `CompiledEncounter.conserves`: a property that nothing reads.
- `CompiledReduction.set_indices`: a field set in `_reductions.py` that nothing reads.
- `BasePass._values_of`: duplicates the core's `BaseLaw._get_values` line for line, plus the `include` and `exclude` filters.
- `engine.runtime.catalog`, `describe`, `create`, and `run`: their only clients are the wrappers of the same name in `skcomplex.models._catalog`.
- The nine `Protocol` classes (`_RenewalProcess`, `_TopologyChanges`, `_SiteStep`, `_SiteState`, `Jacobian`, `Record`, `_Declared`, `_Spec`, `_ConservationLaw`): they exist only to avoid imports that the hierarchy makes possible.
- `Getter` and `VectorFunction` in `problems.__all__`: nothing outside the engine imports them.
- `ExternalSolver` with one child, unless `MesaSolver` moves under it.
- `labels_of` and `partition_system`: no client outside the engine besides their own doctests and surface tests.

#### The user's decisions
1. **Kernels.** The compiled laws become one hierarchy under `BaseKernel`, with `run` replacing `apply`, `evaluate`, `rewrite`, `fire`, and `rates`.
2. **Class variables.** The class variables of passes, solvers, and backends become hooks, such as `_get_support()`, as the analysis's data kind did.
3. **Validation.** `validate` is no public function. The compiler's `check()` validates its settings and its `prepare(system)` validates the system and sets `report_`; `run()` raises the report's errors and returns the problem.
4. **Model catalog.** The catalog moves to `skcomplex.models` whole: registering, listing, describing, and creating models by name. `Experiment` and `Simulation` take a system or a factory; a user who wants a model by name gets it from the models package first.
5. **Kinetic distributions.** `distribution_from` bins the members of a group over the values of one of their states, which is coarse-graining in state space. It becomes a setting of `CoarseGrainAnalysis` that bins a state over given values instead of positions over a grid. `members_from` builds a group, which a group's constructor already does from member values, so it is removed, and the model that samples members from densities draws the values with NumPy.

#### The common API
Every engine class follows one contract, so that a developer who learns one base knows them all, as with scikit-learn's estimators:
- `__init__` only stores the settings.
- `check()` validates the settings and returns the object.
- `prepare(target)` takes the one object the class serves (a system for a compiler, a compilation for a pass, a problem for a kernel, a search, a backend, a solver, an observer, or a visual), validates it against the settings, sets the attributes that end in `_`, and returns the object.
- `run(inputs)` acts on the one input the class transforms (the state for a kernel or a solver, the frame for an observer or a visual, the laws for a lowering) and returns the result or the object.
- The arguments are the same kind of thing for every base: a base never takes extra positional arguments in `prepare` or `run` that another base of the same role does not take; what varies between subclasses goes in settings, never in the verbs' signatures.
- `Compilation` and the model catalog are no exception: a pass's compilation is its prepared target, and the catalog in the models package is a collection read with named methods.

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
