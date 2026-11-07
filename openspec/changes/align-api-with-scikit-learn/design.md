## Context

0.6.0 makes every public object an estimator with `prepare` and `run`: parts, `Simulation`, about fifty analysis
classes, the experiment classes, and the compilers. Compatibility is spread over role specs, solver supports, the
compiler's form inference, the inventory's system vocabulary, and a generated compatibility page that lists every
declaration. scikit-learn separates estimators from the functions that evaluate them, and tells functions what an
estimator can do through tags.

## Goals / Non-Goals

**Goals:**
- One mental model shared with scikit-learn: estimators for what is defined, functions for what is done to them.
- One source of truth for compatibility, the tags, read by every function and by the documentation.
- Documentation and examples that use the new API, and compatibility pages that help a reader decide.

**Non-Goals:**
- Changing the numerics of any solver, backend, law, or system.
- Changing what a system defines, beyond the tags its parts declare.

## Correspondence with scikit-learn

| scikit-learn | scikit-complexity |
|---|---|
| `BaseEstimator`, `get_params`, `set_params`, `clone` | the base of every part |
| `fit` | `prepare` |
| `transform`, `predict` | `run` of a part |
| `NotFittedError` | the error for running an unprepared part |
| `Pipeline`, `ColumnTransformer` | `System` |
| `_parameter_constraints`, `Interval`, `StrOptions` | `ParameterSpec`, `StateSpec`, domains |
| estimator tags, `__sklearn_tags__` | capability tags, `get_tags()` |
| `check_estimator` | the conformance suite, selected by tags |
| `LinearModel`, `Ridge`, `Lasso` | general systems of the core, field realizations such as `SIRSystem` |
| `sklearn.utils.all_estimators()` | `all_systems()` |
| `cross_validate` | `simulate` |
| `cv=` splitters | `solver=` |
| `n_jobs`, joblib backends | `backend=` |
| `scoring=`, `make_scorer` | `observables=`, and the analysis binding helper |
| `sklearn.metrics` | analysis functions |
| `validation_curve`, `learning_curve`, repeated CV | sweeps, ensembles, scenarios, sensitivity |
| `GridSearchCV` | calibration, which stays an estimator |
| `*Display.from_estimator` | visual observers, such as Rerun |
| fitting internals | compilers and problem forms, private |

## Decisions

- **Tags through a plain method.** scikit-learn uses `__sklearn_tags__`. The project forbids magic methods, so parts
  define `get_tags()`, overriding a private `_get_tags()` hook per class the way specs are declared today, and return
  a frozen `Tags` dataclass from `skcomplex.core.specs`.
- **Tags come from existing declarations.** The fields map to what exists: state specs give domains, spaces give
  space, law classes give law kinds, solver supports become supported tags. The inventory's `system` vocabulary is
  generated from the tags of each system instead of written by hand, and the inventory check compares them.
- **System tags are derived, never declared**, so they cannot drift from what the compiler does. The compiler's form
  inference reads the tags.
- **Analyses as functions.** Each analysis class becomes a function named after it in snake case, with the same
  settings as keyword arguments and a frozen result value with the same fields as today's fitted attributes. The base
  `BaseAnalysis` and families with a choice argument become one function with that argument. The binding helper
  `bind_analysis(function, **settings)` returns a callable that experiments and `simulate` accept.
- **Experiments as functions.** `sweep`, `ensemble`, `compare_scenarios`, `sensitivity`, and `basin_stability`
  return tidy tables. `Calibration` stays an estimator.
- **Package layout, which needs the user's approval before any move.** Proposed: `simulate` and `Trajectory` in
  `skcomplex.core.engine` (no move), analysis functions in `skcomplex.core.analysis` (no move), experiment functions
  in `skcomplex.core.experiments` (no move), compilers and problems made private inside `skcomplex.core.engine` (no
  move, only names lose their export). No package is created or moved by this design. Any later move is asked first.
- **Documentation.** The guide's parts keep their task order, rewritten to the new API. The compatibility page and its
  generator are replaced by one generated page per kind of part, under a "Parts" section of the navigation, with
  executed examples of a compatible and an incompatible combination per page. A migration page maps removed names to
  their replacements.
  `docs/generate_parts.py` writes the pages of elements, interactions, laws, solvers, and backends, one private
  function per page. The analyses, experiments, and systems pages come in later waves, once their functions and
  classes declare tags, and the generator should add a function for each of them then.
- **Release.** One breaking release, 0.7.0. No aliases for removed names, as in earlier breaking releases.

- **System classes.** `BaseSystem` holds the verbs, the derived tags, nesting, and a private method that passes model
  parameters down to the parts in `prepare`. Each factory function becomes a class in the same self-contained module,
  named after the system in CamelCase with the suffix `System`. Its class attributes give `field`, `topics`,
  `references`, and the tuned run settings (`until`, `dt`, solver hint, observables, initial state). Model parameters
  become constructor arguments with the factory's defaults.
- **Discovery.** `all_systems()` reads field and topics statically from the class definitions, as the registration
  reader does today, so listing by field or topics imports no system module. The registry, `register_system`,
  `create_system`, `create_simulation`, and `describe` are removed. The conformance suite, the documentation generator,
  and the inventory and gallery checks use `all_systems()`.
- **General systems.** The cross-field duplicate scan of the consolidation lists shared structures. A general system
  enters the core in a later task only when two fields share the structure, starting with mass-action compartments.

## System class template

Chemistry is the first field converted, and every later field follows it. The rules below are what its modules do.

**Discovery during the transition.** `all_systems(field=None, topics=(), tags=None)` of `skcomplex.systems` returns
`(class name, class)` pairs of the converted system classes, sorted by name. The catalog reads every module of
`skcomplex.systems` with `ast`: a module-level class whose body assigns `field` is a system class, and its `field`,
`topics`, and `references` are read as literals or module-level literal constants, as the registration reader does.
Selecting by field and topics is static, so only the modules of the selected classes are imported, to return their
classes. Filtering by tags builds every selected class with its default parameters and compares `get_tags()`. Until
the removal wave, the old catalog serves converted classes too, so nothing that still uses names breaks:

- `catalog()` lists a class under its name without `System` in snake case, with acronyms kept as one word
  (`GrayScottSystem` is `gray_scott`, `BrusselatorODESystem` is `brusselator_ode`), its topics as tags, and its class
  name in `SystemListing.class_name`. The inventory, the gallery, and their checks keep their keys.
- `describe(name)` imports the module and returns a `SystemEntry` whose `factory` takes the model parameters and
  returns the prepared composite `SystemClass(**parameters).prepare().system_`, whose signature is that of the class,
  whose `defaults` are the tuned settings `until`, `dt`, `solver`, and `initial_state` the class gives, and whose
  observables are the class's `observables`.
- `create_system(name)` and `create_simulation(name)` therefore return and run the same composite the factory returned,
  so the catalog conformance tests, the gallery hooks, and the documentation generator need no change.

**One class per factory.** In the module of the factory, `register_system` and the factory are replaced by:

1. `class <CamelName>System(BaseSystem)`, named so that the snake-case rule gives back the registered name. When no
   CamelCase name does, rename the gallery example, its `SYSTEM` constant, and the inventory key, and say so.
2. The class docstring: its first line is the catalog description, without `Return`. The body is the factory's,
   ending with the sentence that names the states from its `Returns`. `Args` lists every constructor argument and
   `label`. `Examples` imports the class from its subfield and checks a parameter on `.prepare().system_` or a tag.
3. Class attributes, in this order: `field`, `topics` (the old `tags`), `references`, then the tuned run settings of
   `defaults`: `until`, `dt`, `solver`, `observables`, and `initial_state`. `field`, `topics`, and `references` are
   literals or module-level literal constants, and every listed class assigns its own, since the reader does not
   follow inheritance. A base class that should not be listed assigns no `field`.
4. `__init__` takes the factory's arguments in the same order and kind, with the same defaults, then
   `label: str = 'system'` as the last keyword-only argument. It calls `super().__init__(label)` and stores every
   argument under its own name, nothing else.
5. `_build_system(self) -> System` holds the factory's body, reading `self.<argument>`, and returns
   `System(self.label, elements=..., interactions=..., ...)`. It validates and caches nothing, since `get_tags`,
   `_check`, and `prepare` call it.
6. Helpers that only this class uses become its private methods, static when they do not read `self`, and are passed
   as `self._helper`. Helpers that several classes of the module use stay private functions of the module. A part that
   several modules of the field use stays in the field's `common`.
7. Every element or interaction of the field that writes `_get_laws` lists its law classes in `law_classes`, such as
   `law_classes = (RateLaw,)`, so that `SystemClass().get_tags().laws` is never `None`. Check it for every class.
8. The subfield `__init__` exports the class in place of the factory, in a sorted `__all__`, and docstrings that named
   the factory name the class.

**Tests and gallery.** Tests build the class and call `simulate(SystemClass(...), until=..., dt=..., random_state=0)`,
omitting the solver to use the tuned one, and read parts on `.prepare().system_`. `describe(name).defaults[...]`
becomes the class attribute or is omitted. A gallery example keeps `SYSTEM = '<snake name>'`, which coverage and the
recording use, keeps `observer = RerunObserver(...)` as a module variable, which the gallery runner finds, and calls
`trajectory = simulate(SystemClass(...), observers=[observer], ...)` with only the settings that differ from the tuned
ones, replacing `simulation.trajectory_` by `trajectory`. Analysis calls are left to the analyses wave. For this,
`simulate` takes `dt` and `observers` besides the arguments of the simulation spec.

**Checks per field.** The field's tests, `tests/systems/test_catalog.py`, doctests of the changed modules,
`PYTHONPATH=docs python docs/inventory_checks.py`, `PYTHONPATH=docs python -m gallery_tools coverage --strict`, running
every changed gallery example, and ruff, black, docformatter `--check`, mypy, and pydoclint on the changed files.

**Open in the template.** Run settings that depend on the model parameters (`register_system(settings=...)`,
`initial_state=...`) and observables built from the system (`observables=lambda system: ...`) have no class attribute
form while `simulate` reads class attributes. Static observables become an `observables` tuple of `ObservableSpec`.


## Analysis names

Every analysis class became a function named after it in snake case without `Analysis`, with a clearer name where
snake case reads badly or clashes (`CollapseAnalysis`, `DiscoveryAnalysis`, `FitAnalysis`, `NetworkAnalysis`,
`OperatorAnalysis`, `ShapeAnalysis`, and some field realizations). The source is the first positional argument, the
former constructor settings are keyword-only with the same defaults, and the function returns a frozen result whose
fields are the former fitted attributes without the trailing underscore. A field that only some kinds set defaults to
`None`. `bind_analysis(function, **settings)` returns a `functools.partial` that `Experiment` and other callers apply to
a trajectory, and an experiment names its outputs `<function name>.<field>`. `BaseAnalysis` is removed: the readers of
sources are private functions of `skcomplex.core.analysis._base`, and a system class that builds its parts is read as
its prepared composite.

| Old class | Function | Result |
|---|---|---|
| `AlignmentAnalysis` | `alignment` | `AlignmentResult` |
| `AutocatalyticSetAnalysis` | `autocatalytic_sets` | `AutocatalyticSetsResult` |
| `AutomatonAnalysis` | `automaton` | `AutomatonResult` |
| `CellComplexAnalysis` | `cell_complex` | `CellComplexResult` |
| `ClusterAnalysis` | `clusters` | `ClustersResult` |
| `CoarseGrainAnalysis` | `coarse_grain` | `CoarseGrainResult` |
| `CollapseAnalysis` | `data_collapse` | `DataCollapseResult` |
| `ComplexityAnalysis` | `complexity` | `ComplexityResult` |
| `CorrelationAnalysis` | `correlation` | `CorrelationResult` |
| `DefectAnalysis` | `defects` | `DefectsResult` |
| `DiscoveryAnalysis` | `applicable_analyses` | `ApplicableAnalysesResult` |
| `DisplacementAnalysis` | `displacement` | `DisplacementResult` |
| `DistributionAnalysis` | `distribution` | `DistributionResult` |
| `DomainAnalysis` | `domains` | `DomainsResult` |
| `ElementaryModeAnalysis` | `elementary_modes` | `ElementaryModesResult` |
| `EmbeddingAnalysis` | `embedding` | `EmbeddingResult` |
| `EmergenceAnalysis` | `emergence` | `EmergenceResult` |
| `EntropyAnalysis` | `entropy` | `EntropyResult` |
| `EpsilonMachineAnalysis` | `epsilon_machine` | `EpsilonMachineResult` |
| `ErgodicityAnalysis` | `ergodicity` | `ErgodicityResult` |
| `EventAnalysis` | `events` | `EventsResult` |
| `FitAnalysis` | `fit_law` | `FitLawResult` |
| `InformationAnalysis` | `information` | `InformationResult` |
| `InvariantAnalysis` | `invariants` | `InvariantsResult` |
| `IrreversibilityAnalysis` | `irreversibility` | `IrreversibilityResult` |
| `KramersMoyalAnalysis` | `kramers_moyal` | `KramersMoyalResult` |
| `LargeDeviationAnalysis` | `large_deviations` | `LargeDeviationsResult` |
| `LinearizationAnalysis` | `linearization` | `LinearizationResult` |
| `LocalInformationAnalysis` | `local_information` | `LocalInformationResult` |
| `LoopAnalysis` | `loops` | `LoopsResult` |
| `LyapunovAnalysis` | `lyapunov` | `LyapunovResult` |
| `MarkovAnalysis` | `markov` | `MarkovResult` |
| `ModeDecompositionAnalysis` | `mode_decomposition` | `ModeDecompositionResult` |
| `MotifAnalysis` | `motifs` | `MotifsResult` |
| `NeighborAnalysis` | `neighbors` | `NeighborsResult` |
| `NetworkAnalysis` | `network_measures` | `NetworkMeasuresResult` |
| `OperatorAnalysis` | `operator_measures` | `OperatorMeasuresResult` |
| `PartialInformationAnalysis` | `partial_information` | `PartialInformationResult` |
| `PassageAnalysis` | `passage` | `PassageResult` |
| `RecurrenceAnalysis` | `recurrence` | `RecurrenceResult` |
| `RenormalizationAnalysis` | `renormalization` | `RenormalizationResult` |
| `ReservoirAnalysis` | `reservoir` | `ReservoirResult` |
| `RollingAnalysis` | `rolling` | `RollingResult` |
| `ScalingAnalysis` | `scaling` | `ScalingResult` |
| `SectionAnalysis` | `section` | `SectionResult` |
| `ShapeAnalysis` | `chain_shape` | `ChainShapeResult` |
| `SignatureAnalysis` | `signatures` | `SignaturesResult` |
| `SparseIdentificationAnalysis` | `sparse_identification` | `SparseIdentificationResult` |
| `SpectrumAnalysis` | `spectrum` | `SpectrumResult` |
| `StabilityAnalysis` | `stability` | `StabilityResult` |
| `SurrogateTestAnalysis` | `surrogate_test` | `SurrogateTestResult` |
| `SymbolicAnalysis` | `symbolic` | `SymbolicResult` |
| `SensitivityAnalysis` (experiments) | `sensitivity` | `SensitivityResult` |

Field realizations, which call the core functions with fixed settings or replace system analyses that subclassed
`BaseAnalysis`:

| Old class | Subfield of `skcomplex.systems` | Function | Result |
|---|---|---|---|
| `EarlyWarningAnalysis` | `physics.climate` | `early_warning` | `EarlyWarningResult` |
| `CracklingNoiseAnalysis` | `physics.criticality` | `crackling_noise` | `CracklingNoiseResult` |
| `KineticTemperatureAnalysis` | `physics.statistical_mechanics` | `kinetic_temperature` | `KineticTemperatureResult` |
| `MaxwellBoltzmannAnalysis` | `physics.statistical_mechanics` | `maxwell_boltzmann` | `MaxwellBoltzmannResult` |
| `SusceptibilityAnalysis` | `physics.statistical_mechanics` | `susceptibility` | `SusceptibilityResult` |
| `GaugeAnalysis` | `physics.waves` | `gauge_observables` | `GaugeObservablesResult` |
| `FuzzyOilDropAnalysis` | `biology.systems_biology` | `fuzzy_oil_drop_measures` | `FuzzyOilDropMeasuresResult` |
| `LabellingAnalysis` | `biology.systems_biology` | `labelling` | `LabellingResult` |
| `FluxVariabilityAnalysis` | `biology.common.metabolism` | `flux_variability` | `FluxVariabilityResult` |
| `ProductionScanAnalysis` | `biology.common.metabolism` | `production_scan` | `ProductionScanResult` |
| `CobraAnalysis` | `biology.common.metabolism` | `cobra_measures` | `CobraMeasuresResult` |
| `StrainDesignAnalysis` | `biology.common.metabolism` | `strain_design` | `StrainDesignResult` |
| `CommunityAnalysis` | `biology.common.metabolism` | `community_growth` | `CommunityGrowthResult` |
| `NeutralNetworkAnalysis` | `biology.evolution` | `neutral_network` | `NeutralNetworkResult` |

## Experiment names

Every experiment class became a function of a system, in the role of the model selection functions of scikit-learn.
The system is the first argument: a `BaseSystem` instance, whose copies take model parameters through `set_params`, or
a factory of systems, such as a system class. A name with a dot is an initial state or a parameter of a part, which
keeps its value for the whole run. Every run goes through `simulate`, so the tuned run settings of the class fill what
a call leaves out. The former settings are keyword arguments, analyses are functions or `bind_analysis` results, and
the results are frozen values. Designs became the `design` argument of `sweep` and the `method` of `sensitivity`.
Calibration stays an estimator, in the role of `GridSearchCV`: it fits model parameters, and `run` sets `system_`.

| Old name | New name | Result |
|---|---|---|
| `Experiment(source, design, bounds)` | `sweep(system, parameters, design=...)` | `ExperimentResult` |
| `Experiment(source, replicates=n)` | `ensemble(system, replicates=n)` | `ExperimentResult` |
| `Experiment(source, scenarios=...)` | `compare_scenarios(system, scenarios)` | `ExperimentResult` |
| `GridDesign()` | `sweep(..., design='grid')`, the default | |
| `RowsDesign(rows)` | `sweep(system, rows)` | |
| `RandomDesign(n)` | `sweep(..., design='random', n=n)` | |
| `LatinHypercubeDesign(n)` | `sweep(..., design='latin_hypercube', n=n)` | |
| `SobolDesign(n)` | `sweep(..., design='sobol', n=n)` | |
| `OneAtATimeDesign()` | `sensitivity(..., method='oat')` | `SensitivityResult` |
| `MorrisDesign(n)` | `sensitivity(..., method='morris', n=n)` | `SensitivityResult` |
| `SaltelliDesign(n)` | `sensitivity(..., method='sobol', n=n)` | `SensitivityResult` |
| `sensitivity(result, ...)` | `sensitivity(system, bounds, output=...)` | `SensitivityResult` |
| `BasinStabilityExperiment` | `basin_stability` | `BasinStabilityResult` |
| `OrbitDiagramExperiment` | `orbit_diagram` | `OrbitDiagramResult` |
| `CalibrationExperiment` | `Calibration`, fitting model parameters | its `system_` |
| `ExperimentResult` | `ExperimentResult`, now a frozen value | |
| `BaseExperiment`, `BaseDesign` | removed, the shared runs are private | |

The tags each analysis needs are the module constant `TAGS` of its module, an `AnalysisTags` of `skcomplex.core.specs`
with the data kind it reads and the capability tags it needs, and `KIND_TAGS` for the kinds of a family that differ.
`applicable_analyses` reads them in place of its table, and an analysis checks them with `_check_tags` against a system,
or the system behind a trajectory, raising `TagError` naming the tag: `lyapunov` needs a deterministic system,
`network_measures`, `motifs`, and `renormalization` a network space, and `section` continuous time. `ensemble` does not
yet check a stochastic randomness, since the tags of many agent-based random systems say `deterministic`.

## Risks / Trade-offs

- **Size:** about fifty analyses, the experiments, the catalog, every gallery example, and most tests change. The work
  runs in waves by package, with the documentation last.
- **Tags too fine or too coarse:** tags must stay few and orthogonal. A new tag needs a function that reads it.
- **Functions with many arguments:** `simulate` takes several keyword arguments. That is the scikit-learn trade, and
  the tuned settings of a catalog entry keep the common case short.

## Decisions after the chemistry pilot

- **Listing imports only what it returns.** `all_systems` selects by field and topics statically and imports only the
  modules of the systems it returns, as `all_estimators` imports what it lists. The zero-import listing of `catalog()`
  goes with the registry in the removal wave.
- **Run settings through one hook.** `simulate` reads the tuned run settings through `BaseSystem.get_run_settings()`,
  which returns the class attributes `until`, `dt`, `solver`, `observables`, and `initial_state` by default. A system
  whose settings depend on its parameters or its built system, such as `active_gels`, `gauge_fields`, or
  `dynamic_fba`, overrides the hook. Class attributes stay the plain case.
- **`simulate` takes `dt` and `observers`**, added by the pilot, so tests choose the time between frames and gallery
  examples record runs.
- **Names.** A class name round-trips to its catalog name by the snake-case rule. Where it cannot, such as names with
  digits, the gallery example and the inventory key are renamed and reported. `label` is the last keyword-only
  argument with the default `'system'`.

## Inventory vocabulary from the tags

- **Computed at check time, never written back.** The time, the space, the laws, and the domains of an implemented
  entry of `docs/overview/inventory.toml` are no longer written by hand. `docs/inventory_checks.py` finds the class of
  every entry with `all_systems()`, by the snake-case name of the class without `System`, builds it with its default
  parameters, and derives its tags. It reports an implemented entry that writes one of these categories, and a class
  whose tags cannot be derived or whose law kinds are unknown. No maintenance script writes the generated values to
  the file, so there is nothing to keep in step.
- **One vocabulary.** The `vocabulary.time`, `vocabulary.space`, `vocabulary.laws`, and `vocabulary.domains` tables are
  removed. Pending and blocked entries, which have no class yet, write these categories with the values of the public
  `TIMES`, `SPACES`, `LAWS`, and `DOMAINS` of `skcomplex.core.specs`, and the check validates them against those. The
  finer hand-written terms, such as the laws `local-rule`, `transfer`, `event`, and `boundary`, or the domains `angle`,
  `tensor`, and `hilbert`, are dropped: the primitives of every entry, such as `local-rules`, `stocks-and-flows`,
  `events`, `boundaries`, `angles`, and `tensor-fields`, already record them.
- **Hand-written where tags say nothing.** `components` and `bindings` stay hand-written, with their vocabulary tables.
  Deriving them from the element and interaction classes of the built system is possible, but the kinds of group
  (variable, per site, per edge, per hyperedge) and of binding are not declared anywhere the tags read, so it is left
  for a later change.
- **Tags describe the default parameters.** A system whose stochastic or noisy mode is off by default, such as
  `SIRSystem`, has the tags of its deterministic form, where the hand-written laws listed every mode.

## General systems of the core

- **`MassActionSystem` in `skcomplex.core.components.mass_action`.** Species in compartments and reactions between
  them at mass-action rates: every species is a `CompartmentElement` named after it, every reaction a
  `ReactionInteraction` between the states `n` of the compartments it names. Its parameters are `species`, the initial
  amounts by name, `reactions`, a `ReactionTable` of `core.types` that maps the label of every reaction to its
  reactants and products, as coefficients by species name, and its rate constant, and `mode`, `'rate'`, `'jump'`, or
  `'langevin'`. It builds on the core's `CompartmentElement` and `ReactionInteraction`, so it adds no element or
  interaction. The new module sits above `system` in the layers of `skcomplex.core.components`.
- **Hooks for field realizations.** A subclass takes the parameters of its model under its own names, as
  `ReactionInteraction` subclasses do, calls `super().__init__(label=label)`, and gives its parts from `_get_species`,
  `_get_reactions`, and `_get_mode`, or gives its reactions as interactions from `_get_interactions` when they are
  `ReactionInteraction` subclasses with roles of their own names. The chemistry realizations give a reaction table, and
  the epidemic ones keep their named reactions, such as `InfectionInteraction`, which tests and doctests read.
- **Realizations.** `SIRSystem`, `SISSystem`, `SIRSSystem`, and `ZombieOutbreakSystem` of `biology.epidemiology`, and
  `MichaelisMentenSystem` and `ChemicalLangevinNetworkSystem` of `chemistry.kinetics`, subclass it. Their parts, tags,
  and run settings are those they built before, element by element and reaction by reaction.
- **Candidates left out.** Recorded here for a later change:
  - The predator-prey and growth systems of `biology.ecology`, such as `LotkaVolterraSystem`, use
    `PopulationElement` with rate interactions whose conversion efficiency `b` differs from the predation rate `a`, so
    they are not mass action with integer stoichiometry. Rewriting them as reactions would change their compiled laws.
  - Chemistry systems whose species are states of one reactor element, such as `BrusselatorODESystem`,
    `OregonatorSystem`, `CubicAutocatalysisSystem`, and the reaction-diffusion fields of `chemistry.pattern_formation`,
    share reactions but not compartments. A reaction-diffusion general system shared with `physics.pattern_formation`
    is a candidate once both build their fields the same way.
  - `NanoparticleCoronaSystem` of `chemistry.kinetics` builds compartments and reactions in a loop over proteins, and
    the reaction systems of `biology.oncology`, `biology.physiology`, and `biology.systems_biology` mix mass-action
    reactions with propensities of their own. They could become realizations once `MassActionSystem` accepts given
    propensities, which no second field needs today.
  - The duplicate scan of `consolidate-system-modules` found no other shared system structure: its cross-field
    duplicates are helpers (`_random_cells`, `_noise`, evenly spaced grids) and the `LorenzElement`, not systems.
