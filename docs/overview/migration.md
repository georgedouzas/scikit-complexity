# Migrating from 0.6 to 0.7

Move code written for 0.6 to 0.7, name by name. Version 0.7 aligns the library with scikit-learn: what you define is an
estimator, and what you do to it is a function. It is one breaking release, and the removed names have no aliases.

In short:

- every ready-made system is a class, such as `SIRSystem`, in place of a factory function and a catalog name,
- `all_systems` finds the classes, in place of `catalog`, `describe`, and the registry,
- `simulate(system, ...)` runs a system and returns its `Trajectory`, in place of `Simulation` and `create_simulation`,
- every analysis is a function that returns a frozen result, in place of an analysis class,
- every experiment is a function of a system, except `Calibration`, which stays an estimator,
- the compilers and most problem classes are private,
- every part has capability tags, from `get_tags()`.

## Ready-made systems

A ready-made system was a factory function registered under a catalog name. It is now a class named after the system
with the suffix `System`, whose constructor takes the arguments of the factory with the same defaults, then `label` as
the last keyword-only argument.

Before:

```python
from skcomplex.systems import create_system, describe

system = create_system('sir', beta=0.0004)
entry = describe('sir')
entry.description, entry.tags, entry.references, entry.defaults
```

After:

```python
from skcomplex.systems import all_systems

SIRSystem = dict(all_systems(field='biology/epidemiology'))['SIRSystem']
system = SIRSystem(beta=0.0004)
SIRSystem.__doc__.splitlines()[0], SIRSystem.topics, SIRSystem.references, system.get_run_settings()
```

A class can also be imported from its subfield package, where the factory was, such as `SIRSystem` from the package of
`biology/epidemiology` in place of `sir`.

The catalog name of a system is the snake case of its class without `System`, as the gallery and the inventory use it.
For most systems the name and the factory stay as they were, `lotka_volterra` and `LotkaVolterraSystem`. These names
changed, because their class names do not round-trip:

| Name in 0.6 | Name in 0.7 | Class |
| --- | --- | --- |
| `boids_3d` | `boids3d` | `Boids3dSystem` |
| `carbon_13_mfa` | `carbon13_mfa` | `Carbon13MFASystem` |
| `intelligent_driver_system` | `intelligent_driver` | `IntelligentDriverSystem` |
| `lorenz_84` | `lorenz84` | `Lorenz84System` |
| `lorenz_96` | `lorenz96` | `Lorenz96System` |
| `random_energy_system` | `random_energy` | `RandomEnergySystem` |
| `trap_system` | `trap` | `TrapSystem` |
| `voter_system` | `voter` | `VoterSystem` |

### The catalog

| 0.6 | 0.7 |
| --- | --- |
| `catalog()` | `all_systems()`, which returns `(name, class)` pairs |
| `catalog(field='biology/ecology')` | `all_systems(field='biology/ecology')` |
| `catalog(tag='oscillation')` | `all_systems(topics='oscillation')` |
| | `all_systems(tags={'randomness': 'stochastic'})`, by capability tags |
| `describe(name)` | the class, its attributes, and its methods |
| `create_system(name, **parameters)` | `SystemClass(**parameters)` |
| `create_simulation(name, **parameters)` | `simulate(SystemClass(**parameters))` |
| `register_system(...)` | a class attribute `field`, `topics`, and `references`, read from the source by `all_systems` |
| `SYSTEMS`, `SystemCatalog` | `all_systems()` |
| `SystemEntry`, `SystemListing` | the class itself |
| `skcomplex.systems.types` | removed |
| `SystemListing.tags`, `SystemEntry.tags` | the class attribute `topics` |
| `SystemEntry.description` | the first line of the docstring of the class |
| `SystemEntry.factory` | the class |
| `SystemEntry.parameters` | `get_params()` |
| `SystemEntry.defaults`, `settings`, `initial_state`, `observables` | `get_run_settings()`, from class attributes such as `until` and `dt` |
| `SystemEntry.formalism` | `get_tags()` |
| `create_system(name).set_params(label=...)` | `SystemClass(label=...)` |

A system class builds its parts in `prepare`, and keeps them as `system_`, the `System` that the factory returned:

```python
SIRSystem().prepare().system_.elements
```

## Simulations

`Simulation` is private. `simulate` takes the system and the run settings as keyword arguments, runs from time 0 to
`until`, and returns the trajectory. An argument left out comes from the tuned run settings of the system class.

Before:

```python
from skcomplex.core.engine.runtime import Simulation
from skcomplex.systems import create_simulation, create_system

trajectory = create_simulation('sir').prepare().run().trajectory_
simulation = Simulation(create_system('lotka_volterra'), dt=0.1).prepare()
trajectory = simulation.run(until=50).trajectory_
```

After:

```python
from skcomplex.core.engine.runtime import simulate

LotkaVolterraSystem = dict(all_systems(field='biology/ecology'))['LotkaVolterraSystem']
trajectory = simulate(SIRSystem())
trajectory = simulate(LotkaVolterraSystem(), until=50, dt=0.1)
```

| 0.6 | 0.7 |
| --- | --- |
| `Simulation(system, **settings).prepare().run(until).trajectory_` | `simulate(system, until=until, **settings)` |
| `Simulation(..., solver=..., backend=..., dt=..., observers=..., schedule=..., observables=..., invariants=..., initial_state=..., random_state=...)` | the keyword arguments of `simulate` with the same names |
| `Simulation(..., compiler=...)` | `simulate(..., compiler=...)`, for a compiler on `BaseCompiler` that a system package defines |
| `simulation.set_params(...)` before a run | the arguments of the next `simulate` call |
| `run(until=...)` | `simulate(..., until=...)` |
| `update_parameters({path: value})` | `simulate(..., schedule={path: function_of_time})` |
| `extend(until)` | `simulate(..., initial_state=...)` from the last frame of the previous trajectory |
| `Simulation(..., clock=...)`, `start`, `pause`, `resume`, `stop`, `wait` | no public replacement, runs go as fast as possible |
| `problem_`, `stepper_`, `time_` | no public replacement, the trajectory holds `times` and the values |
| `clone(simulation)` | a new `simulate` call, since `simulate` never changes the system |

Validation errors that the compiler reported through `PassCompiler().prepare(system).report_` are raised by `simulate`,
such as a `RoleElementError` that names the role, the state, and the fix. A solver or a backend that does not support
the tags of a system raises a `TagError` before anything compiles.

## Compilers and problems

`simulate` chooses the compiler of the form of a system, the way the tags choose the default solver, so the compilers
of the library are private.

| 0.6 | 0.7 |
| --- | --- |
| `PassCompiler`, `OperatorCompiler`, `RewritingCompiler` | private, chosen by `simulate` |
| `Compilation`, `FlattenPass`, `LayoutPass`, `ProblemPass`, `ObservableEvaluator` | private |
| `BaseCompiler`, `BasePass`, `BaseLowering`, `HilbertSpaceError` | public, for a system package that brings a form of its own through the `compiler` run setting |
| `BaseProblemSpec`, `ProblemSpec`, `FrameSpec`, `BaseKernel`, `BaseSearch`, `BasePrinter` | public, unchanged |
| every other name of `skcomplex.core.engine.problems`, such as `OperatorProblemSpec`, `RewritingProblemSpec`, `KineticProblemSpec`, `LearningProblemSpec`, `LinearProgramSpec`, `PartitionSpec`, and their kernels | private |

## Analyses

Every analysis class is a function named after it in snake case without `Analysis`, with a clearer name where snake
case reads badly. The source is the first argument, the former constructor settings are keyword-only with the same
defaults, and the function returns a frozen result whose fields are the former fitted attributes without the trailing
underscore. A first positional setting, such as the kind, is now the keyword argument `kind`, `law` for `fit_law`, `variables`
for `mode_decomposition`, `sparse_identification`, and `reservoir`, `shape` for `coarse_grain`, and `analysis` and
`attribute` for `surrogate_test`.

Before:

```python
from skcomplex.core.analysis import EntropyAnalysis, SymbolicAnalysis

SymbolicAnalysis().prepare(system).run().equations_
EntropyAnalysis('permutation', dimension=4).prepare(values).run().entropy_
```

After:

```python
import numpy as np

from skcomplex.core.analysis import entropy, symbolic

values = np.sin(np.arange(500) * 0.1)
symbolic(SIRSystem()).equations
entropy(values, kind='permutation', dimension=4).entropy
```

`BaseAnalysis` is removed. An analysis given to an experiment was an unprepared instance, and is now a function or the
result of `bind_analysis(function, **settings)`, such as `bind_analysis(entropy, kind='permutation')`. An experiment
names its columns `<function name>.<field>`, such as `entropy.entropy`.

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

Field realizations moved from analysis classes of the system packages to functions of the same subfields:

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

## Experiments

Every experiment class is a function of a system, in the role of the model selection functions of scikit-learn. The
system is the first argument, a system instance or a factory of systems such as a system class, and the results are
frozen values.

Before:

```python
from skcomplex.core.experiments import CalibrationExperiment, Experiment, GridDesign
from skcomplex.systems import describe

sir = describe('sir').factory
result = Experiment(sir, GridDesign(), {'gamma': [0.1, 0.2]}, until=60).prepare().run().result_
fit = CalibrationExperiment(sir, data, bounds).prepare().run()
fit.estimates_
```

After:

```python
from skcomplex.core.experiments import Calibration, sweep

result = sweep(SIRSystem(), {'gamma': [0.1, 0.2]}, until=60, outputs=['I.n'])
times = np.arange(0.0, 60.0, 5.0)
data = {'I.n': (times, simulate(SIRSystem(), until=60).values('I.n')[::5][: len(times)])}
fit = Calibration(SIRSystem(), data, {'gamma': (0.05, 0.2)}).prepare().run()
fit.estimates_, fit.system_.gamma
```

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

## Tags

Every part has capability tags, a frozen `Tags` of `skcomplex.core.specs` from `get_tags()`: its time, its domains,
its space, the kinds of its laws, its randomness, and its structure. The tags of a system are derived from its parts.
They replace the hand-written vocabulary of time, space, laws, and domains in the inventory, and the compatibility
page, which the generated pages under Parts replace. `simulate`, the analyses, and the experiments read them and raise
a `TagError` that names a tag they cannot use.

```python
SIRSystem().get_tags()
```
