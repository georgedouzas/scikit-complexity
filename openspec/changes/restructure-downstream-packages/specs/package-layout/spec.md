## MODIFIED Requirements

### Requirement: Dependencies run from formalisms to engine to core
The top-level packages SHALL import downward in the order `experiments`, `analysis`, `engine`, `core`, so `analysis`
never imports `experiments`, `engine` imports neither, and `core` imports none of them. Within `core` and within
`engine`, each package SHALL import only the packages listed before it.

#### Scenario: Upward import
- **WHEN** a module of `skcomplex.engine.solvers` imports from `skcomplex.engine.backends`
- **THEN** the import contracts of the checks session fail and name the importing module and the imported one

#### Scenario: Core reaching the engine
- **WHEN** a module of `skcomplex.core.elements` imports from `skcomplex.engine.compile`
- **THEN** the import contracts of the checks session fail and name the module

#### Scenario: Analysis reaching experiments
- **WHEN** a module of `skcomplex.analysis` imports from `skcomplex.experiments`
- **THEN** the import contracts of the checks session fail and name the module

### Requirement: The package holds four subpackages
The package SHALL expose exactly five subpackages, which together hold every domain-agnostic package and the models:
- `skcomplex.core` holds what a system is: `base`, `specs`, `types`, `errors`, `spaces`, `laws`, `elements`,
  `interactions`, and `system`.
- `skcomplex.engine` holds what turns a system into a trajectory: `problems`, `compile`, `backends`, `solvers`,
  `runtime`, and `viz`.
- `skcomplex.analysis` holds what measures a trajectory, an array, or a system.
- `skcomplex.experiments` holds what runs many systems.
- `skcomplex.models` holds the domain models.

#### Scenario: Framework imports
- **WHEN** a user imports `Element` from `skcomplex.core.elements`, `Simulation` from `skcomplex.engine.runtime`,
  `RK4Solver` from `skcomplex.engine.solvers`, `EntropyAnalysis` from `skcomplex.analysis`, and `Experiment` from
  `skcomplex.experiments`
- **THEN** every import succeeds

## ADDED Requirements

### Requirement: The engine surface is its pipeline
Every package of `skcomplex.engine` SHALL re-export only the classes of the pipeline from a system to a trajectory, the
bases a user subclasses, and the values a client reads, and SHALL keep every helper of one pass, solver, or backend
private to that class.

#### Scenario: Engine surface listing
- **WHEN** a user lists the public names of `skcomplex.engine.compile`
- **THEN** the list holds `compile_system`, `validate`, the base of passes, and the values a client reads, and none of
  the helpers of a single pass, such as `values_of`, `member_of`, or `hilbert_of`

#### Scenario: A user solver
- **WHEN** a user subclasses the solver base, implements its hooks, and passes an instance to a simulation
- **THEN** the simulation runs with that solver
