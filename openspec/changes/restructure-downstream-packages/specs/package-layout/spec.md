## MODIFIED Requirements

### Requirement: Dependencies run from formalisms to engine to core
The top-level packages SHALL import downward in the order `experiments`, `analysis`, `engine`, `core`, so `analysis`
never imports `experiments`, `engine` imports neither, and `core` imports none of them. Within `core` and within
`engine`, each package SHALL import only the packages listed before it.

#### Scenario: Upward import
- **WHEN** a module of `skcomplex.engine.backends` imports from `skcomplex.engine.solvers`
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

### Requirement: Library packages hold general operations
The core, the engine, analysis, and experiments SHALL hold general operations named mathematically: every public class
SHALL name what it computes or simulates, and its choices SHALL name variants of that operation. A name taken from one
application field SHALL appear only in the models library, as a realization that subclasses the general class in that
field's package and fixes its settings.

#### Scenario: Field name in a library package
- **WHEN** a reviewer lists the public classes and the choices of the core, the engine, analysis, and experiments
- **THEN** none is named after a quantity of one application field, such as the Binder cumulant, the polar order, or
  the segregation index

#### Scenario: Field measure in the models library
- **WHEN** a physics model reports the Binder cumulant of its magnetisation
- **THEN** it uses a realization of the physics field that subclasses the general distribution analysis and only fixes
  its settings


### Requirement: The engine surface is its pipeline
Every package of `skcomplex.engine` SHALL re-export only classes: the bases a developer subclasses, the classes of the
pipeline from a system to a trajectory, and the frozen values a client reads, with `compile_system` the only public
function. Every helper of one class SHALL be a private method of that class.

#### Scenario: Engine surface listing
- **WHEN** a user lists the public names of every package of `skcomplex.engine`
- **THEN** every name is a class derived from a base of its package, or `compile_system`, and none is `validate` or a
  helper of a single pass, such as `values_of`, `member_of`, or `hilbert_of`

#### Scenario: A user solver
- **WHEN** a user subclasses the solver base, implements its hooks, and passes an instance to a simulation
- **THEN** the simulation runs with that solver

### Requirement: Every engine class follows the common API
Every class of the engine SHALL derive from exactly one base of its package, SHALL store only its settings in
`__init__`, SHALL validate them in `check()`, SHALL take the one object it serves in `prepare(target)` and set the
attributes that end in `_`, and SHALL take the one input it acts on in `run(inputs)`. A base SHALL take no argument in
its verbs that its peers do not take, and a variation between subclasses SHALL be a setting.

#### Scenario: Compiling a system
- **WHEN** a user builds the system compiler, checks it, prepares it on a system, and runs it
- **THEN** `prepare` sets `report_` with the issues of the system, and `run` returns the compiled problem or raises the
  report's errors

#### Scenario: A user kernel
- **WHEN** a developer subclasses the kernel base for a new law kind, with the pass that builds it and a solver that
  runs it
- **THEN** the kernel is prepared on the problem and run on the state like every built-in kernel

### Requirement: Engine dataclasses are frozen values
Every dataclass of the engine SHALL be a frozen value with `check()` and plain readers only, and a class that acts SHALL
be a class of the engine's tree instead.

#### Scenario: No acting dataclass
- **WHEN** a reviewer lists the dataclasses of the engine
- **THEN** every one is frozen and has no method besides `check()` and plain readers
