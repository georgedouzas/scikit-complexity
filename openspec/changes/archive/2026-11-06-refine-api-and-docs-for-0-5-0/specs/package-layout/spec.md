# Spec Delta

## ADDED Requirements

### Requirement: The library calls its ready-made items systems
Every identifier, path, documentation page, and sentence of the library SHALL call a ready-made item of the catalog a
system, never a model, including scientific names such as "the SIR system". Identifiers of other libraries, such as
`mesa.Model`, and the titles of cited references SHALL keep their own wording.

#### Scenario: Catalog identifiers
- **WHEN** a user lists the public names of `skcomplex.systems`
- **THEN** none of them contains `model` or `Model`, and the registration function is `register_system`

#### Scenario: Prose of the documentation
- **WHEN** a reviewer searches the README, the user guide, and the docstrings for the word model
- **THEN** it appears only in identifiers of other libraries and in the titles of cited references

## RENAMED Requirements

- FROM: `### Requirement: The package holds four subpackages`
- TO: `### Requirement: The package holds two subpackages`

## MODIFIED Requirements

### Requirement: The package holds two subpackages
The package SHALL expose exactly two subpackages, `skcomplex.core` and `skcomplex.systems`, and no module beside them:
- `skcomplex.core.base`, `skcomplex.core.types`, `skcomplex.core.specs`, and `skcomplex.core.errors` hold the root
  class `BaseConfig`, the type aliases, the specs, and the errors.
- `skcomplex.core.components` holds what a system is: `base`, `spaces`, `laws`, `elements`, `interactions`, `system`,
  and the types and specs that name them.
- `skcomplex.core.engine` holds what turns a system into a trajectory: `problems`, `compile`, `backends`, `solvers`,
  `runtime`, `translators`, and `viz`.
- `skcomplex.core.analysis` holds what measures a trajectory, an array, or a system.
- `skcomplex.core.experiments` holds what runs many systems.
- `skcomplex.systems` holds the catalog of ready-made systems.

#### Scenario: Framework imports
- **WHEN** a user imports `Element` from `skcomplex.core.components.elements`, `Simulation` from
  `skcomplex.core.engine.runtime`, `RK4Solver` from `skcomplex.core.engine.solvers`, `EntropyAnalysis` from
  `skcomplex.core.analysis`, and `Experiment` from `skcomplex.core.experiments`
- **THEN** every import succeeds

#### Scenario: Old paths
- **WHEN** a user imports `skcomplex.engine`, `skcomplex.analysis`, `skcomplex.errors`, or `skcomplex.models`
- **THEN** the import fails, since the old paths keep no alias

### Requirement: Dependencies run from formalisms to engine to core
The packages of `skcomplex.core` SHALL import downward in the order `experiments`, `analysis`, `engine`, `components`,
and the modules `base`, `types`, `specs`, and `errors` of `skcomplex.core` below them, so `analysis` never imports
`experiments`, `engine` imports neither, `components` imports none of them, and those modules import no package.
Within `components` and within `engine`, each package SHALL import only the packages listed before it.

#### Scenario: Upward import
- **WHEN** a module of `skcomplex.core.engine.backends` imports from `skcomplex.core.engine.solvers`
- **THEN** the import contracts of the checks session fail and name the importing module and the imported one

#### Scenario: Core reaching the engine
- **WHEN** a module of `skcomplex.core.components.elements` imports from `skcomplex.core.engine.compile`
- **THEN** the import contracts of the checks session fail and name the module

#### Scenario: Analysis reaching experiments
- **WHEN** a module of `skcomplex.core.analysis` imports from `skcomplex.core.experiments`
- **THEN** the import contracts of the checks session fail and name the module

### Requirement: The core never depends on models
No module of `skcomplex.core` SHALL import `skcomplex.systems`, and every feature of the core SHALL work without the
catalog imported.

#### Scenario: Core without models
- **WHEN** a user builds, compiles, simulates, and analyses a system written from scratch, and `skcomplex.systems` was
  never imported
- **THEN** every step works and `skcomplex.systems` stays out of the loaded modules

#### Scenario: Planted reverse import
- **WHEN** a module of the core imports from `skcomplex.systems`
- **THEN** the import contracts of the checks session fail and name the module

### Requirement: Models use only the public core
Modules of `skcomplex.systems` SHALL import only public names of the packages of `skcomplex.core` and third-party
libraries, never private modules or names.

#### Scenario: Private core import
- **WHEN** a module of `skcomplex.systems` imports a private module, such as `skcomplex.core.engine.compile._compile`
- **THEN** the import contracts of the checks session fail and name the module and the import

### Requirement: Model packages are self-contained
Every subfield package of `skcomplex.systems` SHALL be self-contained: it SHALL NOT import any other subfield package,
of its own field or another, except the shared realizations of its own field in
`skcomplex.systems.<field>.realizations`. A component that more than one field needs SHALL be domain-agnostic and live
in the core.

#### Scenario: Import across model packages
- **WHEN** a module of `skcomplex.systems.biology.collective_behaviour` imports from
  `skcomplex.systems.physics.mechanics`, or from `skcomplex.systems.biology.ecology`
- **THEN** the import contracts of the checks session fail and name the module and the import

#### Scenario: Realizations of the own field
- **WHEN** a module of `skcomplex.systems.physics.mechanics` imports from `skcomplex.systems.physics.realizations`
- **THEN** the import contracts of the checks session pass

#### Scenario: Removing a model package
- **WHEN** any one subfield package is deleted from the library
- **THEN** every other subfield package still imports, and its systems still pass the conformance suite

### Requirement: The test tree mirrors the source tree
Every test module SHALL test one module of `skcomplex` and SHALL live at the path that mirrors it, so
`tests/core/<package>/test_<module>.py` tests `skcomplex.core.<package>._<module>`, and
`tests/systems/<field>/<subfield>/test_<module>.py` tests the module of that subfield package. No test module SHALL
check repository conventions, documentation tooling, or documentation data.

#### Scenario: Test without a mirrored module
- **WHEN** the test modules under `tests` are listed
- **THEN** each one names a module of `skcomplex` at the mirrored path, and none lies outside `tests/core` and
  `tests/systems`

### Requirement: What every package shares lives in root modules
A module used by every package of the core SHALL live at the root of `skcomplex.core`, not in one package:
`skcomplex.core.base` SHALL define `BaseConfig`, the root of every configurable class, `skcomplex.core.errors` SHALL
define every error and warning, `skcomplex.core.specs` SHALL define the shared specs, and `skcomplex.core.types` SHALL
define every type alias built from Python, NumPy, SciPy, and Sage types, one alias per type, such as `Array` for every
float array. An alias that names a class of the library SHALL stay in the module of that class, and the catalog SHALL
keep its own aliases.

#### Scenario: Root imports
- **WHEN** a user imports `BaseConfig` from `skcomplex.core.base`, `ValidationError` from `skcomplex.core.errors`,
  `StateSpec` from `skcomplex.core.specs`, and `Array` from `skcomplex.core.types`
- **THEN** every import succeeds and none of these modules imports a package

#### Scenario: One alias per type
- **WHEN** a module of the components, the engine, analysis, or experiments needs a float array
- **THEN** it imports `Array` from `skcomplex.core.types` instead of defining its own alias

### Requirement: The engine surface is its pipeline
Every package of `skcomplex.core.engine` SHALL re-export only classes: the bases a developer subclasses, the classes of
the pipeline from a system to a trajectory, and the frozen values a client reads, with no public function. Every helper
of one class SHALL be a private method of that class.

#### Scenario: Engine surface listing
- **WHEN** a user lists the public names of every package of `skcomplex.core.engine`
- **THEN** every name is a class derived from a base of its package, and none is `compile_system`, `validate`, or a
  helper of a single pass, such as `values_of`, `member_of`, or `hilbert_of`

#### Scenario: A user solver
- **WHEN** a user subclasses the solver base, implements its hooks, and passes an instance to a simulation
- **THEN** the simulation runs with that solver
