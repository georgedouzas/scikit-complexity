# package-layout Specification

## Purpose
Separates the domain-agnostic framework, the core, from the domain model library, so every model is built only from
the general interface and the interface never depends on a model.

## Requirements

### Requirement: The core holds every domain-agnostic package
The package SHALL expose exactly two subpackages: `skcomplex.core`, holding every domain-agnostic package, and
`skcomplex.models`, holding the domain models. The core's packages SHALL be `base`, `space`, `laws`, `components`,
`interactions`, `system`, `compile`, `solvers`, `backends`, `runtime`, `viz`, `experiments`, `analysis`, and
`formalisms`. Each package SHALL import only the packages listed before it.

#### Scenario: Framework imports
- **WHEN** a user imports `Element` from `skcomplex.core.components`, `Simulation` from `skcomplex.core.runtime`, `RK4`
  from `skcomplex.core.solvers`, and `Domain` from `skcomplex.core.base`
- **THEN** every import succeeds

#### Scenario: Old framework paths are gone
- **WHEN** a user imports `skcomplex.model`, `skcomplex.core.model`, `skcomplex.solvers`, or `skcomplex.runtime`
- **THEN** the import fails

#### Scenario: Upward import
- **WHEN** a module of `skcomplex.core.solvers` imports from `skcomplex.core.backends`
- **THEN** the import contracts of the checks session fail and name the importing module and the imported one

### Requirement: The core never depends on models
No module of `skcomplex.core` SHALL import `skcomplex.models`, and every core feature SHALL work without the model
library imported.

#### Scenario: Core without models
- **WHEN** a user builds, compiles, simulates, and analyses a system written from scratch, and `skcomplex.models` was
  never imported
- **THEN** every step works and `skcomplex.models` stays out of the loaded modules

#### Scenario: Planted reverse import
- **WHEN** a core module imports from `skcomplex.models`
- **THEN** the import contracts of the checks session fail and name the module

### Requirement: Models use only the public core
Modules of `skcomplex.models` SHALL import only public names of `skcomplex.core` packages and third-party libraries,
never private modules or names.

#### Scenario: Private core import
- **WHEN** a model module imports a private module, such as `skcomplex.core.compile._compile`
- **THEN** the import contracts of the checks session fail and name the module and the import

### Requirement: Model packages are self-contained
Every subfield package of `skcomplex.models` SHALL be self-contained: it SHALL NOT import any other model package, of
its own field or another, except the shared realizations of its own field in `skcomplex.models.<field>.realizations`.
A component that more than one field needs SHALL be domain-agnostic and live in the core.

#### Scenario: Import across model packages
- **WHEN** a module of `skcomplex.models.biology.collective_behaviour` imports from
  `skcomplex.models.physics.mechanics`, or from `skcomplex.models.biology.ecology`
- **THEN** the import contracts of the checks session fail and name the module and the import

#### Scenario: Realizations of the own field
- **WHEN** a module of `skcomplex.models.physics.mechanics` imports from `skcomplex.models.physics.realizations`
- **THEN** the import contracts of the checks session pass

#### Scenario: Removing a model package
- **WHEN** any one subfield package is deleted from the library
- **THEN** every other model package still imports, and its models still pass the conformance suite

### Requirement: The test tree mirrors the source tree
Every test module SHALL test one module of `skcomplex` and SHALL live at the path that mirrors it, so
`tests/core/<package>/test_<module>.py` tests `skcomplex.core.<package>._<module>`, and
`tests/models/<field>/<subfield>/test_<module>.py` tests the module of that model package. No test module SHALL check
repository conventions, documentation tooling, or documentation data.

#### Scenario: Test without a mirrored module
- **WHEN** the test modules under `tests` are listed
- **THEN** each one names a module of `skcomplex` at the mirrored path, and none lies outside `tests/core` and
  `tests/models`

### Requirement: The test session runs the examples of docstrings and documents
The test session SHALL run every docstring example of `skcomplex` through pytest's doctest collection, and every Python
block of `README.md` and the user guide through `pytest-markdown-docs`, with no test module for either. A block that
builds on earlier blocks of its document SHALL be marked as their continuation.

#### Scenario: Failing docstring example
- **WHEN** a docstring example of a core module prints a value other than the one it shows
- **THEN** the test session fails and names the module and the example

#### Scenario: Failing user guide block
- **WHEN** a Python block of the user guide raises an error
- **THEN** the test session fails and names the document and the block
