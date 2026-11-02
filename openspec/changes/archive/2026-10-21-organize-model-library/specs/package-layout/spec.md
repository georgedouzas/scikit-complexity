# Spec Delta

## Purpose

Separates the domain-agnostic framework, the core, from the domain model library, so every model is built only from
the general interface and the interface never depends on a model.

## ADDED Requirements

### Requirement: The core holds every domain-agnostic package
The package SHALL expose exactly two subpackages: `skcomplex.core`, holding every domain-agnostic package, and
`skcomplex.models`, holding the domain models. The core's packages SHALL be `base`, `model`, `compile`, `solvers`,
`runtime`, `viz`, `experiments`, `analysis`, and `backends`.

#### Scenario: Framework imports
- **WHEN** a user imports `Element` from `skcomplex.core.model`, `Simulation` from `skcomplex.core.runtime`, `RK4`
  from `skcomplex.core.solvers`, and `Domain` from `skcomplex.core.base`
- **THEN** every import succeeds

#### Scenario: Old framework paths are gone
- **WHEN** a user imports `skcomplex.model`, `skcomplex.solvers`, or `skcomplex.runtime`
- **THEN** the import fails

### Requirement: The core never depends on models
No module of `skcomplex.core` SHALL import `skcomplex.models`, and every core feature SHALL work without the model
library imported.

#### Scenario: Core without models
- **WHEN** a user builds, compiles, simulates, and analyses a system written from scratch, and `skcomplex.models` was
  never imported
- **THEN** every step works and `skcomplex.models` stays out of the loaded modules

#### Scenario: Planted reverse import
- **WHEN** a core module imports from `skcomplex.models`
- **THEN** the layout check fails and names the module

### Requirement: Models use only the public core
Modules of `skcomplex.models` SHALL import only public names of `skcomplex.core` packages and third-party libraries,
never private modules or names.

#### Scenario: Private core import
- **WHEN** a model module imports a private name, such as `skcomplex.core.compile._compiler`
- **THEN** the layout check fails and names the module and the import

### Requirement: Model packages are self-contained
Every subfield package of `skcomplex.models` SHALL be self-contained: it SHALL NOT import any other model package, of
its own field or another. A component that more than one model package needs SHALL be domain-agnostic and live in the
core.

#### Scenario: Import across model packages
- **WHEN** a module of `skcomplex.models.biology.collective_behaviour` imports from
  `skcomplex.models.physics.mechanics`, or from `skcomplex.models.biology.ecology`
- **THEN** the layout check fails and names the module and the import

#### Scenario: Removing a model package
- **WHEN** any one subfield package is deleted from the library
- **THEN** every other model package still imports, and its models still pass the conformance suite
