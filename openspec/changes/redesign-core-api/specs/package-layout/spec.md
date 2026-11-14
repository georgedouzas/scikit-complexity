## ADDED Requirements

### Requirement: Concrete systems consume the core
The concrete systems of a field SHALL be built from the public core and from the common package of their field, and
SHALL live in the systems package of that field. The core SHALL hold no concrete system.

#### Scenario: A falling particle
- **WHEN** a contributor adds a free-fall system
- **THEN** it lives in `skcomplex.systems.physics.mechanics`, built from the point mass and gravity of
  `skcomplex.systems.physics.common.particles` and from the core `System`

### Requirement: Field-specific components live in their field
A component that systems of one field use SHALL live in `skcomplex.systems.<field>.common.<family>`, and a component that
several fields share as a recipe SHALL live in `skcomplex.systems.general.<concept>`. Moving a component out of the core
SHALL keep its behaviour and its tests, and the tests SHALL move to mirror its new place.

#### Scenario: A quantum law
- **WHEN** a reviewer looks for the Lindblad dissipator
- **THEN** it lives in `skcomplex.systems.physics.common.quantum` with the solver that lowers it, and no module of the
  core names it

### Requirement: Engine records are private
A record that only the engine reads SHALL be a private dataclass of the package that defines it, and no record SHALL
be public only because another package imports it.

#### Scenario: Subsystem record
- **WHEN** the operator compiler and the operator lowering name the subsystem of a tensor product
- **THEN** both use one private record, and `skcomplex.core.components.specs` no longer exists as a public package

### Requirement: Components are placed by one ordered test
A structural base, a law kind, an operator generated from one function, or a space that systems of several fields use
SHALL go to the core under a mathematical name. Any other component SHALL go to `skcomplex.systems.general.<concept>`
for several fields, to `systems/<field>/common/<family>` for one field, or to its system's module for one system.
Analyses follow the same test by their users. The core SHALL never import `skcomplex.systems`.

#### Scenario: A law kind
- **WHEN** a contributor asks where `Jump` lives
- **THEN** it lives in the core, since reactions, epidemics, and births and deaths of several fields use it

#### Scenario: A recipe of several fields
- **WHEN** a contributor adds an agent rule that systems of economics and social use
- **THEN** it goes to a concept subpackage of `skcomplex.systems.general`, built on `Interaction` and `Update`

#### Scenario: Source tests guard the rule
- **WHEN** the source tests run
- **THEN** no module under `src/skcomplex/core` imports `skcomplex.systems`, and every public class of
  `skcomplex.core.components` is a base, a law kind, a generated interaction, or a space used by systems of several
  fields

#### Scenario: One text of the rule
- **WHEN** a reviewer reads `openspec/constitution.md`, `CONTRIBUTING.md`, and the extension page of the user guide
- **THEN** each states the same ordered test as this requirement

## MODIFIED Requirements

### Requirement: Estimators for what is defined, functions for what is done
The public API SHALL hold systems and engine classes as estimators, elements, interactions, spaces, and laws as their
configurable parts and values, and public functions for analysing and running experiments. Simulating SHALL be the
method `simulate` of a prepared system, and no free function `simulate` SHALL exist.

#### Scenario: Public surface
- **WHEN** a test lists the public names of `skcomplex.core.engine`
- **THEN** it finds solvers, backends, observers, and `Trajectory`, and no `simulate`

#### Scenario: A system package brings its own form
- **WHEN** a field package defines a solver or a compiler on the public engine bases for a law of its own
- **THEN** a system of that field prepares with it, and the engine core names no class of the field

## REMOVED Requirements

### Requirement: Compile records shared across engine packages are public
**Reason**: Replaced by "Engine records are private". Records that only the engine reads do not belong to the public
API.
**Migration**: None for users. Engine modules import the private records.

### Requirement: Every component has one place by the placement rule
**Reason**: Replaced by "Components are placed by one ordered test", which adds the law kinds and the spaces of several
fields to the core and drops the steps for primitives the engine keyed on.
**Migration**: Place new components by the new test.
