## MODIFIED Requirements

### Requirement: Shared realizations live in one cross-domain package
A realization that systems of two or more fields use SHALL live in a concept subpackage of `skcomplex.systems.general`,
with no loose module at its root. The package SHALL import only the public modules of the core and no field
package. Every field MAY import it, and the core SHALL import nothing from it. A realization used by one field only
SHALL live in that field. A class of `skcomplex.systems.general` SHALL be named by its mathematics or by a neutral
term, never by the term of one field.

#### Scenario: Import contracts
- **WHEN** the checks session runs the import contracts
- **THEN** no module of `skcomplex.systems.general` imports a field package, and no module of the core imports
  `skcomplex.systems.general`

#### Scenario: Users of a shared module
- **WHEN** the catalog tests read the importers of every module of `skcomplex.systems.general`
- **THEN** each module has importers in at least two fields

#### Scenario: Core features return plain composites
- **WHEN** a data-driven function of the core, such as `sparse_identification`, returns a fitted system
- **THEN** it is a `CompositeSystem` of core components, and the core imports nothing from `skcomplex.systems.general`

#### Scenario: No engine support in the package
- **WHEN** a reviewer searches the engine for the classes of `skcomplex.systems.general`
- **THEN** no compile pass, kernel, solver, or backend names any of them
