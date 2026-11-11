## MODIFIED Requirements

### Requirement: Every component has one place by the placement rule
Every new component SHALL be placed by one ordered test. A structural base SHALL go to the core. A component the
engine handles by type, or an operator generated from one function, SHALL go to the core under a mathematical name.
Any other component is a recipe, placed by its users: `skcomplex.systems.general.<concept>` for several fields,
`systems/<field>/common/<family>` for one field, its system's module for one system. Every analysis SHALL be placed by
its users too: an analysis that measures what systems of several fields share SHALL go to `skcomplex.core.analysis`,
and an analysis of one field SHALL go to `systems/<field>/common/<family>`. The core SHALL never import
`skcomplex.systems`.

#### Scenario: Structural base
- **WHEN** a contributor adds a base where laws run once per triangle of a cell complex
- **THEN** it goes to the core, and the source tests accept it as a class that another public core class subclasses

#### Scenario: Handled by type
- **WHEN** a contributor adds a law that a new kernel lowers
- **THEN** it goes to the core under a mathematical name, and the source tests accept it because the engine names it

#### Scenario: Recipe of several fields
- **WHEN** a contributor adds an agent rule that systems of economics and social use
- **THEN** it goes to a concept subpackage of `skcomplex.systems.general`, and the source tests reject it in the core,
  since no engine code names it and it returns only `RuleLaw`

#### Scenario: Missing primitive
- **WHEN** a recipe needs an operation the engine cannot lower
- **THEN** the operation enters the core as a new primitive by the second step, and the recipe keeps no engine code

#### Scenario: Engine support keeps its place
- **WHEN** the expression printer of the engine reads the dialects of `DialectSpec`
- **THEN** `DialectSpec` stays in `core/specs/_dialect_spec.py` as engine support, and `skcomplex.core.specs` does not
  re-export it, since nothing outside the package reads it

#### Scenario: Analysis of several fields
- **WHEN** a contributor adds an analysis of the winding defects of a field of angles, which liquid crystals, active
  nematics, and spiral waves of heart tissue share
- **THEN** it goes to `skcomplex.core.analysis`

#### Scenario: Analysis of one field
- **WHEN** a contributor adds an analysis of the entanglement of the subsystems of a quantum state
- **THEN** it goes to `systems/physics/common/quantum`, and it imports only the public modules of the core

#### Scenario: Source tests guard the rule
- **WHEN** the source tests run
- **THEN** every public class of `skcomplex.core.components` is a structural base, is named in the code of another core
  module, or is the interaction with the fewest law classes among those of its base that list one of its laws other
  than `RuleLaw`, no module under `src/skcomplex/core` imports `skcomplex.systems`, and the roots of `systems/general`
  and of every `systems/<field>/common` hold no module but `__init__.py`

#### Scenario: One text of the rule
- **WHEN** a reviewer reads `openspec/constitution.md`, `CONTRIBUTING.md`, and the extension page of the user guide
- **THEN** each states the same ordered test as this requirement, in the word component, and each says that analyses
  follow it by their users
