## MODIFIED Requirements

### Requirement: The shared core never imports a formalism
No module of `skcomplex.core.components` or `skcomplex.core.engine` SHALL import a formalism, and a system built only from the core
SHALL compile and simulate with the engine and no formalism imported.

#### Scenario: Shared core alone
- **WHEN** a user builds, compiles, and simulates a logistic growth system written with `Element`, `Rate`, and `CompositeSystem`
- **THEN** the run succeeds and no module of `skcomplex.formalisms` is loaded

#### Scenario: Planted import
- **WHEN** a module of `skcomplex.core.engine.compile` imports from `skcomplex.formalisms.lattices`
- **THEN** the import contracts of the checks session fail and name the module
